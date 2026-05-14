"""Custom CUDA first-hit renderer for ShapeNBV voxel GT grids.

The Open3D renderer casts rays against the triangle mesh on CPU and then
copies depth/alpha back to CUDA.  For ShapeNBV training, the downstream
state/reward only needs the first occupied reward-grid voxel per pixel.
This module therefore raycasts directly through the preprocessed
``grid_gt`` occupancy grid and returns:

    target_idx: [N, R, 3] int64 first occupied voxel per ray
    depth:      [N, R]    float distance along the normalized ray
    hit_mask:   [N, R]    bool/uint8 hit flag

It is intentionally optional.  CPU development and CUDA images without a
compiler fall back to a small PyTorch reference in the caller.
"""
from __future__ import annotations

import os
import threading
import warnings
from typing import Optional, Tuple

import torch


_EXTENSION_NAME = "shapenbv_cuda_voxel_renderer_v1"
_EXTENSION = None
_EXTENSION_ERROR: Optional[str] = None
_LOAD_LOCK = threading.Lock()
_WARNED_LOAD_ERROR = False


_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> voxel_first_hit_cuda(
    torch::Tensor grid,
    torch::Tensor bbox_min,
    torch::Tensor voxel_size,
    torch::Tensor eyes,
    torch::Tensor rays_world,
    double occ_threshold);

std::vector<torch::Tensor> voxel_first_hit(
    torch::Tensor grid,
    torch::Tensor bbox_min,
    torch::Tensor voxel_size,
    torch::Tensor eyes,
    torch::Tensor rays_world,
    double occ_threshold) {
  TORCH_CHECK(grid.is_cuda(), "grid must be CUDA");
  TORCH_CHECK(bbox_min.is_cuda(), "bbox_min must be CUDA");
  TORCH_CHECK(voxel_size.is_cuda(), "voxel_size must be CUDA");
  TORCH_CHECK(eyes.is_cuda(), "eyes must be CUDA");
  TORCH_CHECK(rays_world.is_cuda(), "rays_world must be CUDA");
  TORCH_CHECK(grid.dtype() == torch::kFloat32, "grid must be float32");
  TORCH_CHECK(bbox_min.dtype() == torch::kFloat32, "bbox_min must be float32");
  TORCH_CHECK(voxel_size.dtype() == torch::kFloat32, "voxel_size must be float32");
  TORCH_CHECK(eyes.dtype() == torch::kFloat32, "eyes must be float32");
  TORCH_CHECK(rays_world.dtype() == torch::kFloat32, "rays_world must be float32");
  TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
  TORCH_CHECK(bbox_min.is_contiguous(), "bbox_min must be contiguous");
  TORCH_CHECK(voxel_size.is_contiguous(), "voxel_size must be contiguous");
  TORCH_CHECK(eyes.is_contiguous(), "eyes must be contiguous");
  TORCH_CHECK(rays_world.is_contiguous(), "rays_world must be contiguous");
  TORCH_CHECK(grid.dim() == 4, "grid must be [N, G, G, G]");
  TORCH_CHECK(grid.size(1) == grid.size(2) && grid.size(2) == grid.size(3),
              "grid must be cubic");
  TORCH_CHECK(bbox_min.sizes() == torch::IntArrayRef({grid.size(0), 3}),
              "bbox_min must be [N, 3]");
  TORCH_CHECK(voxel_size.sizes() == torch::IntArrayRef({grid.size(0), 3}),
              "voxel_size must be [N, 3]");
  TORCH_CHECK(eyes.sizes() == torch::IntArrayRef({grid.size(0), 3}),
              "eyes must be [N, 3]");
  TORCH_CHECK(rays_world.dim() == 3 && rays_world.size(0) == grid.size(0) &&
              rays_world.size(2) == 3, "rays_world must be [N, R, 3]");

  return voxel_first_hit_cuda(
      grid, bbox_min, voxel_size, eyes, rays_world, occ_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_first_hit", &voxel_first_hit,
        "First-hit raycast through [N,G,G,G] voxel occupancy grids (CUDA)");
}
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ int clamp_i(const int v, const int lo, const int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int sign_from_float(const float v, const float eps) {
  return (v > eps) - (v < -eps);
}

__device__ __forceinline__ bool update_ray_aabb_axis(
    const float o,
    const float d,
    const float lo,
    const float hi,
    float* tmin,
    float* tmax) {
  const float eps_dir = 1e-9f;
  if (fabsf(d) < eps_dir) {
    return o >= lo && o <= hi;
  }
  const float inv = 1.0f / d;
  float t0 = (lo - o) * inv;
  float t1 = (hi - o) * inv;
  if (t0 > t1) {
    const float tmp = t0;
    t0 = t1;
    t1 = tmp;
  }
  *tmin = fmaxf(*tmin, t0);
  *tmax = fminf(*tmax, t1);
  return *tmax >= *tmin;
}

__global__ void voxel_first_hit_kernel(
    const float* __restrict__ grid,
    const float* __restrict__ bbox_min,
    const float* __restrict__ voxel_size,
    const float* __restrict__ eyes,
    const float* __restrict__ rays_world,
    int64_t* __restrict__ target_idx,
    float* __restrict__ depth,
    uint8_t* __restrict__ hit_mask,
    const int64_t num_envs,
    const int64_t num_rays,
    const int G,
    const float occ_threshold) {
  const int64_t global_ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_envs * num_rays;
  if (global_ray >= total) {
    return;
  }

  const int64_t env = global_ray / num_rays;
  const int64_t ray = global_ray - env * num_rays;
  const int64_t ray_base = (env * num_rays + ray) * 3;
  const int64_t env3 = env * 3;

  const float ox = eyes[env3 + 0];
  const float oy = eyes[env3 + 1];
  const float oz = eyes[env3 + 2];
  const float dx = rays_world[ray_base + 0];
  const float dy = rays_world[ray_base + 1];
  const float dz = rays_world[ray_base + 2];

  const float bmin_x = bbox_min[env3 + 0];
  const float bmin_y = bbox_min[env3 + 1];
  const float bmin_z = bbox_min[env3 + 2];
  const float vs_x = voxel_size[env3 + 0];
  const float vs_y = voxel_size[env3 + 1];
  const float vs_z = voxel_size[env3 + 2];
  const float bmax_x = bmin_x + static_cast<float>(G) * vs_x;
  const float bmax_y = bmin_y + static_cast<float>(G) * vs_y;
  const float bmax_z = bmin_z + static_cast<float>(G) * vs_z;

  const int64_t out3 = global_ray * 3;
  target_idx[out3 + 0] = 0;
  target_idx[out3 + 1] = 0;
  target_idx[out3 + 2] = 0;
  depth[global_ray] = 0.0f;
  hit_mask[global_ray] = static_cast<uint8_t>(0);

  float tmin = 0.0f;
  float tmax = 3.402823466e+38F;
  const float eps_dir = 1e-9f;

  if (!update_ray_aabb_axis(ox, dx, bmin_x, bmax_x, &tmin, &tmax)) return;
  if (!update_ray_aabb_axis(oy, dy, bmin_y, bmax_y, &tmin, &tmax)) return;
  if (!update_ray_aabb_axis(oz, dz, bmin_z, bmax_z, &tmin, &tmax)) return;
  if (tmax < fmaxf(tmin, 0.0f)) return;

  const float min_vs = fminf(vs_x, fminf(vs_y, vs_z));
  const float eps_t = fmaxf(min_vs * 1e-4f, 1e-7f);
  float current_t = fmaxf(tmin, 0.0f);
  const float start_t = fminf(current_t + eps_t, tmax);
  const float px = ox + start_t * dx;
  const float py = oy + start_t * dy;
  const float pz = oz + start_t * dz;

  int ix = clamp_i(static_cast<int>(floorf((px - bmin_x) / vs_x)), 0, G - 1);
  int iy = clamp_i(static_cast<int>(floorf((py - bmin_y) / vs_y)), 0, G - 1);
  int iz = clamp_i(static_cast<int>(floorf((pz - bmin_z) / vs_z)), 0, G - 1);

  const int sx = sign_from_float(dx, eps_dir);
  const int sy = sign_from_float(dy, eps_dir);
  const int sz = sign_from_float(dz, eps_dir);
  const float inf = 3.402823466e+38F;

  float t_max_x = inf;
  float t_max_y = inf;
  float t_max_z = inf;
  float t_delta_x = inf;
  float t_delta_y = inf;
  float t_delta_z = inf;

  if (sx != 0) {
    const float boundary = bmin_x + (sx > 0 ? (static_cast<float>(ix + 1) * vs_x)
                                            : (static_cast<float>(ix) * vs_x));
    t_max_x = (boundary - ox) / dx;
    t_delta_x = vs_x / fabsf(dx);
  }
  if (sy != 0) {
    const float boundary = bmin_y + (sy > 0 ? (static_cast<float>(iy + 1) * vs_y)
                                            : (static_cast<float>(iy) * vs_y));
    t_max_y = (boundary - oy) / dy;
    t_delta_y = vs_y / fabsf(dy);
  }
  if (sz != 0) {
    const float boundary = bmin_z + (sz > 0 ? (static_cast<float>(iz + 1) * vs_z)
                                            : (static_cast<float>(iz) * vs_z));
    t_max_z = (boundary - oz) / dz;
    t_delta_z = vs_z / fabsf(dz);
  }

  // A ray through a G^3 box crosses at most 3G voxel slabs.  The extra 3
  // covers boundary/tie cases after the small entry epsilon above.
  const int max_steps = 3 * G + 3;
  const float tie_eps = 1e-7f;
  for (int step = 0; step < max_steps; ++step) {
    if (ix < 0 || ix >= G || iy < 0 || iy >= G || iz < 0 || iz >= G) {
      return;
    }
    if (current_t > tmax + eps_t) {
      return;
    }

    const int64_t flat = (((env * G + ix) * G + iy) * G + iz);
    if (grid[flat] > occ_threshold) {
      target_idx[out3 + 0] = static_cast<int64_t>(ix);
      target_idx[out3 + 1] = static_cast<int64_t>(iy);
      target_idx[out3 + 2] = static_cast<int64_t>(iz);
      depth[global_ray] = fmaxf(current_t, 0.0f);
      hit_mask[global_ray] = static_cast<uint8_t>(1);
      return;
    }

    float next_t = fminf(t_max_x, fminf(t_max_y, t_max_z));
    if (next_t > tmax + eps_t) {
      return;
    }
    const bool adv_x = t_max_x <= next_t + tie_eps;
    const bool adv_y = t_max_y <= next_t + tie_eps;
    const bool adv_z = t_max_z <= next_t + tie_eps;
    if (adv_x) {
      ix += sx;
      t_max_x += t_delta_x;
    }
    if (adv_y) {
      iy += sy;
      t_max_y += t_delta_y;
    }
    if (adv_z) {
      iz += sz;
      t_max_z += t_delta_z;
    }
    current_t = next_t;
  }
}

std::vector<torch::Tensor> voxel_first_hit_cuda(
    torch::Tensor grid,
    torch::Tensor bbox_min,
    torch::Tensor voxel_size,
    torch::Tensor eyes,
    torch::Tensor rays_world,
    double occ_threshold) {
  const int64_t N = grid.size(0);
  const int64_t G = grid.size(1);
  const int64_t R = rays_world.size(1);
  auto idx_options = torch::TensorOptions().dtype(torch::kInt64).device(grid.device());
  auto f_options = torch::TensorOptions().dtype(torch::kFloat32).device(grid.device());
  auto u8_options = torch::TensorOptions().dtype(torch::kUInt8).device(grid.device());
  torch::Tensor target_idx = torch::empty({N, R, 3}, idx_options);
  torch::Tensor depth = torch::empty({N, R}, f_options);
  torch::Tensor hit_mask = torch::empty({N, R}, u8_options);

  const int64_t total = N * R;
  if (total == 0) {
    return {target_idx, depth, hit_mask};
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  voxel_first_hit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      grid.data_ptr<float>(),
      bbox_min.data_ptr<float>(),
      voxel_size.data_ptr<float>(),
      eyes.data_ptr<float>(),
      rays_world.data_ptr<float>(),
      target_idx.data_ptr<int64_t>(),
      depth.data_ptr<float>(),
      hit_mask.data_ptr<uint8_t>(),
      N,
      R,
      static_cast<int>(G),
      static_cast<float>(occ_threshold));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {target_idx, depth, hit_mask};
}
"""


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR, _WARNED_LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        return None
    if os.environ.get("SHAPENBV_DISABLE_CUDA_VOXEL_RENDERER", "0") == "1":
        _EXTENSION_ERROR = "disabled by SHAPENBV_DISABLE_CUDA_VOXEL_RENDERER=1"
        return None
    if not torch.cuda.is_available():
        _EXTENSION_ERROR = "torch.cuda is not available"
        return None

    with _LOAD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            return None
        try:
            from torch.utils.cpp_extension import load_inline

            if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                major, minor = torch.cuda.get_device_capability()
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            _EXTENSION = load_inline(
                name=_EXTENSION_NAME,
                cpp_sources=[_CPP_SRC],
                cuda_sources=[_CUDA_SRC],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=bool(int(os.environ.get("SHAPENBV_VERBOSE_CUDA_BUILD", "0"))),
            )
            return _EXTENSION
        except Exception as exc:  # pragma: no cover - depends on CUDA toolchain.
            _EXTENSION_ERROR = repr(exc)
            if not _WARNED_LOAD_ERROR:
                warnings.warn(
                    "ShapeNBV custom CUDA voxel renderer unavailable; "
                    f"falling back to the requested renderer. Reason: {_EXTENSION_ERROR}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED_LOAD_ERROR = True
            return None


def last_extension_error() -> Optional[str]:
    return _EXTENSION_ERROR


def voxel_first_hit_cuda(
    grid: torch.Tensor,
    bbox_min: torch.Tensor,
    voxel_size: torch.Tensor,
    eyes: torch.Tensor,
    rays_world: torch.Tensor,
    *,
    occ_threshold: float = 0.5,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return first occupied voxel per ray, or ``None`` if CUDA is unavailable."""
    if not (
        grid.is_cuda
        and bbox_min.is_cuda
        and voxel_size.is_cuda
        and eyes.is_cuda
        and rays_world.is_cuda
    ):
        return None
    ext = _load_extension()
    if ext is None:
        return None
    out = ext.voxel_first_hit(
        grid.to(dtype=torch.float32).contiguous(),
        bbox_min.to(dtype=torch.float32).contiguous(),
        voxel_size.to(dtype=torch.float32).contiguous(),
        eyes.to(dtype=torch.float32).contiguous(),
        rays_world.to(dtype=torch.float32).contiguous(),
        float(occ_threshold),
    )
    target_idx, depth, hit_mask_u8 = out
    return target_idx, depth, hit_mask_u8.to(dtype=torch.bool)


def voxel_first_hit_reference(
    grid: torch.Tensor,
    bbox_min: torch.Tensor,
    voxel_size: torch.Tensor,
    eyes: torch.Tensor,
    rays_world: torch.Tensor,
    *,
    occ_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small CPU/GPU reference implementation for tests and fallback.

    This intentionally uses Python loops over rays, so it is only suitable
    for tiny unit tests or as a correctness oracle.
    """
    device = grid.device
    grid_cpu = grid.detach().cpu().float()
    bbox_cpu = bbox_min.detach().cpu().float()
    vs_cpu = voxel_size.detach().cpu().float()
    eyes_cpu = eyes.detach().cpu().float()
    rays_cpu = rays_world.detach().cpu().float()
    N, G = int(grid_cpu.shape[0]), int(grid_cpu.shape[1])
    R = int(rays_cpu.shape[1])
    target = torch.zeros(N, R, 3, dtype=torch.long)
    depth = torch.zeros(N, R, dtype=torch.float32)
    hit = torch.zeros(N, R, dtype=torch.bool)

    for n in range(N):
        lo = bbox_cpu[n]
        vs = vs_cpu[n]
        hi = lo + float(G) * vs
        eye = eyes_cpu[n]
        min_vs = float(vs.min().item())
        for r in range(R):
            d = rays_cpu[n, r]
            tmin = 0.0
            tmax = float("inf")
            ok = True
            for a in range(3):
                da = float(d[a].item())
                oa = float(eye[a].item())
                if abs(da) < 1e-9:
                    if oa < float(lo[a].item()) or oa > float(hi[a].item()):
                        ok = False
                        break
                    continue
                t0 = (float(lo[a].item()) - oa) / da
                t1 = (float(hi[a].item()) - oa) / da
                if t0 > t1:
                    t0, t1 = t1, t0
                tmin = max(tmin, t0)
                tmax = min(tmax, t1)
                if tmax < tmin:
                    ok = False
                    break
            if not ok or tmax < max(tmin, 0.0):
                continue

            t = max(tmin, 0.0)
            p = eye + min(t + max(min_vs * 1e-4, 1e-7), tmax) * d
            idx = torch.floor((p - lo) / vs).long().clamp(0, G - 1)
            step = torch.sign(d).long()
            inf = float("inf")
            t_max = torch.full((3,), inf)
            t_delta = torch.full((3,), inf)
            for a in range(3):
                if step[a] == 0:
                    continue
                boundary = lo[a] + (idx[a] + (1 if step[a] > 0 else 0)) * vs[a]
                t_max[a] = (boundary - eye[a]) / d[a]
                t_delta[a] = vs[a] / d[a].abs()

            for _ in range(3 * G + 3):
                x, y, z = int(idx[0]), int(idx[1]), int(idx[2])
                if not (0 <= x < G and 0 <= y < G and 0 <= z < G):
                    break
                if t > tmax + max(min_vs * 1e-4, 1e-7):
                    break
                if float(grid_cpu[n, x, y, z].item()) > occ_threshold:
                    target[n, r] = idx
                    depth[n, r] = float(max(t, 0.0))
                    hit[n, r] = True
                    break
                next_t = float(t_max.min().item())
                if next_t > tmax + max(min_vs * 1e-4, 1e-7):
                    break
                adv = t_max <= (next_t + 1e-7)
                idx = idx + adv.long() * step
                t_max = t_max + adv.float() * t_delta
                t = next_t

    return target.to(device), depth.to(device), hit.to(device)


def warmup_cuda_voxel_renderer(device: Optional[torch.device | str] = None) -> bool:
    """Compile/load the extension and launch one tiny first-hit render."""
    if device is None:
        device = torch.device("cuda")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    grid = torch.zeros(1, 4, 4, 4, dtype=torch.float32, device=device)
    grid[0, 2, 2, 2] = 1.0
    bbox_min = torch.zeros(1, 3, dtype=torch.float32, device=device)
    voxel_size = torch.full((1, 3), 0.25, dtype=torch.float32, device=device)
    eyes = torch.tensor([[-0.5, 0.625, 0.625]], dtype=torch.float32, device=device)
    rays = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32, device=device)
    out = voxel_first_hit_cuda(grid, bbox_min, voxel_size, eyes, rays)
    return out is not None
