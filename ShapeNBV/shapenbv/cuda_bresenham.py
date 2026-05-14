"""Custom CUDA Bresenham scatter for ShapeNBV free-space updates.

This is a small torch CUDA extension with the same public shape as the
Triton implementation in :mod:`shapenbv.triton_bresenham`.  It follows
GenNBV's PyCUDA design choice that one CUDA thread owns one ray, but it
scatters directly into the per-step uint8 free mask instead of materializing
all trajectory points first.

The module is intentionally optional.  Local CPU development and CUDA
images without a compiler fall back to Triton/PyTorch callers.
"""
from __future__ import annotations

import os
import threading
import warnings
from typing import Optional

import torch


_EXTENSION_NAME = "shapenbv_cuda_bresenham_v4"
_EXTENSION = None
_EXTENSION_ERROR: Optional[str] = None
_LOAD_LOCK = threading.Lock()
_WARNED_LOAD_ERROR = False


_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

void bresenham_scatter_mask_cuda(
    torch::Tensor env_ids,
    torch::Tensor sources,
    torch::Tensor targets,
    torch::Tensor out_mask,
    int64_t grid_size,
    int64_t max_steps,
    bool include_source,
    bool include_target);

void empty_ray_pairs_cuda(
    torch::Tensor eyes,
    torch::Tensor rays_world,
    torch::Tensor hit_pixel_mask,
    torch::Tensor bbox_min,
    torch::Tensor voxel_size,
    torch::Tensor valid,
    torch::Tensor sources,
    torch::Tensor targets,
    int64_t grid_size);

torch::Tensor bresenham_scatter_mask(
    torch::Tensor env_ids,
    torch::Tensor sources,
    torch::Tensor targets,
    torch::Tensor out_mask,
    int64_t grid_size,
    int64_t max_steps,
    bool include_source,
    bool include_target) {
  TORCH_CHECK(env_ids.is_cuda(), "env_ids must be CUDA");
  TORCH_CHECK(sources.is_cuda(), "sources must be CUDA");
  TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
  TORCH_CHECK(out_mask.is_cuda(), "out_mask must be CUDA");
  TORCH_CHECK(env_ids.dtype() == torch::kInt64, "env_ids must be int64");
  TORCH_CHECK(sources.dtype() == torch::kInt64, "sources must be int64");
  TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
  TORCH_CHECK(out_mask.dtype() == torch::kUInt8, "out_mask must be uint8");
  TORCH_CHECK(env_ids.is_contiguous(), "env_ids must be contiguous");
  TORCH_CHECK(sources.is_contiguous(), "sources must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(out_mask.is_contiguous(), "out_mask must be contiguous");
  TORCH_CHECK(sources.dim() == 2 && sources.size(1) == 3, "sources must be [N, 3]");
  TORCH_CHECK(targets.dim() == 2 && targets.size(1) == 3, "targets must be [N, 3]");
  TORCH_CHECK(env_ids.numel() == targets.size(0), "env_ids length must match targets");
  TORCH_CHECK(sources.size(0) == targets.size(0), "sources length must match targets");
  TORCH_CHECK(grid_size > 0, "grid_size must be positive");
  TORCH_CHECK(max_steps >= 0, "max_steps must be non-negative");

  bresenham_scatter_mask_cuda(
      env_ids,
      sources,
      targets,
      out_mask,
      grid_size,
      max_steps,
      include_source,
      include_target);
  return out_mask;
}

std::vector<torch::Tensor> empty_ray_pairs(
    torch::Tensor eyes,
    torch::Tensor rays_world,
    torch::Tensor hit_pixel_mask,
    torch::Tensor bbox_min,
    torch::Tensor voxel_size,
    int64_t grid_size) {
  TORCH_CHECK(eyes.is_cuda(), "eyes must be CUDA");
  TORCH_CHECK(rays_world.is_cuda(), "rays_world must be CUDA");
  TORCH_CHECK(hit_pixel_mask.is_cuda(), "hit_pixel_mask must be CUDA");
  TORCH_CHECK(bbox_min.is_cuda(), "bbox_min must be CUDA");
  TORCH_CHECK(voxel_size.is_cuda(), "voxel_size must be CUDA");
  TORCH_CHECK(eyes.dtype() == torch::kFloat32, "eyes must be float32");
  TORCH_CHECK(rays_world.dtype() == torch::kFloat32, "rays_world must be float32");
  TORCH_CHECK(hit_pixel_mask.dtype() == torch::kBool, "hit_pixel_mask must be bool");
  TORCH_CHECK(bbox_min.dtype() == torch::kFloat32, "bbox_min must be float32");
  TORCH_CHECK(voxel_size.dtype() == torch::kFloat32, "voxel_size must be float32");
  TORCH_CHECK(eyes.is_contiguous(), "eyes must be contiguous");
  TORCH_CHECK(rays_world.is_contiguous(), "rays_world must be contiguous");
  TORCH_CHECK(hit_pixel_mask.is_contiguous(), "hit_pixel_mask must be contiguous");
  TORCH_CHECK(bbox_min.is_contiguous(), "bbox_min must be contiguous");
  TORCH_CHECK(voxel_size.is_contiguous(), "voxel_size must be contiguous");
  TORCH_CHECK(eyes.dim() == 2 && eyes.size(1) == 3, "eyes must be [N, 3]");
  TORCH_CHECK(rays_world.dim() == 3 && rays_world.size(0) == eyes.size(0) &&
              rays_world.size(2) == 3, "rays_world must be [N, R, 3]");
  TORCH_CHECK(hit_pixel_mask.dim() == 2 && hit_pixel_mask.size(0) == eyes.size(0) &&
              hit_pixel_mask.size(1) == rays_world.size(1), "hit_pixel_mask must be [N, R]");
  TORCH_CHECK(bbox_min.dim() == 2 && bbox_min.size(0) == eyes.size(0) &&
              bbox_min.size(1) == 3, "bbox_min must be [N, 3]");
  TORCH_CHECK(voxel_size.dim() == 2 && voxel_size.size(0) == eyes.size(0) &&
              voxel_size.size(1) == 3, "voxel_size must be [N, 3]");
  TORCH_CHECK(grid_size > 0, "grid_size must be positive");

  const auto N = eyes.size(0);
  const auto R = rays_world.size(1);
  auto valid = torch::empty({N, R}, torch::TensorOptions().dtype(torch::kBool).device(eyes.device()));
  auto sources = torch::empty({N, R, 3}, torch::TensorOptions().dtype(torch::kInt64).device(eyes.device()));
  auto targets = torch::empty({N, R, 3}, torch::TensorOptions().dtype(torch::kInt64).device(eyes.device()));

  empty_ray_pairs_cuda(
      eyes,
      rays_world,
      hit_pixel_mask,
      bbox_min,
      voxel_size,
      valid,
      sources,
      targets,
      grid_size);
  return {valid, sources, targets};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bresenham_scatter_mask", &bresenham_scatter_mask,
        "Scatter exact 3D Bresenham paths into a uint8 mask (CUDA)");
  m.def("empty_ray_pairs", &empty_ray_pairs,
        "Compute miss-ray grid entry/exit voxel pairs for a batched ray grid (CUDA)");
}
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

__device__ __forceinline__ int sign_i(const int v) {
  return (v > 0) - (v < 0);
}

__device__ __forceinline__ void store_if_in_bounds(
    uint8_t* __restrict__ out_mask,
    const int64_t env,
    const int x,
    const int y,
    const int z,
    const int G) {
  if (x >= 0 && x < G && y >= 0 && y < G && z >= 0 && z < G) {
    const int64_t flat = (((env * G + x) * G + y) * G + z);
    out_mask[flat] = static_cast<uint8_t>(1);
  }
}

__global__ void bresenham_scatter_kernel(
    const int64_t* __restrict__ env_ids,
    const int64_t* __restrict__ sources,
    const int64_t* __restrict__ targets,
    uint8_t* __restrict__ out_mask,
    const int64_t num_rays,
    const int G,
    const int max_steps,
    const bool include_source,
    const bool include_target) {
  const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ray >= num_rays) {
    return;
  }

  const int64_t env = env_ids[ray];
  const int64_t base = ray * 3;
  int x = static_cast<int>(sources[base + 0]);
  int y = static_cast<int>(sources[base + 1]);
  int z = static_cast<int>(sources[base + 2]);
  const int tx = static_cast<int>(targets[base + 0]);
  const int ty = static_cast<int>(targets[base + 1]);
  const int tz = static_cast<int>(targets[base + 2]);

  const int dx0 = tx - x;
  const int dy0 = ty - y;
  const int dz0 = tz - z;
  const int dx = abs(dx0);
  const int dy = abs(dy0);
  const int dz = abs(dz0);
  const int sx = sign_i(dx0);
  const int sy = sign_i(dy0);
  const int sz = sign_i(dz0);

  int drive_axis = 0;
  int da = dx;
  int sa = sx;
  int db = dy;
  int dc = dz;
  int sb = sy;
  int sc = sz;

  if (!(dx >= dy && dx >= dz)) {
    if (dy >= dz) {
      drive_axis = 1;
      da = dy;
      sa = sy;
      db = dz;
      dc = dx;
      sb = sz;
      sc = sx;
    } else {
      drive_axis = 2;
      da = dz;
      sa = sz;
      db = dx;
      dc = dy;
      sb = sx;
      sc = sy;
    }
  }

  // other_axes = [(drive_axis + 1) % 3, (drive_axis + 2) % 3].
  const int b_axis = (drive_axis + 1) % 3;
  const int c_axis = (drive_axis + 2) % 3;
  int p1 = 2 * db - da;
  int p2 = 2 * dc - da;

  const bool same_as_target = (x == tx && y == ty && z == tz);
  if (include_source && (include_target || !same_as_target)) {
    store_if_in_bounds(out_mask, env, x, y, z, G);
  } else if (!include_source && include_target && same_as_target) {
    // Match bresenham3D_strict's zero-length include_target behavior.
    store_if_in_bounds(out_mask, env, x, y, z, G);
  }

  const int loop_steps = da < max_steps ? da : max_steps;
  #pragma unroll 1
  for (int step = 0; step < loop_steps; ++step) {
    if (p1 >= 0 && da > 0) {
      if (b_axis == 0) {
        x += sb;
      } else if (b_axis == 1) {
        y += sb;
      } else {
        z += sb;
      }
      p1 -= 2 * da;
    }
    if (p2 >= 0 && da > 0) {
      if (c_axis == 0) {
        x += sc;
      } else if (c_axis == 1) {
        y += sc;
      } else {
        z += sc;
      }
      p2 -= 2 * da;
    }

    if (drive_axis == 0) {
      x += sa;
    } else if (drive_axis == 1) {
      y += sa;
    } else {
      z += sa;
    }
    p1 += 2 * db;
    p2 += 2 * dc;

    const bool hit_target = (x == tx && y == ty && z == tz);
    if (include_target || !hit_target) {
      store_if_in_bounds(out_mask, env, x, y, z, G);
    }
  }
}

__device__ __forceinline__ int clamp_i(const int v, const int lo, const int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float safe_dir(const float d) {
  constexpr float eps = 1.0e-9f;
  if (fabsf(d) < eps) {
    return d >= 0.0f ? eps : -eps;
  }
  return d;
}

__global__ void empty_ray_pairs_kernel(
    const float* __restrict__ eyes,
    const float* __restrict__ rays_world,
    const bool* __restrict__ hit_pixel_mask,
    const float* __restrict__ bbox_min,
    const float* __restrict__ voxel_size,
    bool* __restrict__ valid,
    int64_t* __restrict__ sources,
    int64_t* __restrict__ targets,
    const int64_t total_rays,
    const int R,
    const int G) {
  const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (flat >= total_rays) {
    return;
  }

  valid[flat] = false;
  if (hit_pixel_mask[flat]) {
    return;
  }

  const int env = static_cast<int>(flat / R);
  const int64_t ray_base = flat * 3;
  const int64_t env_base = static_cast<int64_t>(env) * 3;

  const float ex = eyes[env_base + 0];
  const float ey = eyes[env_base + 1];
  const float ez = eyes[env_base + 2];
  const float raw_dx = rays_world[ray_base + 0];
  const float raw_dy = rays_world[ray_base + 1];
  const float raw_dz = rays_world[ray_base + 2];
  const float dx = safe_dir(raw_dx);
  const float dy = safe_dir(raw_dy);
  const float dz = safe_dir(raw_dz);

  const float bx0 = bbox_min[env_base + 0];
  const float by0 = bbox_min[env_base + 1];
  const float bz0 = bbox_min[env_base + 2];
  const float vx = voxel_size[env_base + 0];
  const float vy = voxel_size[env_base + 1];
  const float vz = voxel_size[env_base + 2];
  const float bx1 = bx0 + static_cast<float>(G) * vx;
  const float by1 = by0 + static_cast<float>(G) * vy;
  const float bz1 = bz0 + static_cast<float>(G) * vz;

  const float tx0 = (bx0 - ex) / dx;
  const float tx1 = (bx1 - ex) / dx;
  const float ty0 = (by0 - ey) / dy;
  const float ty1 = (by1 - ey) / dy;
  const float tz0 = (bz0 - ez) / dz;
  const float tz1 = (bz1 - ez) / dz;

  const float tmin_x = fminf(tx0, tx1);
  const float tmax_x = fmaxf(tx0, tx1);
  const float tmin_y = fminf(ty0, ty1);
  const float tmax_y = fmaxf(ty0, ty1);
  const float tmin_z = fminf(tz0, tz1);
  const float tmax_z = fmaxf(tz0, tz1);
  const float t_near = fmaxf(fmaxf(tmin_x, tmin_y), tmin_z);
  const float t_far = fminf(fminf(tmax_x, tmax_y), tmax_z);
  const float t_start = fmaxf(t_near, 0.0f);
  if (!(t_far > t_start + 1.0e-6f)) {
    return;
  }

  const float vmin = fminf(fminf(vx, vy), vz);
  const float voxel_eps = vmin * 0.25f;
  const float t_entry = (t_near <= 0.0f) ? 0.0f : (t_start + voxel_eps);
  const float t_exit = fmaxf(t_far - voxel_eps, t_entry);

  const float sxp = __fadd_rn(ex, __fmul_rn(t_entry, raw_dx));
  const float syp = __fadd_rn(ey, __fmul_rn(t_entry, raw_dy));
  const float szp = __fadd_rn(ez, __fmul_rn(t_entry, raw_dz));
  const float exp = __fadd_rn(ex, __fmul_rn(t_exit, raw_dx));
  const float eyp = __fadd_rn(ey, __fmul_rn(t_exit, raw_dy));
  const float ezp = __fadd_rn(ez, __fmul_rn(t_exit, raw_dz));

  int sx = static_cast<int>(floorf((sxp - bx0) / vx));
  int sy = static_cast<int>(floorf((syp - by0) / vy));
  int sz = static_cast<int>(floorf((szp - bz0) / vz));
  int tx = static_cast<int>(floorf((exp - bx0) / vx));
  int ty = static_cast<int>(floorf((eyp - by0) / vy));
  int tz = static_cast<int>(floorf((ezp - bz0) / vz));
  sx = clamp_i(sx, 0, G - 1);
  sy = clamp_i(sy, 0, G - 1);
  sz = clamp_i(sz, 0, G - 1);
  tx = clamp_i(tx, 0, G - 1);
  ty = clamp_i(ty, 0, G - 1);
  tz = clamp_i(tz, 0, G - 1);

  sources[ray_base + 0] = static_cast<int64_t>(sx);
  sources[ray_base + 1] = static_cast<int64_t>(sy);
  sources[ray_base + 2] = static_cast<int64_t>(sz);
  targets[ray_base + 0] = static_cast<int64_t>(tx);
  targets[ray_base + 1] = static_cast<int64_t>(ty);
  targets[ray_base + 2] = static_cast<int64_t>(tz);
  valid[flat] = true;
}

void bresenham_scatter_mask_cuda(
    torch::Tensor env_ids,
    torch::Tensor sources,
    torch::Tensor targets,
    torch::Tensor out_mask,
    int64_t grid_size,
    int64_t max_steps,
    bool include_source,
    bool include_target) {
  const int64_t num_rays = targets.size(0);
  if (num_rays == 0) {
    return;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((num_rays + threads - 1) / threads);
  bresenham_scatter_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      env_ids.data_ptr<int64_t>(),
      sources.data_ptr<int64_t>(),
      targets.data_ptr<int64_t>(),
      out_mask.data_ptr<uint8_t>(),
      num_rays,
      static_cast<int>(grid_size),
      static_cast<int>(max_steps),
      include_source,
      include_target);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void empty_ray_pairs_cuda(
    torch::Tensor eyes,
    torch::Tensor rays_world,
    torch::Tensor hit_pixel_mask,
    torch::Tensor bbox_min,
    torch::Tensor voxel_size,
    torch::Tensor valid,
    torch::Tensor sources,
    torch::Tensor targets,
    int64_t grid_size) {
  const int64_t N = eyes.size(0);
  const int64_t R = rays_world.size(1);
  const int64_t total_rays = N * R;
  if (total_rays == 0) {
    return;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total_rays + threads - 1) / threads);
  empty_ray_pairs_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      eyes.data_ptr<float>(),
      rays_world.data_ptr<float>(),
      hit_pixel_mask.data_ptr<bool>(),
      bbox_min.data_ptr<float>(),
      voxel_size.data_ptr<float>(),
      valid.data_ptr<bool>(),
      sources.data_ptr<int64_t>(),
      targets.data_ptr<int64_t>(),
      total_rays,
      static_cast<int>(R),
      static_cast<int>(grid_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR, _WARNED_LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        return None
    if os.environ.get("SHAPENBV_DISABLE_CUDA_BRESENHAM", "0") == "1":
        _EXTENSION_ERROR = "disabled by SHAPENBV_DISABLE_CUDA_BRESENHAM=1"
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
                    "ShapeNBV custom CUDA Bresenham extension unavailable; "
                    f"falling back to Triton/PyTorch. Reason: {_EXTENSION_ERROR}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED_LOAD_ERROR = True
            return None


def last_extension_error() -> Optional[str]:
    """Return the cached CUDA extension load error, if any."""
    return _EXTENSION_ERROR


def scatter_bresenham3d_to_mask_cuda(
    env_ids: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_envs: int,
    grid_size: int,
    include_source: bool,
    include_target: bool,
    out_mask: Optional[torch.Tensor] = None,
    max_steps: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Scatter exact Bresenham paths into ``[num_envs, G, G, G]`` uint8 mask.

    Returns ``None`` when the CUDA extension cannot be used.  Callers can
    then fall back to Triton or the pure PyTorch reference.
    """
    if not (env_ids.is_cuda and sources.is_cuda and targets.is_cuda):
        return None
    ext = _load_extension()
    if ext is None:
        return None

    if targets.numel() == 0:
        if out_mask is None:
            G0 = int(grid_size)
            return torch.zeros(
                (int(num_envs), G0, G0, G0),
                dtype=torch.uint8,
                device=targets.device,
            )
        return out_mask

    device = targets.device
    env_ids = env_ids.to(dtype=torch.int64, device=device).contiguous().view(-1)
    sources = sources.to(dtype=torch.int64, device=device).contiguous().view(-1, 3)
    targets = targets.to(dtype=torch.int64, device=device).contiguous().view(-1, 3)
    if sources.shape != targets.shape:
        raise ValueError(f"sources {tuple(sources.shape)} != targets {tuple(targets.shape)}")
    if env_ids.shape[0] != targets.shape[0]:
        raise ValueError(f"env_ids {tuple(env_ids.shape)} incompatible with targets {tuple(targets.shape)}")

    G = int(grid_size)
    if out_mask is None:
        out_mask = torch.zeros((int(num_envs), G, G, G), dtype=torch.uint8, device=device)
    else:
        if out_mask.dtype != torch.uint8 or not out_mask.is_cuda:
            raise ValueError("out_mask must be a CUDA uint8 tensor")
        if not out_mask.is_contiguous():
            raise ValueError("out_mask must be contiguous")

    steps = int(max_steps if max_steps is not None else 3 * G)
    ext.bresenham_scatter_mask(
        env_ids,
        sources,
        targets,
        out_mask.view(-1),
        int(G),
        int(steps),
        bool(include_source),
        bool(include_target),
    )
    return out_mask


def empty_ray_pairs_cuda(
    eyes: torch.Tensor,
    rays_world: torch.Tensor,
    hit_pixel_mask: torch.Tensor,
    bbox_min: torch.Tensor,
    voxel_size: torch.Tensor,
    *,
    grid_size: int,
    dedupe: bool = False,
    return_ray_indices: bool = False,
) -> Optional[tuple[torch.Tensor, ...]]:
    """Return miss-ray grid entry/exit pairs for a full batched ray image.

    Args mirror ``TensorBatchEnv._empty_ray_pairs_for_env`` but operate on
    every env/pixel in one CUDA launch.  The returned tensors are compact:
    ``env_ids`` is ``[M]`` and ``sources``/``targets`` are ``[M, 3]``.  When
    ``dedupe`` is true, duplicate ``(env, source, target)`` rows are removed;
    this is optional because downstream mask scatter is idempotent.
    """
    if not (
        eyes.is_cuda
        and rays_world.is_cuda
        and hit_pixel_mask.is_cuda
        and bbox_min.is_cuda
        and voxel_size.is_cuda
    ):
        return None
    ext = _load_extension()
    if ext is None:
        return None

    device = eyes.device
    eyes = eyes.to(dtype=torch.float32, device=device).contiguous().view(-1, 3)
    rays_world = rays_world.to(dtype=torch.float32, device=device).contiguous()
    hit_pixel_mask = hit_pixel_mask.to(dtype=torch.bool, device=device).contiguous()
    bbox_min = bbox_min.to(dtype=torch.float32, device=device).contiguous().view(-1, 3)
    voxel_size = voxel_size.to(dtype=torch.float32, device=device).contiguous().view(-1, 3)
    if rays_world.dim() != 3 or rays_world.shape[0] != eyes.shape[0] or rays_world.shape[2] != 3:
        raise ValueError(f"rays_world must be [N, R, 3], got {tuple(rays_world.shape)}")
    if hit_pixel_mask.shape != rays_world.shape[:2]:
        raise ValueError(
            f"hit_pixel_mask {tuple(hit_pixel_mask.shape)} incompatible with rays {tuple(rays_world.shape)}"
        )
    if bbox_min.shape != eyes.shape or voxel_size.shape != eyes.shape:
        raise ValueError("bbox_min and voxel_size must both be [N, 3]")

    valid, sources_all, targets_all = ext.empty_ray_pairs(
        eyes,
        rays_world,
        hit_pixel_mask,
        bbox_min,
        voxel_size,
        int(grid_size),
    )
    valid_flat = valid.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).flatten()
    before = int(valid_idx.shape[0])
    if before == 0:
        empty_env = torch.empty((0,), dtype=torch.int64, device=device)
        empty_idx = torch.empty((0, 3), dtype=torch.int64, device=device)
        if return_ray_indices:
            return empty_env, empty_idx, empty_idx, 0, 0, empty_env
        return empty_env, empty_idx, empty_idx, 0, 0

    R = int(rays_world.shape[1])
    env_ids = torch.div(valid_idx, R, rounding_mode="floor").to(dtype=torch.int64)
    sources = sources_all.reshape(-1, 3)[valid_idx].contiguous()
    targets = targets_all.reshape(-1, 3)[valid_idx].contiguous()
    if return_ray_indices and dedupe:
        raise ValueError("return_ray_indices=True requires dedupe=False")
    ray_ids = torch.remainder(valid_idx, R).to(dtype=torch.int64)
    if dedupe and sources.numel() > 0:
        env_start_end = torch.cat([env_ids.view(-1, 1), sources, targets], dim=1)
        env_start_end = torch.unique(env_start_end, dim=0, sorted=False)
        env_ids = env_start_end[:, 0].contiguous()
        sources = env_start_end[:, 1:4].contiguous()
        targets = env_start_end[:, 4:7].contiguous()
    after = int(sources.shape[0])
    if return_ray_indices:
        return env_ids, sources, targets, before, after, ray_ids
    return env_ids, sources, targets, before, after


def warmup_cuda_bresenham(device: Optional[torch.device | str] = None) -> bool:
    """Compile/load the extension and launch one tiny scatter.

    ``torch.utils.cpp_extension`` compiles lazily.  Calling this during env
    construction keeps the one-time build cost out of step timing profiles.
    """
    if device is None:
        device = torch.device("cuda")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    env_ids = torch.zeros(1, dtype=torch.int64, device=device)
    sources = torch.zeros(1, 3, dtype=torch.int64, device=device)
    targets = torch.ones(1, 3, dtype=torch.int64, device=device)
    out_mask = torch.zeros(1, 2, 2, 2, dtype=torch.uint8, device=device)
    out = scatter_bresenham3d_to_mask_cuda(
        env_ids,
        sources,
        targets,
        num_envs=1,
        grid_size=2,
        include_source=True,
        include_target=True,
        out_mask=out_mask,
        max_steps=2,
    )
    rays = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    eyes = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device=device)
    hit = torch.zeros((1, 1), dtype=torch.bool, device=device)
    bbox_min = torch.zeros((1, 3), dtype=torch.float32, device=device)
    voxel_size = torch.ones((1, 3), dtype=torch.float32, device=device)
    pairs = empty_ray_pairs_cuda(
        eyes,
        rays,
        hit,
        bbox_min,
        voxel_size,
        grid_size=2,
        dedupe=False,
    )
    if pairs is None:
        return False
    return out is not None
