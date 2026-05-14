"""Pure-PyTorch mesh ray-triangle rasterizer for NBV.

Replaces the earlier native-extension renderer to remove install pain.
For NBV we only need depth + alpha (no shading, no texture), so a
vectorized Möller–Trumbore intersection over (rays × triangles) is both
simpler and free of native deps.

Interface mirrors gsplat_renderer's `SequenceRenderer` so env.py is
backend-agnostic:
    render(position_canon, look_at_canon, save_path=None) -> RenderOutput
    render_batch(positions_canon, look_ats_canon)         -> BatchRenderOutput

Limits:
    - Memory is O(K * H * W * F): K cameras × pixels × triangles. For
      K=1, H=W=64, F<=20k this is ~80MB which is fine on L4. For very
      large ShapeNet meshes (>30k tris) at H=W=256, switch to a BVH
      raycaster (e.g., kaolin's mesh_to_sdf path) — for the smoke test
      it's irrelevant (sphere = 1280 tris).

Coordinate convention (right-handed, canonical frame):
    eye = (x, y, z), look_dir = (at - eye)/||·||
    camera basis: right = up × look_dir, true_up = look_dir × right
    pixel ray (in CAM frame, x→right, y→down, z→forward):
        ray_x = (px - cx) / fx
        ray_y = (py - cy) / fy
        ray_z = 1
    world ray = R_cw @ (ray / ||ray||)
    where R_cw = [right, true_up, look_dir] (cols are camera axes in world).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class RenderOutput:
    rgb: torch.Tensor    # [H, W, 3] zeros (we don't shade)
    depth: torch.Tensor  # [H, W]   positive distance to surface; 0 background
    alpha: torch.Tensor  # [H, W]   1 = hit, 0 = miss
    points: Optional[torch.Tensor] = None  # [H, W, 3] optional surface xyz


@dataclass
class BatchRenderOutput:
    rgb: torch.Tensor    # [K, H, W, 3]
    depth: torch.Tensor  # [K, H, W]
    alpha: torch.Tensor  # [K, H, W]
    points: Optional[torch.Tensor] = None  # [K, H, W, 3] optional surface xyz


@dataclass
class PinholeIntrinsics:
    fov_deg: float
    width: int
    height: int


# ---------------------------------------------------------------------------
# Mesh loading (uses trimesh — already in our deps)
# ---------------------------------------------------------------------------
def _load_obj_verts_faces(mesh_path: Path, max_faces: int = 0):
    """Load a mesh as (verts [V,3], faces [F,3]) numpy arrays.

    If `max_faces > 0` and the loaded mesh has more faces, run quadric
    decimation down to ~`max_faces`. Crucial for ABO furniture (often
    50K-100K tris); without this, our pure-PyTorch Möller-Trumbore
    runs ~26× slower than on synthetic-shape smoke.
    """
    import trimesh
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if hasattr(mesh, "geometry") and not hasattr(mesh, "vertices"):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if max_faces > 0 and len(mesh.faces) > max_faces:
        # Use open3d's quadric decimation directly. trimesh's wrapper
        # requires `fast_simplification` (not in our Modal image), and
        # falls through silently — leaving 100K-tri meshes intact.
        try:
            import open3d as o3d
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
                triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
            )
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                target_number_of_triangles=int(max_faces))
            verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
            faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)
            return verts, faces
        except Exception as e:
            # Open3d sometimes crashes on non-manifold ABO meshes — fall
            # back to the unsimplified mesh (slow but correct).
            print(f"[mesh_renderer] decimation failed for {mesh_path.name}: {e}", flush=True)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


# ---------------------------------------------------------------------------
# Vectorized ray–triangle intersection (Möller–Trumbore, batched)
# ---------------------------------------------------------------------------
def _ray_triangle_intersect(
    ray_o: torch.Tensor,        # [K, R, 3] origins (per camera × ray)
    ray_d: torch.Tensor,        # [K, R, 3] unit dirs
    v0: torch.Tensor,           # [F, 3]
    v1: torch.Tensor,           # [F, 3]
    v2: torch.Tensor,           # [F, 3]
    eps: float = 1e-7,
    f_chunk: int = 1024,
    pair_budget: int = 8_000_000,
):
    """Möller–Trumbore. Returns (t_min [K, R], hit_mask [K, R] bool).

    `t_min` is the distance along the ray to the closest surface; 0
    where no triangle was hit (`hit_any` is False there).

    Chunked over both the ray and triangle dimensions so 400x400
    GenNBV-aligned renders do not materialize K*160k*F intermediates.
    `pair_budget` caps the approximate K*ray_chunk*f_chunk products
    used by the largest broadcast tensors.
    """
    K, R, _ = ray_o.shape
    F = v0.shape[0]
    t_min = torch.full((K, R), float("inf"), device=ray_o.device, dtype=ray_o.dtype)

    f_chunk = max(1, min(int(f_chunk), F))
    pair_budget = max(int(pair_budget), K * f_chunk)
    r_chunk = max(1, min(R, pair_budget // max(K * f_chunk, 1)))

    for r0 in range(0, R, r_chunk):
        r1 = min(R, r0 + r_chunk)
        Rn = r1 - r0
        rd = ray_d[:, r0:r1].unsqueeze(2)        # [K, Rn, 1, 3]
        ro = ray_o[:, r0:r1].unsqueeze(2)        # [K, Rn, 1, 3]
        t_min_r = torch.full((K, Rn), float("inf"), device=ray_o.device, dtype=ray_o.dtype)

        for f0 in range(0, F, f_chunk):
            f1 = min(F, f0 + f_chunk)
            v0c = v0[f0:f1]
            v1c = v1[f0:f1]
            v2c = v2[f0:f1]
            Fc = f1 - f0
            e1 = v1c - v0c                      # [Fc, 3]
            e2 = v2c - v0c                      # [Fc, 3]

            p = torch.cross(
                rd.expand(K, Rn, Fc, 3),
                e2.unsqueeze(0).unsqueeze(0).expand(K, Rn, Fc, 3),
                dim=-1,
            )
            det = torch.sum(e1 * p, dim=-1)     # [K, Rn, Fc]
            det_safe = torch.where(det.abs() > eps, det, torch.full_like(det, eps))
            inv_det = 1.0 / det_safe

            s = ro - v0c.unsqueeze(0).unsqueeze(0)
            u = inv_det * torch.sum(s * p, dim=-1)
            q = torch.cross(s, e1.unsqueeze(0).unsqueeze(0).expand_as(s), dim=-1)
            del s, p
            v = inv_det * torch.sum(rd * q, dim=-1)
            t = inv_det * torch.sum(e2.unsqueeze(0).unsqueeze(0) * q, dim=-1)
            del q

            hit = (det.abs() > eps) & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > eps)
            t = torch.where(hit, t, torch.full_like(t, float("inf")))
            chunk_min, _ = t.min(dim=-1)         # [K, Rn]
            t_min_r = torch.minimum(t_min_r, chunk_min)
            del t, u, v, det, inv_det, det_safe, hit, chunk_min

        t_min[:, r0:r1] = t_min_r
        del rd, ro, t_min_r

    hit_any = torch.isfinite(t_min)
    t_min = torch.where(hit_any, t_min, torch.full_like(t_min, 0.0))
    return t_min, hit_any


def _ray_aabb_intersect(
    ray_o: torch.Tensor,        # [..., 3]
    ray_d: torch.Tensor,        # [..., 3]
    bbox_min: torch.Tensor,     # [3]
    bbox_max: torch.Tensor,     # [3]
    eps: float = 1e-9,
) -> torch.Tensor:
    """Return rays whose positive half-line intersects the mesh AABB."""
    dir_safe = torch.where(
        ray_d.abs() < eps,
        torch.where(ray_d >= 0.0, torch.full_like(ray_d, eps), torch.full_like(ray_d, -eps)),
        ray_d,
    )
    t0 = (bbox_min.view(*((1,) * (ray_o.ndim - 1)), 3) - ray_o) / dir_safe
    t1 = (bbox_max.view(*((1,) * (ray_o.ndim - 1)), 3) - ray_o) / dir_safe
    t_near = torch.minimum(t0, t1).amax(dim=-1)
    t_far = torch.maximum(t0, t1).amin(dim=-1)
    t_start = t_near.clamp(min=0.0)
    return t_far > (t_start + 1e-6)


_INSIDE_TEST_DIRS = (
    (1.0, 0.37139067, 0.52981294),
    (-0.2844273, 1.0, 0.42163702),
    (0.33711252, -0.6117841, 1.0),
    (-1.0, -0.21783091, 0.47391105),
    (0.19421137, -1.0, -0.6823914),
)


def _points_inside_mesh_ray_parity(
    points: torch.Tensor,       # [P, 3]
    v0: torch.Tensor,           # [F, 3]
    v1: torch.Tensor,
    v2: torch.Tensor,
    *,
    eps: float = 1e-6,
    t_dedupe_eps: float = 1e-5,
    f_chunk: int = 4096,
) -> torch.Tensor:
    """Classify points by multi-ray odd/even mesh intersection parity.

    This is a point-in-closed-triangle-mesh test. Several deterministic,
    non-axis-aligned rays are cast per point and a majority vote is used
    to reduce edge/vertex degeneracy. It is intended for camera collision
    checks, not rendering.

    For non-watertight meshes no parity method can be perfect. ShapeNBV
    therefore combines this solid test with the existing surface-voxel
    check in the environments.
    """
    if points.numel() == 0:
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)

    points = points.reshape(-1, 3).to(device=v0.device, dtype=v0.dtype)
    dirs = torch.tensor(_INSIDE_TEST_DIRS, dtype=v0.dtype, device=v0.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=eps)

    P = points.shape[0]
    D = dirs.shape[0]
    R = P * D
    ray_o = points[:, None, :].expand(P, D, 3).reshape(R, 3)
    ray_d = dirs[None, :, :].expand(P, D, 3).reshape(R, 3)

    F = v0.shape[0]
    t_hits = torch.full((R, F), float("inf"), dtype=v0.dtype, device=v0.device)

    for start in range(0, F, int(f_chunk)):
        end = min(start + int(f_chunk), F)
        vv0 = v0[start:end]
        e1 = v1[start:end] - vv0
        e2 = v2[start:end] - vv0

        rd = ray_d[:, None, :]                                      # [R,1,3]
        ro = ray_o[:, None, :]
        pvec = torch.linalg.cross(rd, e2[None, :, :], dim=-1)       # [R,C,3]
        det = (e1[None, :, :] * pvec).sum(dim=-1)                   # [R,C]
        non_parallel = det.abs() > eps
        inv_det = torch.where(non_parallel, 1.0 / det, torch.zeros_like(det))

        tvec = ro - vv0[None, :, :]
        u = (tvec * pvec).sum(dim=-1) * inv_det
        qvec = torch.linalg.cross(tvec, e1[None, :, :], dim=-1)
        v = (rd * qvec).sum(dim=-1) * inv_det
        t = (e2[None, :, :] * qvec).sum(dim=-1) * inv_det

        hit = (
            non_parallel
            & (u >= -eps)
            & (v >= -eps)
            & ((u + v) <= 1.0 + eps)
            & (t > eps)
        )
        t_hits[:, start:end] = torch.where(hit, t, t_hits[:, start:end])

    # Sort finite t values so duplicated intersections from shared
    # triangle edges/vertices count as a single surface crossing.
    sorted_t = torch.sort(t_hits, dim=1).values
    finite = torch.isfinite(sorted_t)
    unique_hit = finite.clone()
    if F > 1:
        unique_hit[:, 1:] = finite[:, 1:] & (
            (sorted_t[:, 1:] - sorted_t[:, :-1]).abs() > t_dedupe_eps
        )
    counts = unique_hit.sum(dim=1)
    inside_votes = (counts % 2) == 1
    inside_votes = inside_votes.reshape(P, D)
    return inside_votes.sum(dim=1) > (D // 2)


def _build_camera_basis(eye: torch.Tensor, at: torch.Tensor):
    """Right-handed camera basis (right, true_up, look_dir) from eye + at.

    Up vector chosen as +Y unless degenerate; falls back to +Z.
    Returns R_cw of shape [3, 3] (cols = cam axes in world).
    """
    look = at - eye
    look = look / (look.norm() + 1e-8)
    up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)
    if abs(float((look * up).sum())) > 0.999:
        up = torch.tensor([0.0, 0.0, 1.0], device=eye.device, dtype=eye.dtype)
    right = torch.cross(up, look, dim=0)
    right = right / (right.norm() + 1e-8)
    true_up = torch.cross(look, right, dim=0)
    return torch.stack([right, true_up, look], dim=1)   # [3, 3]


def _build_camera_basis_np(eye: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Numpy equivalent of _build_camera_basis for CPU BVH raycasting."""
    look = at - eye
    look = look / (np.linalg.norm(look) + 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float((look * up).sum())) > 0.999:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(up, look)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(look, right)
    return np.stack([right, true_up, look], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-mesh renderer
# ---------------------------------------------------------------------------
class MeshSequenceRenderer:
    """Holds one mesh on GPU and rasterizes (depth + alpha) for arbitrary
    pinhole cameras via vectorized Möller–Trumbore.

    Naming kept identical to gsplat_renderer.SequenceRenderer for drop-in
    compatibility with env.py.
    """

    def __init__(
        self,
        mesh_path: Path,
        sequence_name: str,
        device,
        render_size: Tuple[int, int] = (400, 400),
        fov_deg: float = 60.0,
        T_canon: Optional[torch.Tensor] = None,
        max_faces: int = 0,
        bbox_ray_cull: bool = True,
    ):
        self.mesh_path = Path(mesh_path)
        self.sequence_name = sequence_name
        self.device = torch.device(device)
        self.render_size = tuple(render_size)
        self.fov_deg = float(fov_deg)
        self.bbox_ray_cull = bool(bbox_ray_cull)
        self.last_render_stats: Dict[str, float] = {}
        self.intrinsics = PinholeIntrinsics(
            fov_deg=fov_deg, width=render_size[1], height=render_size[0]
        )

        verts_np, faces_np = _load_obj_verts_faces(self.mesh_path, max_faces=max_faces)
        verts = torch.from_numpy(verts_np).to(self.device)
        faces = torch.from_numpy(faces_np).to(self.device).long()
        if T_canon is not None:
            T = T_canon.to(self.device).float()
            verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1, device=self.device)], dim=1)
            verts = (T @ verts_h.T).T[:, :3]

        self._verts = verts                                    # [V, 3]
        self._faces = faces                                    # [F, 3]
        self._v0 = verts[faces[:, 0]]                          # [F, 3]
        self._v1 = verts[faces[:, 1]]
        self._v2 = verts[faces[:, 2]]
        bbox_min_t = verts.min(dim=0).values
        bbox_max_t = verts.max(dim=0).values
        bbox_extent_t = bbox_max_t - bbox_min_t
        # A tiny outward pad prevents numerical false negatives for rays
        # that graze a triangle lying exactly on the mesh AABB.
        bbox_pad = torch.clamp(bbox_extent_t.max() * 1e-6, min=1e-6)
        self._bbox_min_t = bbox_min_t - bbox_pad
        self._bbox_max_t = bbox_max_t + bbox_pad
        self.bbox_min = bbox_min_t.detach().cpu().numpy()
        self.bbox_max = bbox_max_t.detach().cpu().numpy()
        self.center = 0.5 * (self.bbox_min + self.bbox_max)
        self.extent = self.bbox_max - self.bbox_min

        # Pre-compute pixel ray directions in camera space (constant per
        # render — cached so render_batch only does the world transform).
        H, W = self.render_size
        fy = 0.5 * H / math.tan(math.radians(self.fov_deg) * 0.5)
        fx = fy * (W / H)
        cx = 0.5 * W
        cy = 0.5 * H
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        rays_cam = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], dim=-1)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
        self._rays_cam = rays_cam.reshape(-1, 3)               # [H*W, 3]
        self._H, self._W = H, W

    # ----- public API mirrors gsplat_renderer ------------------------------
    def render(
        self,
        position_canon: torch.Tensor,
        look_at_canon: torch.Tensor,
        save_path: Optional[Path] = None,
    ) -> RenderOutput:
        out = self.render_batch(
            position_canon.unsqueeze(0).to(self.device).float(),
            look_at_canon.unsqueeze(0).to(self.device).float(),
        )
        pts = out.points[0] if out.points is not None else None
        return RenderOutput(rgb=out.rgb[0], depth=out.depth[0], alpha=out.alpha[0], points=pts)

    def render_batch(
        self,
        positions_canon: torch.Tensor,    # [K, 3]
        look_ats_canon: torch.Tensor,     # [K, 3]
    ) -> BatchRenderOutput:
        K = positions_canon.shape[0]
        H, W = self._H, self._W
        device = self.device

        # Per-camera rotation cam→world.
        R_world_cam = torch.stack(
            [_build_camera_basis(positions_canon[k], look_ats_canon[k]) for k in range(K)],
            dim=0,
        )   # [K, 3, 3]

        # Per-camera ray directions in world space: R_cw @ ray_cam.
        # rays_cam: [HW, 3], R_world_cam: [K, 3, 3]
        # → world rays: [K, HW, 3] = einsum("kij, hj -> khi")
        rays_world = torch.einsum("kij,hj->khi", R_world_cam, self._rays_cam)
        ray_o = positions_canon[:, None, :].expand(K, H * W, 3)

        total_rays = int(K * H * W)
        if self.bbox_ray_cull:
            active = _ray_aabb_intersect(
                ray_o=ray_o,
                ray_d=rays_world,
                bbox_min=self._bbox_min_t,
                bbox_max=self._bbox_max_t,
            )
            active_count = int(active.sum().detach().cpu().item())
            t_min = torch.zeros((K, H * W), device=device, dtype=torch.float32)
            hit = torch.zeros((K, H * W), device=device, dtype=torch.bool)
            if active_count > 0:
                t_active, hit_active = _ray_triangle_intersect(
                    ray_o=ray_o[active].unsqueeze(0),
                    ray_d=rays_world[active].unsqueeze(0),
                    v0=self._v0,
                    v1=self._v1,
                    v2=self._v2,
                )
                t_min[active] = t_active.squeeze(0)
                hit[active] = hit_active.squeeze(0)
            self.last_render_stats = {
                "bbox_ray_cull": 1.0,
                "active_rays": float(active_count),
                "total_rays": float(total_rays),
                "active_ratio": float(active_count / max(total_rays, 1)),
            }
        else:
            t_min, hit = _ray_triangle_intersect(
                ray_o=ray_o, ray_d=rays_world,
                v0=self._v0, v1=self._v1, v2=self._v2,
            )
            self.last_render_stats = {
                "bbox_ray_cull": 0.0,
                "active_rays": float(total_rays),
                "total_rays": float(total_rays),
                "active_ratio": 1.0,
            }

        depth = t_min.reshape(K, H, W)
        alpha = hit.float().reshape(K, H, W)
        # Zero out depth on misses (matches gsplat convention so
        # downstream `mask = alpha > 0.5` filters identically).
        depth = depth * alpha
        rgb = torch.zeros((K, H, W, 3), device=device, dtype=torch.float32)
        return BatchRenderOutput(rgb=rgb, depth=depth, alpha=alpha)

    @torch.no_grad()
    def points_inside_mesh(self, points_canon: torch.Tensor) -> torch.Tensor:
        """Return a mesh-level inside/collision mask for camera centers.

        Unlike the GT coverage grid, this treats the triangle mesh as a
        closed surface and uses ray parity. It is therefore able to catch
        camera centers in the object interior even when they do not fall
        on a surface voxel.
        """
        pts = points_canon.to(self.device).float().reshape(-1, 3)
        if pts.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        return _points_inside_mesh_ray_parity(pts, self._v0, self._v1, self._v2)


class Open3DMeshSequenceRenderer:
    """CPU BVH raycaster using Open3D's RaycastingScene.

    This preserves the same public API and camera convention as
    MeshSequenceRenderer but replaces O(rays × triangles) dense CUDA
    Möller-Trumbore with Open3D's BVH traversal. It is an infrastructure
    backend: depth is still Euclidean distance along the same normalized
    pixel-center rays and alpha is still first-hit/non-hit.
    """

    def __init__(
        self,
        mesh_path: Path,
        sequence_name: str,
        device,
        render_size: Tuple[int, int] = (400, 400),
        fov_deg: float = 60.0,
        T_canon: Optional[torch.Tensor] = None,
        max_faces: int = 0,
        nthreads: int = 0,
    ):
        import open3d as o3d

        self.mesh_path = Path(mesh_path)
        self.sequence_name = sequence_name
        self.device = torch.device(device)
        self.render_size = tuple(render_size)
        self.fov_deg = float(fov_deg)
        self.last_render_stats: Dict[str, float] = {}

        verts_np, faces_np = _load_obj_verts_faces(self.mesh_path, max_faces=max_faces)
        verts = torch.from_numpy(verts_np).to(self.device)
        faces = torch.from_numpy(faces_np).to(self.device).long()
        if T_canon is not None:
            T = T_canon.to(self.device).float()
            verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1, device=self.device)], dim=1)
            verts = (T @ verts_h.T).T[:, :3]

        self._verts = verts
        self._faces = faces
        self._v0 = verts[faces[:, 0]]
        self._v1 = verts[faces[:, 1]]
        self._v2 = verts[faces[:, 2]]
        self.bbox_min = verts.min(dim=0).values.detach().cpu().numpy()
        self.bbox_max = verts.max(dim=0).values.detach().cpu().numpy()
        self.center = 0.5 * (self.bbox_min + self.bbox_max)
        self.extent = self.bbox_max - self.bbox_min

        H, W = self.render_size
        fy = 0.5 * H / math.tan(math.radians(self.fov_deg) * 0.5)
        fx = fy * (W / H)
        cx = 0.5 * W
        cy = 0.5 * H
        ys, xs = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij",
        )
        rays_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
        rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=-1, keepdims=True)
        self._rays_cam_np = rays_cam.reshape(-1, 3).astype(np.float32)
        self._H, self._W = H, W

        self._o3d = o3d
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=int(nthreads))
        self._scene.add_triangles(
            o3d.core.Tensor(verts.detach().cpu().numpy().astype(np.float32),
                            dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(faces.detach().cpu().numpy().astype(np.uint32),
                            dtype=o3d.core.Dtype.UInt32),
        )

    def render(
        self,
        position_canon: torch.Tensor,
        look_at_canon: torch.Tensor,
        save_path: Optional[Path] = None,
    ) -> RenderOutput:
        out = self.render_batch(
            position_canon.unsqueeze(0).to(self.device).float(),
            look_at_canon.unsqueeze(0).to(self.device).float(),
        )
        pts = out.points[0] if out.points is not None else None
        return RenderOutput(rgb=out.rgb[0], depth=out.depth[0], alpha=out.alpha[0], points=pts)

    def render_batch(
        self,
        positions_canon: torch.Tensor,
        look_ats_canon: torch.Tensor,
    ) -> BatchRenderOutput:
        K = int(positions_canon.shape[0])
        H, W = self._H, self._W
        pos_np = positions_canon.detach().cpu().numpy().astype(np.float32, copy=False)
        at_np = look_ats_canon.detach().cpu().numpy().astype(np.float32, copy=False)
        bases = np.stack([
            _build_camera_basis_np(pos_np[k], at_np[k])
            for k in range(K)
        ], axis=0)
        rays_world = np.einsum("kij,hj->khi", bases, self._rays_cam_np, optimize=True)
        origins = np.broadcast_to(pos_np[:, None, :], (K, H * W, 3))
        rays = np.concatenate([origins, rays_world], axis=-1).reshape(-1, 6).astype(np.float32, copy=False)
        rays_t = self._o3d.core.Tensor(rays, dtype=self._o3d.core.Dtype.Float32)
        ans = self._scene.cast_rays(rays_t)
        t_hit = ans["t_hit"].numpy().reshape(K, H, W).astype(np.float32, copy=False)
        hit = np.isfinite(t_hit)
        depth_np = np.where(hit, t_hit, 0.0).astype(np.float32, copy=False)
        alpha_np = hit.astype(np.float32, copy=False)

        depth = torch.from_numpy(depth_np).to(self.device, non_blocking=True)
        alpha = torch.from_numpy(alpha_np).to(self.device, non_blocking=True)
        rgb = torch.zeros((K, H, W, 3), device=self.device, dtype=torch.float32)
        self.last_render_stats = {
            "backend_open3d": 1.0,
            "hit_rays": float(hit.sum()),
            "active_rays": float(K * H * W),
            "total_rays": float(K * H * W),
            "active_ratio": 1.0,
            "hit_ratio": float(hit.sum() / max(K * H * W, 1)),
        }
        return BatchRenderOutput(rgb=rgb, depth=depth, alpha=alpha)

    @torch.no_grad()
    def points_inside_mesh(self, points_canon: torch.Tensor) -> torch.Tensor:
        pts = points_canon.to(self.device).float().reshape(-1, 3)
        if pts.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        return _points_inside_mesh_ray_parity(pts, self._v0, self._v1, self._v2)


class NvdiffrastMeshSequenceRenderer(MeshSequenceRenderer):
    """GPU triangle rasterizer using NVlabs nvdiffrast.

    This backend keeps the mesh-rendering semantics of Open3D/IsaacGym
    but avoids the CPU raycasting path.  nvdiffrast rasterizes triangles
    in CUDA and interpolates canonical xyz at the visible surface.  The
    TensorBatchEnv hot path consumes those surface points directly for
    voxel hit construction, avoiding an extra depth/backprojection round.
    """

    def __init__(self, *args, near: float = 1e-4, far: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise ValueError("NvdiffrastMeshSequenceRenderer requires a CUDA device")
        try:
            import nvdiffrast.torch as dr
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise ImportError(
                "renderer_backend='nvdiffrast' requires NVlabs nvdiffrast. "
                "Install with: pip install git+https://github.com/NVlabs/nvdiffrast.git "
                "--no-build-isolation"
            ) from exc
        self._dr = dr
        self._dr_ctx = dr.RasterizeCudaContext(device=self.device)
        self._faces_i32 = self._faces.to(device=self.device, dtype=torch.int32).contiguous()
        self.near = float(near)
        self.far = float(far)

    def render_batch(
        self,
        positions_canon: torch.Tensor,
        look_ats_canon: torch.Tensor,
    ) -> BatchRenderOutput:
        K = int(positions_canon.shape[0])
        H, W = self._H, self._W
        device = self.device
        positions = positions_canon.to(device=device, dtype=torch.float32).contiguous()
        look_ats = look_ats_canon.to(device=device, dtype=torch.float32).contiguous()

        R_world_cam = torch.stack(
            [_build_camera_basis(positions[k], look_ats[k]) for k in range(K)],
            dim=0,
        )  # [K, 3, 3], columns are camera axes in world.
        verts = self._verts.to(device=device, dtype=torch.float32)
        rel = verts.unsqueeze(0) - positions[:, None, :]             # [K, V, 3]
        cam = torch.einsum("kji,kvj->kvi", R_world_cam, rel)         # world -> camera

        z = cam[..., 2]
        w = z.clamp(min=self.near)
        tan_half = math.tan(math.radians(self.fov_deg) * 0.5)
        # Pixel-centre convention: ShapeNBV's ray grid uses integer pixels
        # with cx=W/2, cy=H/2. nvdiffrast samples at i+0.5, so shift the
        # projection by half a pixel to keep the same rays.
        x_shift = 1.0 / max(float(W), 1.0)
        y_shift = -1.0 / max(float(H), 1.0)
        clip_x = cam[..., 0] / tan_half + x_shift * w
        clip_y = -cam[..., 1] / tan_half + y_shift * w
        z_ndc = ((w - self.near) / max(self.far - self.near, 1e-6)) * 2.0 - 1.0
        clip_z = z_ndc * w
        pos_clip = torch.stack([clip_x, clip_y, clip_z, w], dim=-1).contiguous()

        rast, _ = self._dr.rasterize(
            self._dr_ctx,
            pos_clip,
            self._faces_i32,
            resolution=(H, W),
            grad_db=False,
        )
        hit = rast[..., 3] > 0
        attrs = verts.unsqueeze(0).expand(K, -1, -1).contiguous()
        pts_world, _ = self._dr.interpolate(attrs, rast, self._faces_i32)
        # nvdiffrast returns rows in the opposite vertical convention from
        # ShapeNBV's cached pixel rays (`ray_y = (row - cy) / f`, positive
        # camera-y points down).  Flip the raster outputs here so alpha,
        # depth and interpolated hit points stay pixel-aligned with
        # TensorBatchEnv._world_rays and the Open3D/torch renderer backends.
        hit = torch.flip(hit, dims=[1])
        pts_world = torch.flip(pts_world, dims=[1])
        depth = torch.linalg.norm(pts_world - positions[:, None, None, :], dim=-1)
        depth = torch.where(hit, depth, torch.zeros_like(depth))
        alpha = hit.to(dtype=torch.float32)
        rgb = torch.zeros((K, H, W, 3), device=device, dtype=torch.float32)
        total_rays = float(K * H * W)
        hit_rays = float(hit.sum().detach().cpu().item())
        self.last_render_stats = {
            "backend_nvdiffrast": 1.0,
            "hit_rays": hit_rays,
            "active_rays": total_rays,
            "total_rays": total_rays,
            "active_ratio": 1.0,
            "hit_ratio": hit_rays / max(total_rays, 1.0),
        }
        return BatchRenderOutput(rgb=rgb, depth=depth, alpha=alpha, points=pts_world)


# ---------------------------------------------------------------------------
# Cache of renderers, mirrors object_nbv_zgr.UCO3DIndex
# ---------------------------------------------------------------------------
class ShapeNetIndex:
    def __init__(
        self,
        entries: Dict[str, Path],
        device: str = "cuda",
        render_size: Tuple[int, int] = (400, 400),
        fov_deg: float = 60.0,
    ):
        self.entries = dict(entries)
        self.device = torch.device(device)
        self.render_size = tuple(render_size)
        self.fov_deg = float(fov_deg)
        self._cache: Dict[str, MeshSequenceRenderer] = {}

    @property
    def sequence_names(self):
        return list(self.entries.keys())

    def get_or_build_renderer(
        self,
        sequence_name: str,
        T_canon: Optional[torch.Tensor] = None,
        device=None,
        render_size=None,
        fov_deg=None,
    ) -> MeshSequenceRenderer:
        if sequence_name in self._cache:
            return self._cache[sequence_name]
        renderer = MeshSequenceRenderer(
            mesh_path=self.entries[sequence_name],
            sequence_name=sequence_name,
            device=device or self.device,
            render_size=render_size or self.render_size,
            fov_deg=fov_deg or self.fov_deg,
            T_canon=T_canon,
        )
        self._cache[sequence_name] = renderer
        return renderer

    def preload_renderers(self, **_kwargs):
        for name in self.entries:
            self.get_or_build_renderer(name)
