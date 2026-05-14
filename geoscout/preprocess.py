"""ShapeNet mesh → canonical points → 128³ GT occupancy grid.

Input:  `model_normalized.obj` (already centered + unit-cube normalized
        by ShapeNet pipeline)
Output: `<name>.pt` with the same dict schema as object_nbv_zgr.preprocess
        so env.py can stay agnostic:

    {
        "T_canon":          [4, 4] identity (mesh already canonical)
        "range_gt":         [6] (x_max, x_min, y_max, y_min, z_max, z_min)
        "voxel_size_gt":    [3]
        "grid_gt":          [G, G, G] binary occupancy
        "num_valid_voxel_gt": scalar
        "grid_size":        int
    }

The GT is built by densely sampling points on the mesh surface and
voxelizing — same pattern as object_nbv_zgr but with a perfect mesh
sampling source instead of SfM points. No noise, no holes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh


def sample_surface_points(mesh_path: Path, n_points: int = 100_000) -> np.ndarray:
    """Uniformly sample points on the mesh surface.

    Uses trimesh's area-weighted sampler. Returns [n_points, 3] in the
    mesh's coordinate frame (canonical for ShapeNet).
    """
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if hasattr(mesh, "geometry") and not hasattr(mesh, "sample"):
        # Scene with multiple sub-meshes — concat.
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float32)


def voxelize_pointcloud(
    points: np.ndarray,
    grid_size: int = 128,
    margin: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize a point cloud into a cubic grid (GenNBV index convention).

    GenNBV's `scanned_pts_to_idx_3D` expects `range_gt` to encode the
    CENTRES of the edge voxels, not their corners — its `xyz_min_voxel`
    line subtracts a half-voxel from `range_gt[:, [1,3,5]]` to recover the
    grid origin. We must match that here, otherwise the agent's runtime
    voxel indices land half a voxel off the GT grid (the canonical
    "agent renders fine but cr stays at 0" failure mode).

    Returns:
        range_gt:      [6] (x_max_centre, x_min_centre, y_max_centre,
                            y_min_centre, z_max_centre, z_min_centre)
        voxel_size_gt: [3]
        grid_gt:       [G, G, G] binary
    """
    bbox_min_corner = points.min(axis=0)
    bbox_max_corner = points.max(axis=0)
    extent = np.maximum(bbox_max_corner - bbox_min_corner, 1e-3)
    bbox_min_corner = bbox_min_corner - margin * extent
    bbox_max_corner = bbox_max_corner + margin * extent
    voxel_size = (bbox_max_corner - bbox_min_corner) / grid_size

    rel = points - bbox_min_corner[None, :]
    idx = np.floor(rel / voxel_size[None, :]).astype(np.int64)
    mask = np.all((idx >= 0) & (idx < grid_size), axis=1)
    idx = idx[mask]

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    if len(idx) > 0:
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    # GenNBV convention: range_gt entries are voxel CENTRES.
    bbox_min_center = bbox_min_corner + 0.5 * voxel_size
    bbox_max_center = bbox_max_corner - 0.5 * voxel_size
    range_gt = np.array([
        bbox_max_center[0], bbox_min_center[0],
        bbox_max_center[1], bbox_min_center[1],
        bbox_max_center[2], bbox_min_center[2],
    ], dtype=np.float32)
    return range_gt, voxel_size.astype(np.float32), grid


def validate_preproc_schema(
    data: dict,
    *,
    expected_grid_size: Optional[int] = None,
    expected_grid_storage_dtype: Optional[str] = None,
    require_points_canon: bool = False,
    min_points_canon: int = 0,
    require_caption_emb: bool = False,
) -> Tuple[bool, str]:
    """Check that a serialized preproc dict matches the runtime contract."""
    required = ("T_canon", "range_gt", "voxel_size_gt", "grid_gt", "num_valid_voxel_gt")
    for key in required:
        if key not in data:
            return False, f"missing key {key!r}"

    grid = data["grid_gt"]
    if grid.ndim != 3 or grid.shape[0] != grid.shape[1] or grid.shape[1] != grid.shape[2]:
        return False, f"grid_gt must be cubic [G,G,G], got {tuple(grid.shape)}"

    grid_size = int(grid.shape[0])
    stored_grid_size = data.get("grid_size")
    if stored_grid_size is not None and int(stored_grid_size) != grid_size:
        return False, f"grid_size={stored_grid_size} but grid_gt shape is {tuple(grid.shape)}"
    if expected_grid_size is not None and grid_size != int(expected_grid_size):
        return False, f"grid_size={grid_size}, expected {int(expected_grid_size)}"

    if tuple(data["T_canon"].shape) != (4, 4):
        return False, f"T_canon shape {tuple(data['T_canon'].shape)} != (4, 4)"
    if data["range_gt"].numel() != 6:
        return False, f"range_gt has {data['range_gt'].numel()} entries, expected 6"
    if data["voxel_size_gt"].numel() != 3:
        return False, f"voxel_size_gt has {data['voxel_size_gt'].numel()} entries, expected 3"
    if bool((data["voxel_size_gt"].float() <= 0).any().item()):
        return False, "voxel_size_gt must be positive"

    valid_count = float(data["num_valid_voxel_gt"].item())
    actual_count = float((grid.float() > 0.5).sum().item())
    if abs(valid_count - actual_count) > 0.5:
        return False, (
            f"num_valid_voxel_gt={valid_count:.0f} but grid_gt has "
            f"{actual_count:.0f} occupied voxels"
        )

    if expected_grid_storage_dtype:
        expected_dtype = {
            "float32": torch.float32,
            "uint8": torch.uint8,
        }.get(expected_grid_storage_dtype)
        if expected_dtype is None:
            return False, f"unknown expected_grid_storage_dtype={expected_grid_storage_dtype!r}"
        if grid.dtype != expected_dtype:
            return False, f"grid_gt dtype {grid.dtype} != {expected_dtype}"

    if require_points_canon:
        pts = data.get("points_canon")
        if pts is None:
            return False, "missing points_canon"
        if pts.ndim != 2 or pts.shape[1] != 3:
            return False, f"points_canon shape {tuple(pts.shape)} != [N, 3]"
        if int(pts.shape[0]) < int(min_points_canon):
            return False, (
                f"points_canon has {int(pts.shape[0])} points, "
                f"expected at least {int(min_points_canon)}"
            )

    if require_caption_emb and "caption_emb" not in data:
        return False, "missing caption_emb"

    return True, "ok"


def preproc_file_matches_config(
    path: Path,
    *,
    grid_size: int,
    grid_storage_dtype: str,
    n_surface_points: int = 0,
    require_caption_emb: bool = False,
) -> Tuple[bool, str]:
    """Return whether an existing `.pt` can be safely reused."""
    try:
        data = load_preproc(path, map_location="cpu")
    except Exception as exc:
        return False, f"could not load existing preproc: {type(exc).__name__}: {exc}"
    return validate_preproc_schema(
        data,
        expected_grid_size=grid_size,
        expected_grid_storage_dtype=grid_storage_dtype,
        require_points_canon=n_surface_points > 0,
        min_points_canon=max(0, int(n_surface_points)),
        require_caption_emb=require_caption_emb,
    )


def preprocess_mesh(
    mesh_path: Path,
    grid_size: int = 128,
    n_surface_points: int = 100_000,
    margin: float = 0.1,
    caption_emb: Optional[torch.Tensor] = None,
    synset: Optional[str] = None,
    category: Optional[str] = None,
    grid_storage_dtype: str = "uint8",
) -> dict:
    """Full preprocessing for one ShapeNet mesh.

    Returns dict of torch tensors with the same schema as
    object_nbv_zgr.preprocess.preprocess_sequence so env.py stays
    backend-agnostic. T_canon is identity (ShapeNet's mesh is already
    canonical) — kept in the dict so the env still calls
    `_preproc["T_canon"]` without branching.

    Phase 1 caption injection (optional):
        - `caption_emb`: pre-encoded sentence-transformer vector (e.g.
          384-d for MiniLM-L6) for this object's CATEGORY. Same vector
          for all objects in the same synset (Phase 1 simplification);
          attached here so the env can read `_preproc["caption_emb"]`
          and concatenate into the policy obs.
        - `synset` / `category`: stored as strings for diagnostics.

    With `caption_emb=None`, callers fall back to a zero vector at env
    construction time — consistent with the Phase 0 "no-caption" path.
    """
    # We always sample at high count for the voxelization (denser sampling
    # → cleaner grid_gt). The serialized `points_canon` is OPTIONAL — at
    # 100k×3 floats it's the dominant cost (~1.2 MB / object × 51k
    # ShapeNet ≈ 60 GB).  Set `n_surface_points <= 0` to skip storing
    # them; validate.py can re-derive a coarse GT pcd from the
    # `grid_gt` voxel centres on the fly.
    sampling_count = max(int(n_surface_points), 100_000)
    pts = sample_surface_points(mesh_path, sampling_count)
    range_gt, voxel_size, grid = voxelize_pointcloud(
        pts, grid_size=grid_size, margin=margin
    )
    if grid_storage_dtype == "uint8":
        grid_for_save = grid.astype(np.uint8)
    elif grid_storage_dtype == "float32":
        grid_for_save = grid.astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"grid_storage_dtype must be 'float32' or 'uint8', got {grid_storage_dtype}"
        )

    T_canon = np.eye(4, dtype=np.float32)
    out = {
        "T_canon": torch.from_numpy(T_canon),
        "range_gt": torch.from_numpy(range_gt),
        "voxel_size_gt": torch.from_numpy(voxel_size),
        "grid_gt": torch.from_numpy(grid_for_save),
        "num_valid_voxel_gt": torch.tensor(float(grid.sum())),
        "grid_size": grid_size,
        "grid_storage_dtype": grid_storage_dtype,
    }
    if n_surface_points > 0:
        # Optional: keep dense surface points so validate.py can show a
        # high-resolution GT pcd. Skip (n_surface_points=0) to save
        # ~1.2 MB / object — 35× total preproc shrinkage.
        if n_surface_points < sampling_count:
            sub = np.random.choice(pts.shape[0], n_surface_points, replace=False)
            out["points_canon"] = torch.from_numpy(pts[sub])
        else:
            out["points_canon"] = torch.from_numpy(pts)
    if caption_emb is not None:
        out["caption_emb"] = caption_emb.detach().cpu().float()
    if synset is not None:
        out["synset"] = synset
    if category is not None:
        out["category"] = category
    return out


def save_preproc(out_path: Path, data: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)


def load_preproc(path: Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
