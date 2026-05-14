"""Smoke test for the voxel utility functions.

Verifies bresenham3D_strict / bresenham3D / scanned_pts_to_idx_3D /
pose_coord_to_idx_3D / grid_occupancy_tri_cls match the GenNBV
semantics on a small synthetic case (single env, known voxel layout).
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch


def _setup():
    """Single env, 10×10×10 grid spanning [-0.5, 0.5]^3.

    GenNBV convention: range_gt entries are CENTRES of the edge voxels,
    not corners. With voxel_size=0.1 and corner [-0.5, 0.5], the edge
    voxel centres are at [-0.45, +0.45]. The runtime function then adds
    back the half-voxel (xyz_min_voxel = -0.45 - 0.05 = -0.5) to get
    the corner.
    """
    range_gt = torch.tensor([[0.45, -0.45, 0.45, -0.45, 0.45, -0.45]], dtype=torch.float32)
    voxel_size_gt = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    map_size = 10
    return range_gt, voxel_size_gt, map_size


def test_pose_to_idx_origin():
    from shapenbv.voxel_utils import pose_coord_to_idx_3D
    range_gt, voxel_size_gt, map_size = _setup()
    poses = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    idx = pose_coord_to_idx_3D(poses, range_gt, voxel_size_gt, map_size=map_size)
    print(f"pose origin → idx={idx[0].tolist()}  (expected ~[5, 5, 5])")
    assert tuple(idx[0].tolist()) == (5, 5, 5), idx


def test_pts_to_idx_corner():
    from shapenbv.voxel_utils import scanned_pts_to_idx_3D
    range_gt, voxel_size_gt, map_size = _setup()
    pts = torch.tensor([[0.45, -0.45, 0.45], [-0.45, 0.45, -0.45]], dtype=torch.float32)
    out = scanned_pts_to_idx_3D([pts], range_gt, voxel_size_gt, map_size=map_size)
    idx = out[0]
    print(f"corner pts → idx={idx.tolist()}  (expected [9,0,9],[0,9,0])")
    assert sorted(map(tuple, idx.tolist())) == sorted([(9, 0, 9), (0, 9, 0)]), idx


def test_bresenham_axis():
    from shapenbv.voxel_utils import bresenham3D
    src = torch.tensor([[0, 5, 5]], dtype=torch.long)
    tgt = torch.tensor([[9, 5, 5]], dtype=torch.long)
    paths = bresenham3D(src, tgt, map_size=10)
    print(f"bresenham(0,5,5)→(9,5,5): {paths.shape[0]} voxels along x.")
    assert paths.shape[0] >= 9, paths
    assert (paths[:, 1] == 5).all()
    assert (paths[:, 2] == 5).all()


def test_bresenham_strict_axis_full_free_path():
    from shapenbv.voxel_utils import bresenham3D_strict
    src = torch.tensor([[0, 7, 7]], dtype=torch.long)
    tgt = torch.tensor([[31, 7, 7]], dtype=torch.long)
    paths = bresenham3D_strict(
        src,
        tgt,
        map_size=32,
        include_source=True,
        include_target=False,
    )
    got = set(map(tuple, paths.tolist()))
    expected = {(x, 7, 7) for x in range(31)}
    print(f"strict axis free path length={len(got)}  (expected 31, > fixed K=16)")
    assert got == expected, sorted(got)
    assert (31, 7, 7) not in got
    assert (0, 7, 7) in got


def test_bresenham_strict_diagonal_full_free_path():
    from shapenbv.voxel_utils import bresenham3D_strict
    src = torch.tensor([[0, 0, 0]], dtype=torch.long)
    tgt = torch.tensor([[31, 31, 31]], dtype=torch.long)
    paths = bresenham3D_strict(
        src,
        tgt,
        map_size=32,
        include_source=True,
        include_target=False,
    )
    got = set(map(tuple, paths.tolist()))
    expected = {(i, i, i) for i in range(31)}
    print(f"strict diagonal free path length={len(got)}  (expected 31, > fixed K=16)")
    assert got == expected, sorted(got)
    assert (31, 31, 31) not in got
    assert (0, 0, 0) in got


def test_tensor_env_free_update_uses_full_axis_traversal():
    try:
        from shapenbv.tensor_env import TensorBatchEnv, LOG_ODDS_FREE
    except ImportError as e:
        pytest.skip(f"tensor_env import failed: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = object.__new__(TensorBatchEnv)
    env.n_free_samples_per_ray = 1      # legacy gate only; no longer path resolution
    env.grid_size = 32
    env._bbox_min = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=torch.float32, device=device,
    )
    env._voxel_size = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32, device=device,
    )
    env._prob_grid = torch.zeros(2, 32, 32, 32, dtype=torch.float32, device=device)

    eyes = torch.tensor(
        [[0.5, 7.5, 7.5], [11.0, 11.0, 11.0]], dtype=torch.float32, device=device,
    )  # source idx [0,7,7] and [0,0,0]
    target_idx = torch.tensor([[[31, 7, 7]], [[0, 31, 0]]], dtype=torch.long, device=device)
    valid_mask = torch.tensor([[True], [True]], dtype=torch.bool, device=device)

    TensorBatchEnv._update_free_voxels(env, eyes, target_idx, valid_mask)
    got = {
        (int(x), 7, 7)
        for x in torch.nonzero(env._prob_grid[0, :, 7, 7] < 0, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    }
    expected = {(x, 7, 7) for x in range(31)}
    got_env1 = {
        (0, int(y), 0)
        for y in torch.nonzero(env._prob_grid[1, 0, :, 0] < 0, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    }
    expected_env1 = {(0, y, 0) for y in range(31)}
    print(f"tensor_env free axis path length={len(got)} / {len(got_env1)}")
    assert got == expected, sorted(got)
    assert got_env1 == expected_env1, sorted(got_env1)
    assert float(env._prob_grid[0, 31, 7, 7]) == 0.0
    assert float(env._prob_grid[1, 0, 31, 0]) == 0.0
    assert torch.allclose(
        env._prob_grid[0, :31, 7, 7],
        torch.full((31,), LOG_ODDS_FREE, dtype=torch.float32, device=device),
    )
    assert torch.allclose(
        env._prob_grid[1, 0, :31, 0],
        torch.full((31,), LOG_ODDS_FREE, dtype=torch.float32, device=device),
    )


def test_tensor_env_empty_ray_update_marks_grid_box_path():
    try:
        from shapenbv.tensor_env import TensorBatchEnv, LOG_ODDS_FREE
    except ImportError as e:
        pytest.skip(f"tensor_env import failed: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = object.__new__(TensorBatchEnv)
    env.n_free_samples_per_ray = 1
    env.update_empty_rays = True
    env.grid_size = 8
    env._bbox_min = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    env._voxel_size = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    env._prob_grid = torch.zeros(1, 8, 8, 8, dtype=torch.float32, device=device)

    eyes = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device=device)
    target_idx = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
    valid_mask = torch.tensor([[False]], dtype=torch.bool, device=device)
    rays_world = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    hit_pixel_mask = torch.tensor([[False]], dtype=torch.bool, device=device)

    TensorBatchEnv._update_free_voxels(
        env,
        eyes,
        target_idx,
        valid_mask,
        rays_world=rays_world,
        hit_pixel_mask=hit_pixel_mask,
    )
    got = {
        (0, 0, int(z))
        for z in torch.nonzero(env._prob_grid[0, 0, 0, :] < 0, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    }
    expected = {(0, 0, z) for z in range(8)}
    print(f"empty miss ray free path length={len(got)}  (expected 8)")
    assert got == expected, sorted(got)
    assert torch.allclose(
        env._prob_grid[0, 0, 0, :],
        torch.full((8,), LOG_ODDS_FREE, dtype=torch.float32, device=device),
    )


def test_tensor_env_empty_ray_pair_dedupe_preserves_path_set():
    try:
        from shapenbv.tensor_env import TensorBatchEnv
    except ImportError as e:
        pytest.skip(f"tensor_env import failed: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = object.__new__(TensorBatchEnv)
    env.grid_size = 8
    env._bbox_min = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    env._voxel_size = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)

    eye = torch.tensor([0.5, 0.5, -1.0], dtype=torch.float32, device=device)
    rays_world = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    miss_mask = torch.tensor([True, True, True], dtype=torch.bool, device=device)

    env.dedupe_empty_ray_pairs = False
    paths_full = TensorBatchEnv._empty_ray_paths_for_env(
        env, 0, eye, rays_world, miss_mask,
    )
    env.dedupe_empty_ray_pairs = True
    stats = {}
    paths_dedup = TensorBatchEnv._empty_ray_paths_for_env(
        env, 0, eye, rays_world, miss_mask, stats=stats,
    )

    full_set = set(map(tuple, paths_full.detach().cpu().tolist()))
    dedup_set = set(map(tuple, paths_dedup.detach().cpu().tolist()))
    print(
        "empty ray pair dedupe "
        f"before={stats['empty_ray_pairs_before_unique']} "
        f"after={stats['empty_ray_pairs_after_unique']} "
        f"path_rows={paths_full.shape[0]}->{paths_dedup.shape[0]}"
    )
    assert full_set == dedup_set == {(0, 0, z) for z in range(8)}
    assert stats["empty_ray_pairs_before_unique"] == 3
    assert stats["empty_ray_pairs_after_unique"] == 1


def test_mesh_ray_parity_inside_cube():
    from shapenbv.mesh_renderer import _points_inside_mesh_ray_parity

    verts = torch.tensor(
        [
            (-0.5, -0.5, -0.5), (+0.5, -0.5, -0.5), (+0.5, +0.5, -0.5), (-0.5, +0.5, -0.5),
            (-0.5, -0.5, +0.5), (+0.5, -0.5, +0.5), (+0.5, +0.5, +0.5), (-0.5, +0.5, +0.5),
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (2, 6, 7), (2, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (0, 3, 7), (0, 7, 4),
        ],
        dtype=torch.long,
    )
    points = torch.tensor(
        [
            (0.0, 0.0, 0.0),
            (0.45, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (0.0, 0.0, 0.75),
        ],
        dtype=torch.float32,
    )
    inside = _points_inside_mesh_ray_parity(
        points,
        verts[faces[:, 0]],
        verts[faces[:, 1]],
        verts[faces[:, 2]],
    )
    print(f"cube ray-parity inside={inside.tolist()}  (expected [T,T,F,F])")
    assert inside.tolist() == [True, True, False, False], inside


def test_tri_class():
    from shapenbv.voxel_utils import grid_occupancy_tri_cls
    g = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
    g[0, 0, 0, 0] = 0.8       # > 0.5 → +1
    g[0, 1, 1, 1] = -0.1      # < 0   → -1
    g[0, 2, 2, 2] = 0.2       #  in [0, 0.5] → 0
    tri = grid_occupancy_tri_cls(g, return_tri_cls_only=True)
    print(f"tri @ (0,0,0)={int(tri[0,0,0,0])}  (1,1,1)={int(tri[0,1,1,1])}  "
          f"(2,2,2)={int(tri[0,2,2,2])}")
    assert int(tri[0, 0, 0, 0]) == 1
    assert int(tri[0, 1, 1, 1]) == -1
    assert int(tri[0, 2, 2, 2]) == 0


if __name__ == "__main__":
    tests = [test_pose_to_idx_origin, test_pts_to_idx_corner,
             test_bresenham_axis, test_bresenham_strict_axis_full_free_path,
             test_bresenham_strict_diagonal_full_free_path,
             test_tensor_env_free_update_uses_full_axis_traversal,
             test_tensor_env_empty_ray_update_marks_grid_box_path,
             test_mesh_ray_parity_inside_cube,
             test_tri_class]
    ok = True
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except Exception as e:
            ok = False
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)
