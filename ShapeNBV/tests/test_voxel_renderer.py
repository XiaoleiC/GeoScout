"""Voxel first-hit renderer correctness tests."""
from __future__ import annotations

import torch


def test_voxel_first_hit_reference_hits_expected_cell():
    from shapenbv.cuda_voxel_renderer import voxel_first_hit_reference

    grid = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
    grid[0, 2, 2, 2] = 1.0
    bbox_min = torch.zeros(1, 3, dtype=torch.float32)
    voxel_size = torch.full((1, 3), 0.25, dtype=torch.float32)
    eyes = torch.tensor([[-0.5, 0.625, 0.625]], dtype=torch.float32)
    rays = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

    target, depth, hit = voxel_first_hit_reference(
        grid, bbox_min, voxel_size, eyes, rays,
    )

    assert bool(hit[0, 0])
    assert target[0, 0].tolist() == [2, 2, 2]
    assert 0.9 <= float(depth[0, 0]) <= 1.1


def test_voxel_first_hit_reference_misses_parallel_empty_line():
    from shapenbv.cuda_voxel_renderer import voxel_first_hit_reference

    grid = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
    grid[0, 2, 2, 2] = 1.0
    bbox_min = torch.zeros(1, 3, dtype=torch.float32)
    voxel_size = torch.full((1, 3), 0.25, dtype=torch.float32)
    eyes = torch.tensor([[-0.5, 0.1, 0.1]], dtype=torch.float32)
    rays = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

    target, depth, hit = voxel_first_hit_reference(
        grid, bbox_min, voxel_size, eyes, rays,
    )

    assert not bool(hit[0, 0])
    assert target[0, 0].tolist() == [0, 0, 0]
    assert float(depth[0, 0]) == 0.0


def test_voxel_first_hit_cuda_matches_reference_when_available():
    if not torch.cuda.is_available():
        return
    from shapenbv.cuda_voxel_renderer import voxel_first_hit_cuda, voxel_first_hit_reference

    device = torch.device("cuda")
    grid = torch.zeros(2, 8, 8, 8, dtype=torch.float32, device=device)
    grid[0, 4, 4, 4] = 1.0
    grid[1, 1, 6, 2] = 1.0
    bbox_min = torch.zeros(2, 3, dtype=torch.float32, device=device)
    voxel_size = torch.full((2, 3), 0.125, dtype=torch.float32, device=device)
    eyes = torch.tensor(
        [[-0.25, 0.5625, 0.5625], [0.1875, 1.25, 0.3125]],
        dtype=torch.float32,
        device=device,
    )
    rays = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
        device=device,
    )

    ref_target, ref_depth, ref_hit = voxel_first_hit_reference(
        grid, bbox_min, voxel_size, eyes, rays,
    )
    out = voxel_first_hit_cuda(grid, bbox_min, voxel_size, eyes, rays)
    assert out is not None
    cuda_target, cuda_depth, cuda_hit = out

    assert torch.equal(cuda_hit, ref_hit)
    assert torch.equal(cuda_target, ref_target)
    assert torch.allclose(cuda_depth, ref_depth, atol=1e-5)
