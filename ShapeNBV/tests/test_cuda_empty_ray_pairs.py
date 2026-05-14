from __future__ import annotations

import pytest


def test_cuda_empty_ray_pairs_match_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA empty-ray pair builder requires CUDA")

    from shapenbv.cuda_bresenham import empty_ray_pairs_cuda
    from shapenbv.tensor_env import TensorBatchEnv

    device = torch.device("cuda")
    env = object.__new__(TensorBatchEnv)
    env.grid_size = 8
    env.dedupe_empty_ray_pairs = False
    env._bbox_min = torch.tensor(
        [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
        dtype=torch.float32,
        device=device,
    )
    env._voxel_size = torch.tensor(
        [[1.0, 1.0, 1.0], [0.25, 0.25, 0.25]],
        dtype=torch.float32,
        device=device,
    )
    eyes = torch.tensor(
        [[0.5, 0.5, -1.0], [0.0, -2.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    rays = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.2, 0.1, 1.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.3, 1.0, 0.1],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    rays = rays / rays.norm(dim=-1, keepdim=True)
    hit = torch.tensor(
        [
            [False, False, True, False],
            [False, True, False, False],
        ],
        dtype=torch.bool,
        device=device,
    )

    got_env, got_src, got_tgt, before, after = empty_ray_pairs_cuda(
        eyes,
        rays,
        hit,
        env._bbox_min,
        env._voxel_size,
        grid_size=env.grid_size,
        dedupe=False,
    )

    ref_env_parts = []
    ref_src_parts = []
    ref_tgt_parts = []
    for env_idx in range(2):
        src, tgt = TensorBatchEnv._empty_ray_pairs_for_env(
            env,
            env_idx,
            eyes[env_idx],
            rays[env_idx],
            ~hit[env_idx],
        )
        if src.numel() == 0:
            continue
        ref_env_parts.append(torch.full((src.shape[0],), env_idx, dtype=torch.long, device=device))
        ref_src_parts.append(src)
        ref_tgt_parts.append(tgt)
    ref_env = torch.cat(ref_env_parts, dim=0)
    ref_src = torch.cat(ref_src_parts, dim=0)
    ref_tgt = torch.cat(ref_tgt_parts, dim=0)

    assert before == int(ref_src.shape[0])
    assert after == before
    assert torch.equal(got_env, ref_env)
    assert torch.equal(got_src, ref_src)
    assert torch.equal(got_tgt, ref_tgt)


def test_cuda_empty_ray_pair_dedupe_matches_reference_set():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA empty-ray pair builder requires CUDA")

    from shapenbv.cuda_bresenham import empty_ray_pairs_cuda
    from shapenbv.tensor_env import TensorBatchEnv

    device = torch.device("cuda")
    G = 8
    bbox_min = torch.zeros((1, 3), dtype=torch.float32, device=device)
    voxel_size = torch.ones((1, 3), dtype=torch.float32, device=device)
    eyes = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device=device)
    rays = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    hit = torch.zeros((1, 3), dtype=torch.bool, device=device)

    got_env, got_src, got_tgt, before, after = empty_ray_pairs_cuda(
        eyes,
        rays,
        hit,
        bbox_min,
        voxel_size,
        grid_size=G,
        dedupe=True,
    )

    env = object.__new__(TensorBatchEnv)
    env.grid_size = G
    env.dedupe_empty_ray_pairs = True
    env._bbox_min = bbox_min
    env._voxel_size = voxel_size
    ref_src, ref_tgt = TensorBatchEnv._empty_ray_pairs_for_env(
        env,
        0,
        eyes[0],
        rays[0],
        ~hit[0],
    )

    got = set(
        zip(
            got_env.detach().cpu().tolist(),
            map(tuple, got_src.detach().cpu().tolist()),
            map(tuple, got_tgt.detach().cpu().tolist()),
        )
    )
    ref = {
        (0, tuple(src), tuple(tgt))
        for src, tgt in zip(ref_src.detach().cpu().tolist(), ref_tgt.detach().cpu().tolist())
    }
    assert before == 3
    assert after == 1
    assert got == ref
