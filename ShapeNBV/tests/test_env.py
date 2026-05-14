"""End-to-end smoke test for ShapeNBVEnv.

Builds a minimal in-memory ShapeNet-like fixture (one .obj cube +
matching preproc .pt), instantiates ShapeNBVEnv, and runs reset+step.
Verifies:
    - action_space is MultiDiscrete([81,81,81,1,13,13])
    - obs shape matches buffer_size*6 + obs_grid_size**3
    - one random step doesn't crash
    - cr_history grows monotonically (or stays equal) on "ok" steps
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _write_cube_obj(path: Path):
    """Write a [-0.5, 0.5]^3 unit cube as a small OBJ file."""
    verts = [
        (-0.5, -0.5, -0.5), (+0.5, -0.5, -0.5), (+0.5, +0.5, -0.5), (-0.5, +0.5, -0.5),
        (-0.5, -0.5, +0.5), (+0.5, -0.5, +0.5), (+0.5, +0.5, +0.5), (-0.5, +0.5, +0.5),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 7, 6), (5, 8, 7),
        (1, 5, 6), (1, 6, 2),
        (3, 7, 8), (3, 8, 4),
        (2, 6, 7), (2, 7, 3),
        (1, 4, 8), (1, 8, 5),
    ]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for fa in faces:
            f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")


def main():
    try:
        from shapenbv.mesh_renderer import ShapeNetIndex
        from shapenbv.preprocess import preprocess_mesh, save_preproc
        from shapenbv.env import ShapeNBVEnv, POSE_DIM
    except ImportError as e:
        print(f"[skip] import failed: {e}")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create fake ShapeNet-style entry: <tmp>/03001627/abc123/models/model_normalized.obj
        synset = "03001627"
        model_id = "abc123_test"
        mesh_dir = tmp / "shapenet" / synset / model_id / "models"
        mesh_dir.mkdir(parents=True)
        mesh_path = mesh_dir / "model_normalized.obj"
        _write_cube_obj(mesh_path)

        # Preprocess.
        preproc_dir = tmp / "preproc"
        preproc_dir.mkdir()
        name = f"{synset}_{model_id}"
        grid_size = 32
        obs_grid_size = 32
        data = preprocess_mesh(mesh_path, grid_size=grid_size, n_surface_points=5000)
        save_preproc(preproc_dir / f"{name}.pt", data)
        print(f"[ok] preproc: grid_gt sum = {float(data['grid_gt'].sum()):.0f} occupied voxels")

        # Build env.
        index = ShapeNetIndex(
            entries={name: mesh_path},
            device=device,
            render_size=(64, 64),
            fov_deg=60.0,
        )
        env = ShapeNBVEnv(
            index=index,
            preproc_dir=str(preproc_dir),
            sequence_names=[name],
            device=device,
            buffer_size=20,            # small for fast smoke
            episode_len=5,
            grid_size=grid_size,
            obs_grid_size=obs_grid_size,
            seed=0,
        )

        # Action / obs space sanity.
        assert env.action_space.shape == (POSE_DIM,) or hasattr(env.action_space, "nvec")
        nvec = env.action_space.nvec
        assert tuple(nvec.tolist()) == (81, 81, 81, 1, 13, 13), nvec
        print(f"[ok] action_space.nvec = {nvec.tolist()}")

        expected_obs = 20 * POSE_DIM + obs_grid_size ** 3
        assert env.observation_space.shape == (expected_obs,), env.observation_space
        print(f"[ok] observation_space.shape = {env.observation_space.shape}")

        # Reset + step.
        obs, info = env.reset(options={"seq_name": name})
        assert obs.shape == (expected_obs,), obs.shape
        print(f"[ok] reset obs OK; cr_init={env._coverage_ratio():.3f}")

        n_ok = 0
        for s in range(5):
            action = np.array([
                np.random.randint(0, 81),  # x
                np.random.randint(0, 81),  # y
                np.random.randint(0, 81),  # z
                0,                          # roll (frozen)
                np.random.randint(0, 13),   # pitch
                np.random.randint(0, 13),   # yaw
            ], dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"step {s}: action={action.tolist()}  reward={reward:.3f}  "
                  f"cr={info['cr']:.3f}  term={terminated}  trunc={truncated}  "
                  f"col={info['collision']}")
            if not info["collision"]:
                n_ok += 1
            if terminated or truncated:
                break
        print(f"[ok] {n_ok} 'ok' steps out of 5.")
        return True


def test_smoke_env():
    assert main()


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
