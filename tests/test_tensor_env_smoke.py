"""End-to-end smoke test for TensorBatchEnv."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch


def _write_cube_obj(path: Path):
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
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def test_tensor_env_reset_step_and_collision():
    from geoscout.preprocess import preprocess_mesh, save_preproc
    from geoscout.tensor_env import TensorBatchEnv, POSE_DIM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mesh_path = tmp_path / "cube.obj"
        preproc_path = tmp_path / "cube.pt"
        _write_cube_obj(mesh_path)
        data = preprocess_mesh(
            mesh_path,
            grid_size=32,
            n_surface_points=5000,
            grid_storage_dtype="uint8",
        )
        save_preproc(preproc_path, data)

        env = TensorBatchEnv(
            num_envs=2,
            mesh_paths=[mesh_path],
            preproc_paths=[preproc_path],
            device=device,
            buffer_size=4,
            grid_size=32,
            obs_grid_size=16,
            episode_len=3,
            render_size=32,
            fov_deg=60.0,
            auto_lookat_center=True,
            skip_free_raycast=True,
            update_empty_rays=True,
            coverage_hit_dilate_radius=1,
            collision_penalty=10.0,
            seed=0,
        )

        obs = env.reset()
        expected_obs = 4 * POSE_DIM + 16 ** 3
        assert obs.shape == (2, expected_obs)

        actions = np.array(
            [
                [40, 40, 40, 0, 0, 0],  # eye at origin, inside cube
                [40, 40, 80, 0, 0, 0],  # top view, outside cube
            ],
            dtype=np.int64,
        )
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()

        assert obs.shape == (2, expected_obs)
        assert dones[0]
        assert infos[0]["collision"]
        assert rewards[0] <= -9.9
        assert "terminal_observation" in infos[0]
        assert not infos[1]["collision"]
        env.close()
