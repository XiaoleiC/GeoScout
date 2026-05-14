"""Minimal Modal entrypoint for GeoScout visual smoke validation.

This file intentionally avoids the larger `modal_app.py` caption/download
entrypoints and their optional secrets. It is meant for account migration and
checkpoint-validation work where only `/data`, `/runs`, CUDA, and the local
GeoScout package are needed.
"""
from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("geoscout-smoke")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "wget", "unzip")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy",
        "scipy",
        "trimesh",
        "matplotlib",
        "open3d",
        "stable-baselines3==2.3.2",
        "gymnasium",
        "pillow",
        "wheel",
        "ninja",
        "wandb",
    )
    .pip_install(
        "numpy==1.26.4",
        "jax[cuda12]==0.4.35",
        "flax==0.10.4",
        "optax==0.2.4",
        "distrax==0.1.5",
        "ml_collections",
        "tqdm",
    )
    .env({
        "PYTHONPATH": "/workspace/GeoScout",
        "CC": "gcc",
        "CXX": "g++",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9;9.0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    .run_commands(
        "python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"
    )
    # Keep this smoke app lean: the full GeoScout tree contains hundreds of MB
    # of prior visualization/report artifacts. Upload only importable code.
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parents[1] / "geoscout"),
        remote_path="/workspace/GeoScout/geoscout",
    )
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parents[1] / "scripts"),
        remote_path="/workspace/GeoScout/scripts",
    )
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parents[1] / "offline_rl"),
        remote_path="/workspace/GeoScout/offline_rl",
    )
)

vol_data = modal.Volume.from_name("geoscout-data", create_if_missing=True)
vol_runs = modal.Volume.from_name("geoscout-runs", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=30 * 60,
    memory=8 * 1024,
    retries=0,
)
def probe_caption_counterfactual_policy(
    ckpt_subdir: str = "train600-24m-discrete-s1-judged-l40s-n128-wandb-0508",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2_judged_v1",
    out_subdir: str = "fig4_caption_counterfactual_probe_0512",
    seq_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "04379243_156d606fa86ba19c4eb174a255d0ec5e,"
        "04256520_1050790962944624febad4f49b26ec52"
    ),
    labels: str = "chair,table,sofa",
    buffer_size: int = 30,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
):
    """Export caption-counterfactual action marginals for the Fig. 4 probe.

    The probe fixes pose history and belief grid to the same initial state
    (zeros: no camera history and unknown tri-class grid) and swaps only the
    caption embedding. The resulting MultiCategorical marginals reveal how the
    trained caption-conditioned PPO prior moves mass over the NBV action grid.
    """
    import json
    import math
    import sys
    import types
    from pathlib import Path

    import numpy as np
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO

    from geoscout.tensor_env import DEFAULT_ACTION_LOW_WORLD, DEFAULT_ACTION_UNIT, NVEC

    ckpt_path = Path("/runs") / ckpt_subdir / "ppo_geoscout.zip"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    seq_list = [s.strip() for s in seq_names.split(",") if s.strip()]
    label_list = [s.strip() for s in labels.split(",") if s.strip()]
    if len(seq_list) != len(label_list):
        raise ValueError("seq_names and labels must have the same length")

    # Some PPO checkpoints were cloudpickled under NumPy 2.x, whose module path
    # for internals is `numpy._core.*`.  The lean smoke image pins NumPy 1.26 for
    # JAX/FQL compatibility, where the same modules live under `numpy.core.*`.
    # Register aliases so cloudpickle can resolve the saved observation/action
    # spaces without changing the runtime numerics.
    if "numpy._core" not in sys.modules:
        np_core_mod = types.ModuleType("numpy._core")
        np_core_mod.__dict__.update(np.core.__dict__)
        sys.modules["numpy._core"] = np_core_mod
    try:
        import numpy.core.numeric as np_numeric

        sys.modules.setdefault("numpy._core.numeric", np_numeric)
    except Exception:
        pass

    grid_dim = int(obs_grid_size) ** 3
    pose_dim = int(buffer_size) * 6
    obs_dim = pose_dim + grid_dim + int(caption_dim)
    custom_objects = {
        "observation_space": spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        ),
        "action_space": spaces.MultiDiscrete(np.asarray(NVEC, dtype=np.int64)),
    }
    model = PPO.load(str(ckpt_path), device="cpu", custom_objects=custom_objects)
    if pose_dim <= 0 or pose_dim % 6 != 0:
        raise ValueError(
            f"unexpected obs_dim={obs_dim} for obs_grid_size={obs_grid_size} "
            f"caption_dim={caption_dim}"
        )

    def load_caption_emb(seq: str) -> np.ndarray:
        pp = Path(preproc_dir) / f"{seq}.pt"
        if not pp.exists():
            raise FileNotFoundError(f"missing preproc: {pp}")
        payload = torch.load(str(pp), map_location="cpu")
        emb = payload.get("caption_emb")
        if emb is None:
            raise KeyError(f"caption_emb missing in {pp}")
        arr = emb.float().view(-1).cpu().numpy().astype(np.float32)
        if arr.shape[0] != int(caption_dim):
            raise ValueError(f"{seq}: caption dim {arr.shape[0]} != {caption_dim}")
        return arr

    caption_embs = {
        label: {
            "seq_name": seq,
            "embedding": load_caption_emb(seq),
        }
        for label, seq in zip(label_list, seq_list)
    }
    caption_embs["no caption"] = {
        "seq_name": "",
        "embedding": np.zeros(int(caption_dim), dtype=np.float32),
    }

    action_low = np.asarray(DEFAULT_ACTION_LOW_WORLD, dtype=np.float32)
    action_unit = np.asarray(DEFAULT_ACTION_UNIT, dtype=np.float32)
    nvec = np.asarray(NVEC, dtype=np.int64)

    model.policy.set_training_mode(False)
    conditions = []
    for label, item in caption_embs.items():
        obs = np.zeros((1, obs_dim), dtype=np.float32)
        obs[:, pose_dim + grid_dim:] = item["embedding"][None, :]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
        cats = getattr(dist, "distribution", None)
        if not isinstance(cats, list):
            raise TypeError(f"expected MultiCategorical distribution list, got {type(cats)!r}")
        probs = [c.probs.detach().cpu().numpy()[0].astype(float) for c in cats]
        det_action, _ = model.predict(obs, deterministic=True)
        det = np.asarray(det_action[0], dtype=np.int64)
        det_pose = det.astype(np.float32) * action_unit + action_low

        # Position marginal entropy is a compact way to quantify whether the
        # caption creates a sharp directional prior or leaves the policy broad.
        pos_entropy = 0.0
        for axis_probs in probs[:3]:
            p = np.asarray(axis_probs, dtype=float)
            p = p / max(float(p.sum()), 1e-12)
            pos_entropy += float(-(p * np.log(p + 1e-12)).sum())
        pos_entropy_norm = pos_entropy / float(sum(math.log(int(n)) for n in nvec[:3]))

        conditions.append({
            "label": label,
            "seq_name": item["seq_name"],
            "det_action": det.tolist(),
            "det_pose": det_pose.tolist(),
            "pos_entropy_norm": pos_entropy_norm,
            "marginals": [p.tolist() for p in probs],
        })

    out_dir = Path("/runs") / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": str(ckpt_path),
        "preproc_dir": str(preproc_dir),
        "obs_dim": obs_dim,
        "pose_dim": pose_dim,
        "grid_dim": grid_dim,
        "caption_dim": int(caption_dim),
        "nvec": nvec.tolist(),
        "action_low": action_low.tolist(),
        "action_unit": action_unit.tolist(),
        "conditions": conditions,
    }
    out_path = out_dir / "probe.json"
    out_path.write_text(json.dumps(payload, indent=2))
    vol_runs.commit()
    print(f"[fig4-probe] wrote {out_path}", flush=True)
    return {"out_path": str(out_path), "conditions": [c["label"] for c in conditions]}


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    memory=24 * 1024,
    retries=0,
)
def export_caption_swap_rollouts(
    ckpt_subdir: str = "train600-24m-discrete-s1-judged-l40s-n128-wandb-0508",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2_judged_v1",
    out_subdir: str = "fig4_hard_geometry_caption_contrast_0513",
    fixed_seq_names: str = (
        "table=04379243_14922c38b2cbfce6fa31c88352968918,"
        "chair=03001627_18f2f833d95ad137111c729c2fe5f751,"
        "sofa=04256520_1b25f96d97a94b05125abe33bf4f0061"
    ),
    simple_caption_seq_names: str = (
        "table=04379243_121a3040c28295829e4b5aa807bb4e7,"
        "chair=03001627_115b11a77b8d8c3c110a27d1d78196,"
        "sofa=04256520_17a768e79ba434b91ca25a4447d3477e"
    ),
    detailed_caption_seq_names: str = (
        "table=04379243_1509a8710d2fce3c4785a5d3b6c47521,"
        "chair=03001627_11e521e41ff6a64922e4620665c23c97,"
        "sofa=04256520_16dca17207a6a2b87f6fd4fd84c364f4"
    ),
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    max_faces: int = 5000,
    seed: int = 0,
):
    """Roll out hard objects under geometry, simple, detailed, and no captions.

    Each category uses one fixed hard geometry and swaps only the caption
    embedding. This is an exploratory probe for deciding how to redesign Fig. 4.
    """
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from scripts.evaluate_visual_rollouts import install_numpy_pickle_compat
    from geoscout.data import list_shapenet
    from geoscout.tensor_env import TensorBatchEnv

    def parse_map(text: str) -> dict[str, str]:
        out = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
        return out

    fixed = parse_map(fixed_seq_names)
    simple = parse_map(simple_caption_seq_names)
    detailed = parse_map(detailed_caption_seq_names)
    categories = [c for c in ("table", "chair", "sofa") if c in fixed]
    if set(categories) - set(simple):
        raise ValueError("simple_caption_seq_names must include every fixed category")
    if set(categories) - set(detailed):
        raise ValueError("detailed_caption_seq_names must include every fixed category")

    entries = list_shapenet(
        Path("/data/ShapeNetCore.v2"),
        synsets=["03001627", "04256520", "04379243"],
        require_obj=True,
    )
    mesh_by_name = {entry.name: Path(entry.mesh_path) for entry in entries}
    mesh_paths = []
    preproc_paths = []
    for cat in categories:
        seq = fixed[cat]
        if seq not in mesh_by_name:
            raise FileNotFoundError(f"missing ShapeNet mesh for {seq}")
        pp = Path(preproc_dir) / f"{seq}.pt"
        if not pp.exists():
            raise FileNotFoundError(f"missing fixed preproc for {seq}: {pp}")
        mesh_paths.append(mesh_by_name[seq])
        preproc_paths.append(pp)

    env = TensorBatchEnv(
        num_envs=len(categories),
        mesh_paths=mesh_paths,
        preproc_paths=preproc_paths,
        device="cuda",
        buffer_size=30,
        grid_size=grid_size,
        obs_grid_size=obs_grid_size,
        episode_len=episode_len,
        render_size=image_size,
        cr_success_threshold=0.99,
        coverage_reward_scale=20.0,
        termination_bonus=1.0,
        collision_penalty=10.0,
        short_path_grace=30,
        short_path_clip=2.0,
        short_path_scale=0.1,
        only_positive_rewards=True,
        coverage_hit_dilate_radius=1,
        caption_dim=caption_dim,
        auto_lookat_center=True,
        action_space_type="discrete",
        max_faces=max_faces,
        renderer_backend="voxel_cuda",
        free_raycast_backend="cuda",
        free_mask_apply_mode="triton",
        triton_bresenham_block_rays=64,
        seed=seed,
    )

    ckpt_path = Path("/runs") / ckpt_subdir / "ppo_geoscout.zip"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    install_numpy_pickle_compat()
    model = PPO.load(
        str(ckpt_path),
        device="cuda",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )

    def caption_embedding(seq: str) -> torch.Tensor:
        pp = Path(preproc_dir) / f"{seq}.pt"
        if not pp.exists():
            raise FileNotFoundError(f"missing caption-source preproc for {seq}: {pp}")
        payload = torch.load(str(pp), map_location="cpu")
        emb = payload.get("caption_emb")
        if emb is None:
            raise KeyError(f"caption_emb missing in {pp}")
        return emb.float().view(-1).to(env.device)

    conditions = {
        "geometry_caption": [caption_embedding(fixed[cat]) for cat in categories],
        "simple_caption": [caption_embedding(simple[cat]) for cat in categories],
        "detailed_caption": [caption_embedding(detailed[cat]) for cat in categories],
        "no_caption": [torch.zeros(caption_dim, dtype=torch.float32, device=env.device) for _ in categories],
    }
    caption_sources = {
        "geometry_caption": fixed,
        "simple_caption": simple,
        "detailed_caption": detailed,
        "no_caption": {cat: "" for cat in categories},
    }

    all_rollouts: dict[str, dict[str, list[dict]]] = {cat: {} for cat in categories}
    mesh_ids = np.arange(len(categories), dtype=np.int64)
    for condition_name, embs in conditions.items():
        obs = env.reset_to_mesh_ids(mesh_ids)
        if env._caption_emb is None:
            raise RuntimeError("environment has no caption channel")
        env._caption_emb[:] = torch.stack(embs, dim=0)
        obs = env._build_observation_np()
        active = np.ones(len(categories), dtype=bool)
        rows_by_env = {i: [] for i in range(len(categories))}

        for _step in range(1, episode_len + 1):
            actions, _ = model.predict(obs, deterministic=True)
            actions = np.asarray(actions, dtype=np.int64)
            action_t = torch.as_tensor(actions, dtype=torch.long, device=env.device)
            pose6_t, eyes_t, ats_t = env._decode_actions(action_t)
            pose6 = pose6_t.detach().cpu().numpy().astype(np.float32)
            eyes = eyes_t.detach().cpu().numpy().astype(np.float32)
            ats = ats_t.detach().cpu().numpy().astype(np.float32)

            obs, rewards, dones, infos = env.step(actions)
            for i, cat in enumerate(categories):
                if not active[i]:
                    continue
                info = infos[i]
                row = {
                    "policy": condition_name,
                    "category": cat,
                    "fixed_seq_name": fixed[cat],
                    "caption_source_seq_name": caption_sources[condition_name][cat],
                    "step": int(len(rows_by_env[i]) + 1),
                    "cr": float(info.get("cr", 0.0)),
                    "coverage_delta": float(info.get("coverage_delta", 0.0)),
                    "reward": float(rewards[i]),
                    "collision": bool(info.get("collision", False)),
                    "done": bool(dones[i]),
                    "early_stopped": bool(info.get("early_stopped", False)),
                    "timeout": bool(info.get("TimeLimit.truncated", False)),
                    "action": np.asarray(actions[i]).reshape(-1).tolist(),
                    "pose6": pose6[i].reshape(-1).tolist(),
                    "eye": eyes[i].reshape(-1).tolist(),
                    "look_at": ats[i].reshape(-1).tolist(),
                }
                rows_by_env[i].append(row)
                if bool(dones[i]):
                    active[i] = False
            if not active.any():
                break

        for i, cat in enumerate(categories):
            all_rollouts[cat][condition_name] = rows_by_env[i]

    env.close()
    out_dir = Path("/runs") / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": str(ckpt_path),
        "preproc_dir": preproc_dir,
        "categories": categories,
        "fixed_seq_names": fixed,
        "simple_caption_seq_names": simple,
        "detailed_caption_seq_names": detailed,
        "conditions": list(conditions.keys()),
        "rollouts": all_rollouts,
    }
    out_path = out_dir / "caption_swap_rollouts.json"
    out_path.write_text(json.dumps(payload, indent=2))
    vol_runs.commit()
    print(f"[caption-swap] wrote {out_path}", flush=True)
    return {"out_path": str(out_path), "categories": categories}


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    memory=32 * 1024,
    retries=0,
)
def export_caption_contrast_grid_rollouts(
    ckpt_subdir: str = "train600-24m-discrete-s1-judged-l40s-n128-wandb-0508",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2_judged_v1",
    out_subdir: str = "fig4_caption_contrast_grid_0513",
    fixed_seq_names: str = (
        "chair=03001627_1190af00b6c86c99c3bd24f986301745|03001627_18f2f833d95ad137111c729c2fe5f751|03001627_1049953406c81b237eaeab1f0c9120b7,"
        "table=04379243_14922c38b2cbfce6fa31c88352968918|04379243_146f90f6a4d8c7bd142fb08fcc642f29|04379243_124c4b3afa6a3e56eaf288f952624966,"
        "sofa=04256520_1b25f96d97a94b05125abe33bf4f0061|04256520_157ed8452a7edab161412053ff521f64|04256520_1543a5ea73b6ab10df2fa7eaa812363c"
    ),
    simple_caption_seq_names: str = (
        "table=04379243_121a3040c28295829e4b5aa807bb4e7,"
        "chair=03001627_115b11a77b8d8c3c110a27d1d78196,"
        "sofa=04256520_17a768e79ba434b91ca25a4447d3477e"
    ),
    detailed_caption_seq_names: str = (
        "table=04379243_1509a8710d2fce3c4785a5d3b6c47521,"
        "chair=03001627_11e521e41ff6a64922e4620665c23c97,"
        "sofa=04256520_16dca17207a6a2b87f6fd4fd84c364f4"
    ),
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    max_faces: int = 5000,
    seed: int = 0,
):
    """Roll out a 3 x 3 hard-geometry caption contrast grid.

    For each fixed object, only the caption embedding changes:
    its own matched caption, a simple same-category caption, or a detailed
    same-category caption. No no-caption baseline is included here because this
    probe is intended to isolate caption-text semantics rather than absence of
    caption prior.
    """
    import json
    import os
    from pathlib import Path

    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from scripts.evaluate_visual_rollouts import install_numpy_pickle_compat
    from geoscout.data import list_shapenet
    from geoscout.tensor_env import TensorBatchEnv

    def parse_seq_groups(text: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=", 1)
            out[key.strip()] = [seq.strip() for seq in value.split("|") if seq.strip()]
        return out

    def parse_map(text: str) -> dict[str, str]:
        out = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
        return out

    fixed = parse_seq_groups(fixed_seq_names)
    simple = parse_map(simple_caption_seq_names)
    detailed = parse_map(detailed_caption_seq_names)
    category_order = [c for c in ("chair", "table", "sofa") if c in fixed]
    if set(category_order) - set(simple):
        raise ValueError("simple_caption_seq_names must include every fixed category")
    if set(category_order) - set(detailed):
        raise ValueError("detailed_caption_seq_names must include every fixed category")

    samples = [
        {"category": cat, "category_index": idx + 1, "seq_name": seq}
        for cat in category_order
        for idx, seq in enumerate(fixed[cat])
    ]

    entries = list_shapenet(
        Path("/data/ShapeNetCore.v2"),
        synsets=["03001627", "04256520", "04379243"],
        require_obj=True,
    )
    mesh_by_name = {entry.name: Path(entry.mesh_path) for entry in entries}
    mesh_paths = []
    preproc_paths = []
    for sample in samples:
        seq = sample["seq_name"]
        if seq not in mesh_by_name:
            raise FileNotFoundError(f"missing ShapeNet mesh for {seq}")
        pp = Path(preproc_dir) / f"{seq}.pt"
        if not pp.exists():
            raise FileNotFoundError(f"missing fixed preproc for {seq}: {pp}")
        mesh_paths.append(mesh_by_name[seq])
        preproc_paths.append(pp)

    env = TensorBatchEnv(
        num_envs=len(samples),
        mesh_paths=mesh_paths,
        preproc_paths=preproc_paths,
        device="cuda",
        buffer_size=30,
        grid_size=grid_size,
        obs_grid_size=obs_grid_size,
        episode_len=episode_len,
        render_size=image_size,
        cr_success_threshold=0.99,
        coverage_reward_scale=20.0,
        termination_bonus=1.0,
        collision_penalty=10.0,
        short_path_grace=30,
        short_path_clip=2.0,
        short_path_scale=0.1,
        only_positive_rewards=True,
        coverage_hit_dilate_radius=1,
        caption_dim=caption_dim,
        auto_lookat_center=True,
        action_space_type="discrete",
        max_faces=max_faces,
        renderer_backend="voxel_cuda",
        free_raycast_backend="cuda",
        free_mask_apply_mode="triton",
        triton_bresenham_block_rays=64,
        seed=seed,
    )

    ckpt_path = Path("/runs") / ckpt_subdir / "ppo_geoscout.zip"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    install_numpy_pickle_compat()
    model = PPO.load(
        str(ckpt_path),
        device="cuda",
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )

    def caption_embedding(seq: str) -> torch.Tensor:
        pp = Path(preproc_dir) / f"{seq}.pt"
        if not pp.exists():
            raise FileNotFoundError(f"missing caption-source preproc for {seq}: {pp}")
        payload = torch.load(str(pp), map_location="cpu")
        emb = payload.get("caption_emb")
        if emb is None:
            raise KeyError(f"caption_emb missing in {pp}")
        return emb.float().view(-1).to(env.device)

    conditions = {
        "matched_caption": [caption_embedding(sample["seq_name"]) for sample in samples],
        "simple_caption": [caption_embedding(simple[sample["category"]]) for sample in samples],
        "detailed_caption": [caption_embedding(detailed[sample["category"]]) for sample in samples],
    }

    def caption_source(condition_name: str, sample: dict) -> str:
        if condition_name == "matched_caption":
            return sample["seq_name"]
        if condition_name == "simple_caption":
            return simple[sample["category"]]
        if condition_name == "detailed_caption":
            return detailed[sample["category"]]
        raise KeyError(condition_name)

    all_rollouts: dict[str, dict[str, list[dict]]] = {sample["seq_name"]: {} for sample in samples}
    mesh_ids = np.arange(len(samples), dtype=np.int64)
    for condition_name, embs in conditions.items():
        obs = env.reset_to_mesh_ids(mesh_ids)
        if env._caption_emb is None:
            raise RuntimeError("environment has no caption channel")
        env._caption_emb[:] = torch.stack(embs, dim=0)
        obs = env._build_observation_np()
        active = np.ones(len(samples), dtype=bool)
        rows_by_env = {i: [] for i in range(len(samples))}

        for _step in range(1, episode_len + 1):
            actions, _ = model.predict(obs, deterministic=True)
            actions = np.asarray(actions, dtype=np.int64)
            action_t = torch.as_tensor(actions, dtype=torch.long, device=env.device)
            pose6_t, eyes_t, ats_t = env._decode_actions(action_t)
            pose6 = pose6_t.detach().cpu().numpy().astype(np.float32)
            eyes = eyes_t.detach().cpu().numpy().astype(np.float32)
            ats = ats_t.detach().cpu().numpy().astype(np.float32)

            obs, rewards, dones, infos = env.step(actions)
            for i, sample in enumerate(samples):
                if not active[i]:
                    continue
                seq = sample["seq_name"]
                info = infos[i]
                row = {
                    "policy": condition_name,
                    "category": sample["category"],
                    "category_index": int(sample["category_index"]),
                    "fixed_seq_name": seq,
                    "caption_source_seq_name": caption_source(condition_name, sample),
                    "step": int(len(rows_by_env[i]) + 1),
                    "cr": float(info.get("cr", 0.0)),
                    "coverage_delta": float(info.get("coverage_delta", 0.0)),
                    "reward": float(rewards[i]),
                    "collision": bool(info.get("collision", False)),
                    "done": bool(dones[i]),
                    "early_stopped": bool(info.get("early_stopped", False)),
                    "timeout": bool(info.get("TimeLimit.truncated", False)),
                    "action": np.asarray(actions[i]).reshape(-1).tolist(),
                    "pose6": pose6[i].reshape(-1).tolist(),
                    "eye": eyes[i].reshape(-1).tolist(),
                    "look_at": ats[i].reshape(-1).tolist(),
                }
                rows_by_env[i].append(row)
                if bool(dones[i]):
                    active[i] = False
            if not active.any():
                break

        for i, sample in enumerate(samples):
            all_rollouts[sample["seq_name"]][condition_name] = rows_by_env[i]

    env.close()
    out_dir = Path("/runs") / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": str(ckpt_path),
        "preproc_dir": preproc_dir,
        "categories": category_order,
        "samples": samples,
        "fixed_seq_names": fixed,
        "simple_caption_seq_names": simple,
        "detailed_caption_seq_names": detailed,
        "conditions": list(conditions.keys()),
        "rollouts": all_rollouts,
    }
    out_path = out_dir / "caption_contrast_grid_rollouts.json"
    out_path.write_text(json.dumps(payload, indent=2))
    vol_runs.commit()
    print(f"[caption-grid] wrote {out_path}", flush=True)
    return {"out_path": str(out_path), "num_samples": len(samples), "conditions": list(conditions.keys())}

@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=6 * 3600,
    memory=32 * 1024,
    retries=0,
)
def smoke_validate_8ckpt_visual(
    out_subdir: str = "eval_8ckpt_smoke_validation",
    seq_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "04256520_1050790962944624febad4f49b26ec52,"
        "04379243_156d606fa86ba19c4eb174a255d0ec5e"
    ),
    action_modes: str = "discrete_det,continuous_det,axis6",
    discrete_ckpt_subdir: str = "train600-24m-discrete-s0-l40s-n128-wandb-0506",
    continuous_ckpt_subdir: str = "train600-24m-continuous-s0-l40s-n128-wandb-0506",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    caption_jsonl: str = (
        "/data/geoscout_captions/"
        "full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl"
    ),
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
    seed: int = 0,
):
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.smoke_validate_8ckpt_visual",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--caption_jsonl", caption_jsonl,
        "--out_dir", out_dir,
        "--seq_names", seq_names,
        "--action_modes", action_modes,
        "--discrete_ckpt", f"/runs/{discrete_ckpt_subdir}/ppo_geoscout.zip",
        "--continuous_ckpt", f"/runs/{continuous_ckpt_subdir}/ppo_geoscout.zip",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--image_size", str(image_size),
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--caption_dim", str(caption_dim),
        "--coverage_hit_dilate_radius", "1",
        "--coverage_threshold", "0.99",
        "--coverage_reward_scale", "20",
        "--termination_bonus", "1",
        "--collision_penalty", "10",
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--max_faces", str(max_faces),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    print("[smoke-8ckpt] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[smoke-8ckpt] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=6 * 3600,
    memory=32 * 1024,
    retries=0,
)
def evaluate_best_ckpt_numeric(
    ckpt_subdir: str,
    out_subdir: str,
    action_space_type: str,
    policies: str = "ppo,axis6,random,fibonacci,ring",
    deterministic: bool = True,
    n_episodes: int = 600,
    n_envs: int = 64,
    episode_len: int = 50,
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    synsets: str = "03001627,04256520,04379243",
    limit_per_synset: int = 200,
    max_meshes: int = 0,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
    seed: int = 0,
):
    """Numerically evaluate one selected GeoScout checkpoint.

    Unlike the legacy `modal_app.py::evaluate_baselines_shapenet`, this entry
    keeps `auto_lookat_center=False` by default and exposes
    `action_space_type`, matching the 24M GeoScout training contract.
    """
    import os
    import subprocess
    import sys

    if action_space_type not in {"discrete", "continuous_tanh"}:
        raise ValueError(
            "action_space_type must be 'discrete' or 'continuous_tanh', "
            f"got {action_space_type!r}"
        )

    ckpt_path = f"/runs/{ckpt_subdir}/ppo_geoscout.zip"
    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_baselines",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--out_dir", out_dir,
        "--policies", policies,
        "--ckpt", ckpt_path,
        "--n_episodes", str(n_episodes),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", str(image_size),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_hit_dilate_radius", "1",
        "--action_space_type", action_space_type,
        "--caption_dim", str(caption_dim),
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--max_faces", str(max_faces),
        "--coverage_threshold", "0.99",
        "--coverage_reward_scale", "20",
        "--coverage_reward_type", "linear",
        "--termination_bonus", "1",
        "--collision_penalty", "10",
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
    ]
    if deterministic:
        cmd += ["--deterministic"]
    if max_meshes > 0:
        cmd += ["--max_meshes", str(max_meshes)]
    print("[best-ckpt-eval] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[best-ckpt-eval] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=12 * 3600,
    memory=32 * 1024,
    retries=0,
)
def visualize_selected_eval_cases(
    out_subdir: str = "eval_selected_cases_visual_0507",
    selected_cases_json: str = "/runs/eval_selected_cases_0507/reports/selected_cases.json",
    action_modes: str = "discrete_det,continuous_det,fibonacci,axis6,random",
    discrete_ckpt_subdir: str = "train600-24m-discrete-s1-l40s-n128-wandb-0506",
    continuous_ckpt_subdir: str = "train600-24m-continuous-s1-l40s-n128-wandb-0506",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    caption_jsonl: str = (
        "/data/geoscout_captions/"
        "full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl"
    ),
    contact_sheet_dir: str = "/runs/caption_full_results_zup_v1/contact_sheets_600_attr_v2",
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
    step_frame_mode: str = "key",
    max_step_frames: int = 12,
    seed: int = 0,
):
    """Replay selected validation cases and build a visual atlas.

    The selected-case JSON usually comes from local case mining and is uploaded
    into the `/runs` volume before this function is called. The rollout path is
    the same TensorBatchEnv path used by the numeric eval.
    """
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.smoke_validate_8ckpt_visual",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--caption_jsonl", caption_jsonl,
        "--selected_cases_json", selected_cases_json,
        "--contact_sheet_dir", contact_sheet_dir,
        "--out_dir", out_dir,
        "--action_modes", action_modes,
        "--discrete_ckpt", f"/runs/{discrete_ckpt_subdir}/ppo_geoscout.zip",
        "--continuous_ckpt", f"/runs/{continuous_ckpt_subdir}/ppo_geoscout.zip",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--image_size", str(image_size),
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--caption_dim", str(caption_dim),
        "--coverage_hit_dilate_radius", "1",
        "--coverage_threshold", "0.99",
        "--coverage_reward_scale", "20",
        "--termination_bonus", "1",
        "--collision_penalty", "10",
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--max_faces", str(max_faces),
        "--step_frame_mode", step_frame_mode,
        "--max_step_frames", str(max_step_frames),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    print("[selected-case-viz] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[selected-case-viz] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=12 * 3600,
    memory=48 * 1024,
    retries=0,
)
def visualize_full_eval_policy(
    policy: str,
    out_subdir: str = "eval_full600_visual_0507",
    discrete_ckpt_subdir: str = "train600-24m-discrete-s1-l40s-n128-wandb-0506",
    continuous_ckpt_subdir: str = "train600-24m-continuous-s1-l40s-n128-wandb-0506",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    caption_jsonl: str = (
        "/data/geoscout_captions/"
        "full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl"
    ),
    contact_sheet_dir: str = "/runs/caption_full_results_zup_v1/contact_sheets_600_attr_v2",
    n_envs: int = 64,
    max_meshes: int = 0,
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
    write_trajectory_sheets: bool = False,
    seed: int = 0,
    synsets: str = "03001627,04256520,04379243",
    limit_per_synset: int = 200,
    seq_names: str = "",
):
    """Run one full-600 visual rollout policy on one GPU.

    Launch this function once per policy to get policy-level parallelism.
    The script uses deterministic mesh assignment, so every resolved sample is
    evaluated exactly once for this policy.

    `synsets`, `limit_per_synset`, and `seq_names` mirror the underlying
    evaluate_visual_rollouts CLI; defaults preserve prior behavior. Override
    them for OOD-only or custom-subset evals.
    """
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_visual_rollouts",
        "--mode", "run_policy",
        "--policy", policy,
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--caption_jsonl", caption_jsonl,
        "--contact_sheet_dir", contact_sheet_dir,
        "--out_dir", out_dir,
        "--discrete_ckpt", f"/runs/{discrete_ckpt_subdir}/ppo_geoscout.zip",
        "--continuous_ckpt", f"/runs/{continuous_ckpt_subdir}/ppo_geoscout.zip",
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--n_envs", str(n_envs),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--image_size", str(image_size),
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--caption_dim", str(caption_dim),
        "--coverage_hit_dilate_radius", "1",
        "--coverage_threshold", "0.99",
        "--coverage_reward_scale", "20",
        "--termination_bonus", "1",
        "--collision_penalty", "10",
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--max_faces", str(max_faces),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    if max_meshes > 0:
        cmd += ["--max_meshes", str(max_meshes)]
    if seq_names:
        cmd += ["--seq_names", seq_names]
    if write_trajectory_sheets:
        cmd += ["--write_trajectory_sheets"]
    else:
        cmd += ["--no_write_trajectory_sheets"]
    print("[full-visual-policy] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[full-visual-policy] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    volumes={"/data": vol_data, "/runs": vol_runs},
    cpu=8.0,
    timeout=6 * 3600,
    memory=32 * 1024,
    retries=0,
)
def build_full_eval_visual_report(
    out_subdir: str = "eval_full600_visual_0507",
    contact_sheet_dir: str = "/runs/caption_full_results_zup_v1/contact_sheets_600_attr_v2",
    report_workers: int = 8,
):
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_visual_rollouts",
        "--mode", "build_report",
        "--out_dir", out_dir,
        "--contact_sheet_dir", contact_sheet_dir,
        "--report_workers", str(report_workers),
    ]
    print("[full-visual-report] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[full-visual-report] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=24 * 3600,
    memory=64 * 1024,
    retries=0,
)
def export_fql_replay_shapenet(
    out_subdir: str = "fql_replay_judged_full600_v1",
    sample_mode: str = "full",
    policy_mix: str = "ppo_det:1,ppo_stoch:4",
    ckpt_subdir: str = "train600-24m-discrete-s1-judged-l40s-n128-wandb-0508",
    policy_name: str = "ppo_judged_discrete_s1_24m",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2_judged_v1",
    hard_cases_file: str = "/workspace/GeoScout/offline_rl/hard_cases_judged_0509.txt",
    n_envs: int = 64,
    max_samples: int = 0,
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
    auto_lookat_center: bool = False,
    seed: int = 0,
    synsets: str = "03001627,04256520,04379243",
    limit_per_synset: int = 200,
    shard_size: int = 1024,
    obs_storage: str = "flat",
    compress: bool = True,
    overwrite: bool = False,
):
    """Export GeoScout rollouts as sharded FQL/offline-RL replay data."""
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.export_fql_replay",
        "--out_dir", out_dir,
        "--ckpt", f"/runs/{ckpt_subdir}/ppo_geoscout.zip",
        "--policy_name", policy_name,
        "--policy_mix", policy_mix,
        "--sample_mode", sample_mode,
        "--hard_cases_file", hard_cases_file,
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--n_envs", str(n_envs),
        "--max_samples", str(max_samples),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--image_size", str(image_size),
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--caption_dim", str(caption_dim),
        "--coverage_hit_dilate_radius", "1",
        "--coverage_threshold", "0.99",
        "--coverage_reward_scale", "20",
        "--termination_bonus", "1",
        "--collision_penalty", "10",
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--max_faces", str(max_faces),
        "--seed", str(seed),
        "--device", "cuda",
        "--shard_size", str(shard_size),
        "--obs_storage", obs_storage,
    ]
    if compress:
        cmd += ["--compress"]
    else:
        cmd += ["--no_compress"]
    if auto_lookat_center:
        cmd += ["--auto_lookat_center"]
    if overwrite:
        cmd += ["--overwrite"]
    print("[fql-replay-export] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[fql-replay-export] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=12 * 3600,
    memory=96 * 1024,
    retries=0,
)
def train_fql_offline_rl(
    out_subdir: str = "fql_train_full_compact8_smoke_0512",
    replay_subdirs: str = "fql_replay_judged_full600_large_v1",
    offline_steps: int = 10000,
    max_shards_per_replay: int = 0,
    shard_selection: str = "first",
    max_transitions: int = 0,
    obs_mode: str = "compact",
    grid_bins: int = 8,
    action_key: str = "action_norm5",
    alpha: float = 1.0,
    batch_size: int = 1024,
    q_agg: str = "min",
    seed: int = 0,
    log_interval: int = 1000,
    save_interval: int = 0,
    eval_after_train: bool = True,
    eval_max_samples: int = 600,
    eval_n_envs: int = 64,
    eval_episode_len: int = 50,
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2_judged_v1",
    synsets: str = "03001627,04256520,04379243",
    limit_per_synset: int = 200,
    image_size: int = 400,
    grid_size: int = 128,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
):
    """Train official JAX FQL on exported GeoScout replay shards and eval it."""
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    replay_dirs = ",".join(
        f"/runs/{x.strip()}" for x in str(replay_subdirs).split(",") if x.strip()
    )
    if not replay_dirs:
        raise ValueError("replay_subdirs resolved to no replay directories")

    cmd = [
        sys.executable, "-u", "-m", "offline_rl.train_fql_from_replay",
        "--replay_dirs", replay_dirs,
        "--out_dir", out_dir,
        "--fql_dir", "/tmp/fql",
        "--obs_mode", obs_mode,
        "--grid_bins", str(grid_bins),
        "--action_key", action_key,
        "--offline_steps", str(offline_steps),
        "--batch_size", str(batch_size),
        "--alpha", str(alpha),
        "--q_agg", q_agg,
        "--normalize_q_loss",
        "--max_shards_per_replay", str(max_shards_per_replay),
        "--shard_selection", shard_selection,
        "--max_transitions", str(max_transitions),
        "--seed", str(seed),
        "--gpu_label", "L4",
        "--log_interval", str(log_interval),
        "--save_interval", str(save_interval),
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--device", "cuda",
        "--image_size", str(image_size),
        "--grid_size", str(grid_size),
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--max_faces", str(max_faces),
        "--eval_max_samples", str(eval_max_samples),
        "--eval_n_envs", str(eval_n_envs),
        "--eval_episode_len", str(eval_episode_len),
    ]
    if eval_after_train:
        cmd += ["--eval_after_train"]

    print("[fql-train-modal] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(
            cmd,
            env={**os.environ, "PYTHONPATH": "/workspace/GeoScout:/tmp/fql"},
        )
    finally:
        print("[fql-train-modal] committing /runs volume", flush=True)
        vol_runs.commit()
