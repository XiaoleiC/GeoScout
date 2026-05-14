"""PPO trainer for GeoScout.

GenNBV-faithful defaults:
    n_steps=128, batch_size=128, n_epochs=5, clip_range=0.2,
    learning_rate=1e-4, ent_coef=0.0, gamma=0.99, target_kl=0.05.

Single-machine entrypoint; for Modal cloud see `scripts/modal_app.py`.

Usage:
    python -m scripts.train \
        --shapenet_root /data/ShapeNetCore.v2 \
        --preproc_dir /data/geoscout_preproc_g128 \
        --synsets 03001627,02958343 \
        --total_timesteps 14000000 \
        --n_envs 32 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from geoscout.data import list_shapenet
from geoscout.env import GeoScoutEnv, POSE_DIM
from geoscout.hybrid_encoder import Hybrid_Encoder
from geoscout.mesh_renderer import ShapeNetIndex


# ----------------------------------------------------------------------
# Env factory
# ----------------------------------------------------------------------
def make_env(
    shapenet_root: str,
    preproc_dir: str,
    dataset: str,
    sequences: Optional[List[str]],
    device: str,
    seed: int,
    rank: int,
    image_size: int,
    fov_deg: float,
    buffer_size: int,
    episode_len: int,
    grid_size: int,
    obs_grid_size: int,
    coverage_hit_dilate_radius: int,
    coverage_reward_scale: float,
    short_path_grace_steps: int,
    short_path_max_extra: int,
    short_path_scale: float,
    termination_bonus: float,
    collision_penalty: float,
    only_positive_rewards: bool,
    coverage_threshold: float,
    caption_dim: int = 0,
    auto_lookat_center: bool = False,
    action_space_type: str = "discrete",
    skip_free_raycast: bool = False,
    update_empty_rays: bool = True,
    debug_dir_env0: Optional[str] = None,
):
    def _fn():
        # Build a per-worker ShapeNetIndex so each subprocess gets its own
        # mesh cache (no cross-process tensor sharing).
        if dataset == "abo":
            from geoscout.abo import list_abo
            entries = list_abo(Path(shapenet_root))
        else:
            entries = list_shapenet(Path(shapenet_root))
        name_to_path = {e.name: e.mesh_path for e in entries}
        if sequences is not None:
            name_to_path = {n: name_to_path[n] for n in sequences if n in name_to_path}
        index = ShapeNetIndex(
            entries=name_to_path,
            device=device,
            render_size=(image_size, image_size),
            fov_deg=fov_deg,
        )
        # NB: render_size + fov_deg are owned by ShapeNetIndex (renderer
        # config), not the env. Reward scales are passed through so the
        # single-env fallback stays aligned with TensorBatchEnv.
        env = GeoScoutEnv(
            index=index,
            preproc_dir=preproc_dir,
            sequence_names=list(name_to_path.keys()),
            device=device,
            buffer_size=buffer_size,
            episode_len=episode_len,
            grid_size=grid_size,
            obs_grid_size=obs_grid_size,
            cr_success_threshold=coverage_threshold,
            coverage_reward_scale=coverage_reward_scale,
            short_path_grace=short_path_grace_steps,
            short_path_clip=float(short_path_max_extra),
            short_path_scale=short_path_scale,
            termination_bonus=termination_bonus,
            collision_penalty=collision_penalty,
            only_positive_rewards=only_positive_rewards,
            caption_dim=caption_dim,
            auto_lookat_center=auto_lookat_center,
            skip_free_raycast=skip_free_raycast,
            update_empty_rays=update_empty_rays,
            coverage_hit_dilate_radius=coverage_hit_dilate_radius,
            seed=seed,
            debug_dir=debug_dir_env0 if (rank == 0 and debug_dir_env0) else None,
        )
        env = Monitor(env, info_keywords=("cr_history",))
        return env
    return _fn


# ----------------------------------------------------------------------
# Per-rollout metrics callback (mirrors object_nbv_zgr's version, but
# stripped to GeoScout-relevant fields).
# ----------------------------------------------------------------------
class GeoScoutMetricsCallback(BaseCallback):
    def __init__(self, buffer_len: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self._ep_stats: deque = deque(maxlen=buffer_len)
        self._rollout_t0 = 0.0
        self._rollout_step0 = 0

    def _on_rollout_start(self) -> None:
        self._rollout_t0 = time.perf_counter()
        self._rollout_step0 = int(self.num_timesteps)
        reset_stats = getattr(self.training_env, "reset_continuous_action_stats", None)
        if callable(reset_stats):
            reset_stats()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            self._ep_stats.append({
                "cr": float(info.get("cr", 0.0)),
                "collision": int(info.get("collision", False)),
                "early_stopped": int(info.get("early_stopped", False)),
                "ep_len": int(info.get("step_idx", 0)),
                "cr_history": info.get("cr_history") or [],
                "novelty_ratio": float(info.get("episode_novelty_ratio", 0.0)),
                "redundancy_ratio": float(info.get("episode_redundancy_ratio", 0.0)),
                "new_gt_voxels": float(info.get("episode_new_gt_voxels", 0.0)),
                "visible_gt_voxels": float(info.get("episode_visible_gt_voxels", 0.0)),
                "revisit_penalty_mean": float(info.get("episode_revisit_penalty_mean", 0.0)),
            })
        return True

    def _on_rollout_end(self) -> None:
        dt = max(time.perf_counter() - self._rollout_t0, 1e-9)
        rollout_env_steps = max(int(self.num_timesteps) - int(self._rollout_step0), 0)
        self.logger.record("geoscout/rollout_fps", float(rollout_env_steps / dt))
        self.logger.record("geoscout/rollout_wall_time_s", float(dt))
        self.logger.record("geoscout/rollout_env_steps", int(rollout_env_steps))
        action_stats_fn = getattr(self.training_env, "get_continuous_action_stats", None)
        if callable(action_stats_fn):
            action_stats = action_stats_fn()
            if action_stats.get("stat_steps", 0.0) > 0.0:
                self.logger.record("geoscout/action_raw_abs_mean",
                                   float(action_stats["raw_abs_mean"]))
                self.logger.record("geoscout/action_raw_abs_gt3_frac",
                                   float(action_stats["raw_abs_gt3_frac"]))
                self.logger.record("geoscout/action_raw_clip_frac",
                                   float(action_stats["raw_clip_frac"]))
                self.logger.record("geoscout/action_raw_abs_max",
                                   float(action_stats["raw_abs_max"]))
        if not self._ep_stats:
            return
        crs = np.array([s["cr"] for s in self._ep_stats])
        ep_lens = np.array([s["ep_len"] for s in self._ep_stats])
        self.logger.record("geoscout/cr_final_mean", float(crs.mean()))
        self.logger.record("geoscout/cr_final_max", float(crs.max()))
        self.logger.record("geoscout/cr_final_p10", float(np.percentile(crs, 10)))
        self.logger.record("geoscout/cr_final_p25", float(np.percentile(crs, 25)))
        self.logger.record("geoscout/cr_final_p50", float(np.median(crs)))
        self.logger.record("geoscout/cr_final_p75", float(np.percentile(crs, 75)))
        self.logger.record("geoscout/cr_final_p90", float(np.percentile(crs, 90)))
        self.logger.record("geoscout/reach_cr_50_rate", float((crs >= 0.5).mean()))
        self.logger.record("geoscout/reach_cr_80_rate", float((crs >= 0.8).mean()))
        self.logger.record("geoscout/ep_len_mean", float(ep_lens.mean()))
        self.logger.record("geoscout/collision_rate",
                           float(np.mean([s["collision"] for s in self._ep_stats])))
        self.logger.record("geoscout/early_stopped_rate",
                           float(np.mean([s["early_stopped"] for s in self._ep_stats])))
        self.logger.record("geoscout/novelty_ratio_mean",
                           float(np.mean([s["novelty_ratio"] for s in self._ep_stats])))
        self.logger.record("geoscout/redundancy_ratio_mean",
                           float(np.mean([s["redundancy_ratio"] for s in self._ep_stats])))
        self.logger.record("geoscout/new_gt_voxels_mean",
                           float(np.mean([s["new_gt_voxels"] for s in self._ep_stats])))
        self.logger.record("geoscout/visible_gt_voxels_mean",
                           float(np.mean([s["visible_gt_voxels"] for s in self._ep_stats])))
        self.logger.record("geoscout/revisit_penalty_mean",
                           float(np.mean([s["revisit_penalty_mean"] for s in self._ep_stats])))


class PeriodicCheckpointCallback(BaseCallback):
    """Save PPO checkpoints by environment timestep.

    Stable-Baselines3's built-in CheckpointCallback counts callback calls,
    which are vector-env steps rather than environment timesteps. GeoScout
    usually trains with many envs per GPU, so using model.num_timesteps keeps
    the CLI semantics direct: `--checkpoint_freq_steps 1000000` means roughly
    every one million environment transitions.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        freq_steps: int,
        keep_last: int = 5,
        name_prefix: str = "ppo_geoscout",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.freq_steps = int(freq_steps)
        self.keep_last = int(keep_last)
        self.name_prefix = str(name_prefix)
        self._next_step = 0

    def _on_training_start(self) -> None:
        if self.freq_steps <= 0:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        current = int(getattr(self.model, "num_timesteps", 0))
        self._next_step = ((current // self.freq_steps) + 1) * self.freq_steps
        if self.verbose:
            print(
                f"[checkpoint] saving every {self.freq_steps:,} env steps "
                f"to {self.checkpoint_dir} (next={self._next_step:,}, "
                f"keep_last={self.keep_last})",
                flush=True,
            )

    def _on_step(self) -> bool:
        if self.freq_steps <= 0:
            return True
        current = int(self.model.num_timesteps)
        if current >= self._next_step:
            # If a very large vectorized step jumps across multiple intervals,
            # advance the schedule past all missed boundaries but save once at
            # the actual current timestep.
            while current >= self._next_step:
                self._next_step += self.freq_steps
            self._save_checkpoint(current)
        return True

    def _save_checkpoint(self, num_timesteps: int) -> None:
        path = self.checkpoint_dir / f"{self.name_prefix}_step_{num_timesteps}.zip"
        latest = self.checkpoint_dir / f"{self.name_prefix}_latest.zip"
        meta_path = self.checkpoint_dir / f"{self.name_prefix}_latest.json"

        self.model.save(path)
        shutil.copy2(path, latest)
        meta = {
            "checkpoint": path.name,
            "latest": latest.name,
            "num_timesteps": int(num_timesteps),
            "wall_time_unix": float(time.time()),
            "next_checkpoint_step": int(self._next_step),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        if self.verbose:
            print(f"[checkpoint] saved {path} and updated {latest}", flush=True)
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return
        prefix = f"{self.name_prefix}_step_"
        files = []
        for path in self.checkpoint_dir.glob(f"{prefix}*.zip"):
            stem = path.name[:-4] if path.name.endswith(".zip") else path.name
            try:
                step = int(stem.rsplit("_step_", 1)[1])
            except (IndexError, ValueError):
                continue
            files.append((step, path))
        files.sort()
        for _, path in files[:-self.keep_last]:
            try:
                path.unlink()
                if self.verbose:
                    print(f"[checkpoint] pruned old checkpoint {path}", flush=True)
            except OSError as exc:
                print(f"[checkpoint] warning: could not prune {path}: {exc}", flush=True)


def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--shapenet_root", type=str, required=True)
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--synsets", type=str, default="")
    p.add_argument("--categories", type=str, default="")
    p.add_argument("--limit_per_synset", type=int, default=0)
    p.add_argument("--seq_names", type=str, default="",
                   help="Optional comma-separated exact sequence names. "
                        "Applied after synset/category enumeration and "
                        "preserves the requested order.")
    # Sim
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image_size", type=int, default=400)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--episode_len", type=int, default=100,
                   help="GenNBV default 100.")
    p.add_argument("--buffer_size", type=int, default=100,
                   help="Pose history length. GenNBV default 100.")
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--obs_grid_size", type=int, default=32,
                   help="Policy-observation grid resolution. 0 = same as "
                        "--grid_size. Use e.g. --grid_size 128 "
                        "--obs_grid_size 32 to compute reward at high "
                        "resolution without exploding the 3D CNN.")
    p.add_argument("--coverage_hit_dilate_radius", type=int, default=1,
                   help="Dilate rendered hit voxels by this many cells before "
                        "matching GT coverage. 0 preserves exact matching.")
    p.add_argument("--renderer_backend", type=str, default="nvdiffrast",
                   choices=["torch", "open3d", "nvdiffrast", "voxel_cuda"],
                   help="Renderer implementation for TensorBatchEnv. "
                   "'torch' is the pure CUDA ray-triangle reference; "
                   "'open3d' uses CPU BVH raycasting with the same "
                        "pixel-center rays; 'nvdiffrast' uses GPU mesh "
                        "rasterization; 'voxel_cuda' raycasts directly "
                        "through the preprocessed reward grid on GPU.")
    p.add_argument("--free_mask_apply_mode", type=str, default="triton",
                   choices=["index", "dense", "triton"],
                   help="Infra-only free-mask writeback path. 'index' is "
                        "the older nonzero/index_put path; 'triton' keeps "
                        "the same set-union semantics with a dense CUDA pass.")
    p.add_argument("--free_raycast_backend", type=str, default="auto",
                   choices=["auto", "cuda", "triton", "torch"],
                   help="Infra-only Bresenham scatter backend. 'auto' tries "
                        "the custom CUDA extension, then Triton, then the "
                        "PyTorch reference fallback.")
    p.add_argument("--triton_bresenham_block_rays", type=int, default=64,
                   help="Number of rays handled by each Triton Bresenham "
                        "program. Infra-only performance knob.")
    # Reward (GenNBV effective scales after IsaacGym dt multiplier)
    p.add_argument("--coverage_reward_scale", type=float, default=20.0)
    p.add_argument("--short_path_grace_steps", type=int, default=30)
    p.add_argument("--short_path_max_extra", type=int, default=2)
    p.add_argument("--short_path_scale", type=float, default=0.1)
    p.add_argument("--termination_bonus", type=float, default=1.0)
    p.add_argument("--coverage_threshold", type=float, default=0.99)
    p.add_argument("--coverage_reward_type", type=str, default="linear",
                   choices=["linear", "log", "remaining", "information_gain"],
                   help="`linear` (GenNBV default, reward = cr_t - cr_{t-1}) "
                        "or `log` (reward ∝ log(1-cr_prev) - log(1-cr); "
                        "steeper near cr=1, useful when random-policy "
                        "baseline already gets high cr like ShapeNet). "
                        "`remaining` normalizes new voxels by remaining "
                        "uncovered voxels. `information_gain` combines "
                        "new coverage, novelty, remaining-gain and "
                        "overlap penalties.")
    p.add_argument("--novelty_reward_scale", type=float, default=0.0,
                   help="information_gain only: bonus for the fraction of "
                        "currently visible GT voxels that are newly covered.")
    p.add_argument("--remaining_reward_scale", type=float, default=0.0,
                   help="information_gain only: bonus for new voxels "
                        "normalized by remaining uncovered GT voxels.")
    p.add_argument("--redundancy_penalty_scale", type=float, default=0.0,
                   help="information_gain only: penalty for visible GT "
                        "voxels already covered by previous views.")
    p.add_argument("--view_revisit_penalty_scale", type=float, default=0.0,
                   help="Penalty for choosing a camera direction too close "
                        "to a previous view in the episode.")
    p.add_argument("--view_revisit_angle_deg", type=float, default=12.0)
    p.add_argument("--collision_penalty", type=float, default=10.0,
                   help="Negative reward applied on collision step (eye in "
                        "object / invalid camera center). Applied after the "
                        "only_positive clamp so collision remains costly.")
    p.add_argument("--only_positive_rewards", dest="only_positive_rewards",
                   action="store_true", default=True)
    p.add_argument("--no_only_positive_rewards", dest="only_positive_rewards",
                   action="store_false",
                   help="Disable the GenNBV-style nonnegative clamp on "
                        "coverage/short-path reward before terminal terms.")
    p.add_argument("--skip_free_raycast", action="store_true",
                   help="Skip the free-voxel bresenham every step "
                        "(prob_grid only marks occupied endpoints). "
                        "Halves per-step env time at the cost of less "
                        "informative observation. Recommended for smoke "
                        "tests on single objects.")
    p.add_argument("--no_update_empty_rays", action="store_true",
                   help="Disable free-space updates for alpha-miss rays. "
                        "By default, miss rays are traversed through the "
                        "current grid AABB and marked free.")
    # PPO
    p.add_argument("--total_timesteps", type=int, default=14_000_000)
    p.add_argument("--n_envs", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--clip_range_vf", type=float, default=-1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--target_kl", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subproc", action="store_true",
                   help="Use SubprocVecEnv instead of DummyVecEnv.")
    p.add_argument("--tensor_env", action="store_true",
                   help="Use TensorBatchEnv (IsaacGym-style tensor-batched, "
                        "single process, all N envs as one [N,...] tensor "
                        "batch on GPU). 5-20× faster than DummyVecEnv. "
                        "Single-mesh only — for multi-object training "
                        "stick with --subproc until per-mesh batching is "
                        "added.")
    p.add_argument("--tensor_env_n_envs", type=int, default=32,
                   help="Override --n_envs when --tensor_env is set "
                        "(tensor-batch can fit many more envs per GPU).")
    p.add_argument("--caption_dim", type=int, default=0,
                   help="Dimension of per-episode caption_emb appended to "
                        "the observation. 0 = disabled. 384 = "
                        "sentence-transformers MiniLM-L6 (Phase 1).")
    p.add_argument("--auto_lookat_center", action="store_true",
                   help="Ignore action's pitch/yaw dims and force "
                        "look-at = object centre. This must match "
                        "between training and validation.")
    p.add_argument("--action_space_type",
                   choices=["discrete", "continuous_tanh"],
                   default="discrete",
                   help="Action representation for TensorBatchEnv. "
                        "`discrete` is GenNBV-style Cube Mode "
                        "MultiDiscrete([81,81,81,1,13,13]); "
                        "`continuous_tanh` uses a Gaussian Box policy over "
                        "raw R^5 actions and maps tanh(raw) into "
                        "x/y/z, pitch, yaw.")
    p.add_argument("--max_faces", type=int, default=0,
                   help="Quadric-decimate each loaded mesh to ~max_faces "
                        "triangles. 0 disables. ABO furniture has 50K-100K "
                        "tris — without decimation the PyTorch ray-tracer "
                        "drops fps from ~500 to ~20. 5000 is a good default.")
    p.add_argument("--dataset", type=str, default="shapenet",
                   choices=["shapenet", "abo"],
                   help="Source dataset enumerator. `shapenet` uses "
                        "geoscout.data.list_shapenet (ShapeNetCore.v2 "
                        "directory layout); `abo` uses geoscout.abo."
                        "list_abo (Amazon Berkeley Objects manifest).")
    # Logging
    p.add_argument("--log_dir", type=str, default="/tmp/geoscout_run")
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--checkpoint_freq_steps", type=int, default=0,
                   help="Save a PPO checkpoint every N environment "
                        "timesteps into <log_dir>/checkpoints. 0 disables "
                        "periodic checkpoints.")
    p.add_argument("--checkpoint_keep_last", type=int, default=5,
                   help="How many numbered periodic checkpoints to keep. "
                        "The ppo_geoscout_latest.zip convenience copy is "
                        "always updated when checkpointing is enabled.")
    # Wandb
    p.add_argument("--wandb_project", type=str, default="geoscout")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="disabled")
    args = p.parse_args()
    if not args.tensor_env:
        non_linear_reward = (
            args.coverage_reward_type != "linear"
            or args.novelty_reward_scale != 0.0
            or args.remaining_reward_scale != 0.0
            or args.redundancy_penalty_scale != 0.0
            or args.view_revisit_penalty_scale != 0.0
        )
        if non_linear_reward:
            sys.exit(
                "[train] non-linear reward shaping is implemented in "
                "TensorBatchEnv only. Use --tensor_env or keep "
                "--coverage_reward_type linear with shaping scales at 0."
            )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    debug_dir_env0 = log_dir / "debug_env0"

    # Resolve sequences from synsets/categories filter. `--dataset`
    # picks the enumerator: ShapeNet (synset-id directory layout) vs
    # ABO (Amazon manifest CSV + listings NDJSON).
    synsets = [s.strip() for s in args.synsets.split(",") if s.strip()] or None
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    if args.dataset == "abo":
        from geoscout.abo import list_abo
        # ABO uses `--shapenet_root` for the ABO root path (kept the same
        # CLI flag name so callers don't have to special-case).
        entries = list_abo(
            Path(args.shapenet_root),
            categories=categories,
            limit=args.limit_per_synset * len(categories) if (categories and args.limit_per_synset) else 0,
        )
    else:
        entries = list_shapenet(
            Path(args.shapenet_root),
            synsets=synsets, categories=categories,
            limit_per_synset=args.limit_per_synset,
        )
    if args.seq_names:
        wanted = [s.strip() for s in args.seq_names.split(",") if s.strip()]
        by_name = {e.name: e for e in entries}
        missing = [name for name in wanted if name not in by_name]
        entries = [by_name[name] for name in wanted if name in by_name]
        if missing:
            print(
                f"[train] warning: {len(missing)} requested seq_names not found "
                f"after dataset filters: {missing[:10]}",
                flush=True,
            )

    sequences = [e.name for e in entries]
    print(f"[train] {len(sequences)} sequences after filter (synsets={synsets}, "
          f"categories={categories}, limit={args.limit_per_synset}).")
    if not sequences:
        sys.exit("[train] No sequences after filter — abort.")

    if args.tensor_env:
        # Tensor-batched single-process VecEnv. Now accepts a POOL of
        # meshes; per-env mesh assignment is randomly resampled at every
        # episode reset. Length-1 pool degenerates to single-mesh.
        from geoscout.tensor_env import TensorBatchEnv
        name_to_path = {e.name: e.mesh_path for e in entries}
        mesh_paths_all = []
        preproc_paths_all = []
        for seq_name in sequences:
            mp = name_to_path.get(seq_name)
            pp = Path(args.preproc_dir) / f"{seq_name}.pt"
            if mp is None or not pp.exists():
                continue
            mesh_paths_all.append(mp)
            preproc_paths_all.append(pp)
        if not mesh_paths_all:
            sys.exit("[train] tensor_env: no mesh+preproc pair found.")
        print(f"[train] tensor_env pool size = {len(mesh_paths_all)} meshes")

        n = args.tensor_env_n_envs if args.tensor_env_n_envs > 0 else args.n_envs
        args.n_envs = n
        vec_env = TensorBatchEnv(
            num_envs=n,
            mesh_paths=mesh_paths_all,
            preproc_paths=preproc_paths_all,
            device=args.device,
            buffer_size=args.buffer_size,
            grid_size=args.grid_size,
            obs_grid_size=args.obs_grid_size,
            episode_len=args.episode_len,
            render_size=args.image_size,
            fov_deg=args.fov_deg,
            cr_success_threshold=args.coverage_threshold,
            coverage_reward_scale=args.coverage_reward_scale,
            short_path_grace=args.short_path_grace_steps,
            short_path_clip=float(args.short_path_max_extra),
            short_path_scale=args.short_path_scale,
            only_positive_rewards=args.only_positive_rewards,
            skip_free_raycast=args.skip_free_raycast,
            update_empty_rays=not args.no_update_empty_rays,
            coverage_hit_dilate_radius=args.coverage_hit_dilate_radius,
            caption_dim=args.caption_dim,
            auto_lookat_center=args.auto_lookat_center,
            action_space_type=args.action_space_type,
            max_faces=args.max_faces,
            renderer_backend=args.renderer_backend,
            free_raycast_backend=args.free_raycast_backend,
            free_mask_apply_mode=args.free_mask_apply_mode,
            triton_bresenham_block_rays=args.triton_bresenham_block_rays,
            coverage_reward_type=args.coverage_reward_type,
            termination_bonus=args.termination_bonus,
            novelty_reward_scale=args.novelty_reward_scale,
            remaining_reward_scale=args.remaining_reward_scale,
            redundancy_penalty_scale=args.redundancy_penalty_scale,
            view_revisit_penalty_scale=args.view_revisit_penalty_scale,
            view_revisit_angle_deg=args.view_revisit_angle_deg,
            collision_penalty=args.collision_penalty,
            seed=args.seed,
        )
        print(f"[train] using TensorBatchEnv with {n} envs, "
              f"{len(mesh_paths_all)}-mesh pool on {args.device} "
            f"(auto_lookat_center={args.auto_lookat_center}, "
              f"action_space_type={args.action_space_type}, "
              f"grid_size={args.grid_size}, "
              f"obs_grid_size={args.obs_grid_size or args.grid_size}, "
              f"renderer_backend={args.renderer_backend}, "
              f"free_raycast_backend={args.free_raycast_backend}, "
              f"free_mask_apply_mode={args.free_mask_apply_mode}, "
              f"triton_bresenham_block_rays={args.triton_bresenham_block_rays}, "
              f"reward={args.coverage_reward_type})")
    else:
        env_fns = [
            make_env(
                shapenet_root=args.shapenet_root,
                preproc_dir=args.preproc_dir,
                dataset=args.dataset,
                sequences=sequences,
                device=args.device,
                seed=args.seed + rank,
                rank=rank,
                image_size=args.image_size,
                fov_deg=args.fov_deg,
                buffer_size=args.buffer_size,
                episode_len=args.episode_len,
                grid_size=args.grid_size,
                obs_grid_size=args.obs_grid_size,
                coverage_hit_dilate_radius=args.coverage_hit_dilate_radius,
                coverage_reward_scale=args.coverage_reward_scale,
                short_path_grace_steps=args.short_path_grace_steps,
                short_path_max_extra=args.short_path_max_extra,
                short_path_scale=args.short_path_scale,
                termination_bonus=args.termination_bonus,
                collision_penalty=args.collision_penalty,
                only_positive_rewards=args.only_positive_rewards,
                coverage_threshold=args.coverage_threshold,
                caption_dim=args.caption_dim,
                auto_lookat_center=args.auto_lookat_center,
                skip_free_raycast=args.skip_free_raycast,
                update_empty_rays=not args.no_update_empty_rays,
                debug_dir_env0=str(debug_dir_env0),
            )
            for rank in range(args.n_envs)
        ]
        if args.subproc:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

    # ----- Wandb (init early so URL prints before slow PPO/model build) -----
    wandb_run = None
    if args.wandb_mode != "disabled":
        import datetime
        import wandb
        run_name = args.wandb_run_name or f"geoscout-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        print(f"[train] wandb.init project={args.wandb_project} run={run_name} mode={args.wandb_mode} ...", flush=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=run_name,
            id=run_name,
            resume="allow",
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(log_dir),
        )
        print(f"[train] wandb run: {wandb_run.url} (mode={args.wandb_mode})", flush=True)

    obs_grid_size = args.obs_grid_size if args.obs_grid_size > 0 else args.grid_size
    state_input_shape = (args.buffer_size * POSE_DIM,)
    visual_input_shape = (args.buffer_size, args.image_size, args.image_size)

    policy_kwargs = dict(
        features_extractor_class=Hybrid_Encoder,
        features_extractor_kwargs=dict(
            encoder_param={"hidden_shapes": [256, 256], "visual_dim": 256},
            net_param={
                "transformer_params": [[1, 256], [1, 256]],
                "append_hidden_shapes": [256, 256],
            },
            state_input_shape=state_input_shape,
            visual_input_shape=visual_input_shape,
            state_input_only=True,
            grid_size=obs_grid_size,
            caption_dim=args.caption_dim,
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.device,
        seed=args.seed,
    )
    if args.clip_range_vf > 0:
        ppo_kwargs["clip_range_vf"] = args.clip_range_vf
    if args.target_kl > 0:
        ppo_kwargs["target_kl"] = args.target_kl

    if args.resume_from:
        print(f"[train] resuming from {args.resume_from}")
        model = PPO.load(args.resume_from, env=vec_env, device=args.device)
        # Reset hparams in case the user changed them on resume.
        for k, v in ppo_kwargs.items():
            if hasattr(model, k) and k not in ("policy", "env", "policy_kwargs"):
                setattr(model, k, v)
        # PPO._setup_model() wraps clip_range / clip_range_vf as schedule
        # callables; the setattr above can replace them with raw floats,
        # which then breaks `self.clip_range(progress)` in PPO.train().
        from stable_baselines3.common.utils import get_schedule_fn
        if not callable(model.clip_range):
            model.clip_range = get_schedule_fn(model.clip_range)
        if model.clip_range_vf is not None and not callable(model.clip_range_vf):
            model.clip_range_vf = get_schedule_fn(model.clip_range_vf)
    else:
        model = PPO(**ppo_kwargs)

    # ----- Wandb callback + a custom logger Output that pipes every
    # logger.dump() (rollout/, train/, time/, geoscout/) to wandb.log.
    # WandbCallback alone only records gradient/system stats unless
    # sync_tensorboard=True is set — but that requires `tensorboard`
    # in the runtime image. The output-format approach skips that dep.
    # Hook is attached at training_start because PPO._logger is not
    # initialized until model.learn() calls _setup_learn().
    callbacks = [GeoScoutMetricsCallback(verbose=1)]
    if args.checkpoint_freq_steps > 0:
        callbacks.append(
            PeriodicCheckpointCallback(
                checkpoint_dir=log_dir / "checkpoints",
                freq_steps=args.checkpoint_freq_steps,
                keep_last=args.checkpoint_keep_last,
                verbose=1,
            )
        )
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        from stable_baselines3.common.logger import KVWriter
        import wandb as _wandb

        class WandbOutputFormat(KVWriter):
            def write(self, key_values, key_excluded, step=0):
                payload = {k: v for k, v in key_values.items()
                           if isinstance(v, (int, float, np.integer, np.floating))}
                if payload:
                    _wandb.log(payload, step=int(step))

            def close(self):
                pass

        class _AttachWandbOutput(BaseCallback):
            def _on_training_start(self) -> None:
                self.model.logger.output_formats.append(WandbOutputFormat())

            def _on_step(self) -> bool:
                return True

        callbacks.append(_AttachWandbOutput(verbose=0))
        callbacks.append(WandbCallback(verbose=1, model_save_freq=0))

    print(f"[train] PPO total_timesteps={args.total_timesteps:,}, "
          f"n_envs={args.n_envs}, log_dir={log_dir}, "
          f"wandb={args.wandb_mode}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not bool(args.resume_from),
    )
    if wandb_run is not None:
        wandb_run.finish()

    ckpt = log_dir / "ppo_geoscout.zip"
    model.save(ckpt)
    print(f"[train] saved checkpoint to {ckpt}")
    if args.checkpoint_freq_steps > 0:
        checkpoint_dir = log_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest = checkpoint_dir / "ppo_geoscout_latest.zip"
        shutil.copy2(ckpt, latest)
        meta = {
            "checkpoint": ckpt.name,
            "latest": latest.name,
            "num_timesteps": int(model.num_timesteps),
            "wall_time_unix": float(time.time()),
            "final": True,
        }
        (checkpoint_dir / "ppo_geoscout_latest.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[train] updated latest checkpoint to {latest}")


if __name__ == "__main__":
    main()
