"""Export ShapeNBV rollouts as an offline-RL replay buffer.

The exporter intentionally mirrors ``scripts.evaluate_visual_rollouts`` so the
dataset is generated with the same TensorBatchEnv settings as the paper evals.
It writes sharded ``.npz`` files plus JSON/CSV manifests that are easy to load
from JAX, PyTorch, or lightweight FQL baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.evaluate_visual_rollouts import (
    action_table,
    category_from_name,
    make_env,
    random_actions,
    resolve_pairs,
)


POSE_DIM = 6
CONT_EPS = 1e-4
SUPPORTED_BEHAVIORS = {"ppo_det", "ppo_stoch", "random", "fibonacci", "axis6", "ring"}


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    count: int

    @property
    def deterministic(self) -> bool:
        return self.name != "ppo_stoch" and self.name != "random"


def parse_policy_mix(value: str) -> List[BehaviorSpec]:
    specs: List[BehaviorSpec] = []
    aliases = {
        "det": "ppo_det",
        "deterministic": "ppo_det",
        "ppo": "ppo_det",
        "stoch": "ppo_stoch",
        "stochastic": "ppo_stoch",
    }
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            raw_name, raw_count = item.split(":", 1)
            count = int(raw_count)
        else:
            raw_name, count = item, 1
        name = aliases.get(raw_name.strip(), raw_name.strip())
        if name not in SUPPORTED_BEHAVIORS:
            raise ValueError(f"Unsupported behavior {raw_name!r}; choose from {sorted(SUPPORTED_BEHAVIORS)}")
        if count <= 0:
            continue
        specs.append(BehaviorSpec(name=name, count=count))
    if not specs:
        raise ValueError("policy_mix resolved to no behaviors")
    return specs


def read_hard_case_ids(path: Path, threshold: float) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"hard case source not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            out = []
            for row in reader:
                seq = str(row.get("seq_name") or row.get("object_id") or "").strip()
                if not seq:
                    continue
                try:
                    cr = float(row.get("final_cr", "nan"))
                except ValueError:
                    cr = float("nan")
                timeout = str(row.get("timeout", "")).lower() == "true"
                collision = str(row.get("collision", "")).lower() == "true"
                if timeout or collision or cr <= threshold:
                    out.append(seq)
            return out
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line.split()[0])
    return out


def default_hard_case_file() -> Path:
    return Path(__file__).resolve().parents[1] / "offline_rl" / "hard_cases_judged_0509.txt"


def obs_layout(args: argparse.Namespace) -> Dict[str, object]:
    pose_end = int(args.buffer_size) * POSE_DIM
    grid_end = pose_end + int(args.obs_grid_size) ** 3
    cap_end = grid_end + int(args.caption_dim)
    return {
        "flat_dim": cap_end,
        "pose_history": {
            "slice": [0, pose_end],
            "shape": [int(args.buffer_size), POSE_DIM],
            "dtype": "float32",
            "description": "Decoded pose6 history ring buffer in env order.",
        },
        "belief_grid": {
            "slice": [pose_end, grid_end],
            "shape": [int(args.obs_grid_size), int(args.obs_grid_size), int(args.obs_grid_size)],
            "dtype": "float32 in flat obs; int8 in split obs",
            "description": "Tri-class occupancy grid flattened in TensorBatchEnv order: -1 free, 0 unknown, +1 occupied.",
        },
        "caption_emb": {
            "slice": [grid_end, cap_end],
            "shape": [int(args.caption_dim)],
            "dtype": "float32",
            "description": "Sentence-transformers all-MiniLM-L6-v2 object caption embedding from the preproc .pt file.",
        },
    }


def split_obs(prefix: str, obs: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    pose_end = int(args.buffer_size) * POSE_DIM
    grid_dim = int(args.obs_grid_size) ** 3
    grid_end = pose_end + grid_dim
    out = {
        f"{prefix}_pose_history": obs[:, :pose_end].reshape(-1, int(args.buffer_size), POSE_DIM).astype(np.float32),
        f"{prefix}_belief_grid": obs[:, pose_end:grid_end].reshape(
            -1, int(args.obs_grid_size), int(args.obs_grid_size), int(args.obs_grid_size)
        ).astype(np.int8),
    }
    if int(args.caption_dim) > 0:
        out[f"{prefix}_caption_emb"] = obs[:, grid_end:grid_end + int(args.caption_dim)].astype(np.float32)
    return out


def pose6_to_norm5(pose6: np.ndarray) -> np.ndarray:
    """Map env pose6 to the 5D normalized continuous action convention."""
    pose = np.asarray(pose6, dtype=np.float32)
    norm = np.empty(5, dtype=np.float32)
    norm[:3] = pose[:3]
    norm[3] = pose[4] / (0.5 * math.pi)
    norm[4] = pose[5] / math.pi - 1.0
    return np.clip(norm, -1.0 + CONT_EPS, 1.0 - CONT_EPS).astype(np.float32)


class ReplayWriter:
    def __init__(self, out_dir: Path, args: argparse.Namespace):
        self.out_dir = out_dir
        self.args = args
        self.rows: List[dict] = []
        self.shards: List[dict] = []
        self.episode_summaries: List[dict] = []
        self.shard_idx = 0
        self.transition_count = 0

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= int(self.args.shard_size):
            self.flush()

    def add_episode_summary(self, row: dict) -> None:
        self.episode_summaries.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        rows = self.rows
        self.rows = []
        path = self.out_dir / f"shard_{self.shard_idx:06d}.npz"
        obs = np.stack([r["obs"] for r in rows]).astype(np.float32)
        next_obs = np.stack([r["next_obs"] for r in rows]).astype(np.float32)
        arrays: Dict[str, np.ndarray] = {
            "action": np.stack([r["action"] for r in rows]).astype(np.int64),
            "action_discrete": np.stack([r["action"] for r in rows]).astype(np.int64),
            "action_norm5": np.stack([r["action_norm5"] for r in rows]).astype(np.float32),
            "action_raw5": np.stack([r["action_raw5"] for r in rows]).astype(np.float32),
            "reward": np.asarray([r["reward"] for r in rows], dtype=np.float32),
            "done": np.asarray([r["done"] for r in rows], dtype=np.bool_),
            "timeout": np.asarray([r["timeout"] for r in rows], dtype=np.bool_),
            "truncated": np.asarray([r["truncated"] for r in rows], dtype=np.bool_),
            "object_id": np.asarray([r["object_id"] for r in rows], dtype=str),
            "category": np.asarray([r["category"] for r in rows], dtype=str),
            "timestep": np.asarray([r["timestep"] for r in rows], dtype=np.int32),
            "episode_id": np.asarray([r["episode_id"] for r in rows], dtype=np.int64),
            "sample_index": np.asarray([r["sample_index"] for r in rows], dtype=np.int32),
            "rollout_index": np.asarray([r["rollout_index"] for r in rows], dtype=np.int32),
            "coverage": np.asarray([r["coverage"] for r in rows], dtype=np.float32),
            "cr": np.asarray([r["cr"] for r in rows], dtype=np.float32),
            "coverage_delta": np.asarray([r["coverage_delta"] for r in rows], dtype=np.float32),
            "collision": np.asarray([r["collision"] for r in rows], dtype=np.bool_),
            "policy_name": np.asarray([r["policy_name"] for r in rows], dtype=str),
            "behavior": np.asarray([r["behavior"] for r in rows], dtype=str),
            "deterministic": np.asarray([r["deterministic"] for r in rows], dtype=np.bool_),
            "pose6": np.stack([r["pose6"] for r in rows]).astype(np.float32),
            "eye": np.stack([r["eye"] for r in rows]).astype(np.float32),
            "look_at": np.stack([r["look_at"] for r in rows]).astype(np.float32),
        }
        if self.args.obs_storage in ("flat", "both"):
            arrays["obs"] = obs
            arrays["next_obs"] = next_obs
        if self.args.obs_storage in ("split", "both"):
            arrays.update(split_obs("obs", obs, self.args))
            arrays.update(split_obs("next_obs", next_obs, self.args))
        if self.args.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        shard = {
            "path": path.name,
            "num_transitions": len(rows),
            "first_episode_id": int(arrays["episode_id"][0]),
            "last_episode_id": int(arrays["episode_id"][-1]),
        }
        self.shards.append(shard)
        self.shard_idx += 1
        self.transition_count += len(rows)
        print(f"[fql-export] wrote {path} transitions={len(rows)}", flush=True)

    def finish(self, metadata: dict) -> None:
        self.flush()
        metadata = dict(metadata)
        metadata.update({
            "num_shards": len(self.shards),
            "num_transitions": int(self.transition_count),
            "num_episodes": len(self.episode_summaries),
            "shards": self.shards,
        })
        (self.out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        fields = [
            "episode_id", "object_id", "category", "sample_index", "rollout_index",
            "policy_name", "behavior", "deterministic", "steps", "final_cr",
            "success", "timeout", "collision", "episode_reward",
        ]
        with (self.out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.episode_summaries:
                writer.writerow({k: row.get(k, "") for k in fields})
        print(
            f"[fql-export] DONE shards={len(self.shards)} transitions={self.transition_count} "
            f"episodes={len(self.episode_summaries)} out={self.out_dir}",
            flush=True,
        )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"out_dir is non-empty; pass --overwrite to replace exporter outputs: {path}")
        for child in path.iterdir():
            if child.is_file() and (
                child.name.startswith("shard_")
                or child.name in {"manifest.json", "episodes.csv", "hard_cases_used.txt"}
            ):
                child.unlink()
            elif child.is_dir() and child.name == "tmp":
                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)


def filter_samples(args: argparse.Namespace, names: List[str], meshes: List[Path], preprocs: List[Path]) -> Tuple[List[str], List[Path], List[Path], List[int], List[str]]:
    original_indices = list(range(len(names)))
    hard_ids: List[str] = []
    if args.sample_mode == "hard":
        hard_path = Path(args.hard_cases_file) if args.hard_cases_file else default_hard_case_file()
        hard_ids = read_hard_case_ids(hard_path, float(args.hard_threshold))
        hard_set = set(hard_ids)
        keep = [i for i, name in enumerate(names) if name in hard_set]
        names = [names[i] for i in keep]
        meshes = [meshes[i] for i in keep]
        preprocs = [preprocs[i] for i in keep]
        original_indices = [original_indices[i] for i in keep]
        missing = [x for x in hard_ids if x not in set(names)]
        if missing:
            print(f"[fql-export] warning: {len(missing)} hard ids were not resolved by current sample filters", flush=True)
    if args.max_samples > 0:
        names = names[: args.max_samples]
        meshes = meshes[: args.max_samples]
        preprocs = preprocs[: args.max_samples]
        original_indices = original_indices[: args.max_samples]
    if not names:
        raise RuntimeError("No samples left after filtering")
    return names, meshes, preprocs, original_indices, hard_ids


@torch.no_grad()
def export_replay(args: argparse.Namespace) -> None:
    policy_mix = parse_policy_mix(args.policy_mix)
    names, mesh_paths, preproc_paths = resolve_pairs(args)
    names, mesh_paths, preproc_paths, sample_indices, hard_ids = filter_samples(args, names, mesh_paths, preproc_paths)
    out_dir = Path(args.out_dir)
    prepare_output_dir(out_dir, bool(args.overwrite))
    if hard_ids:
        (out_dir / "hard_cases_used.txt").write_text("\n".join(names) + "\n")

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, mesh_paths, preproc_paths, "discrete")
    model: Optional[PPO] = None
    if any(spec.name.startswith("ppo_") for spec in policy_mix):
        print(f"[fql-export] loading PPO checkpoint {args.ckpt}", flush=True)
        model = PPO.load(args.ckpt, device=args.device)
        if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
            raise RuntimeError(
                f"observation shape mismatch: model={model.observation_space.shape} env={env.observation_space.shape}"
            )

    tables = {
        name: action_table(name, "discrete", int(args.episode_len), float(args.view_radius))
        for name in ("fibonacci", "axis6", "ring")
    }
    writer = ReplayWriter(out_dir, args)
    start = time.perf_counter()
    episode_id = 0
    behavior_rollout_offset = 0

    print(
        f"[fql-export] samples={len(names)} sample_mode={args.sample_mode} "
        f"policy_mix={','.join(f'{s.name}:{s.count}' for s in policy_mix)} n_envs={args.n_envs}",
        flush=True,
    )

    for spec in policy_mix:
        for rep in range(spec.count):
            rollout_index = behavior_rollout_offset + rep
            torch.manual_seed(int(args.seed) + 1009 * rollout_index)
            for batch_start in range(0, len(names), int(args.n_envs)):
                batch_ids = list(range(batch_start, min(batch_start + int(args.n_envs), len(names))))
                if len(batch_ids) < int(args.n_envs):
                    batch_mesh_ids = batch_ids + [batch_ids[0]] * (int(args.n_envs) - len(batch_ids))
                else:
                    batch_mesh_ids = batch_ids
                obs = env.reset_to_mesh_ids(batch_mesh_ids)
                active = np.zeros(int(args.n_envs), dtype=bool)
                active[: len(batch_ids)] = True
                phase = np.zeros(int(args.n_envs), dtype=np.int64)
                ep_rewards = np.zeros(len(batch_ids), dtype=np.float64)
                ep_steps = np.zeros(len(batch_ids), dtype=np.int32)
                ep_final_cr = np.zeros(len(batch_ids), dtype=np.float32)
                ep_timeout = np.zeros(len(batch_ids), dtype=bool)
                ep_collision = np.zeros(len(batch_ids), dtype=bool)
                ep_ids = np.arange(episode_id, episode_id + len(batch_ids), dtype=np.int64)

                for step_call in range(1, int(args.episode_len) + 1):
                    if spec.name.startswith("ppo_"):
                        assert model is not None
                        actions, _ = model.predict(obs, deterministic=(spec.name == "ppo_det"))
                        actions = np.asarray(actions, dtype=np.int64)
                    elif spec.name == "random":
                        actions = random_actions(rng, int(args.n_envs), "discrete")
                    else:
                        table = tables[spec.name]
                        if table is None:
                            raise RuntimeError(f"missing action table for {spec.name}")
                        actions = table[phase % len(table)]

                    action_t = torch.as_tensor(actions, dtype=torch.long, device=env.device)
                    pose6_t, eyes_t, ats_t = env._decode_actions(action_t)
                    pose6 = pose6_t.detach().cpu().numpy().astype(np.float32)
                    eyes = eyes_t.detach().cpu().numpy().astype(np.float32)
                    ats = ats_t.detach().cpu().numpy().astype(np.float32)

                    obs_before = obs
                    obs_after, rewards, dones, infos = env.step(actions)
                    phase += 1

                    for local_i, mesh_id in enumerate(batch_ids):
                        if not active[local_i]:
                            continue
                        info = infos[local_i]
                        done = bool(dones[local_i])
                        timeout = bool(info.get("TimeLimit.truncated", False))
                        next_obs = np.asarray(info.get("terminal_observation", obs_after[local_i]) if done else obs_after[local_i], dtype=np.float32)
                        reward = float(rewards[local_i])
                        cr = float(info.get("cr", 0.0))
                        collision = bool(info.get("collision", False))
                        ep_rewards[local_i] += reward
                        ep_steps[local_i] += 1
                        ep_final_cr[local_i] = cr
                        ep_timeout[local_i] = timeout
                        ep_collision[local_i] = ep_collision[local_i] or collision

                        writer.add({
                            "obs": np.asarray(obs_before[local_i], dtype=np.float32),
                            "action": np.asarray(actions[local_i], dtype=np.int64).reshape(-1),
                            "action_norm5": pose6_to_norm5(pose6[local_i]),
                            "action_raw5": np.arctanh(pose6_to_norm5(pose6[local_i])).astype(np.float32),
                            "reward": reward,
                            "next_obs": next_obs,
                            "done": done,
                            "timeout": timeout,
                            "truncated": timeout,
                            "object_id": names[mesh_id],
                            "category": category_from_name(names[mesh_id]),
                            "timestep": int(ep_steps[local_i] - 1),
                            "episode_id": int(ep_ids[local_i]),
                            "sample_index": int(sample_indices[mesh_id]),
                            "rollout_index": int(rollout_index),
                            "coverage": cr,
                            "cr": cr,
                            "coverage_delta": float(info.get("coverage_delta", 0.0)),
                            "collision": collision,
                            "policy_name": str(args.policy_name),
                            "behavior": spec.name,
                            "deterministic": spec.deterministic,
                            "pose6": pose6[local_i],
                            "eye": eyes[local_i],
                            "look_at": ats[local_i],
                        })
                        if done:
                            active[local_i] = False

                    obs = obs_after
                    if not active.any():
                        break

                for local_i, mesh_id in enumerate(batch_ids):
                    writer.add_episode_summary({
                        "episode_id": int(ep_ids[local_i]),
                        "object_id": names[mesh_id],
                        "category": category_from_name(names[mesh_id]),
                        "sample_index": int(sample_indices[mesh_id]),
                        "rollout_index": int(rollout_index),
                        "policy_name": str(args.policy_name),
                        "behavior": spec.name,
                        "deterministic": spec.deterministic,
                        "steps": int(ep_steps[local_i]),
                        "final_cr": float(ep_final_cr[local_i]),
                        "success": bool(ep_final_cr[local_i] > float(args.coverage_threshold)),
                        "timeout": bool(ep_timeout[local_i]),
                        "collision": bool(ep_collision[local_i]),
                        "episode_reward": float(ep_rewards[local_i]),
                    })

                episode_id += len(batch_ids)
                print(
                    f"[fql-export] behavior={spec.name} rep={rep + 1}/{spec.count} "
                    f"batch={batch_start // int(args.n_envs) + 1} episodes={episode_id} "
                    f"pending_transitions={len(writer.rows)}",
                    flush=True,
                )
        behavior_rollout_offset += spec.count

    env.close()
    elapsed_s = time.perf_counter() - start
    metadata = {
        "format_version": "shapenbv_fql_replay_v1",
        "created_by": "scripts.export_fql_replay",
        "elapsed_s": elapsed_s,
        "sample_mode": args.sample_mode,
        "num_samples": len(names),
        "sample_ids": names,
        "policy_name": args.policy_name,
        "policy_checkpoint": args.ckpt,
        "policy_mix": [{"name": s.name, "count": s.count} for s in policy_mix],
        "obs_storage": args.obs_storage,
        "obs_layout": obs_layout(args),
        "action_space": "MultiDiscrete([81,81,81,1,13,13])",
        "action_encoding": {
            "action": "Original discrete MultiDiscrete indices from the behavior rollout.",
            "action_discrete": "Alias of action for loaders that reserve action for continuous control.",
            "action_norm5": "Continuous tanh-space [x,y,z,pitch_norm,yaw_norm] in [-1,1], derived from decoded pose6.",
            "action_raw5": "atanh(action_norm5), clipped to finite values and compatible with TensorBatchEnv continuous_tanh Box([-8,8]^5) raw action input.",
        },
        "continuous_action_space": {
            "env_action_space_type": "continuous_tanh",
            "shape": [5],
            "box_low": -8.0,
            "box_high": 8.0,
            "decode": [
                "pose6[0:3] = tanh(raw[0:3])",
                "pose6[3] = 0.0",
                "pose6[4] = tanh(raw[3]) * pi/2",
                "pose6[5] = (tanh(raw[4]) + 1) * pi",
            ],
            "auto_lookat_center": bool(args.auto_lookat_center),
        },
        "success_definition": f"cr > {float(args.coverage_threshold)}; hard cases use cr <= {float(args.hard_threshold)} or timeout/collision",
        "env": {
            "preproc_dir": args.preproc_dir,
            "shapenet_root": args.shapenet_root,
            "synsets": args.synsets,
            "limit_per_synset": int(args.limit_per_synset),
            "episode_len": int(args.episode_len),
            "buffer_size": int(args.buffer_size),
            "grid_size": int(args.grid_size),
            "obs_grid_size": int(args.obs_grid_size),
            "caption_dim": int(args.caption_dim),
            "coverage_hit_dilate_radius": int(args.coverage_hit_dilate_radius),
            "renderer_backend": args.renderer_backend,
            "free_raycast_backend": args.free_raycast_backend,
            "free_mask_apply_mode": args.free_mask_apply_mode,
        },
    }
    writer.finish(metadata)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", default="/runs/train600-24m-discrete-s1-judged-l40s-n128-wandb-0508/ppo_shapenbv.zip")
    p.add_argument("--policy_name", default="ppo_judged_discrete_s1_24m")
    p.add_argument("--policy_mix", default="ppo_det:1,ppo_stoch:4")
    p.add_argument("--sample_mode", choices=["full", "hard"], default="full")
    p.add_argument("--hard_cases_file", default="")
    p.add_argument("--hard_threshold", type=float, default=0.99)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--shapenet_root", default="/data/ShapeNetCore.v2")
    p.add_argument("--preproc_dir", default="/data/shapenbv_preproc_g128_attr_v2_judged_v1")
    p.add_argument("--seq_names", default="")
    p.add_argument("--synsets", default="03001627,04256520,04379243")
    p.add_argument("--categories", default="")
    p.add_argument("--limit_per_synset", type=int, default=200)
    p.add_argument("--max_meshes", type=int, default=0)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--image_size", type=int, default=400)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--episode_len", type=int, default=50)
    p.add_argument("--buffer_size", type=int, default=30)
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--obs_grid_size", type=int, default=32)
    p.add_argument("--caption_dim", type=int, default=384)
    p.add_argument("--coverage_hit_dilate_radius", type=int, default=1)
    p.add_argument("--coverage_threshold", type=float, default=0.99)
    p.add_argument("--coverage_reward_scale", type=float, default=20.0)
    p.add_argument("--coverage_reward_type", choices=["linear", "log", "remaining", "information_gain"], default="linear")
    p.add_argument("--termination_bonus", type=float, default=1.0)
    p.add_argument("--novelty_reward_scale", type=float, default=0.0)
    p.add_argument("--remaining_reward_scale", type=float, default=0.0)
    p.add_argument("--redundancy_penalty_scale", type=float, default=0.0)
    p.add_argument("--view_revisit_penalty_scale", type=float, default=0.0)
    p.add_argument("--view_revisit_angle_deg", type=float, default=12.0)
    p.add_argument("--collision_penalty", type=float, default=10.0)
    p.add_argument("--short_path_grace_steps", type=int, default=30)
    p.add_argument("--short_path_max_extra", type=int, default=2)
    p.add_argument("--short_path_scale", type=float, default=0.1)
    p.add_argument("--only_positive_rewards", dest="only_positive_rewards", action="store_true", default=True)
    p.add_argument("--no_only_positive_rewards", dest="only_positive_rewards", action="store_false")
    p.add_argument("--skip_free_raycast", action="store_true")
    p.add_argument("--no_update_empty_rays", action="store_true")
    p.add_argument("--auto_lookat_center", action="store_true", default=False)
    p.add_argument("--max_faces", type=int, default=5000)
    p.add_argument("--renderer_backend", choices=["torch", "open3d", "nvdiffrast", "voxel_cuda"], default="voxel_cuda")
    p.add_argument("--free_raycast_backend", choices=["auto", "cuda", "triton", "torch"], default="cuda")
    p.add_argument("--free_mask_apply_mode", choices=["index", "dense", "triton"], default="triton")
    p.add_argument("--triton_bresenham_block_rays", type=int, default=64)
    p.add_argument("--view_radius", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_size", type=int, default=1024)
    p.add_argument("--obs_storage", choices=["flat", "split", "both"], default="flat")
    p.add_argument("--compress", dest="compress", action="store_true", default=True)
    p.add_argument("--no_compress", dest="compress", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    export_replay(args)


if __name__ == "__main__":
    main()
