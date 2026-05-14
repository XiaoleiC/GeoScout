"""TensorBatchEnv baseline evaluator for GeoScout.

This intentionally evaluates PPO and hand-written policies in the same
TensorBatchEnv used for training. `validate.py` remains the richer
visualization route; this script is the cleaner PPO-vs-baseline
comparison for reward-design debugging.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from geoscout.data import list_shapenet
from geoscout.tensor_env import (
    DEFAULT_ACTION_LOW_WORLD,
    DEFAULT_ACTION_UNIT,
    DEFAULT_CLIP_POSE_IDX_LOW,
    DEFAULT_CLIP_POSE_IDX_UP,
    NVEC,
    TensorBatchEnv,
)


_CONT_EPS = 1e-4


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_pairs(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path]]:
    synsets = _split_csv(args.synsets) or None
    categories = _split_csv(args.categories) or None
    if args.dataset == "abo":
        from geoscout.abo import list_abo

        limit = args.limit_per_synset * len(categories) if (categories and args.limit_per_synset) else 0
        entries = list_abo(Path(args.shapenet_root), categories=categories, limit=limit)
    else:
        entries = list_shapenet(
            Path(args.shapenet_root),
            synsets=synsets,
            categories=categories,
            limit_per_synset=args.limit_per_synset,
        )

    if args.seq_names:
        wanted = set(_split_csv(args.seq_names))
        entries = [e for e in entries if e.name in wanted]

    names: List[str] = []
    mesh_paths: List[Path] = []
    preproc_paths: List[Path] = []
    preproc_root = Path(args.preproc_dir)
    for entry in entries:
        pp = preproc_root / f"{entry.name}.pt"
        if not pp.exists():
            continue
        names.append(entry.name)
        mesh_paths.append(Path(entry.mesh_path))
        preproc_paths.append(pp)

    if args.max_meshes > 0:
        names = names[: args.max_meshes]
        mesh_paths = mesh_paths[: args.max_meshes]
        preproc_paths = preproc_paths[: args.max_meshes]

    if not mesh_paths:
        sys.exit("[eval] no mesh+preproc pairs resolved.")
    return names, mesh_paths, preproc_paths


def _pose_to_cube_action_idx(position: Sequence[float]) -> np.ndarray:
    """Map a camera position looking at the origin onto the 6D action grid."""
    pos = np.asarray(position, dtype=np.float32)
    pose = np.zeros(6, dtype=np.float32)
    pose[:3] = pos

    direction = -pos
    norm = float(np.linalg.norm(direction))
    if norm > 1e-8:
        direction = direction / norm
        pose[4] = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        pose[5] = yaw if yaw >= 0 else yaw + 2.0 * math.pi

    idx = np.zeros(6, dtype=np.int64)
    for j in range(6):
        unit = float(DEFAULT_ACTION_UNIT[j])
        if unit == 0.0:
            idx[j] = 0
        else:
            idx[j] = int(round((float(pose[j]) - float(DEFAULT_ACTION_LOW_WORLD[j])) / unit))
    return np.clip(idx, DEFAULT_CLIP_POSE_IDX_LOW, DEFAULT_CLIP_POSE_IDX_UP).astype(np.int64)


def _position_to_pose6_lookat_origin(position: Sequence[float]) -> np.ndarray:
    """Continuous Cube Mode pose for a camera position looking at origin."""
    pos = np.asarray(position, dtype=np.float32)
    pose = np.zeros(6, dtype=np.float32)
    pose[:3] = pos
    direction = -pos
    norm = float(np.linalg.norm(direction))
    if norm > 1e-8:
        direction = direction / norm
        pose[4] = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        pose[5] = yaw if yaw >= 0 else yaw + 2.0 * math.pi
    return pose


def _pose6_to_continuous_raw(pose6: Sequence[float]) -> np.ndarray:
    """Inverse tanh map used by TensorBatchEnv(action_space_type=continuous_tanh)."""
    pose = np.asarray(pose6, dtype=np.float32)
    norm = np.empty(5, dtype=np.float32)
    norm[:3] = pose[:3]
    norm[3] = pose[4] / (0.5 * math.pi)
    norm[4] = pose[5] / math.pi - 1.0
    norm = np.clip(norm, -1.0 + _CONT_EPS, 1.0 - _CONT_EPS)
    return np.arctanh(norm).astype(np.float32)


def _position_to_continuous_raw(position: Sequence[float]) -> np.ndarray:
    return _pose6_to_continuous_raw(_position_to_pose6_lookat_origin(position))


def _fibonacci_positions(n: int, radius: float) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n, dtype=np.float32)
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (radius * np.stack([x, y, z], axis=1)).astype(np.float32)


def _axis_positions(radius: float) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, radius],
            [radius, 0.0, 0.0],
            [0.0, 0.0, -radius],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
        ],
        dtype=np.float32,
    )


def _ring_positions(n: int, radius: float, height: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, num=max(n, 1), endpoint=False, dtype=np.float32)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(x, height)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _action_table(policy: str, args: argparse.Namespace) -> np.ndarray:
    if policy == "fibonacci":
        positions = _fibonacci_positions(args.episode_len, args.view_radius)
    elif policy == "axis6":
        positions = _axis_positions(args.view_radius)
    elif policy == "ring":
        positions = _ring_positions(args.episode_len, args.view_radius, height=0.25)
    elif policy == "repeat_top":
        positions = np.asarray([[0.0, 0.0, args.view_radius]], dtype=np.float32)
    else:
        raise ValueError(f"no deterministic action table for policy={policy}")
    if args.action_space_type == "continuous_tanh":
        return np.stack([_position_to_continuous_raw(p) for p in positions], axis=0).astype(np.float32)
    return np.stack([_pose_to_cube_action_idx(p) for p in positions], axis=0).astype(np.int64)


def _make_env(
    args: argparse.Namespace,
    mesh_paths: List[Path],
    preproc_paths: List[Path],
    seed: int,
) -> TensorBatchEnv:
    return TensorBatchEnv(
        num_envs=args.n_envs,
        mesh_paths=mesh_paths,
        preproc_paths=preproc_paths,
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
        action_space_type=args.action_space_type,
        seed=seed,
    )


def _random_grid_actions(rng: np.random.Generator, n_envs: int, args: argparse.Namespace) -> np.ndarray:
    if args.action_space_type == "continuous_tanh":
        # Uniform in the physical normalized Cube Mode domain, then map
        # through atanh so env.tanh(raw) recovers that uniform sample.
        norm = rng.uniform(
            low=-1.0 + _CONT_EPS,
            high=1.0 - _CONT_EPS,
            size=(n_envs, 5),
        ).astype(np.float32)
        return np.arctanh(norm).astype(np.float32)
    nvec = NVEC
    actions = np.zeros((n_envs, len(nvec)), dtype=np.int64)
    for j, n in enumerate(nvec):
        actions[:, j] = rng.integers(0, int(n), size=n_envs, dtype=np.int64)
    return actions


def _oracle_action_table(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    if args.oracle_candidate_source == "random_cube":
        dtype = np.float32 if args.action_space_type == "continuous_tanh" else np.int64
        return _random_grid_actions(rng, args.oracle_candidates, args).astype(dtype)
    if args.oracle_candidate_source == "fibonacci":
        positions = _fibonacci_positions(args.oracle_candidates, args.view_radius)
        if args.action_space_type == "continuous_tanh":
            return np.stack([_position_to_continuous_raw(p) for p in positions], axis=0).astype(np.float32)
        return np.stack([_pose_to_cube_action_idx(p) for p in positions], axis=0).astype(np.int64)
    raise ValueError(f"unknown oracle_candidate_source={args.oracle_candidate_source}")


@torch.no_grad()
def _greedy_oracle_actions(
    env: TensorBatchEnv,
    candidate_actions: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    device = env.device
    candidate_dtype = torch.float32 if env.action_space_type == "continuous_tanh" else torch.long
    candidates = torch.as_tensor(candidate_actions, dtype=candidate_dtype, device=device)
    chunk_size = max(1, int(chunk_size))
    n_candidates = int(candidates.shape[0])
    action_dim = 5 if env.action_space_type == "continuous_tanh" else 6
    actions_out = torch.empty((env.num_envs, action_dim), dtype=candidate_dtype, device=device)
    grid_size = int(env.grid_size)

    for env_i in range(env.num_envs):
        mesh_id = int(env._env_mesh_id[env_i].item())
        renderer = env._renderers[mesh_id]
        bbox_min = env._bbox_min[env_i]
        voxel_size = env._voxel_size[env_i]
        unseen_gt = ((env._grid_gt[env_i] > 0.5) & (env._scanned_gt_grid[env_i] < 0.5)).reshape(-1)

        best_count = -1
        best_action = candidates[0]
        for start in range(0, n_candidates, chunk_size):
            chunk = candidates[start : start + chunk_size]
            _, eyes, look_ats = env._decode_actions(chunk)
            eye_idx = torch.floor((eyes - bbox_min.view(1, 3)) / voxel_size.view(1, 3)).long()
            in_box = ((eye_idx >= 0) & (eye_idx < grid_size)).all(dim=-1)
            eye_idx_clamp = eye_idx.clamp(0, grid_size - 1)
            surface_collision = torch.zeros(chunk.shape[0], dtype=torch.bool, device=device)
            if in_box.any():
                surface_collision[in_box] = env._grid_gt[
                    env_i,
                    eye_idx_clamp[in_box, 0],
                    eye_idx_clamp[in_box, 1],
                    eye_idx_clamp[in_box, 2],
                ] > 0.5
            mesh_inside = renderer.points_inside_mesh(eyes)
            valid_candidate = ~(surface_collision | mesh_inside)
            if not valid_candidate.any():
                continue
            render = renderer.render_batch(eyes, look_ats)
            depth_flat = render.depth.reshape(chunk.shape[0], -1)
            alpha_flat = render.alpha.reshape(chunk.shape[0], -1)
            rays_world = env._world_rays(eyes, look_ats)
            if render.points is None:
                points_world = eyes[:, None, :] + depth_flat[..., None] * rays_world
            else:
                points_world = render.points.reshape(chunk.shape[0], -1, 3)

            target_idx = torch.floor((points_world - bbox_min.view(1, 1, 3)) / voxel_size.view(1, 1, 3)).long()
            in_grid = (
                (alpha_flat > 0.5)
                & (target_idx[..., 0] >= 0)
                & (target_idx[..., 0] < grid_size)
                & (target_idx[..., 1] >= 0)
                & (target_idx[..., 1] < grid_size)
                & (target_idx[..., 2] >= 0)
                & (target_idx[..., 2] < grid_size)
            )
            target_idx = target_idx.clamp(0, grid_size - 1)
            linear_idx = (
                target_idx[..., 0] * (grid_size * grid_size)
                + target_idx[..., 1] * grid_size
                + target_idx[..., 2]
            )

            for local_i in range(chunk.shape[0]):
                if not bool(valid_candidate[local_i].item()):
                    count = -1
                else:
                    visible = linear_idx[local_i][in_grid[local_i]]
                    if visible.numel() == 0:
                        count = 0
                    else:
                        visible = visible[unseen_gt[visible]]
                        count = int(torch.unique(visible).numel()) if visible.numel() else 0
                if count > best_count:
                    best_count = count
                    best_action = chunk[local_i]

        actions_out[env_i] = best_action

    return actions_out.detach().cpu().numpy()


def _summarize(
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
    policy: str,
    elapsed_s: float,
    step_calls: int,
) -> Dict[str, object]:
    cr = np.asarray([float(r["cr"]) for r in rows], dtype=np.float32)
    ep_len = np.asarray([int(r["length"]) for r in rows], dtype=np.float32)
    reward = np.asarray([float(r["episode_reward"]) for r in rows], dtype=np.float32)
    novelty = np.asarray([float(r.get("episode_novelty_ratio", 0.0)) for r in rows], dtype=np.float32)
    redundancy = np.asarray([float(r.get("episode_redundancy_ratio", 0.0)) for r in rows], dtype=np.float32)
    revisit = np.asarray([float(r.get("episode_revisit_penalty_mean", 0.0)) for r in rows], dtype=np.float32)
    new_gt = np.asarray([float(r.get("episode_new_gt_voxels", 0.0)) for r in rows], dtype=np.float32)
    visible_gt = np.asarray([float(r.get("episode_visible_gt_voxels", 0.0)) for r in rows], dtype=np.float32)
    total_env_steps = int(step_calls) * int(args.n_envs)
    elapsed_safe = max(float(elapsed_s), 1e-9)
    out: Dict[str, object] = {
        "policy": policy,
        "dataset": args.dataset,
        "coverage_hit_dilate_radius": int(args.coverage_hit_dilate_radius),
        "n_episodes": int(len(rows)),
        "elapsed_s": float(elapsed_s),
        "env_step_calls": int(step_calls),
        "total_env_steps": total_env_steps,
        "env_steps_per_sec": float(total_env_steps / elapsed_safe),
        "batch_steps_per_sec": float(step_calls / elapsed_safe),
        "batch_step_ms": float(1000.0 * elapsed_safe / max(int(step_calls), 1)),
        "episodes_per_sec": float(len(rows) / elapsed_safe),
        "cr_mean": float(cr.mean()) if len(cr) else 0.0,
        "cr_std": float(cr.std()) if len(cr) else 0.0,
        "cr_min": float(cr.min()) if len(cr) else 0.0,
        "cr_p10": float(np.percentile(cr, 10)) if len(cr) else 0.0,
        "cr_p25": float(np.percentile(cr, 25)) if len(cr) else 0.0,
        "cr_p50": float(np.percentile(cr, 50)) if len(cr) else 0.0,
        "cr_p75": float(np.percentile(cr, 75)) if len(cr) else 0.0,
        "cr_p90": float(np.percentile(cr, 90)) if len(cr) else 0.0,
        "cr_max": float(cr.max()) if len(cr) else 0.0,
        "reach_cr_50_rate": float((cr >= 0.50).mean()) if len(cr) else 0.0,
        "reach_cr_80_rate": float((cr >= 0.80).mean()) if len(cr) else 0.0,
        "reach_threshold_rate": float((cr > args.coverage_threshold).mean()) if len(cr) else 0.0,
        "ep_len_mean": float(ep_len.mean()) if len(ep_len) else 0.0,
        "ep_len_p50": float(np.percentile(ep_len, 50)) if len(ep_len) else 0.0,
        "episode_reward_mean": float(reward.mean()) if len(reward) else 0.0,
        "episode_novelty_ratio_mean": float(novelty.mean()) if len(novelty) else 0.0,
        "episode_redundancy_ratio_mean": float(redundancy.mean()) if len(redundancy) else 0.0,
        "episode_revisit_penalty_mean": float(revisit.mean()) if len(revisit) else 0.0,
        "episode_new_gt_voxels_mean": float(new_gt.mean()) if len(new_gt) else 0.0,
        "episode_visible_gt_voxels_mean": float(visible_gt.mean()) if len(visible_gt) else 0.0,
        "collision_rate": float(np.mean([bool(r["collision"]) for r in rows])) if rows else 0.0,
        "early_stop_rate": float(np.mean([bool(r["early_stopped"]) for r in rows])) if rows else 0.0,
        "timeout_rate": float(np.mean([bool(r["timeout"]) for r in rows])) if rows else 0.0,
    }
    return out


def _write_outputs(out_dir: Path, policy: str, rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    policy_dir = out_dir / policy
    policy_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "policy",
        "episode",
        "seq_name",
        "mesh_id",
        "cr",
        "length",
        "episode_reward",
        "episode_new_gt_voxels",
        "episode_visible_gt_voxels",
        "episode_novelty_ratio",
        "episode_redundancy_ratio",
        "episode_revisit_penalty_mean",
        "collision",
        "early_stopped",
        "timeout",
    ]
    with open(policy_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    with open(policy_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(policy_dir / "summary.txt", "w") as f:
        f.write(
            f"policy={policy} dataset={summary['dataset']} n={summary['n_episodes']}\n"
            f"cr_mean={summary['cr_mean']:.4f} p50={summary['cr_p50']:.4f} "
            f"p90={summary['cr_p90']:.4f} max={summary['cr_max']:.4f}\n"
            f"reach50={summary['reach_cr_50_rate']:.3f} "
            f"reach80={summary['reach_cr_80_rate']:.3f} "
            f"reach_threshold={summary['reach_threshold_rate']:.3f}\n"
            f"ep_len_mean={summary['ep_len_mean']:.2f} "
            f"collision={summary['collision_rate']:.3f} "
            f"early_stop={summary['early_stop_rate']:.3f} "
            f"timeout={summary['timeout_rate']:.3f}\n"
            f"reward={summary['episode_reward_mean']:.1f} "
            f"novelty={summary['episode_novelty_ratio_mean']:.3f} "
            f"redundancy={summary['episode_redundancy_ratio_mean']:.3f} "
            f"revisit={summary['episode_revisit_penalty_mean']:.3f}\n"
            f"fps={summary['env_steps_per_sec']:.1f} "
            f"batch_step_ms={summary['batch_step_ms']:.2f} "
            f"elapsed_s={summary['elapsed_s']:.1f}\n"
        )


def evaluate_policy(
    policy: str,
    args: argparse.Namespace,
    mesh_paths: List[Path],
    preproc_paths: List[Path],
) -> Dict[str, object]:
    seed = args.seed + args.policy_seed_stride * args.policy_index
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = _make_env(args, mesh_paths, preproc_paths, seed)
    obs = env.reset()

    model = None
    if policy == "ppo":
        if not args.ckpt:
            raise ValueError(f"--ckpt is required when evaluating policy={policy}")
        print(f"[eval:{policy}] loading PPO from {args.ckpt}", flush=True)
        model = PPO.load(args.ckpt, device=args.device)

    table = None
    if policy in {"fibonacci", "axis6", "ring", "repeat_top"}:
        table = _action_table(policy, args)
    oracle_table = None
    if policy in {"greedy_oracle", "candidate_oracle", "oracle"}:
        oracle_table = _oracle_action_table(args, rng)
        print(
            f"[eval:{policy}] oracle candidates={len(oracle_table)} "
            f"source={args.oracle_candidate_source} chunk={args.oracle_chunk_size}",
            flush=True,
        )

    rows: List[Dict[str, object]] = []
    phase = np.zeros(args.n_envs, dtype=np.int64)
    started = time.time()
    step_calls = 0
    while len(rows) < args.n_episodes:
        if policy == "ppo":
            actions, _ = model.predict(obs, deterministic=args.deterministic)
            dtype = np.float32 if args.action_space_type == "continuous_tanh" else np.int64
            actions = np.asarray(actions, dtype=dtype)
        elif policy == "random":
            actions = _random_grid_actions(rng, args.n_envs, args)
        elif oracle_table is not None:
            actions = _greedy_oracle_actions(env, oracle_table, args.oracle_chunk_size)
        elif table is not None:
            actions = table[phase % len(table)]
        else:
            raise ValueError(f"unknown policy: {policy}")

        obs, _, dones, infos = env.step(actions)
        phase += 1
        step_calls += 1
        for env_i, done in enumerate(dones):
            if not done:
                continue
            info = infos[env_i]
            ep_info = info.get("episode") or {}
            row = {
                "policy": policy,
                "episode": len(rows),
                "seq_name": info.get("seq_name", ""),
                "mesh_id": int(info.get("mesh_id", -1)),
                "cr": float(info.get("cr", 0.0)),
                "length": int(info.get("step_idx", ep_info.get("l", 0))),
                "episode_reward": float(ep_info.get("r", 0.0)),
                "episode_new_gt_voxels": float(info.get("episode_new_gt_voxels", 0.0)),
                "episode_visible_gt_voxels": float(info.get("episode_visible_gt_voxels", 0.0)),
                "episode_novelty_ratio": float(info.get("episode_novelty_ratio", 0.0)),
                "episode_redundancy_ratio": float(info.get("episode_redundancy_ratio", 0.0)),
                "episode_revisit_penalty_mean": float(info.get("episode_revisit_penalty_mean", 0.0)),
                "collision": bool(info.get("collision", False)),
                "early_stopped": bool(info.get("early_stopped", False)),
                "timeout": bool(info.get("TimeLimit.truncated", False)),
            }
            rows.append(row)
            phase[env_i] = 0
            if len(rows) >= args.n_episodes:
                break

        if step_calls % args.log_every_steps == 0:
            partial = _summarize(rows, args, policy, time.time() - started, step_calls) if rows else {}
            print(
                f"[eval:{policy}] step_calls={step_calls} episodes={len(rows)}/"
                f"{args.n_episodes} cr_mean={partial.get('cr_mean', 0.0):.3f} "
                f"p50={partial.get('cr_p50', 0.0):.3f} "
                f"fps={partial.get('env_steps_per_sec', 0.0):.1f}",
                flush=True,
            )

    elapsed = time.time() - started
    env.close()
    summary = _summarize(rows[: args.n_episodes], args, policy, elapsed, step_calls)
    _write_outputs(Path(args.out_dir), policy, rows[: args.n_episodes], summary)
    print(
        f"[eval:{policy}] DONE cr_mean={summary['cr_mean']:.4f} "
        f"p50={summary['cr_p50']:.4f} reach80={summary['reach_cr_80_rate']:.3f} "
        f"len={summary['ep_len_mean']:.2f} collision={summary['collision_rate']:.3f} "
        f"fps={summary['env_steps_per_sec']:.1f}",
        flush=True,
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["shapenet", "abo"], default="shapenet")
    p.add_argument("--shapenet_root", type=str, required=True)
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--policies", type=str, default="random,repeat_top,ppo")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--oracle_candidates", type=int, default=96)
    p.add_argument("--oracle_candidate_source", choices=["fibonacci", "random_cube"], default="random_cube")
    p.add_argument("--oracle_chunk_size", type=int, default=4)
    p.add_argument("--seq_names", type=str, default="")
    p.add_argument("--synsets", type=str, default="")
    p.add_argument("--categories", type=str, default="")
    p.add_argument("--limit_per_synset", type=int, default=0)
    p.add_argument("--max_meshes", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_envs", type=int, default=32)
    p.add_argument("--n_episodes", type=int, default=128)
    p.add_argument("--image_size", type=int, default=400)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--episode_len", type=int, default=50)
    p.add_argument("--buffer_size", type=int, default=30)
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--obs_grid_size", type=int, default=32)
    p.add_argument("--coverage_hit_dilate_radius", type=int, default=1)
    p.add_argument("--caption_dim", type=int, default=384)
    p.add_argument("--coverage_threshold", type=float, default=0.99)
    p.add_argument("--coverage_reward_scale", type=float, default=20.0)
    p.add_argument("--coverage_reward_type",
                   choices=["linear", "log", "remaining", "information_gain"],
                   default="linear")
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
    p.add_argument("--only_positive_rewards", dest="only_positive_rewards",
                   action="store_true", default=True)
    p.add_argument("--no_only_positive_rewards", dest="only_positive_rewards",
                   action="store_false")
    p.add_argument("--skip_free_raycast", action="store_true")
    p.add_argument("--no_update_empty_rays", action="store_true")
    p.add_argument("--auto_lookat_center", action="store_true")
    p.add_argument("--action_space_type",
                   choices=["discrete", "continuous_tanh"],
                   default="discrete")
    p.add_argument("--max_faces", type=int, default=5000)
    p.add_argument("--renderer_backend", choices=["torch", "open3d", "nvdiffrast", "voxel_cuda"], default="nvdiffrast")
    p.add_argument("--free_raycast_backend", choices=["auto", "cuda", "triton", "torch"], default="auto")
    p.add_argument("--free_mask_apply_mode", choices=["index", "dense", "triton"], default="triton")
    p.add_argument("--triton_bresenham_block_rays", type=int, default=64)
    p.add_argument("--view_radius", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy_seed_stride", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--log_every_steps", type=int, default=20)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = _split_csv(args.policies)
    if not policies:
        sys.exit("[eval] no policies requested.")

    names, mesh_paths, preproc_paths = _resolve_pairs(args)
    print(
        f"[eval] dataset={args.dataset} pairs={len(names)} n_envs={args.n_envs} "
        f"n_episodes={args.n_episodes} policies={policies}",
        flush=True,
    )

    summaries = []
    for i, policy in enumerate(policies):
        args.policy_index = i
        summaries.append(evaluate_policy(policy, args, mesh_paths, preproc_paths))

    with open(out_dir / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    with open(out_dir / "summary.txt", "w") as f:
        for s in summaries:
            f.write(
                f"{s['policy']:<14s} cr_mean={s['cr_mean']:.4f} "
                f"p50={s['cr_p50']:.4f} reach80={s['reach_cr_80_rate']:.3f} "
                f"len={s['ep_len_mean']:.2f} coll={s['collision_rate']:.3f} "
                f"rew={s['episode_reward_mean']:.1f} "
                f"nov={s['episode_novelty_ratio_mean']:.3f} "
                f"red={s['episode_redundancy_ratio_mean']:.3f} "
                f"fps={s['env_steps_per_sec']:.1f}\n"
            )
    print(f"[eval] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
