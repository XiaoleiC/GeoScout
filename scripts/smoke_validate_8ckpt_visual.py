"""Visual smoke validation for the eight GeoScout PPO checkpoints.

This script intentionally uses TensorBatchEnv, not the older single-env
validate.py path, so the smoke test exercises the same renderer/free-space
infrastructure used by training. It runs a small fixed set of objects through:

  - one discrete PPO checkpoint
  - one continuous_tanh PPO checkpoint
  - one axis6 open-loop baseline

For each episode it writes per-step camera images, rollout CSV/JSON, coverage
curves, final trajectory/coverage plots, and a lightweight HTML index.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from stable_baselines3 import PPO

from geoscout.data import SYNSET_TO_CATEGORY, list_shapenet
from geoscout.tensor_env import (
    DEFAULT_ACTION_LOW_WORLD,
    DEFAULT_ACTION_UNIT,
    DEFAULT_CLIP_POSE_IDX_LOW,
    DEFAULT_CLIP_POSE_IDX_UP,
    NVEC,
    TensorBatchEnv,
)
from geoscout.viz import (
    _draw_camera_frustum,
    _draw_shaded_mesh,
    _y_up_to_z_up,
    save_camera_trajectory_pro,
    save_coverage_curve,
    save_coverage_heatmap,
    save_filmstrip,
)


DEFAULT_DISCRETE_CKPT = (
    "/runs/train600-24m-discrete-s0-l40s-n128-wandb-0506/ppo_geoscout.zip"
)
DEFAULT_CONTINUOUS_CKPT = (
    "/runs/train600-24m-continuous-s0-l40s-n128-wandb-0506/ppo_geoscout.zip"
)
DEFAULT_CAPTION_JSONL = (
    "/data/geoscout_captions/"
    "full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl"
)
DEFAULT_SEQ_NAMES = (
    "03001627_1006be65e7bc937e9141f9b58470d646,"
    "04256520_1050790962944624febad4f49b26ec52,"
    "04379243_156d606fa86ba19c4eb174a255d0ec5e"
)

CAMERA_DISPLAY_FLIP_Y = True
CONT_EPS = 1e-4


@dataclass
class PolicySpec:
    name: str
    kind: str
    action_space_type: str
    ckpt: str = ""
    deterministic: bool = True


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def category_from_name(seq_name: str) -> str:
    synset = seq_name.split("_", 1)[0]
    return SYNSET_TO_CATEGORY.get(synset, synset)


def load_caption_lookup(path: str) -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, dict] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            object_id = str(row.get("object_id") or "")
            if object_id:
                out[object_id] = row
    return out


def caption_text(caption_row: Optional[dict]) -> str:
    if not caption_row:
        return ""
    cap = caption_row.get("caption")
    if isinstance(cap, dict):
        return str(cap.get("embedding_caption") or cap.get("final_caption") or "")
    return str(cap or "")


def caption_attrs(caption_row: Optional[dict]) -> Dict[str, int]:
    if not caption_row:
        return {}
    cap = caption_row.get("caption")
    if isinstance(cap, dict) and isinstance(cap.get("attributes"), dict):
        out = {}
        for k, v in cap["attributes"].items():
            try:
                out[str(k)] = int(v)
            except Exception:
                pass
        return out
    return {}


def resolve_pairs(
    *,
    shapenet_root: str,
    preproc_dir: str,
    seq_names: Sequence[str],
) -> Tuple[List[str], List[Path], List[Path]]:
    entries = list_shapenet(
        Path(shapenet_root),
        synsets=["03001627", "04256520", "04379243"],
        require_obj=True,
    )
    by_name = {e.name: e.mesh_path for e in entries}
    names: List[str] = []
    mesh_paths: List[Path] = []
    preproc_paths: List[Path] = []
    missing = []
    for name in seq_names:
        pp = Path(preproc_dir) / f"{name}.pt"
        mp = by_name.get(name)
        if mp is None or not pp.exists():
            missing.append(name)
            continue
        names.append(name)
        mesh_paths.append(Path(mp))
        preproc_paths.append(pp)
    if missing:
        raise FileNotFoundError(
            "Missing requested smoke samples: " + ", ".join(missing[:10])
        )
    if not names:
        raise RuntimeError("No smoke samples resolved.")
    return names, mesh_paths, preproc_paths


def camera_display_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    return np.flipud(img) if CAMERA_DISPLAY_FLIP_Y else img


def render_depth(depth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    visible = alpha > 0.5
    out = np.zeros_like(depth, dtype=np.float32)
    if visible.any():
        vals = depth[visible]
        lo, hi = float(vals.min()), float(vals.max())
        out[visible] = 1.0 if hi <= lo else (depth[visible] - lo) / (hi - lo)
    return out


def voxel_centers(indices: np.ndarray, bbox_min: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.float32).reshape(-1, 3)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return bbox_min[None, :] + (idx + 0.5) * voxel_size[None, :]


def _pose_to_cube_action_idx(position: Sequence[float]) -> np.ndarray:
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


def axis6_action(step_idx: int, radius: float = 0.95) -> np.ndarray:
    positions = np.asarray(
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
    return _pose_to_cube_action_idx(positions[step_idx % len(positions)]).reshape(1, 6)


def fibonacci_positions(n: int, radius: float) -> np.ndarray:
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


def ring_positions(n: int, radius: float, height: float = 0.25) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, num=max(n, 1), endpoint=False, dtype=np.float32)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(x, height)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def position_to_pose6_lookat_origin(position: Sequence[float]) -> np.ndarray:
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


def pose6_to_continuous_raw(pose6: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose6, dtype=np.float32)
    norm = np.empty(5, dtype=np.float32)
    norm[:3] = pose[:3]
    norm[3] = pose[4] / (0.5 * math.pi)
    norm[4] = pose[5] / math.pi - 1.0
    norm = np.clip(norm, -1.0 + CONT_EPS, 1.0 - CONT_EPS)
    return np.arctanh(norm).astype(np.float32)


def position_to_action(position: Sequence[float], action_space_type: str) -> np.ndarray:
    if action_space_type == "continuous_tanh":
        return pose6_to_continuous_raw(position_to_pose6_lookat_origin(position))
    return _pose_to_cube_action_idx(position)


def deterministic_baseline_action(
    *,
    policy_kind: str,
    step_idx: int,
    action_space_type: str,
    episode_len: int,
    radius: float,
) -> np.ndarray:
    if policy_kind == "axis6":
        positions = np.asarray(
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
    elif policy_kind == "fibonacci":
        positions = fibonacci_positions(episode_len, radius)
    elif policy_kind == "ring":
        positions = ring_positions(episode_len, radius)
    else:
        raise ValueError(f"unknown deterministic baseline: {policy_kind}")
    action = position_to_action(positions[step_idx % len(positions)], action_space_type)
    return action.reshape(1, -1)


def random_action(rng: np.random.Generator, action_space_type: str) -> np.ndarray:
    if action_space_type == "continuous_tanh":
        norm = rng.uniform(
            low=-1.0 + CONT_EPS,
            high=1.0 - CONT_EPS,
            size=(1, 5),
        ).astype(np.float32)
        return np.arctanh(norm).astype(np.float32)
    actions = np.zeros((1, len(NVEC)), dtype=np.int64)
    for j, n in enumerate(NVEC):
        actions[:, j] = rng.integers(0, int(n), size=1, dtype=np.int64)
    return actions


def compute_collision(env: TensorBatchEnv, eyes: torch.Tensor) -> torch.Tensor:
    eye_idx = torch.floor((eyes - env._bbox_min) / env._voxel_size).long()
    in_box = ((eye_idx >= 0) & (eye_idx < env.grid_size)).all(dim=-1)
    eye_idx_clamp = eye_idx.clamp(0, env.grid_size - 1)
    env_arange = torch.arange(env.num_envs, device=env.device)
    eye_on_gt_surface = env._grid_gt[
        env_arange, eye_idx_clamp[:, 0], eye_idx_clamp[:, 1], eye_idx_clamp[:, 2]
    ] > 0.5
    surface_collision = in_box & eye_on_gt_surface
    mesh_inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for mid in torch.unique(env._env_mesh_id).tolist():
        sel = (env._env_mesh_id == int(mid)).nonzero().flatten()
        mesh_inside[sel] = env._renderers[int(mid)].points_inside_mesh(eyes[sel])
    return surface_collision | mesh_inside


@torch.no_grad()
def preview_action(
    env: TensorBatchEnv,
    action_np: np.ndarray,
) -> Dict[str, object]:
    dtype = torch.float32 if env.action_space_type == "continuous_tanh" else torch.long
    action_t = torch.as_tensor(action_np, dtype=dtype, device=env.device)
    pose6, eyes, ats = env._decode_actions(action_t)
    rays_world = env._world_rays(eyes, ats)
    target_idx, depth_flat, hit_mask = env._voxel_first_hit_render(eyes, rays_world)
    collision = compute_collision(env, eyes)
    active_hit_mask = hit_mask & (~collision).view(env.num_envs, 1)
    return {
        "pose6": pose6.detach().clone(),
        "eyes": eyes.detach().clone(),
        "ats": ats.detach().clone(),
        "rays_world": rays_world.detach(),
        "target_idx": target_idx.detach().clone(),
        "depth_flat": depth_flat.detach().clone(),
        "hit_mask": active_hit_mask.detach().clone(),
        "raw_hit_mask": hit_mask.detach().clone(),
        "collision": collision.detach().clone(),
    }


@torch.no_grad()
def predicted_coverage_update(
    env: TensorBatchEnv,
    preview: Dict[str, object],
    prev_scanned: torch.Tensor,
    dilate_radius: int,
) -> Dict[str, object]:
    G = int(env.grid_size)
    hit_grid = torch.zeros_like(env._grid_gt, dtype=torch.float32)
    in_grid = preview["hit_mask"]
    target_idx = preview["target_idx"]
    env_ids = torch.arange(env.num_envs, device=env.device).view(env.num_envs, 1)
    env_ids = env_ids.expand(env.num_envs, env.render_size * env.render_size)
    env_ids_flat = env_ids[in_grid]
    flat_idx = target_idx[in_grid].clamp(0, G - 1)
    if flat_idx.numel() > 0:
        hit_grid[env_ids_flat, flat_idx[:, 0], flat_idx[:, 1], flat_idx[:, 2]] = 1.0
    coverage_hit_grid = hit_grid
    if dilate_radius > 0:
        r = int(dilate_radius)
        coverage_hit_grid = F.max_pool3d(
            hit_grid.unsqueeze(1),
            kernel_size=2 * r + 1,
            stride=1,
            padding=r,
        ).squeeze(1)
    hit_gt = coverage_hit_grid * env._grid_gt
    new_gt = hit_gt * (1.0 - prev_scanned)
    post_scanned = torch.clamp(prev_scanned + hit_gt, 0.0, 1.0)
    cr = post_scanned.sum(dim=(1, 2, 3)) / env._num_valid_gt_per_env.clamp(min=1.0)
    return {
        "hit_grid": hit_grid.detach().clone(),
        "coverage_hit_grid": coverage_hit_grid.detach().clone(),
        "new_gt": new_gt.detach().clone(),
        "post_scanned": post_scanned.detach().clone(),
        "cr": cr.detach().clone(),
    }


def save_step_frame(
    *,
    alpha: np.ndarray,
    depth: np.ndarray,
    row: dict,
    cr_history: Sequence[float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
    })
    fig = plt.figure(figsize=(13.5, 4.6), dpi=170)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.25, 1.35], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(camera_display_image(alpha), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.set_title("alpha mask (+Y-up display)", loc="left", fontweight="bold")
    ax0.set_xticks([]); ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(camera_display_image(render_depth(depth, alpha)), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("foreground depth, normalized", loc="left", fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(cr_history), dtype=np.int64)
    ax2.plot(x, cr_history, color="#2563eb", marker="o", markersize=3, linewidth=1.5)
    for y, c in [(0.80, "#94a3b8"), (0.90, "#64748b"), (0.95, "#f97316"), (0.99, "#ef4444")]:
        ax2.axhline(y, color=c, linestyle="--", linewidth=0.8, alpha=0.75)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel("step")
    ax2.set_ylabel("CR")
    ax2.grid(alpha=0.25)
    ax2.set_title("coverage so far", loc="left", fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    pose = row["pose6"]
    action = row["action"]
    text = [
        f"policy: {row['policy']}",
        f"sample: {row['seq_name']}",
        f"step: {row['step']}",
        f"reward: {row['reward']:+.4f}",
        f"CR: {row['cr']:.4f}  delta: {row['coverage_delta']:.4f}",
        f"hit pixels: {row['hit_pixels']} / {row['total_pixels']}",
        f"new GT voxels: {row['new_gt_voxels']}",
        f"visible GT voxels: {row['visible_gt_voxels']}",
        f"free voxels: {row['free_voxels_after']}",
        f"occupied voxels: {row['occupied_voxels_after']}",
        f"collision: {row['collision']}  done: {row['done']}",
        f"action: {action}",
        "eye: " + np.array2string(np.asarray(row["eye"]), precision=3, separator=", "),
        "look-at: " + np.array2string(np.asarray(row["look_at"]), precision=3, separator=", "),
        "pose6: " + np.array2string(np.asarray(pose), precision=3, separator=", "),
    ]
    ax3.text(
        0.0,
        1.0,
        "\n".join(text),
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.5,
        color="#0f172a",
    )
    fig.suptitle(
        f"GeoScout smoke validation | {row['policy']} | {row['seq_name']} | step {row['step']}",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def select_key_frame_records(
    frame_records: Sequence[Tuple[int, np.ndarray, np.ndarray, dict, List[float]]],
    max_frames: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, dict, List[float]]]:
    """Pick a compact, review-friendly subset of step frames.

    We keep the first/last frame, threshold-crossing frames, and the
    highest information-gain frames. This mirrors what we care about in NBV:
    where the rollout starts, when it crosses useful CR milestones, which
    views add the most geometry, and where it ends.
    """
    if not frame_records:
        return []
    max_frames = max(1, int(max_frames))
    by_step = {int(r[0]): r for r in frame_records}
    selected_steps = {int(frame_records[0][0]), int(frame_records[-1][0])}

    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
    for th in thresholds:
        for step, _, _, row, _ in frame_records:
            if float(row.get("cr", 0.0)) >= th:
                selected_steps.add(int(step))
                break

    ranked_gain = sorted(
        frame_records,
        key=lambda r: float(r[3].get("coverage_delta", 0.0)),
        reverse=True,
    )
    for step, *_ in ranked_gain[: max_frames]:
        selected_steps.add(int(step))

    if len(selected_steps) < max_frames and len(frame_records) > 2:
        evenly = np.linspace(0, len(frame_records) - 1, num=min(max_frames, len(frame_records)))
        for idx in evenly.astype(int).tolist():
            selected_steps.add(int(frame_records[idx][0]))

    selected = [by_step[s] for s in sorted(selected_steps) if s in by_step]
    if len(selected) <= max_frames:
        return selected

    # Preserve endpoints and the largest-gain/threshold frames under budget.
    endpoint_steps = {int(frame_records[0][0]), int(frame_records[-1][0])}
    middle = [r for r in selected if int(r[0]) not in endpoint_steps]
    middle = sorted(
        middle,
        key=lambda r: (
            int(float(r[3].get("cr", 0.0)) >= 0.99),
            float(r[3].get("coverage_delta", 0.0)),
            int(r[0]),
        ),
        reverse=True,
    )
    keep = [by_step[s] for s in sorted(endpoint_steps) if s in by_step]
    keep += middle[: max(0, max_frames - len(keep))]
    return sorted(keep, key=lambda r: int(r[0]))


def save_action_plot(rows: Sequence[dict], path: Path, action_space_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    steps = np.asarray([int(r["step"]) for r in rows])
    cr = np.asarray([float(r["cr"]) for r in rows])
    poses = np.asarray([np.asarray(r["pose6"], dtype=np.float32) for r in rows])
    fig, axes = plt.subplots(6, 1, figsize=(10, 10), dpi=160, sharex=True)
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, ax in enumerate(axes):
        ax.plot(steps, poses[:, i], marker="o", markersize=2.5, linewidth=1.0)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("step")
    axes[0].set_title(f"decoded pose6 over time ({action_space_type})", loc="left", fontweight="bold")
    ax2 = axes[0].twinx()
    ax2.plot(steps, cr, color="#16a34a", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("CR", color="#16a34a")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_numbered_policy_trajectory(
    *,
    rows: Sequence[dict],
    mesh_path: Path,
    path: Path,
    fov_deg: float,
    title: str,
) -> None:
    """Per-policy trajectory plot with every executed camera pose labelled.

    This is intentionally separate from the compact dashboard trajectory:
    it prioritizes auditability over minimal clutter. Every step gets a
    numbered marker and a frustum, so we can inspect the exact episode path.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    eyes = _y_up_to_z_up(np.asarray([r["eye"] for r in rows], dtype=np.float32))
    ats = _y_up_to_z_up(np.asarray([r["look_at"] for r in rows], dtype=np.float32))
    T = len(rows)

    fig = plt.figure(figsize=(11.0, 9.5), dpi=180, facecolor="#f8fafc")
    ax = fig.add_subplot(111, projection="3d", facecolor="#ffffff")
    _draw_shaded_mesh(ax, mesh_path, alpha=0.56)

    cmap = matplotlib.colormaps["viridis"]
    ax.plot(eyes[:, 0], eyes[:, 1], eyes[:, 2], color="#334155", linewidth=1.1, alpha=0.75)
    for i, (eye, at) in enumerate(zip(eyes, ats), start=1):
        color = cmap((i - 1) / max(T - 1, 1))
        ax.scatter(
            [eye[0]],
            [eye[1]],
            [eye[2]],
            color=[color],
            s=38,
            edgecolor="white",
            linewidth=0.6,
            depthshade=True,
        )
        _draw_camera_frustum(
            ax,
            eye=eye,
            at=at,
            color=color,
            depth=0.11,
            fov_deg=fov_deg,
            linewidth=0.62,
            alpha=0.72 if T <= 20 else 0.46,
        )
        ax.text(
            eye[0],
            eye[1],
            eye[2],
            str(i),
            color="#0f172a",
            fontsize=6.0 if T <= 20 else 5.0,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

    pts = np.concatenate([eyes, ats], axis=0)
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    center = 0.5 * (pmin + pmax)
    half = max(float(np.max(pmax - pmin)) * 0.58, 0.9)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=T))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.58, pad=0.05)
    cb.set_label("step index")
    ax.view_init(elev=22, azim=-52)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (up)")
    ax.set_title(
        f"{title}\nEvery numbered marker is one executed camera pose; frustum top edge marks image up.",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_episode_dashboard(ep_dir: Path, rows: Sequence[dict], title: str) -> None:
    assets = [
        ("numbered trajectory", ep_dir / "numbered_trajectory.png"),
        ("trajectory", ep_dir / "trajectory.png"),
        ("coverage", ep_dir / "coverage_heatmap.png"),
        ("curve", ep_dir / "cr_curve.png"),
        ("actions", ep_dir / "action_pose6.png"),
        ("filmstrip", ep_dir / "filmstrip_steps.png"),
    ]
    imgs = []
    for _, p in assets:
        if p.exists():
            imgs.append((p, Image.open(p).convert("RGB")))
    if not imgs:
        return
    thumb_w = 640
    thumb_h = 480
    tiles = []
    for p, img in imgs:
        img.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        tiles.append((p, canvas))
    cols = 2
    rows_n = int(math.ceil(len(tiles) / cols))
    header_h = 78
    out = Image.new("RGB", (cols * thumb_w, rows_n * thumb_h + header_h), "#f8fafc")
    for i, (_, img) in enumerate(tiles):
        x = (i % cols) * thumb_w
        y = header_h + (i // cols) * thumb_h
        out.paste(img, (x, y))
    fig = plt.figure(figsize=(cols * 4.2, 0.7), dpi=160)
    fig.text(0.01, 0.72, title, fontsize=12, fontweight="bold", color="#0f172a")
    if rows:
        fig.text(
            0.01,
            0.26,
            f"final CR={float(rows[-1]['cr']):.4f} | steps={len(rows)} | "
            f"reward={sum(float(r['reward']) for r in rows):.4f}",
            fontsize=9,
            color="#475569",
        )
    tmp = ep_dir / "_dashboard_header.png"
    fig.savefig(tmp, facecolor="#f8fafc")
    plt.close(fig)
    header = Image.open(tmp).convert("RGB").resize((cols * thumb_w, header_h))
    out.paste(header, (0, 0))
    tmp.unlink(missing_ok=True)
    out.save(ep_dir / "dashboard.png", quality=95)


def write_episode_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "seq_name",
        "category",
        "step",
        "cr",
        "coverage_delta",
        "reward",
        "hit_pixels",
        "new_gt_voxels",
        "visible_gt_voxels",
        "free_voxels_after",
        "occupied_voxels_after",
        "collision",
        "done",
        "early_stopped",
        "timeout",
        "action",
        "eye",
        "look_at",
        "pose6",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            for k in ("action", "eye", "look_at", "pose6"):
                out[k] = json.dumps(np.asarray(out[k]).tolist())
            writer.writerow(out)


def make_env_for_sample(args, mesh_path: Path, preproc_path: Path, action_space_type: str, seed: int) -> TensorBatchEnv:
    return TensorBatchEnv(
        num_envs=1,
        mesh_paths=[mesh_path],
        preproc_paths=[preproc_path],
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
        action_space_type=action_space_type,
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
        seed=seed,
    )


def verify_model_space(model: PPO, env: TensorBatchEnv, policy_name: str) -> None:
    if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
        raise RuntimeError(
            f"{policy_name}: observation shape mismatch: "
            f"model={model.observation_space.shape} env={env.observation_space.shape}"
        )
    if tuple(model.action_space.shape or ()) != tuple(env.action_space.shape or ()):
        # MultiDiscrete has shape metadata too; compare nvec separately below.
        if not hasattr(model.action_space, "nvec"):
            raise RuntimeError(
                f"{policy_name}: action shape mismatch: "
                f"model={model.action_space} env={env.action_space}"
            )
    if hasattr(model.action_space, "nvec") or hasattr(env.action_space, "nvec"):
        model_nvec = getattr(model.action_space, "nvec", None)
        env_nvec = getattr(env.action_space, "nvec", None)
        if model_nvec is None or env_nvec is None or not np.array_equal(model_nvec, env_nvec):
            raise RuntimeError(
                f"{policy_name}: action nvec mismatch: model={model.action_space} env={env.action_space}"
            )


def run_episode(
    *,
    args,
    policy: PolicySpec,
    model: Optional[PPO],
    seq_name: str,
    mesh_path: Path,
    preproc_path: Path,
    caption_row: Optional[dict],
    out_dir: Path,
    seed: int,
) -> dict:
    ep_tag = f"{policy.name}__{seq_name}"
    ep_dir = out_dir / "episodes" / ep_tag
    step_dir = ep_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] episode start {ep_tag}", flush=True)

    env = make_env_for_sample(args, mesh_path, preproc_path, policy.action_space_type, seed)
    obs = env.reset()
    if model is not None:
        verify_model_space(model, env, policy.name)

    rows: List[dict] = []
    cr_history = [0.0]
    scanned_history: List[np.ndarray] = []
    eye_history: List[np.ndarray] = []
    at_history: List[np.ndarray] = []
    step_frames: List[Path] = []
    frame_records: List[Tuple[int, np.ndarray, np.ndarray, dict, List[float]]] = []
    predicted_last_scanned: Optional[torch.Tensor] = None
    total_reward = 0.0
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 9173)

    for step_idx in range(1, args.episode_len + 1):
        if policy.kind == "ppo":
            action, _ = model.predict(obs, deterministic=policy.deterministic)
            action = np.asarray(
                action,
                dtype=np.float32 if policy.action_space_type == "continuous_tanh" else np.int64,
            )
        elif policy.kind in {"axis6", "fibonacci", "ring"}:
            action = deterministic_baseline_action(
                policy_kind=policy.kind,
                step_idx=step_idx - 1,
                action_space_type=policy.action_space_type,
                episode_len=args.episode_len,
                radius=args.view_radius,
            )
        elif policy.kind == "random":
            action = random_action(rng, policy.action_space_type)
        else:
            raise ValueError(f"unknown policy kind: {policy.kind}")

        prev_scanned = env._scanned_gt_grid.detach().clone()
        preview = preview_action(env, action)
        predicted = predicted_coverage_update(
            env,
            preview,
            prev_scanned=prev_scanned,
            dilate_radius=args.coverage_hit_dilate_radius,
        )

        alpha = preview["hit_mask"][0].reshape(args.image_size, args.image_size).detach().cpu().numpy().astype(np.float32)
        depth = preview["depth_flat"][0].reshape(args.image_size, args.image_size).detach().cpu().numpy().astype(np.float32)

        obs, rewards, dones, infos = env.step(action)
        info = infos[0]
        reward = float(rewards[0])
        done = bool(dones[0])
        cr_env = float(info.get("cr", 0.0))
        cr_pred = float(predicted["cr"][0].detach().cpu().item())
        if abs(cr_env - cr_pred) > 2e-4:
            raise RuntimeError(
                f"{ep_tag} step {step_idx}: predicted CR {cr_pred:.6f} "
                f"does not match env CR {cr_env:.6f}"
            )

        pose6 = preview["pose6"][0].detach().cpu().numpy().astype(np.float32)
        eye = preview["eyes"][0].detach().cpu().numpy().astype(np.float32)
        at = preview["ats"][0].detach().cpu().numpy().astype(np.float32)
        bbox_min = env._bbox_min[0].detach().cpu().numpy().astype(np.float32)
        voxel_size = env._voxel_size[0].detach().cpu().numpy().astype(np.float32)
        hit_pixels = int(alpha.sum())
        free_count = 0
        occ_count = 0
        if not done:
            prob = env._prob_grid[0].detach()
            free_count = int((prob < 0.0).sum().cpu().item())
            occ_count = int((prob > 0.5).sum().cpu().item())
        else:
            # TensorBatchEnv auto-resets done envs before returning. Keep
            # final coverage via our verified predicted update, and mark
            # free/occupied as unavailable for this terminal post-step state.
            free_count = -1
            occ_count = -1

        row = {
            "policy": policy.name,
            "policy_kind": policy.kind,
            "action_space_type": policy.action_space_type,
            "deterministic": bool(policy.deterministic),
            "seq_name": seq_name,
            "category": category_from_name(seq_name),
            "caption": caption_text(caption_row),
            "step": int(step_idx),
            "cr": cr_env,
            "predicted_cr": cr_pred,
            "coverage_delta": float(info.get("coverage_delta", 0.0)),
            "reward": reward,
            "hit_pixels": hit_pixels,
            "total_pixels": int(args.image_size * args.image_size),
            "new_gt_voxels": int(round(float(info.get("new_gt_voxels", 0.0)))),
            "visible_gt_voxels": int(round(float(info.get("visible_gt_voxels", 0.0)))),
            "free_voxels_after": free_count,
            "occupied_voxels_after": occ_count,
            "collision": bool(info.get("collision", False)),
            "done": done,
            "early_stopped": bool(info.get("early_stopped", False)),
            "timeout": bool(info.get("TimeLimit.truncated", False)),
            "action": np.asarray(action).reshape(-1).tolist(),
            "pose6": pose6.tolist(),
            "eye": eye.tolist(),
            "look_at": at.tolist(),
        }
        rows.append(row)
        total_reward += reward
        cr_history.append(cr_env)
        scanned_history.append(predicted["post_scanned"][0].detach().cpu().numpy().astype(np.float32))
        predicted_last_scanned = predicted["post_scanned"].detach().clone()
        eye_history.append(eye)
        at_history.append(at)

        if args.step_frame_mode != "none":
            if args.step_frame_mode == "all":
                frame_path = step_dir / f"step_{step_idx:03d}.png"
                save_step_frame(
                    alpha=alpha,
                    depth=depth,
                    row=row,
                    cr_history=cr_history,
                    out_path=frame_path,
                )
                step_frames.append(frame_path)
            else:
                frame_records.append(
                    (
                        int(step_idx),
                        alpha.astype(np.float32, copy=True),
                        depth.astype(np.float32, copy=True),
                        dict(row),
                        [float(x) for x in cr_history],
                    )
                )

        if done:
            break

    elapsed = time.perf_counter() - t0
    final_cr = float(rows[-1]["cr"]) if rows else 0.0
    final_steps = len(rows)
    final_reward = float(total_reward)

    preproc = torch.load(preproc_path, map_location="cpu", weights_only=False)
    grid_gt = preproc["grid_gt"].detach().cpu().numpy().astype(np.float32)
    range_gt = preproc["range_gt"].detach().cpu().numpy().ravel().astype(np.float32)
    voxel_size_cpu = preproc["voxel_size_gt"].detach().cpu().numpy().ravel().astype(np.float32)
    gt_idx = np.argwhere(grid_gt > 0.5)
    bbox_min_centres = np.asarray([range_gt[1], range_gt[3], range_gt[5]], dtype=np.float32)
    bbox_min_corner = bbox_min_centres - 0.5 * voxel_size_cpu
    gt_points = voxel_centers(gt_idx, bbox_min_corner, voxel_size_cpu)
    if len(gt_points) > 60000:
        rng = np.random.default_rng(seed)
        gt_points_plot = gt_points[rng.choice(len(gt_points), size=60000, replace=False)]
    else:
        gt_points_plot = gt_points

    if scanned_history:
        scanned_np = np.stack(scanned_history, axis=0)
    else:
        scanned_np = np.zeros((1,) + grid_gt.shape, dtype=np.float32)

    save_coverage_curve(cr_history, ep_dir / "cr_curve.png")
    save_camera_trajectory_pro(
        eye_positions=eye_history,
        look_ats=at_history,
        bbox_min=env._action_box_min,
        bbox_max=env._action_box_max,
        path=ep_dir / "trajectory.png",
        object_pointcloud=gt_points_plot,
        mesh_path=mesh_path,
        fov_deg=args.fov_deg,
        title=f"{policy.name} | {seq_name}",
    )
    save_numbered_policy_trajectory(
        rows=rows,
        mesh_path=mesh_path,
        path=ep_dir / "numbered_trajectory.png",
        fov_deg=args.fov_deg,
        title=f"{policy.name} | {seq_name} | final CR={final_cr:.4f}",
    )
    save_coverage_heatmap(
        grid_gt=grid_gt,
        scanned_history=scanned_np,
        range_gt=range_gt,
        voxel_size=voxel_size_cpu,
        path=ep_dir / "coverage_heatmap.png",
        title=f"{policy.name} | {seq_name}",
    )
    save_action_plot(rows, ep_dir / "action_pose6.png", policy.action_space_type)
    if args.step_frame_mode == "key":
        for step, alpha_i, depth_i, row_i, cr_i in select_key_frame_records(
            frame_records,
            max_frames=args.max_step_frames,
        ):
            frame_path = step_dir / f"step_{step:03d}.png"
            save_step_frame(
                alpha=alpha_i,
                depth=depth_i,
                row=row_i,
                cr_history=cr_i,
                out_path=frame_path,
            )
            step_frames.append(frame_path)
    film_paths = step_frames
    captions = [f"s{r['step']} CR={r['cr']:.2f}" for r in rows]
    if args.step_frame_mode == "key":
        frame_step_set = {int(p.stem.split("_")[-1]) for p in step_frames}
        captions = [f"s{r['step']} CR={r['cr']:.2f}" for r in rows if int(r["step"]) in frame_step_set]
    save_filmstrip(
        film_paths,
        ep_dir / "filmstrip_steps.png",
        n_cols=5,
        thumbnail_px=150,
        captions=captions,
        title=f"{policy.name} | all executed steps",
    )
    make_episode_dashboard(
        ep_dir,
        rows,
        title=f"{policy.name} | {seq_name} | final CR={final_cr:.4f}",
    )
    write_episode_csv(rows, ep_dir / "steps.csv")
    ep_summary = {
        "policy": policy.name,
        "policy_kind": policy.kind,
        "action_space_type": policy.action_space_type,
        "deterministic": bool(policy.deterministic),
        "seq_name": seq_name,
        "category": category_from_name(seq_name),
        "caption": caption_text(caption_row),
        "attributes": caption_attrs(caption_row),
        "mesh_path": str(mesh_path),
        "preproc_path": str(preproc_path),
        "final_cr": final_cr,
        "steps": final_steps,
        "episode_reward": final_reward,
        "elapsed_s": float(elapsed),
        "fps_env_steps": float(final_steps / max(elapsed, 1e-9)),
        "early_stopped": bool(rows[-1]["early_stopped"]) if rows else False,
        "timeout": bool(rows[-1]["timeout"]) if rows else False,
        "collision": bool(any(r["collision"] for r in rows)),
        "cr_history": [float(x) for x in cr_history],
    }
    (ep_dir / "summary.json").write_text(json.dumps(ep_summary, indent=2, sort_keys=True))
    (ep_dir / "steps.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    if predicted_last_scanned is None:
        pass
    env.close()
    print(
        f"[smoke] episode done {ep_tag} steps={final_steps} "
        f"cr={final_cr:.4f} reward={final_reward:.4f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return ep_summary


def make_comparison_plots(out_dir: Path, summaries: Sequence[dict]) -> None:
    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    by_seq: Dict[str, List[dict]] = {}
    for s in summaries:
        by_seq.setdefault(str(s["seq_name"]), []).append(s)
    for seq, rows in by_seq.items():
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
        for s in rows:
            cr = np.asarray(s.get("cr_history", []), dtype=np.float32)
            label = f"{s['policy']} final={s['final_cr']:.3f}"
            ax.plot(np.arange(len(cr)), cr, marker="o", markersize=2.5, linewidth=1.5, label=label)
        for y, c in [(0.80, "#94a3b8"), (0.90, "#64748b"), (0.95, "#f97316"), (0.99, "#ef4444")]:
            ax.axhline(y, color=c, linestyle="--", linewidth=0.8, alpha=0.75)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
        ax.set_ylabel("coverage ratio")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"Policy comparison | {seq}", loc="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(comp_dir / f"{seq}_coverage_compare.png")
        plt.close(fig)


def load_selected_cases(path: str) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        cases = data.get("selected_cases", [])
    else:
        cases = data
    return [c for c in cases if isinstance(c, dict) and c.get("seq_name")]


def _policy_label(policy: str) -> str:
    return {
        "discrete_s1_det": "Discrete PPO",
        "continuous_s1_det": "Continuous PPO",
        "fibonacci": "Fibonacci",
        "axis6": "Axis6",
        "random": "Random",
        "ring": "Ring",
    }.get(policy, policy)


def _metric_color(value: float, good_high: bool = True) -> str:
    v = float(np.clip(value, 0.0, 1.0))
    if not good_high:
        v = 1.0 - v
    r = int(239 * (1.0 - v) + 34 * v)
    g = int(68 * (1.0 - v) + 197 * v)
    b = int(68 * (1.0 - v) + 94 * v)
    return f"#{r:02x}{g:02x}{b:02x}"


def _copy_contact_sheet(seq_name: str, contact_sheet_dir: str, out_path: Path) -> Optional[Path]:
    if not contact_sheet_dir:
        return None
    src = Path(contact_sheet_dir) / f"{seq_name}.png"
    if not src.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img.save(out_path, quality=92)
    return out_path


def save_case_summary_panel(
    *,
    case: dict,
    summaries: Sequence[dict],
    out_path: Path,
    contact_sheet_path: Optional[Path],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    policies = [s["policy"] for s in summaries]
    labels = [_policy_label(p) for p in policies]
    final_cr = np.asarray([float(s["final_cr"]) for s in summaries], dtype=np.float32)
    steps = np.asarray([int(s["steps"]) for s in summaries], dtype=np.float32)
    rewards = np.asarray([float(s["episode_reward"]) for s in summaries], dtype=np.float32)

    plt.rcParams.update({
        "font.size": 8.5,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
    })
    fig = plt.figure(figsize=(14.5, 8.5), dpi=170)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.25, 1.05], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.axis("off")
    if contact_sheet_path and contact_sheet_path.exists():
        img = Image.open(contact_sheet_path).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_title("multi-view object sheet", loc="left", fontweight="bold")
    else:
        ax_img.text(0.5, 0.5, "contact sheet unavailable", ha="center", va="center", color="#64748b")
        ax_img.set_title("multi-view object sheet", loc="left", fontweight="bold")

    ax_curve = fig.add_subplot(gs[0, 1])
    colors = {
        "discrete_s1_det": "#2563eb",
        "continuous_s1_det": "#16a34a",
        "fibonacci": "#f97316",
        "axis6": "#7c3aed",
        "random": "#64748b",
        "ring": "#0f766e",
    }
    for s in summaries:
        cr = np.asarray(s.get("cr_history", []), dtype=np.float32)
        ax_curve.plot(
            np.arange(len(cr)),
            cr,
            marker="o",
            markersize=2.0,
            linewidth=1.4,
            color=colors.get(s["policy"], None),
            label=f"{_policy_label(s['policy'])} {float(s['final_cr']):.3f}/{int(s['steps'])}",
        )
    for y, c in [(0.80, "#94a3b8"), (0.90, "#64748b"), (0.95, "#f97316"), (0.99, "#ef4444")]:
        ax_curve.axhline(y, color=c, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("coverage ratio")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=6.7, loc="lower right")
    ax_curve.set_title("CR trajectory comparison", loc="left", fontweight="bold")

    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")
    rows = []
    for label, cr, step, rew, s in zip(labels, final_cr, steps, rewards, summaries):
        rows.append([label, f"{cr:.4f}", f"{int(step)}", f"{rew:.2f}", "yes" if s["collision"] else "no"])
    table = ax_table.table(
        cellText=rows,
        colLabels=["policy", "CR", "steps", "reward", "coll."],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif c == 1:
            cell.set_facecolor(_metric_color(float(rows[r - 1][1])))
            cell.set_text_props(color="white", weight="bold")
    ax_table.set_title("terminal metrics", loc="left", fontweight="bold")

    ax_bar = fig.add_subplot(gs[1, 0])
    order = np.arange(len(labels))
    ax_bar.barh(order, final_cr, color=[colors.get(p, "#64748b") for p in policies], alpha=0.86)
    ax_bar.axvline(0.99, color="#ef4444", linestyle="--", linewidth=1.0)
    ax_bar.set_xlim(max(0.0, min(0.92, float(final_cr.min()) - 0.02)), 1.005)
    ax_bar.set_yticks(order)
    ax_bar.set_yticklabels(labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("final CR")
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.set_title("final coverage", loc="left", fontweight="bold")

    ax_steps = fig.add_subplot(gs[1, 1])
    ax_steps.barh(order, steps, color=[colors.get(p, "#64748b") for p in policies], alpha=0.86)
    ax_steps.set_yticks(order)
    ax_steps.set_yticklabels(labels)
    ax_steps.invert_yaxis()
    ax_steps.set_xlabel("episode length")
    ax_steps.grid(axis="x", alpha=0.2)
    ax_steps.set_title("steps to stop or timeout", loc="left", fontweight="bold")

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    attrs = case.get("attributes") or {}
    on_attrs = [k.replace("has_", "") for k, v in attrs.items() if int(v) == 1]
    text = [
        f"seq: {case.get('seq_name', '')}",
        f"category: {case.get('category', '')}",
        f"selection: {case.get('selection_reason', '')}",
        f"score: {float(case.get('selection_score', 0.0)):.3f}",
        "",
        str(case.get("selection_explanation", "")),
        "",
        "caption:",
        str(case.get("caption", "")),
        "",
        "geometry tags:",
        ", ".join((case.get("shape_tags") or [])[:12]),
        "",
        "positive attributes:",
        ", ".join(on_attrs[:12]),
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(text),
        va="top",
        ha="left",
        fontsize=7.5,
        color="#0f172a",
        wrap=True,
    )

    fig.suptitle(
        f"GeoScout selected-case validation | {case.get('category', '')} | {case.get('selection_reason', '')}",
        x=0.015,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_trajectory_panel(
    *,
    out_dir: Path,
    seq_name: str,
    summaries: Sequence[dict],
    out_path: Path,
    fov_deg: float,
) -> None:
    """One 3D object view with all policy camera poses overlaid.

    This is the main qualitative NBV sanity plot: all policies are drawn in
    the same world frame around the shaded object, with every executed camera
    center numbered by step. It makes short-path behavior and view redundancy
    visible without switching between per-policy dashboards.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return
    if not summaries:
        return

    colors = {
        "discrete_s1_det": "#2563eb",
        "continuous_s1_det": "#16a34a",
        "fibonacci": "#f97316",
        "axis6": "#7c3aed",
        "random": "#64748b",
        "ring": "#0f766e",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.5, 10.5), dpi=180, facecolor="#f8fafc")
    ax = fig.add_subplot(111, projection="3d", facecolor="#ffffff")

    mesh_path = summaries[0].get("mesh_path")
    drew_mesh = False
    if mesh_path:
        drew_mesh = _draw_shaded_mesh(ax, mesh_path, alpha=0.54)
    if not drew_mesh:
        ax.text2D(0.02, 0.96, "mesh unavailable", transform=ax.transAxes, color="#64748b")

    all_eyes = []
    all_ats = []
    for s in summaries:
        policy = str(s["policy"])
        steps_json = out_dir / "episodes" / f"{policy}__{seq_name}" / "steps.json"
        if not steps_json.exists():
            continue
        rows = json.loads(steps_json.read_text())
        if not rows:
            continue
        eyes = _y_up_to_z_up(np.asarray([r["eye"] for r in rows], dtype=np.float32))
        ats = _y_up_to_z_up(np.asarray([r["look_at"] for r in rows], dtype=np.float32))
        all_eyes.append(eyes)
        all_ats.append(ats)
        color = colors.get(policy, "#0f172a")
        label = f"{_policy_label(policy)} ({len(rows)} steps, CR={float(s['final_cr']):.3f})"
        ax.plot(
            eyes[:, 0],
            eyes[:, 1],
            eyes[:, 2],
            color=color,
            linewidth=1.7,
            alpha=0.9,
            label=label,
        )
        ax.scatter(
            eyes[:, 0],
            eyes[:, 1],
            eyes[:, 2],
            color=color,
            s=24,
            edgecolor="white",
            linewidth=0.5,
            depthshade=True,
        )
        # Draw every frustum, but keep them small and transparent so 50-step
        # baselines stay readable in the shared plot.
        for i, (eye, at) in enumerate(zip(eyes, ats), start=1):
            _draw_camera_frustum(
                ax,
                eye=eye,
                at=at,
                color=color,
                depth=0.105,
                fov_deg=fov_deg,
                linewidth=0.42,
                alpha=0.34 if len(rows) > 20 else 0.55,
            )
            ax.text(
                eye[0],
                eye[1],
                eye[2],
                str(i),
                color=color,
                fontsize=5.2 if len(rows) > 20 else 6.2,
                ha="center",
                va="center",
                zorder=10,
            )

    if all_eyes:
        pts = np.concatenate(all_eyes + all_ats, axis=0)
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        center = 0.5 * (pmin + pmax)
        half = max(float(np.max(pmax - pmin)) * 0.58, 0.9)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (up)")
    ax.view_init(elev=22, azim=-52)
    ax.legend(loc="upper left", fontsize=7.4, frameon=True)
    ax.set_title(
        f"All-policy camera trajectories | {seq_name}\n"
        "Each numbered marker is one executed camera pose; frustum top edge marks image up.",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_overview_plots(out_dir: Path, summaries: Sequence[dict], cases: Sequence[dict]) -> None:
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    policies = sorted({str(s["policy"]) for s in summaries})
    colors = {
        "discrete_s1_det": "#2563eb",
        "continuous_s1_det": "#16a34a",
        "fibonacci": "#f97316",
        "axis6": "#7c3aed",
        "random": "#64748b",
        "ring": "#0f766e",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=170)
    agg = []
    for p in policies:
        rows = [s for s in summaries if s["policy"] == p]
        agg.append(
            {
                "policy": p,
                "cr_mean": float(np.mean([float(s["final_cr"]) for s in rows])),
                "steps_mean": float(np.mean([int(s["steps"]) for s in rows])),
                "success": float(np.mean([float(s["final_cr"]) >= 0.99 for s in rows])),
                "collision": float(np.mean([bool(s["collision"]) for s in rows])),
            }
        )
    x = np.arange(len(agg))
    axes[0].bar(x, [a["cr_mean"] for a in agg], color=[colors.get(a["policy"], "#64748b") for a in agg])
    axes[0].axhline(0.99, color="#ef4444", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.90, 1.005)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_policy_label(a["policy"]) for a in agg], rotation=25, ha="right")
    axes[0].set_title("mean final CR", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, [a["steps_mean"] for a in agg], color=[colors.get(a["policy"], "#64748b") for a in agg])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_policy_label(a["policy"]) for a in agg], rotation=25, ha="right")
    axes[1].set_title("mean episode length", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(x, [a["success"] for a in agg], color=[colors.get(a["policy"], "#64748b") for a in agg])
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([_policy_label(a["policy"]) for a in agg], rotation=25, ha="right")
    axes[2].set_title("success@0.99 on selected cases", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "overview_policy_bars.png", bbox_inches="tight")
    plt.close(fig)

    seqs = [str(c.get("seq_name")) for c in cases if c.get("seq_name")]
    by_key = {(str(s["seq_name"]), str(s["policy"])): s for s in summaries}
    heat = np.full((len(policies), len(seqs)), np.nan, dtype=np.float32)
    for i, p in enumerate(policies):
        for j, seq in enumerate(seqs):
            s = by_key.get((seq, p))
            if s is not None:
                heat[i, j] = float(s["final_cr"])
    fig, ax = plt.subplots(figsize=(max(12, len(seqs) * 0.22), 3.8), dpi=170)
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0.90, vmax=1.0)
    ax.set_yticks(np.arange(len(policies)))
    ax.set_yticklabels([_policy_label(p) for p in policies])
    ax.set_xticks(np.arange(len(seqs)))
    ax.set_xticklabels([s.split("_", 1)[0] + "\n" + s[-4:] for s in seqs], fontsize=5.5, rotation=0)
    ax.set_title("final CR heatmap across selected cases", loc="left", fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.01)
    cb.set_label("final CR")
    fig.tight_layout()
    fig.savefig(report_dir / "selected_case_cr_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    (report_dir / "selected_case_rollout_summary.json").write_text(
        json.dumps({"aggregate": agg, "episodes": list(summaries)}, indent=2, sort_keys=True)
    )


def write_case_atlas(out_dir: Path, summaries: Sequence[dict], selected_cases: Sequence[dict], args) -> None:
    if not selected_cases:
        return
    cases_by_seq = {str(c["seq_name"]): c for c in selected_cases}
    by_seq: Dict[str, List[dict]] = {}
    for s in summaries:
        by_seq.setdefault(str(s["seq_name"]), []).append(s)

    make_overview_plots(out_dir, summaries, selected_cases)

    case_cards = []
    metrics_rows = []
    for case in selected_cases:
        seq = str(case["seq_name"])
        rows = sorted(by_seq.get(seq, []), key=lambda s: str(s["policy"]))
        if not rows:
            continue
        case_dir = out_dir / "cases" / seq
        case_dir.mkdir(parents=True, exist_ok=True)
        contact = _copy_contact_sheet(seq, args.contact_sheet_dir, case_dir / "contact_sheet.png")
        save_case_summary_panel(
            case=case,
            summaries=rows,
            out_path=case_dir / "case_summary.png",
            contact_sheet_path=contact,
        )
        save_combined_trajectory_panel(
            out_dir=out_dir,
            seq_name=seq,
            summaries=rows,
            out_path=case_dir / "all_policy_trajectory.png",
            fov_deg=args.fov_deg,
        )
        for r in rows:
            metrics_rows.append({
                "seq_name": seq,
                "category": case.get("category", ""),
                "selection_reason": case.get("selection_reason", ""),
                "policy": r["policy"],
                "final_cr": r["final_cr"],
                "steps": r["steps"],
                "reward": r["episode_reward"],
                "collision": r["collision"],
                "early_stopped": r["early_stopped"],
                "timeout": r["timeout"],
            })
        links = []
        for r in rows:
            ep_rel = Path("episodes") / f"{r['policy']}__{seq}"
            links.append(
                f"<a href='{ep_rel / 'dashboard.png'}'>{html.escape(_policy_label(r['policy']))}</a>"
            )
        card = f"""
<section class="case-card" data-reason="{html.escape(str(case.get('selection_reason', '')))}" data-category="{html.escape(str(case.get('category', '')))}">
  <h3>{html.escape(str(case.get('category', '')))} · {html.escape(seq)}</h3>
  <p class="reason">{html.escape(str(case.get('selection_reason', '')))} — {html.escape(str(case.get('selection_explanation', '')))}</p>
  <img src="{(case_dir / 'case_summary.png').relative_to(out_dir)}" loading="lazy">
  <img src="{(case_dir / 'all_policy_trajectory.png').relative_to(out_dir)}" loading="lazy">
  <p class="links">policy dashboards: {' · '.join(links)} · <a href="{Path('comparisons') / (seq + '_coverage_compare.png')}">coverage compare</a> · <a href="{(case_dir / 'all_policy_trajectory.png').relative_to(out_dir)}">all-policy 3D trajectory</a></p>
</section>
"""
        case_cards.append(card)

    with (out_dir / "reports" / "case_metrics.csv").open("w", newline="") as f:
        fieldnames = ["seq_name", "category", "selection_reason", "policy", "final_cr", "steps", "reward", "collision", "early_stopped", "timeout"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>GeoScout selected-case visual atlas</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 26px; color: #0f172a; background: #f8fafc; }}
h1 {{ margin-bottom: 6px; }}
p {{ color: #475569; }}
.overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin: 20px 0 26px; }}
.overview img, .case-card img {{ width: 100%; height: auto; display: block; border: 1px solid #e2e8f0; background: white; }}
.case-card {{ background: white; padding: 14px; border: 1px solid #e2e8f0; margin: 18px 0; }}
.case-card h3 {{ margin: 0 0 4px; font-size: 17px; }}
.reason {{ margin: 0 0 10px; font-size: 13px; }}
.links {{ font-size: 13px; }}
code {{ background: #e2e8f0; padding: 2px 4px; border-radius: 4px; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head><body>
<h1>GeoScout selected-case visual atlas</h1>
<p>52 candidate cases are replayed through the current TensorBatchEnv infra. Each case compares learned discrete PPO, learned continuous PPO, Fibonacci, axis6, and random policies. Camera images use the same +Y-up display convention as the smoke probe; 3D panels are Z-up for human inspection.</p>
<p>Config: renderer=<code>{html.escape(args.renderer_backend)}</code>, free=<code>{html.escape(args.free_raycast_backend)}/{html.escape(args.free_mask_apply_mode)}</code>, auto_lookat_center=<code>{args.auto_lookat_center}</code>, grid=<code>{args.grid_size}</code>, image=<code>{args.image_size}</code>, key step frames=<code>{args.max_step_frames}</code>.</p>
<div class="overview">
  <div><h2>Policy Summary</h2><img src="reports/overview_policy_bars.png"></div>
  <div><h2>Case CR Heatmap</h2><img src="reports/selected_case_cr_heatmap.png"></div>
</div>
{''.join(case_cards)}
</body></html>
"""
    (out_dir / "case_atlas.html").write_text(html_text)


def write_index(out_dir: Path, summaries: Sequence[dict], args) -> None:
    rows_html = []
    for s in summaries:
        ep_rel = Path("episodes") / f"{s['policy']}__{s['seq_name']}"
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(s['policy'])}</td>"
            f"<td>{html.escape(s['seq_name'])}</td>"
            f"<td>{html.escape(s['category'])}</td>"
            f"<td>{s['final_cr']:.4f}</td>"
            f"<td>{s['steps']}</td>"
            f"<td>{s['episode_reward']:.4f}</td>"
            f"<td>{s['fps_env_steps']:.3f}</td>"
            f"<td>{s['collision']}</td>"
            f"<td>{s['early_stopped']}</td>"
            f"<td><a href='{ep_rel / 'dashboard.png'}'>dashboard</a> | "
            f"<a href='{ep_rel / 'steps.csv'}'>steps.csv</a> | "
            f"<a href='{ep_rel / 'steps'}'>step frames</a></td>"
            "</tr>"
        )
    config = {
        "shapenet_root": args.shapenet_root,
        "preproc_dir": args.preproc_dir,
        "caption_jsonl": args.caption_jsonl,
        "grid_size": args.grid_size,
        "obs_grid_size": args.obs_grid_size,
        "image_size": args.image_size,
        "episode_len": args.episode_len,
        "buffer_size": args.buffer_size,
        "auto_lookat_center": args.auto_lookat_center,
        "renderer_backend": args.renderer_backend,
        "free_raycast_backend": args.free_raycast_backend,
        "free_mask_apply_mode": args.free_mask_apply_mode,
        "coverage_threshold": args.coverage_threshold,
        "coverage_reward_scale": args.coverage_reward_scale,
        "collision_penalty": args.collision_penalty,
    }
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (out_dir / "reports" / "summary.json").write_text(json.dumps(list(summaries), indent=2, sort_keys=True))
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>GeoScout 8-checkpoint smoke validation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #0f172a; background: #f8fafc; }}
h1 {{ margin-bottom: 6px; }}
p {{ color: #475569; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; font-size: 13px; text-align: left; }}
th {{ background: #e2e8f0; }}
code {{ background: #e2e8f0; padding: 2px 4px; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin-top: 20px; }}
.card {{ background: white; padding: 12px; border: 1px solid #e2e8f0; }}
.card img {{ width: 100%; height: auto; display: block; }}
</style></head>
<body>
<h1>GeoScout 8-checkpoint smoke validation</h1>
<p>This smoke run uses <code>TensorBatchEnv</code> with the current fast renderer/free-space configuration. Camera images are displayed with a human-facing +Y-up vertical flip; the renderer tensors remain pixel-aligned with the env.</p>
<p>Config: renderer=<code>{html.escape(args.renderer_backend)}</code>, free=<code>{html.escape(args.free_raycast_backend)}/{html.escape(args.free_mask_apply_mode)}</code>, auto_lookat_center=<code>{args.auto_lookat_center}</code>, grid=<code>{args.grid_size}</code>, obs_grid=<code>{args.obs_grid_size}</code>, image=<code>{args.image_size}</code>.</p>
<table>
<thead><tr><th>Policy</th><th>Sample</th><th>Category</th><th>Final CR</th><th>Steps</th><th>Reward</th><th>Step FPS</th><th>Collision</th><th>Early stop</th><th>Links</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
<h2>Coverage Comparisons</h2>
<div class="grid">
"""
    comp_dir = out_dir / "comparisons"
    for p in sorted(comp_dir.glob("*_coverage_compare.png")):
        html_text += (
            f"<div class='card'><h3>{html.escape(p.stem)}</h3>"
            f"<img src='{p.relative_to(out_dir)}'></div>\n"
        )
    html_text += "</div></body></html>\n"
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapenet_root", default="/data/ShapeNetCore.v2")
    p.add_argument("--preproc_dir", default="/data/geoscout_preproc_g128_attr_v2")
    p.add_argument("--caption_jsonl", default=DEFAULT_CAPTION_JSONL)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seq_names", default=DEFAULT_SEQ_NAMES)
    p.add_argument("--selected_cases_json", default="")
    p.add_argument("--contact_sheet_dir", default="")
    p.add_argument("--discrete_ckpt", default=DEFAULT_DISCRETE_CKPT)
    p.add_argument("--continuous_ckpt", default=DEFAULT_CONTINUOUS_CKPT)
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
    p.add_argument("--auto_lookat_center", action="store_true")
    p.add_argument("--action_modes", default="discrete_det,continuous_det,axis6")
    p.add_argument("--step_frame_mode", choices=["all", "key", "none"], default="key")
    p.add_argument("--max_step_frames", type=int, default=12)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--renderer_backend", choices=["torch", "open3d", "nvdiffrast", "voxel_cuda"], default="voxel_cuda")
    p.add_argument("--free_raycast_backend", choices=["auto", "cuda", "triton", "torch"], default="cuda")
    p.add_argument("--free_mask_apply_mode", choices=["index", "dense", "triton"], default="triton")
    p.add_argument("--triton_bresenham_block_rays", type=int, default=64)
    p.add_argument("--max_faces", type=int, default=5000)
    p.add_argument("--view_radius", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = load_selected_cases(args.selected_cases_json)
    if selected_cases:
        seq_names = []
        seen_seq = set()
        for case in selected_cases:
            seq = str(case["seq_name"])
            if seq not in seen_seq:
                seq_names.append(seq)
                seen_seq.add(seq)
    else:
        seq_names = split_csv(args.seq_names)
    names, mesh_paths, preproc_paths = resolve_pairs(
        shapenet_root=args.shapenet_root,
        preproc_dir=args.preproc_dir,
        seq_names=seq_names,
    )
    captions = load_caption_lookup(args.caption_jsonl)

    policy_specs = []
    for mode in split_csv(args.action_modes):
        if mode == "discrete_det":
            policy_specs.append(PolicySpec("discrete_s1_det", "ppo", "discrete", args.discrete_ckpt, True))
        elif mode == "discrete_stoch":
            policy_specs.append(PolicySpec("discrete_s1_stoch", "ppo", "discrete", args.discrete_ckpt, False))
        elif mode == "continuous_det":
            policy_specs.append(PolicySpec("continuous_s1_det", "ppo", "continuous_tanh", args.continuous_ckpt, True))
        elif mode == "continuous_stoch":
            policy_specs.append(PolicySpec("continuous_s1_stoch", "ppo", "continuous_tanh", args.continuous_ckpt, False))
        elif mode == "axis6":
            policy_specs.append(PolicySpec("axis6", "axis6", "discrete", "", True))
        elif mode == "axis6_continuous":
            policy_specs.append(PolicySpec("axis6_continuous", "axis6", "continuous_tanh", "", True))
        elif mode == "fibonacci":
            policy_specs.append(PolicySpec("fibonacci", "fibonacci", "discrete", "", True))
        elif mode == "fibonacci_continuous":
            policy_specs.append(PolicySpec("fibonacci_continuous", "fibonacci", "continuous_tanh", "", True))
        elif mode == "random":
            policy_specs.append(PolicySpec("random", "random", "discrete", "", True))
        elif mode == "random_continuous":
            policy_specs.append(PolicySpec("random_continuous", "random", "continuous_tanh", "", True))
        elif mode == "ring":
            policy_specs.append(PolicySpec("ring", "ring", "discrete", "", True))
        else:
            raise ValueError(f"unknown action mode: {mode}")

    print(
        f"[smoke] samples={names} policies={[p.name for p in policy_specs]} "
        f"renderer={args.renderer_backend} free={args.free_raycast_backend}/{args.free_mask_apply_mode} "
        f"auto_lookat_center={args.auto_lookat_center}",
        flush=True,
    )

    models: Dict[str, PPO] = {}
    for spec in policy_specs:
        if spec.kind == "ppo" and spec.ckpt not in models:
            print(f"[smoke] loading {spec.ckpt}", flush=True)
            models[spec.ckpt] = PPO.load(spec.ckpt, device=args.device)

    summaries: List[dict] = []
    for policy_i, spec in enumerate(policy_specs):
        model = models.get(spec.ckpt)
        for sample_i, (name, mp, pp) in enumerate(zip(names, mesh_paths, preproc_paths)):
            ep_dir = out_dir / "episodes" / f"{spec.name}__{name}"
            summary_path = ep_dir / "summary.json"
            required_outputs = [
                summary_path,
                ep_dir / "steps.json",
                ep_dir / "numbered_trajectory.png",
                ep_dir / "dashboard.png",
            ]
            if args.resume and all(p.exists() for p in required_outputs):
                try:
                    summary = json.loads(summary_path.read_text())
                    summaries.append(summary)
                    print(
                        f"[smoke] resume skip {spec.name}__{name} "
                        f"steps={summary.get('steps')} cr={float(summary.get('final_cr', 0.0)):.4f}",
                        flush=True,
                    )
                    continue
                except Exception as exc:
                    print(
                        f"[smoke] resume found corrupt summary for {spec.name}__{name}: {exc}; rerunning",
                        flush=True,
                    )
            summaries.append(
                run_episode(
                    args=args,
                    policy=spec,
                    model=model,
                    seq_name=name,
                    mesh_path=mp,
                    preproc_path=pp,
                    caption_row=captions.get(name),
                    out_dir=out_dir,
                    seed=int(args.seed + policy_i * 100 + sample_i),
                )
            )

    make_comparison_plots(out_dir, summaries)
    write_case_atlas(out_dir, summaries, selected_cases, args)
    write_index(out_dir, summaries, args)
    print(f"[smoke] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
