"""Full-sample visual rollout evaluator for GeoScout.

This is the production replacement for the earlier smoke visualizer.  It is
designed around two constraints:

1. evaluation must cover each requested ShapeNet sample exactly once;
2. visualizations must be readable at 600-sample scale.

The script runs one policy per invocation.  Launching one Modal job per policy
therefore gives true GPU-level parallelism while avoiding one giant sequential
report job.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
from geoscout.viz import _y_up_to_z_up


CONT_EPS = 1e-4
POLICY_COLORS = {
    "discrete_s1_det": "#2563eb",
    "continuous_s1_det": "#16a34a",
    "fibonacci": "#f97316",
    "axis6": "#7c3aed",
    "random": "#64748b",
    "ring": "#0f766e",
}
POLICY_LABELS = {
    "discrete_s1_det": "Discrete PPO",
    "continuous_s1_det": "Continuous PPO",
    "fibonacci": "Fibonacci",
    "axis6": "Axis6",
    "random": "Random",
    "ring": "Ring",
}


def install_numpy_pickle_compat() -> None:
    """Allow SB3 checkpoints pickled with NumPy 2 to load under NumPy 1.x."""
    import sys

    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric
    try:
        import numpy.random._pickle as random_pickle
    except Exception:
        return

    bit_generators = getattr(random_pickle, "BitGenerators", {})
    original_ctor = getattr(random_pickle, "__bit_generator_ctor", None)

    def bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            return bit_generator_name()
        if bit_generator_name in bit_generators:
            return bit_generators[bit_generator_name]()
        if original_ctor is not None:
            return original_ctor(bit_generator_name)
        raise ValueError(f"{bit_generator_name} is not a known BitGenerator module.")

    random_pickle.__bit_generator_ctor = bit_generator_ctor


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def category_from_name(seq_name: str) -> str:
    return SYNSET_TO_CATEGORY.get(seq_name.split("_", 1)[0], seq_name.split("_", 1)[0])


def policy_label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy)


def resolve_pairs(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path]]:
    entries = list_shapenet(
        Path(args.shapenet_root),
        synsets=split_csv(args.synsets) or None,
        categories=split_csv(args.categories) or None,
        limit_per_synset=args.limit_per_synset,
        require_obj=True,
    )
    if args.seq_names:
        wanted = set(split_csv(args.seq_names))
        entries = [e for e in entries if e.name in wanted]
    names: List[str] = []
    meshes: List[Path] = []
    preprocs: List[Path] = []
    preproc_root = Path(args.preproc_dir)
    for e in entries:
        pp = preproc_root / f"{e.name}.pt"
        if not pp.exists():
            continue
        names.append(e.name)
        meshes.append(Path(e.mesh_path))
        preprocs.append(pp)
    if args.max_meshes > 0:
        names = names[: args.max_meshes]
        meshes = meshes[: args.max_meshes]
        preprocs = preprocs[: args.max_meshes]
    if not names:
        raise RuntimeError("No mesh/preprocessed pairs resolved.")
    return names, meshes, preprocs


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
            oid = str(row.get("object_id") or "")
            if oid:
                out[oid] = row
    return out


def caption_text(row: Optional[dict]) -> str:
    if not row:
        return ""
    cap = row.get("caption")
    if isinstance(cap, dict):
        return str(cap.get("embedding_caption") or cap.get("final_caption") or "")
    return str(cap or "")


def caption_attrs(row: Optional[dict]) -> Dict[str, int]:
    if not row:
        return {}
    cap = row.get("caption")
    if not isinstance(cap, dict) or not isinstance(cap.get("attributes"), dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in cap["attributes"].items():
        try:
            out[str(k)] = int(v)
        except Exception:
            pass
    return out


def pose_to_cube_action_idx(position: Sequence[float]) -> np.ndarray:
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
        idx[j] = 0 if unit == 0.0 else int(round((float(pose[j]) - float(DEFAULT_ACTION_LOW_WORLD[j])) / unit))
    return np.clip(idx, DEFAULT_CLIP_POSE_IDX_LOW, DEFAULT_CLIP_POSE_IDX_UP).astype(np.int64)


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


def fibonacci_positions(n: int, radius: float) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n, dtype=np.float32)
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden * i
    return (radius * np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)).astype(np.float32)


def axis_positions(radius: float) -> np.ndarray:
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


def ring_positions(n: int, radius: float, height: float = 0.25) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, num=max(n, 1), endpoint=False, dtype=np.float32)
    return np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, height)],
        axis=1,
    ).astype(np.float32)


def action_table(policy: str, action_space_type: str, episode_len: int, radius: float) -> Optional[np.ndarray]:
    if policy == "fibonacci":
        positions = fibonacci_positions(episode_len, radius)
    elif policy == "axis6":
        positions = axis_positions(radius)
    elif policy == "ring":
        positions = ring_positions(episode_len, radius)
    else:
        return None
    if action_space_type == "continuous_tanh":
        return np.stack([pose6_to_continuous_raw(position_to_pose6_lookat_origin(p)) for p in positions], axis=0)
    return np.stack([pose_to_cube_action_idx(p) for p in positions], axis=0)


def random_actions(rng: np.random.Generator, n_envs: int, action_space_type: str) -> np.ndarray:
    if action_space_type == "continuous_tanh":
        norm = rng.uniform(-1.0 + CONT_EPS, 1.0 - CONT_EPS, size=(n_envs, 5)).astype(np.float32)
        return np.arctanh(norm).astype(np.float32)
    out = np.zeros((n_envs, len(NVEC)), dtype=np.int64)
    for j, n in enumerate(NVEC):
        out[:, j] = rng.integers(0, int(n), size=n_envs, dtype=np.int64)
    return out


def make_env(args: argparse.Namespace, mesh_paths: List[Path], preproc_paths: List[Path], action_space_type: str) -> TensorBatchEnv:
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
        seed=args.seed,
    )


def voxel_centers_from_preproc(preproc_path: Path, max_points: int, seed: int) -> np.ndarray:
    pre = torch.load(preproc_path, map_location="cpu", weights_only=False)
    grid = pre["grid_gt"].detach().cpu().numpy().astype(np.float32)
    range_gt = pre["range_gt"].detach().cpu().numpy().ravel().astype(np.float32)
    voxel = pre["voxel_size_gt"].detach().cpu().numpy().ravel().astype(np.float32)
    idx = np.argwhere(grid > 0.5).astype(np.float32)
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    bbox_min_centres = np.asarray([range_gt[1], range_gt[3], range_gt[5]], dtype=np.float32)
    bbox_min_corner = bbox_min_centres - 0.5 * voxel
    pts = bbox_min_corner[None, :] + (idx + 0.5) * voxel[None, :]
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    return pts.astype(np.float32)


def selected_label_steps(rows: Sequence[dict]) -> set[int]:
    if not rows:
        return set()
    keep = {1, int(rows[-1]["step"])}
    for th in (0.5, 0.8, 0.9, 0.95, 0.99):
        for r in rows:
            if float(r["cr"]) >= th:
                keep.add(int(r["step"]))
                break
    for r in sorted(rows, key=lambda x: float(x.get("coverage_delta", 0.0)), reverse=True)[:4]:
        keep.add(int(r["step"]))
    return keep


def draw_projection(
    ax,
    obj: np.ndarray,
    eyes: np.ndarray,
    ats: np.ndarray,
    dims: Tuple[int, int],
    title: str,
    color: str,
    label_steps: set[int],
) -> None:
    a, b = dims
    if obj.size:
        ax.scatter(obj[:, a], obj[:, b], s=0.25, c="#94a3b8", alpha=0.16, linewidths=0)
    t = np.linspace(0.0, 1.0, max(len(eyes), 1))
    cmap = matplotlib.colormaps["viridis"]
    if len(eyes):
        ax.plot(eyes[:, a], eyes[:, b], color="#334155", linewidth=0.8, alpha=0.35, zorder=2)
    for i, (eye, at) in enumerate(zip(eyes, ats), start=1):
        c = cmap(t[i - 1])
        ax.scatter([eye[a]], [eye[b]], s=18, color=[c], edgecolor="white", linewidth=0.35, zorder=4)
        vec = at - eye
        v2 = np.asarray([vec[a], vec[b]], dtype=np.float32)
        n = float(np.linalg.norm(v2))
        if n > 1e-6:
            v2 = v2 / n * 0.11
            ax.arrow(
                eye[a],
                eye[b],
                v2[0],
                v2[1],
                head_width=0.025,
                head_length=0.035,
                linewidth=0.55,
                color=c,
                alpha=0.65,
                length_includes_head=True,
                zorder=3,
            )
        if i in label_steps:
            ax.text(
                eye[a],
                eye[b],
                str(i),
                fontsize=6.5,
                color="#0f172a",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                zorder=5,
            )
    pts = [obj[:, [a, b]]] if obj.size else []
    if len(eyes):
        pts.append(eyes[:, [a, b]])
        pts.append(ats[:, [a, b]])
    if pts:
        p = np.concatenate(pts, axis=0)
        lo = np.nanmin(p, axis=0)
        hi = np.nanmax(p, axis=0)
        center = 0.5 * (lo + hi)
        half = max(float(np.max(hi - lo)) * 0.56, 0.62)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=7)


def save_policy_trajectory_sheet(
    *,
    rows: Sequence[dict],
    summary: dict,
    preproc_path: Path,
    contact_sheet_dir: str,
    out_path: Path,
    seed: int,
    object_points_zup: Optional[np.ndarray] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    color = POLICY_COLORS.get(str(summary["policy"]), "#2563eb")
    if object_points_zup is None:
        obj_yup = voxel_centers_from_preproc(preproc_path, max_points=45000, seed=seed)
        obj = _y_up_to_z_up(obj_yup)
    else:
        obj = np.asarray(object_points_zup, dtype=np.float32)
    eyes = _y_up_to_z_up(np.asarray([r["eye"] for r in rows], dtype=np.float32))
    ats = _y_up_to_z_up(np.asarray([r["look_at"] for r in rows], dtype=np.float32))
    cr = np.asarray([0.0] + [float(r["cr"]) for r in rows], dtype=np.float32)
    label_steps = selected_label_steps(rows)

    plt.rcParams.update({
        "font.size": 8.5,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
    })
    fig = plt.figure(figsize=(14.5, 8.3), dpi=170)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.05], height_ratios=[0.95, 1.05], wspace=0.28, hspace=0.32)

    ax_xy = fig.add_subplot(gs[0, 0])
    draw_projection(ax_xy, obj, eyes, ats, (0, 1), "top projection: X-Y", color, label_steps)
    ax_xz = fig.add_subplot(gs[0, 1])
    draw_projection(ax_xz, obj, eyes, ats, (0, 2), "front projection: X-Z", color, label_steps)
    ax_yz = fig.add_subplot(gs[0, 2])
    draw_projection(ax_yz, obj, eyes, ats, (1, 2), "side projection: Y-Z", color, label_steps)

    ax_curve = fig.add_subplot(gs[0, 3])
    ax_curve.plot(np.arange(len(cr)), cr, color=color, marker="o", markersize=2.5, linewidth=1.5)
    ax_curve.fill_between(np.arange(len(cr)), 0, cr, color=color, alpha=0.12)
    for th, c in [(0.8, "#94a3b8"), (0.9, "#64748b"), (0.95, "#f97316"), (0.99, "#ef4444")]:
        ax_curve.axhline(th, color=c, linestyle="--", linewidth=0.8, alpha=0.75)
    ax_curve.set_ylim(0.0, 1.02)
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("coverage ratio")
    ax_curve.grid(alpha=0.22)
    ax_curve.set_title("coverage curve", loc="left", fontsize=9.5, fontweight="bold")

    ax_meta = fig.add_subplot(gs[1, 0])
    ax_meta.axis("off")
    sheet_path = Path(contact_sheet_dir) / f"{summary['seq_name']}.png" if contact_sheet_dir else None
    if sheet_path and sheet_path.exists():
        img = Image.open(sheet_path).convert("RGB")
        ax_meta.imshow(img)
        ax_meta.set_title("caption contact sheet", loc="left", fontsize=9.5, fontweight="bold")
    else:
        text = (
            "contact sheet unavailable\n\n"
            f"{summary['category']}\n{summary['seq_name']}\n\n"
            f"{summary.get('caption', '')[:420]}"
        )
        ax_meta.text(0.02, 0.98, text, va="top", ha="left", fontsize=7.4, color="#334155", wrap=True)
        ax_meta.set_title("sample metadata", loc="left", fontsize=9.5, fontweight="bold")

    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    metrics = [
        ["final CR", f"{float(summary['final_cr']):.4f}"],
        ["steps", f"{int(summary['steps'])}"],
        ["reward", f"{float(summary['episode_reward']):.3f}"],
        ["success@0.99", "yes" if float(summary["final_cr"]) > 0.99 else "no"],
        ["collision", "yes" if summary.get("collision") else "no"],
        ["timeout", "yes" if summary.get("timeout") else "no"],
        ["mean delta", f"{float(np.mean([r['coverage_delta'] for r in rows])):.4f}" if rows else "0"],
        ["max delta", f"{float(np.max([r['coverage_delta'] for r in rows])):.4f}" if rows else "0"],
    ]
    table = ax_table.table(cellText=metrics, colLabels=["metric", "value"], loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif c == 1:
            cell.set_text_props(weight="bold")
    ax_table.set_title("terminal metrics", loc="left", fontsize=9.5, fontweight="bold")

    ax_step = fig.add_subplot(gs[1, 2:])
    deltas = np.asarray([float(r["coverage_delta"]) for r in rows], dtype=np.float32)
    x = np.arange(1, len(deltas) + 1)
    ax_step.bar(x, deltas, color=color, alpha=0.75, width=0.82)
    ax_step.set_xlabel("step")
    ax_step.set_ylabel("delta CR")
    ax_step.grid(axis="y", alpha=0.22)
    ax_step.set_title("per-step information gain", loc="left", fontsize=9.5, fontweight="bold")
    if len(deltas):
        for i in np.argsort(-deltas)[: min(5, len(deltas))]:
            ax_step.text(int(i) + 1, float(deltas[i]), str(int(i) + 1), ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"{policy_label(str(summary['policy']))} | {summary['category']} | {summary['seq_name']} | final CR={float(summary['final_cr']):.4f}",
        x=0.012,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.012,
        0.955,
        "Every dot is an executed camera pose; arrows show look direction. Labels mark start/end, CR-threshold crossings, and highest-gain views.",
        ha="left",
        va="top",
        fontsize=8.4,
        color="#475569",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_steps_csv(rows: Sequence[dict], path: Path) -> None:
    fieldnames = [
        "policy",
        "seq_name",
        "category",
        "step",
        "cr",
        "coverage_delta",
        "reward",
        "new_gt_voxels",
        "visible_gt_voxels",
        "collision",
        "done",
        "early_stopped",
        "timeout",
        "action",
        "pose6",
        "eye",
        "look_at",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            for k in ("action", "pose6", "eye", "look_at"):
                out[k] = json.dumps(out[k])
            writer.writerow(out)


def summarize_episode(policy: str, seq_name: str, rows: Sequence[dict], elapsed_s: float, caption_row: Optional[dict]) -> dict:
    final = rows[-1] if rows else {}
    return {
        "policy": policy,
        "seq_name": seq_name,
        "category": category_from_name(seq_name),
        "caption": caption_text(caption_row),
        "attributes": caption_attrs(caption_row),
        "final_cr": float(final.get("cr", 0.0)),
        "steps": int(len(rows)),
        "episode_reward": float(sum(float(r.get("reward", 0.0)) for r in rows)),
        "episode_new_gt_voxels": float(sum(float(r.get("new_gt_voxels", 0.0)) for r in rows)),
        "episode_visible_gt_voxels": float(sum(float(r.get("visible_gt_voxels", 0.0)) for r in rows)),
        "collision": bool(any(bool(r.get("collision", False)) for r in rows)),
        "early_stopped": bool(final.get("early_stopped", False)),
        "timeout": bool(final.get("timeout", False)),
        "elapsed_s": float(elapsed_s),
        "cr_history": [0.0] + [float(r.get("cr", 0.0)) for r in rows],
    }


def policy_action_space(policy: str) -> str:
    return "continuous_tanh" if policy.startswith("continuous") else "discrete"


def policy_checkpoint(policy: str, args: argparse.Namespace) -> str:
    if policy.startswith("discrete"):
        return args.discrete_ckpt
    if policy.startswith("continuous"):
        return args.continuous_ckpt
    return ""


@torch.no_grad()
def evaluate_policy(args: argparse.Namespace) -> None:
    names, mesh_paths, preproc_paths = resolve_pairs(args)
    captions = load_caption_lookup(args.caption_jsonl)
    policy = args.policy
    action_space_type = policy_action_space(policy)
    out_dir = Path(args.out_dir)
    policy_dir = out_dir / "policies" / policy
    episodes_dir = policy_dir / "episodes"
    reports_dir = out_dir / "reports"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[dict] = []
    all_rows_flat: List[dict] = []
    remaining_names: List[str] = []
    remaining_mesh_paths: List[Path] = []
    remaining_preproc_paths: List[Path] = []
    remaining_global_indices: List[int] = []
    reused = 0
    for sample_i, (name, mp, pp) in enumerate(zip(names, mesh_paths, preproc_paths)):
        ep_dir = episodes_dir / name
        summary_path = ep_dir / "summary.json"
        steps_path = ep_dir / "steps.json"
        if args.resume and summary_path.exists() and steps_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                steps = json.loads(steps_path.read_text())
                all_summaries.append(summary)
                all_rows_flat.extend(steps)
                reused += 1
                continue
            except Exception as exc:
                print(f"[visual-eval:{policy}] ignoring corrupt resume data for {name}: {exc}", flush=True)
        remaining_names.append(name)
        remaining_mesh_paths.append(mp)
        remaining_preproc_paths.append(pp)
        remaining_global_indices.append(sample_i)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"[visual-eval:{policy}] samples={len(names)} resume_reused={reused} "
        f"remaining={len(remaining_names)} n_envs={args.n_envs} "
        f"action_space={action_space_type} renderer={args.renderer_backend} "
        f"free={args.free_raycast_backend}/{args.free_mask_apply_mode}",
        flush=True,
    )
    if not remaining_names:
        summary_stats = summarize_policy(policy, all_summaries, 0.0, 0, args.n_envs)
        write_policy_outputs(policy_dir, all_summaries, all_rows_flat, summary_stats)
        print(f"[visual-eval:{policy}] all {len(all_summaries)} episodes already complete; wrote summary", flush=True)
        return

    env = make_env(args, remaining_mesh_paths, remaining_preproc_paths, action_space_type)

    model = None
    ckpt = policy_checkpoint(policy, args)
    if ckpt:
        print(f"[visual-eval:{policy}] loading {ckpt}", flush=True)
        install_numpy_pickle_compat()
        model = PPO.load(
            ckpt,
            device=args.device,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )
        if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
            raise RuntimeError(
                f"observation shape mismatch: model={model.observation_space.shape} env={env.observation_space.shape}"
            )
    table = action_table(policy, action_space_type, args.episode_len, args.view_radius)

    started_all = time.perf_counter()
    total_step_calls = 0

    for batch_start in range(0, len(remaining_names), args.n_envs):
        batch_ids = list(range(batch_start, min(batch_start + args.n_envs, len(remaining_names))))
        if len(batch_ids) < args.n_envs:
            batch_mesh_ids = batch_ids + [batch_ids[0]] * (args.n_envs - len(batch_ids))
        else:
            batch_mesh_ids = batch_ids
        obs = env.reset_to_mesh_ids(batch_mesh_ids)
        active = np.zeros(args.n_envs, dtype=bool)
        active[: len(batch_ids)] = True
        rows_by_env: Dict[int, List[dict]] = {i: [] for i in range(len(batch_ids))}
        phase = np.zeros(args.n_envs, dtype=np.int64)
        batch_started = time.perf_counter()

        for step_call in range(1, args.episode_len + 1):
            if model is not None:
                actions, _ = model.predict(obs, deterministic=True)
                dtype = np.float32 if action_space_type == "continuous_tanh" else np.int64
                actions = np.asarray(actions, dtype=dtype)
            elif policy == "random":
                actions = random_actions(rng, args.n_envs, action_space_type)
            elif table is not None:
                actions = table[phase % len(table)]
            else:
                raise ValueError(f"unsupported policy {policy}")

            action_t = torch.as_tensor(
                actions,
                dtype=torch.float32 if action_space_type == "continuous_tanh" else torch.long,
                device=env.device,
            )
            pose6_t, eyes_t, ats_t = env._decode_actions(action_t)
            pose6 = pose6_t.detach().cpu().numpy().astype(np.float32)
            eyes = eyes_t.detach().cpu().numpy().astype(np.float32)
            ats = ats_t.detach().cpu().numpy().astype(np.float32)

            obs, rewards, dones, infos = env.step(actions)
            total_step_calls += 1
            phase += 1

            for local_i, mesh_id in enumerate(batch_ids):
                if not active[local_i]:
                    continue
                info = infos[local_i]
                row = {
                    "policy": policy,
                    "seq_name": remaining_names[mesh_id],
                    "category": category_from_name(remaining_names[mesh_id]),
                    "step": int(len(rows_by_env[local_i]) + 1),
                    "cr": float(info.get("cr", 0.0)),
                    "coverage_delta": float(info.get("coverage_delta", 0.0)),
                    "reward": float(rewards[local_i]),
                    "new_gt_voxels": float(info.get("new_gt_voxels", 0.0)),
                    "visible_gt_voxels": float(info.get("visible_gt_voxels", 0.0)),
                    "collision": bool(info.get("collision", False)),
                    "done": bool(dones[local_i]),
                    "early_stopped": bool(info.get("early_stopped", False)),
                    "timeout": bool(info.get("TimeLimit.truncated", False)),
                    "action": np.asarray(actions[local_i]).reshape(-1).tolist(),
                    "pose6": pose6[local_i].reshape(-1).tolist(),
                    "eye": eyes[local_i].reshape(-1).tolist(),
                    "look_at": ats[local_i].reshape(-1).tolist(),
                }
                rows_by_env[local_i].append(row)
                if bool(dones[local_i]):
                    active[local_i] = False

            if not active.any():
                break

        for local_i, mesh_id in enumerate(batch_ids):
            seq = remaining_names[mesh_id]
            rows = rows_by_env[local_i]
            ep_dir = episodes_dir / seq
            ep_dir.mkdir(parents=True, exist_ok=True)
            elapsed = time.perf_counter() - batch_started
            summary = summarize_episode(policy, seq, rows, elapsed, captions.get(seq))
            summary.update({
                "mesh_path": str(remaining_mesh_paths[mesh_id]),
                "preproc_path": str(remaining_preproc_paths[mesh_id]),
                "sample_index": int(remaining_global_indices[mesh_id]),
            })
            (ep_dir / "steps.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
            (ep_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
            write_steps_csv(rows, ep_dir / "steps.csv")
            if args.write_trajectory_sheets:
                save_policy_trajectory_sheet(
                    rows=rows,
                    summary=summary,
                    preproc_path=preproc_paths[mesh_id],
                    contact_sheet_dir=args.contact_sheet_dir,
                    out_path=ep_dir / "trajectory_sheet.png",
                    seed=args.seed + mesh_id,
                )
            all_summaries.append(summary)
            all_rows_flat.extend(rows)

        done_count = reused + min(batch_start + len(batch_ids), len(remaining_names))
        cr_mean = float(np.mean([s["final_cr"] for s in all_summaries])) if all_summaries else 0.0
        print(
            f"[visual-eval:{policy}] batch {batch_start // args.n_envs + 1} "
            f"done={done_count}/{len(names)} cr_mean_so_far={cr_mean:.4f} "
            f"step_calls={total_step_calls}",
            flush=True,
        )

    elapsed_all = time.perf_counter() - started_all
    env.close()
    summary_stats = summarize_policy(policy, all_summaries, elapsed_all, total_step_calls, args.n_envs)
    write_policy_outputs(policy_dir, all_summaries, all_rows_flat, summary_stats)
    print(
        f"[visual-eval:{policy}] DONE n={len(all_summaries)} "
        f"cr_mean={summary_stats['cr_mean']:.4f} success={summary_stats['success_rate']:.3f} "
        f"steps={summary_stats['steps_mean']:.2f} fps={summary_stats['env_steps_per_sec']:.1f}",
        flush=True,
    )


def summarize_policy(policy: str, summaries: Sequence[dict], elapsed_s: float, step_calls: int, n_envs: int) -> dict:
    cr = np.asarray([float(s["final_cr"]) for s in summaries], dtype=np.float32)
    steps = np.asarray([int(s["steps"]) for s in summaries], dtype=np.float32)
    rewards = np.asarray([float(s["episode_reward"]) for s in summaries], dtype=np.float32)
    return {
        "policy": policy,
        "n_episodes": int(len(summaries)),
        "elapsed_s": float(elapsed_s),
        "env_step_calls": int(step_calls),
        "env_steps_per_sec": float(step_calls * n_envs / max(elapsed_s, 1e-9)),
        "episodes_per_sec": float(len(summaries) / max(elapsed_s, 1e-9)),
        "cr_mean": float(cr.mean()) if len(cr) else 0.0,
        "cr_std": float(cr.std()) if len(cr) else 0.0,
        "cr_min": float(cr.min()) if len(cr) else 0.0,
        "cr_p10": float(np.percentile(cr, 10)) if len(cr) else 0.0,
        "cr_p50": float(np.percentile(cr, 50)) if len(cr) else 0.0,
        "cr_p90": float(np.percentile(cr, 90)) if len(cr) else 0.0,
        "cr_max": float(cr.max()) if len(cr) else 0.0,
        "success_rate": float((cr > 0.99).mean()) if len(cr) else 0.0,
        "steps_mean": float(steps.mean()) if len(steps) else 0.0,
        "steps_p50": float(np.percentile(steps, 50)) if len(steps) else 0.0,
        "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
        "collision_rate": float(np.mean([bool(s["collision"]) for s in summaries])) if summaries else 0.0,
        "timeout_rate": float(np.mean([bool(s["timeout"]) for s in summaries])) if summaries else 0.0,
    }


def write_policy_outputs(policy_dir: Path, summaries: Sequence[dict], step_rows: Sequence[dict], stats: dict) -> None:
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    (policy_dir / "episodes.json").write_text(json.dumps(list(summaries), indent=2, sort_keys=True))
    fields = [
        "policy",
        "sample_index",
        "seq_name",
        "category",
        "final_cr",
        "steps",
        "episode_reward",
        "episode_new_gt_voxels",
        "episode_visible_gt_voxels",
        "collision",
        "early_stopped",
        "timeout",
        "caption",
    ]
    with (policy_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fields})
    with (policy_dir / "all_steps.csv").open("w", newline="") as f:
        fields2 = [
            "policy",
            "seq_name",
            "category",
            "step",
            "cr",
            "coverage_delta",
            "reward",
            "new_gt_voxels",
            "visible_gt_voxels",
            "collision",
            "done",
            "early_stopped",
            "timeout",
        ]
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        for r in step_rows:
            writer.writerow({k: r.get(k, "") for k in fields2})
    with (policy_dir / "summary.txt").open("w") as f:
        f.write(
            f"policy={stats['policy']} n={stats['n_episodes']}\n"
            f"cr_mean={stats['cr_mean']:.4f} p10={stats['cr_p10']:.4f} "
            f"p50={stats['cr_p50']:.4f} p90={stats['cr_p90']:.4f} min={stats['cr_min']:.4f}\n"
            f"success={stats['success_rate']:.3f} steps_mean={stats['steps_mean']:.2f} "
            f"collision={stats['collision_rate']:.3f} timeout={stats['timeout_rate']:.3f}\n"
            f"reward_mean={stats['reward_mean']:.3f} fps={stats['env_steps_per_sec']:.1f} "
            f"elapsed_s={stats['elapsed_s']:.1f}\n"
        )


def build_report(args: argparse.Namespace) -> None:
    root = Path(args.out_dir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    policy_dirs = sorted((root / "policies").glob("*"))
    all_eps: List[dict] = []
    stats: List[dict] = []
    for pd in policy_dirs:
        ep_path = pd / "episodes.json"
        st_path = pd / "summary.json"
        if ep_path.exists():
            all_eps.extend(json.loads(ep_path.read_text()))
        if st_path.exists():
            stats.append(json.loads(st_path.read_text()))
    if not all_eps:
        raise RuntimeError(f"No policy episodes found under {root / 'policies'}")
    ensure_trajectory_sheets(root, all_eps, args.contact_sheet_dir, args.seed, workers=args.report_workers)
    write_global_plots(root, all_eps, stats)
    write_case_cards(root, all_eps, args.contact_sheet_dir, workers=args.report_workers)
    write_html_atlas(root, all_eps, stats)


def _trajectory_sheet_worker(payload: dict) -> Tuple[str, str, str]:
    """Worker entry point for CPU report generation.

    The GPU rollout stage writes small JSON files.  Each sheet can therefore be
    regenerated independently, which makes the post-process embarrassingly
    parallel and safe to resume.
    """
    summary = payload["summary"]
    ep_dir = Path(payload["ep_dir"])
    out_path = ep_dir / "trajectory_sheet.png"
    policy = str(summary["policy"])
    seq = str(summary["seq_name"])
    if out_path.exists() and not payload.get("force", False):
        return policy, seq, "cached"
    steps_path = ep_dir / "steps.json"
    if not steps_path.exists():
        return policy, seq, "missing_steps"
    rows = json.loads(steps_path.read_text())
    preproc_path = Path(str(summary.get("preproc_path") or ""))
    object_points_zup = _y_up_to_z_up(
        voxel_centers_from_preproc(
            preproc_path,
            max_points=45000,
            seed=int(payload["seed"]),
        )
    )
    save_policy_trajectory_sheet(
        rows=rows,
        summary=summary,
        preproc_path=preproc_path,
        contact_sheet_dir=str(payload["contact_sheet_dir"]),
        out_path=out_path,
        seed=int(payload["seed"]),
        object_points_zup=object_points_zup,
    )
    return policy, seq, "written"


def ensure_trajectory_sheets(
    root: Path,
    episodes: Sequence[dict],
    contact_sheet_dir: str,
    seed: int,
    workers: int,
) -> None:
    """Generate missing per-policy trajectory sheets as a CPU post-process."""
    tasks = []
    total = len(episodes)
    for i, summary in enumerate(episodes, start=1):
        policy = str(summary["policy"])
        seq = str(summary["seq_name"])
        ep_dir = root / "policies" / policy / "episodes" / seq
        out_path = ep_dir / "trajectory_sheet.png"
        if out_path.exists():
            continue
        tasks.append(
            {
                "summary": summary,
                "ep_dir": str(ep_dir),
                "contact_sheet_dir": contact_sheet_dir,
                "seed": seed + int(summary.get("sample_index", i)),
            }
        )
    cached = total - len(tasks)
    if not tasks:
        print(f"[visual-report] trajectory sheets {total}/{total} cached", flush=True)
        return
    if workers <= 0:
        workers = max(1, min(8, os.cpu_count() or 1))
    workers = max(1, min(int(workers), len(tasks)))
    print(
        f"[visual-report] trajectory sheets cached={cached} missing={len(tasks)} total={total} workers={workers}",
        flush=True,
    )
    written = 0
    skipped = 0
    if workers == 1:
        for task in tasks:
            _, seq, status = _trajectory_sheet_worker(task)
            if status == "written":
                written += 1
            else:
                skipped += 1
                if status != "cached":
                    print(f"[visual-report] {status}: {seq}", flush=True)
            done = cached + written + skipped
            if done % 50 == 0 or done == total:
                print(f"[visual-report] trajectory sheets {done}/{total}", flush=True)
        return

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_trajectory_sheet_worker, task) for task in tasks]
        for fut in as_completed(futures):
            _, seq, status = fut.result()
            if status == "written":
                written += 1
            else:
                skipped += 1
                if status != "cached":
                    print(f"[visual-report] {status}: {seq}", flush=True)
            done = cached + written + skipped
            if done % 50 == 0 or done == total:
                print(f"[visual-report] trajectory sheets {done}/{total}", flush=True)


def write_global_plots(root: Path, episodes: Sequence[dict], stats: Sequence[dict]) -> None:
    reports = root / "reports"
    policies = [str(s["policy"]) for s in stats]
    labels = [policy_label(p) for p in policies]
    colors = [POLICY_COLORS.get(p, "#64748b") for p in policies]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=170)
    cr_data = [[float(e["final_cr"]) for e in episodes if e["policy"] == p] for p in policies]
    step_data = [[float(e["steps"]) for e in episodes if e["policy"] == p] for p in policies]
    bp = axes[0, 0].boxplot(cr_data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.82)
    axes[0, 0].axhline(0.99, color="#ef4444", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_title("final CR distribution", loc="left", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].grid(axis="y", alpha=0.22)

    bp = axes[0, 1].boxplot(step_data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.82)
    axes[0, 1].set_title("episode length distribution", loc="left", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].grid(axis="y", alpha=0.22)

    x = np.arange(len(stats))
    axes[0, 2].bar(x, [float(s["success_rate"]) for s in stats], color=colors)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 2].set_ylim(0, 1.02)
    axes[0, 2].set_title("success rate (CR > 0.99)", loc="left", fontweight="bold")
    axes[0, 2].grid(axis="y", alpha=0.22)

    for p, label, c, data in zip(policies, labels, colors, cr_data):
        arr = np.sort(np.asarray(data, dtype=np.float32))
        y = np.arange(1, len(arr) + 1) / max(len(arr), 1)
        axes[1, 0].plot(arr, y, color=c, linewidth=1.8, label=label)
    axes[1, 0].axvline(0.99, color="#ef4444", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("final CR")
    axes[1, 0].set_ylabel("ECDF")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].set_title("CR empirical CDF", loc="left", fontweight="bold")

    categories = sorted({str(e["category"]) for e in episodes})
    heat = np.zeros((len(policies), len(categories)), dtype=np.float32)
    for i, p in enumerate(policies):
        for j, cat in enumerate(categories):
            vals = [float(e["final_cr"]) for e in episodes if e["policy"] == p and e["category"] == cat]
            heat[i, j] = float(np.mean(vals)) if vals else np.nan
    im = axes[1, 1].imshow(heat, vmin=0.85, vmax=1.0, aspect="auto", cmap="viridis")
    axes[1, 1].set_yticks(np.arange(len(policies)))
    axes[1, 1].set_yticklabels(labels)
    axes[1, 1].set_xticks(np.arange(len(categories)))
    axes[1, 1].set_xticklabels(categories, rotation=25, ha="right")
    axes[1, 1].set_title("mean CR by category", loc="left", fontweight="bold")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.75)

    best_by_seq: Dict[str, dict] = {}
    for e in episodes:
        seq = str(e["seq_name"])
        score = (float(e["final_cr"]), -int(e["steps"]))
        if seq not in best_by_seq or score > (
            float(best_by_seq[seq]["final_cr"]),
            -int(best_by_seq[seq]["steps"]),
        ):
            best_by_seq[seq] = e
    win_counts = {p: 0 for p in policies}
    for e in best_by_seq.values():
        win_counts[str(e["policy"])] += 1
    axes[1, 2].bar(x, [win_counts[p] for p in policies], color=colors)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels, rotation=25, ha="right")
    axes[1, 2].set_title("case wins: higher CR, then fewer steps", loc="left", fontweight="bold")
    axes[1, 2].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(reports / "global_policy_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def _case_card_worker(payload: dict) -> Tuple[str, str]:
    root = Path(payload["root"])
    seq = str(payload["seq"])
    rows = sorted(payload["rows"], key=lambda r: str(r["policy"]))
    contact_sheet_dir = str(payload.get("contact_sheet_dir") or "")
    out = root / "cases" / seq / "case_summary.png"
    if out.exists() and not payload.get("force", False):
        return seq, "cached"

    fig = plt.figure(figsize=(13.5, 6.6), dpi=170)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.28)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.axis("off")
    sheet = Path(contact_sheet_dir) / f"{seq}.png" if contact_sheet_dir else None
    if sheet and sheet.exists():
        ax_img.imshow(Image.open(sheet).convert("RGB"))
        ax_img.set_title("object views", loc="left", fontweight="bold")
    else:
        ax_img.text(0.5, 0.5, "contact sheet unavailable", ha="center", va="center", color="#64748b")
        ax_img.set_title("object views", loc="left", fontweight="bold")

    ax_curve = fig.add_subplot(gs[0, 1])
    for r in rows:
        cr = np.asarray(r.get("cr_history", []), dtype=np.float32)
        p = str(r["policy"])
        ax_curve.plot(
            np.arange(len(cr)),
            cr,
            color=POLICY_COLORS.get(p, "#64748b"),
            linewidth=1.6,
            marker="o",
            markersize=2.2,
            label=f"{policy_label(p)} {float(r['final_cr']):.3f}/{int(r['steps'])}",
        )
    for th, c in [(0.8, "#94a3b8"), (0.9, "#64748b"), (0.95, "#f97316"), (0.99, "#ef4444")]:
        ax_curve.axhline(th, color=c, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("coverage ratio")
    ax_curve.grid(alpha=0.22)
    ax_curve.legend(fontsize=7, loc="lower right")
    ax_curve.set_title("coverage trajectories", loc="left", fontweight="bold")

    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")
    table_rows = [
        [
            policy_label(str(r["policy"])),
            f"{float(r['final_cr']):.4f}",
            str(int(r["steps"])),
            "Y" if float(r["final_cr"]) > 0.99 else "N",
            "Y" if r.get("collision") else "N",
        ]
        for r in rows
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["policy", "CR", "steps", "succ", "coll"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.4)
    for (rr, cc), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if rr == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif cc == 1:
            val = float(table_rows[rr - 1][1])
            cell.set_facecolor("#16a34a" if val > 0.99 else "#f97316" if val > 0.9 else "#ef4444")
            cell.set_text_props(color="white", weight="bold")
    ax_table.set_title("terminal metrics", loc="left", fontweight="bold")
    cat = rows[0].get("category", "")
    fig.suptitle(f"{cat} | {seq}", x=0.015, y=0.98, ha="left", fontsize=12.5, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return seq, "written"


def write_case_cards(root: Path, episodes: Sequence[dict], contact_sheet_dir: str, workers: int) -> None:
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    by_seq: Dict[str, List[dict]] = {}
    for e in episodes:
        by_seq.setdefault(str(e["seq_name"]), []).append(e)
    tasks = [
        {"root": str(root), "seq": seq, "rows": rows, "contact_sheet_dir": contact_sheet_dir}
        for seq, rows in by_seq.items()
    ]
    total = len(tasks)
    if workers <= 0:
        workers = max(1, min(8, os.cpu_count() or 1))
    workers = max(1, min(int(workers), total))
    print(f"[visual-report] case summaries total={total} workers={workers}", flush=True)
    done = 0
    if workers == 1:
        for task in tasks:
            _case_card_worker(task)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[visual-report] case summaries {done}/{total}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_case_card_worker, task) for task in tasks]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[visual-report] case summaries {done}/{total}", flush=True)


def write_html_atlas(root: Path, episodes: Sequence[dict], stats: Sequence[dict]) -> None:
    by_seq: Dict[str, List[dict]] = {}
    for e in episodes:
        by_seq.setdefault(str(e["seq_name"]), []).append(e)
    rows_html = []
    def case_sort_key(item: Tuple[str, List[dict]]) -> Tuple[float, int, str]:
        seq, eps = item
        best = max(eps, key=lambda e: (float(e["final_cr"]), -int(e["steps"])))
        # Hard cases first: lower best CR, then longer best episode.
        return (float(best["final_cr"]), -int(best["steps"]), seq)

    for seq, eps in sorted(by_seq.items(), key=case_sort_key):
        eps = sorted(eps, key=lambda e: str(e["policy"]))
        best = max(eps, key=lambda e: (float(e["final_cr"]), -int(e["steps"])))
        links = []
        for e in eps:
            p = str(e["policy"])
            rel = Path("policies") / p / "episodes" / seq / "trajectory_sheet.png"
            links.append(f"<a href='{rel}'>{html.escape(policy_label(p))}</a>")
        cells = "".join(
            f"<td class='metric'>{float(e['final_cr']):.3f}<br><span>{int(e['steps'])} steps</span></td>"
            for e in eps
        )
        rows_html.append(
            f"<tr data-category='{html.escape(str(eps[0].get('category', '')))}'>"
            f"<td><a href='cases/{seq}/case_summary.png'>{html.escape(seq)}</a><br>"
            f"<span>{html.escape(str(eps[0].get('category', '')))}</span></td>"
            f"<td>{html.escape(policy_label(str(best['policy'])))}<br><span>CR={float(best['final_cr']):.3f}, {int(best['steps'])} steps</span></td>"
            f"{cells}"
            f"<td>{' · '.join(links)}</td>"
            "</tr>"
        )
    policy_headers = "".join(f"<th>{html.escape(policy_label(str(s['policy'])))}</th>" for s in sorted(stats, key=lambda x: str(x["policy"])))
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>GeoScout full 600 visual evaluation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #0f172a; background: #f8fafc; }}
h1 {{ margin-bottom: 4px; }}
p {{ color: #475569; }}
img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; background: white; }}
table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 18px; }}
th, td {{ padding: 8px 9px; border-bottom: 1px solid #e2e8f0; font-size: 12.5px; vertical-align: top; }}
th {{ background: #e2e8f0; text-align: left; position: sticky; top: 0; }}
td span {{ color: #64748b; font-size: 11px; }}
td.metric {{ text-align: center; font-variant-numeric: tabular-nums; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.grid {{ display: grid; grid-template-columns: minmax(480px, 1fr); gap: 16px; margin: 18px 0; }}
code {{ background: #e2e8f0; padding: 2px 4px; border-radius: 4px; }}
</style></head>
<body>
<h1>GeoScout full 600 visual evaluation</h1>
<p>Each policy was evaluated in a separate GPU job.  Every sample appears once per policy.  Policy trajectory sheets use three orthographic projections, time-colored camera centers, and look-direction arrows to avoid unreadable 3D clutter.</p>
<div class="grid">
  <div><h2>Global Policy Dashboard</h2><img src="reports/global_policy_dashboard.png"></div>
</div>
<table>
<thead><tr><th>sample</th><th>best policy</th>{policy_headers}<th>trajectory sheets</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</body></html>
"""
    (root / "full_eval_atlas.html").write_text(html_text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["run_policy", "build_report"], default="run_policy")
    p.add_argument("--policy", choices=["discrete_s1_det", "continuous_s1_det", "fibonacci", "axis6", "random", "ring"], default="discrete_s1_det")
    p.add_argument("--shapenet_root", default="/data/ShapeNetCore.v2")
    p.add_argument("--preproc_dir", default="/data/geoscout_preproc_g128_attr_v2")
    p.add_argument("--caption_jsonl", default="/data/geoscout_captions/full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl")
    p.add_argument("--contact_sheet_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--discrete_ckpt", default="/runs/train600-24m-discrete-s1-l40s-n128-wandb-0506/ppo_geoscout.zip")
    p.add_argument("--continuous_ckpt", default="/runs/train600-24m-continuous-s1-l40s-n128-wandb-0506/ppo_geoscout.zip")
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
    p.add_argument("--auto_lookat_center", action="store_true")
    p.add_argument("--max_faces", type=int, default=5000)
    p.add_argument("--renderer_backend", choices=["torch", "open3d", "nvdiffrast", "voxel_cuda"], default="voxel_cuda")
    p.add_argument("--free_raycast_backend", choices=["auto", "cuda", "triton", "torch"], default="cuda")
    p.add_argument("--free_mask_apply_mode", choices=["index", "dense", "triton"], default="triton")
    p.add_argument("--triton_bresenham_block_rays", type=int, default=64)
    p.add_argument("--view_radius", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write_trajectory_sheets", action="store_true", default=False)
    p.add_argument("--no_write_trajectory_sheets", dest="write_trajectory_sheets", action="store_false")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--report_workers", type=int, default=0)
    args = p.parse_args()

    if args.mode == "run_policy":
        evaluate_policy(args)
    else:
        build_report(args)


if __name__ == "__main__":
    main()
