"""Deterministic-rollout viz dumper for ShapeNBV.

Loads a PPO checkpoint, runs N episodes on held-out sequences, and dumps
per-episode visualizations:
  - step_NNN_render.png       — depth / alpha / (zero-filled) RGB strip
  - step_NNN_backproject.ply  — per-step back-projected pcd
  - step_NNN_grid.png         — per-step occupancy tri-class slices
  - cr_curve.png              — coverage vs step
  - trajectory.png            — camera trajectory (with mesh-sampled GT pts overlay)
  - gt.ply                    — full GT pcd (mesh-sampled, 100k pts)
  - predicted.ply             — accumulated agent reconstruction
Plus `summary.txt` for the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from stable_baselines3 import PPO

from shapenbv.data import list_shapenet
from shapenbv.env import ShapeNBVEnv
from shapenbv.mesh_renderer import ShapeNetIndex
from shapenbv.viz import save_pointcloud_ply, save_trajectory_plot, save_coverage_curve


def _load_ply_xyz(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float32)
    except ImportError:
        pass
    pts = []
    with open(path) as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                pts.append([float(p) for p in parts[:3]])
    return np.asarray(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)


def _accumulate_predicted(episode_dir: Path) -> np.ndarray:
    plys = sorted(episode_dir.glob("step_*_backproject.ply"))
    if not plys:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate([_load_ply_xyz(p) for p in plys], axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--shapenet_root", type=str, required=True)
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=12)
    p.add_argument("--seq_names", type=str, default="",
                   help="Comma-separated sequence names. Empty = first "
                        "n_episodes entries from preproc_dir.")
    p.add_argument("--device", type=str, default="cuda")
    # Match training defaults — these MUST mirror what the policy
    # was trained with or obs shapes won't match.
    p.add_argument("--episode_len", type=int, default=100)
    p.add_argument("--buffer_size", type=int, default=100)
    p.add_argument("--image_size", type=int, default=400)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--obs_grid_size", type=int, default=32)
    p.add_argument("--coverage_hit_dilate_radius", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic argmax. Default off — "
                        "stochastic sampling almost always gives better "
                        "MultiCategorical coverage (see object_nbv_zgr "
                        "validate v1 vs v2 in README).")
    p.add_argument("--auto_lookat_center", action="store_true",
                   help="Force the camera to look at the object centre "
                        "every step. MUST match the training-time setting "
                        "(see scripts/train.py --auto_lookat_center).")
    p.add_argument("--caption_dim", type=int, default=0,
                   help="Caption-emb dim (default 0). MUST match the "
                        "training-time --caption_dim or the policy "
                        "obs space won't line up with the ckpt.")
    p.add_argument("--dataset", type=str, default="shapenet",
                   choices=["shapenet", "abo"],
                   help="Source dataset enumerator. Pass `abo` when "
                        "validating an ABO-trained ckpt (uses "
                        "shapenbv.abo.list_abo + Amazon model_id "
                        "directory layout).")
    p.add_argument("--synsets", type=str, default="",
                   help="Comma-separated synset ids to filter the auto-"
                        "selected sequences (when --seq_names is empty). "
                        "ShapeNet name format is `<synset>_<id>` so the "
                        "filter is a prefix match. E.g. for chair/sofa/"
                        "table: `03001627,04256520,04379243`.")
    p.add_argument("--skip_step_dumps", action="store_true",
                   help="Skip per-step .ply / occupancy-slice .png dumps. "
                        "Per-episode dashboards (trajectory_pro / "
                        "coverage_heatmap / dashboard / cr_curve) still "
                        "render. Drops ~360 file writes per 12-ep run "
                        "(~5-10× faster volume I/O).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build seq list.
    if args.seq_names:
        seq_list = [s.strip() for s in args.seq_names.split(",") if s.strip()]
    else:
        all_pt = sorted(Path(args.preproc_dir).glob("*.pt"))
        all_names = [p.stem for p in all_pt]
        if args.synsets:
            wanted = [s.strip() for s in args.synsets.split(",") if s.strip()]
            # Round-robin across requested synsets so we get a balanced
            # mix of categories rather than n_episodes of just the first.
            buckets = {s: [n for n in all_names if n.startswith(s + "_")]
                       for s in wanted}
            seq_list = []
            i = 0
            while len(seq_list) < args.n_episodes:
                progress = False
                for s in wanted:
                    if i < len(buckets[s]) and len(seq_list) < args.n_episodes:
                        seq_list.append(buckets[s][i])
                        progress = True
                if not progress:
                    break
                i += 1
        else:
            seq_list = all_names[: args.n_episodes]
    if not seq_list:
        sys.exit("[validate] No sequences resolved.")
    print(f"[validate] {len(seq_list)} sequences.")

    if args.dataset == "abo":
        from shapenbv.abo import list_abo
        entries = list_abo(Path(args.shapenet_root))
    else:
        entries = list_shapenet(Path(args.shapenet_root))
    name_to_path = {e.name: e.mesh_path for e in entries}
    name_to_path = {n: name_to_path[n] for n in seq_list if n in name_to_path}
    missing = set(seq_list) - set(name_to_path.keys())
    if missing:
        print(f"[validate] WARN: {len(missing)} seqs missing on disk: {sorted(missing)[:5]}...")

    index = ShapeNetIndex(
        entries=name_to_path,
        device=args.device,
        render_size=(args.image_size, args.image_size),
        fov_deg=args.fov_deg,
    )
    # render_size + fov_deg owned by ShapeNetIndex (above), not the env.
    env = ShapeNBVEnv(
        index=index,
        preproc_dir=args.preproc_dir,
        sequence_names=list(name_to_path.keys()),
        device=args.device,
        buffer_size=args.buffer_size,
        episode_len=args.episode_len,
        grid_size=args.grid_size,
        obs_grid_size=args.obs_grid_size,
        coverage_hit_dilate_radius=args.coverage_hit_dilate_radius,
        auto_lookat_center=args.auto_lookat_center,
        caption_dim=args.caption_dim,
        seed=args.seed,
        debug_dir=str(out_dir),
        skip_step_dumps=args.skip_step_dumps,
    )

    print(f"[validate] loading PPO from {args.ckpt}")
    model = PPO.load(args.ckpt, device=args.device)

    rows = []
    for i, seq in enumerate(list(name_to_path.keys())):
        print(f"\n[validate] === episode {i} : {seq} ===")
        obs, _ = env.reset(options={"seq_name": seq})
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

        ep_dir = env._cur_episode_dir
        env._finalize_episode_dump()

        cr = float(env._cr_history[-1]) if env._cr_history else 0.0
        # GT pcd is the dense surface samples saved in preproc.
        gt_pts = env._preproc.get("points_canon")
        if gt_pts is not None:
            gt_pts = (gt_pts.detach().cpu().numpy()
                      if torch.is_tensor(gt_pts) else np.asarray(gt_pts))
            save_pointcloud_ply(gt_pts, ep_dir / "gt.ply")
        # Predicted pcd: prefer in-memory accumulator (when
        # skip_step_dumps), fall back to globbing per-step .ply files.
        if args.skip_step_dumps and getattr(env, "_accumulated_pred", None):
            pred_pts = np.concatenate(env._accumulated_pred, axis=0).astype(np.float32)
        else:
            pred_pts = _accumulate_predicted(ep_dir) if ep_dir else np.empty((0, 3))
        if pred_pts.size > 0:
            save_pointcloud_ply(pred_pts, ep_dir / "predicted.ply")

        # `box_min/box_max` come from env._range_gt (axis-aligned bbox of
        # the action grid) — derive unconditionally so the pred-pcd plot
        # works even when preproc skipped `points_canon` (ABO default).
        rg_t = env._range_gt
        rg = rg_t.detach().cpu().numpy().ravel() if torch.is_tensor(rg_t) else np.asarray(rg_t).ravel()
        box_min = np.array([rg[1], rg[3], rg[5]], dtype=np.float32)
        box_max = np.array([rg[0], rg[2], rg[4]], dtype=np.float32)
        if gt_pts is not None and gt_pts.size > 0:
            save_trajectory_plot(
                eye_positions=env._eye_history, look_ats=env._at_history,
                bbox_min=box_min, bbox_max=box_max,
                path=ep_dir / "trajectory_with_gt.png",
                object_pointcloud=gt_pts,
            )
        if pred_pts.size > 0:
            save_trajectory_plot(
                eye_positions=env._eye_history, look_ats=env._at_history,
                bbox_min=box_min, bbox_max=box_max,
                path=ep_dir / "trajectory_with_pred.png",
                object_pointcloud=pred_pts,
            )
        save_coverage_curve(env._cr_history, ep_dir / "cr_curve.png")

        print(f"[validate] seq={seq}  steps={steps}  cr={cr:.3f}  "
              f"gt={len(gt_pts) if gt_pts is not None else 0}  pred={len(pred_pts)}")
        rows.append({"seq": seq, "steps": steps, "cr": cr,
                     "n_gt": int(len(gt_pts)) if gt_pts is not None else 0,
                     "n_pred": int(len(pred_pts))})

    summary = out_dir / "summary.txt"
    with open(summary, "w") as f:
        f.write(f"{'seq':<50s}  steps    cr      n_gt    n_pred\n")
        for r in rows:
            f.write(f"{r['seq']:<50s}  {r['steps']:>5d}  {r['cr']:.3f}  "
                    f"{r['n_gt']:>6d}  {r['n_pred']:>6d}\n")
        if rows:
            mc = sum(r["cr"] for r in rows) / len(rows)
            ms = sum(r["steps"] for r in rows) / len(rows)
            f.write(f"\nMEAN: cr={mc:.3f}  steps={ms:.1f}  n_eps={len(rows)}\n")
    print(f"\n[validate] wrote {summary}")


if __name__ == "__main__":
    main()
