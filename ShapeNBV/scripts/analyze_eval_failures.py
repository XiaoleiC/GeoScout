"""Analyze ShapeNBV full-eval rollout results.

The evaluator writes one ``episodes.json`` per policy plus per-case visual
artifacts.  This script turns those raw rollout records into auditable failure
analysis tables and figures.  It deliberately avoids inferring anything from a
single metric: each ranked case keeps links back to the case summary and policy
trajectory sheets.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


POLICY_ORDER = ["discrete_s1_det", "continuous_s1_det", "fibonacci", "axis6", "random"]
POLICY_LABELS = {
    "discrete_s1_det": "Discrete PPO",
    "continuous_s1_det": "Continuous PPO",
    "fibonacci": "Fibonacci",
    "axis6": "Axis6",
    "random": "Random",
}
POLICY_COLORS = {
    "discrete_s1_det": "#2563eb",
    "continuous_s1_det": "#16a34a",
    "fibonacci": "#f97316",
    "axis6": "#7c3aed",
    "random": "#64748b",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


def success(cr: float, threshold: float = 0.99) -> bool:
    return float(cr) > threshold


def load_episodes(root: Path) -> List[dict]:
    rows: List[dict] = []
    for policy_dir in sorted((root / "policies").glob("*")):
        ep_path = policy_dir / "episodes.json"
        if not ep_path.exists():
            continue
        episodes = read_json(ep_path)
        rows.extend(episodes)
    if not rows:
        raise RuntimeError(f"No episodes.json found under {root / 'policies'}")
    return rows


def better(a: dict, b: dict) -> dict:
    """Return better episode by final CR, then fewer steps."""
    if float(a["final_cr"]) > float(b["final_cr"]) + 1e-8:
        return a
    if float(b["final_cr"]) > float(a["final_cr"]) + 1e-8:
        return b
    return a if int(a["steps"]) <= int(b["steps"]) else b


def policy_link(root: Path, policy: str, seq: str) -> str:
    return f"policies/{policy}/episodes/{seq}/trajectory_sheet.png"


def case_link(seq: str) -> str:
    return f"cases/{seq}/case_summary.png"


def summarize_cases(root: Path, episodes: Sequence[dict], threshold: float) -> List[dict]:
    by_seq: Dict[str, Dict[str, dict]] = {}
    for ep in episodes:
        by_seq.setdefault(str(ep["seq_name"]), {})[str(ep["policy"])] = ep

    cases: List[dict] = []
    for seq, per_policy in sorted(by_seq.items()):
        missing = [p for p in POLICY_ORDER if p not in per_policy]
        if missing:
            raise RuntimeError(f"{seq} missing policies: {missing}")
        discrete = per_policy["discrete_s1_det"]
        continuous = per_policy["continuous_s1_det"]
        fibonacci = per_policy["fibonacci"]
        axis6 = per_policy["axis6"]
        random = per_policy["random"]
        learned_best = better(discrete, continuous)
        heuristic_best = better(fibonacci, axis6)
        overall = learned_best
        for p in ["fibonacci", "axis6", "random"]:
            overall = better(overall, per_policy[p])
        learned_success = success(learned_best["final_cr"], threshold)
        heuristic_success = success(heuristic_best["final_cr"], threshold)
        case = {
            "seq_name": seq,
            "category": discrete.get("category", ""),
            "caption": discrete.get("caption", ""),
            "learned_best_policy": learned_best["policy"],
            "learned_best_cr": float(learned_best["final_cr"]),
            "learned_best_steps": int(learned_best["steps"]),
            "heuristic_best_policy": heuristic_best["policy"],
            "heuristic_best_cr": float(heuristic_best["final_cr"]),
            "heuristic_best_steps": int(heuristic_best["steps"]),
            "overall_best_policy": overall["policy"],
            "overall_best_cr": float(overall["final_cr"]),
            "overall_best_steps": int(overall["steps"]),
            "learned_success": learned_success,
            "heuristic_success": heuristic_success,
            "discrete_cr": float(discrete["final_cr"]),
            "continuous_cr": float(continuous["final_cr"]),
            "fibonacci_cr": float(fibonacci["final_cr"]),
            "axis6_cr": float(axis6["final_cr"]),
            "random_cr": float(random["final_cr"]),
            "discrete_steps": int(discrete["steps"]),
            "continuous_steps": int(continuous["steps"]),
            "fibonacci_steps": int(fibonacci["steps"]),
            "axis6_steps": int(axis6["steps"]),
            "random_steps": int(random["steps"]),
            "discrete_success": success(discrete["final_cr"], threshold),
            "continuous_success": success(continuous["final_cr"], threshold),
            "fibonacci_success": success(fibonacci["final_cr"], threshold),
            "axis6_success": success(axis6["final_cr"], threshold),
            "random_success": success(random["final_cr"], threshold),
            "heuristic_minus_learned_cr": float(heuristic_best["final_cr"]) - float(learned_best["final_cr"]),
            "learned_minus_axis6_cr": float(learned_best["final_cr"]) - float(axis6["final_cr"]),
            "learned_minus_fibonacci_cr": float(learned_best["final_cr"]) - float(fibonacci["final_cr"]),
            "case_summary": case_link(seq),
            "discrete_sheet": policy_link(root, "discrete_s1_det", seq),
            "continuous_sheet": policy_link(root, "continuous_s1_det", seq),
            "fibonacci_sheet": policy_link(root, "fibonacci", seq),
            "axis6_sheet": policy_link(root, "axis6", seq),
            "random_sheet": policy_link(root, "random", seq),
        }
        attrs = discrete.get("attributes") or {}
        for k, v in sorted(attrs.items()):
            case[f"attr_{k}"] = int(v) if isinstance(v, (int, bool)) else v
        cases.append(case)
    return cases


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_policy(rows: Sequence[dict], threshold: float) -> List[dict]:
    out = []
    for policy in POLICY_ORDER:
        eps = [r for r in rows if r["policy"] == policy]
        cr = np.asarray([float(e["final_cr"]) for e in eps], dtype=np.float64)
        steps = np.asarray([int(e["steps"]) for e in eps], dtype=np.float64)
        out.append(
            {
                "policy": policy,
                "n": len(eps),
                "cr_mean": float(cr.mean()),
                "cr_p10": float(np.quantile(cr, 0.10)),
                "cr_p50": float(np.quantile(cr, 0.50)),
                "cr_p90": float(np.quantile(cr, 0.90)),
                "success_rate": float((cr > threshold).mean()),
                "steps_mean": float(steps.mean()),
                "steps_p50": float(np.quantile(steps, 0.50)),
            }
        )
    return out


def aggregate_by_category(cases: Sequence[dict], threshold: float) -> List[dict]:
    cats = sorted({str(c["category"]) for c in cases})
    out = []
    for cat in cats:
        rows = [c for c in cases if c["category"] == cat]
        for policy in POLICY_ORDER:
            cr = np.asarray([float(r[f"{policy_to_key(policy)}_cr"]) for r in rows], dtype=np.float64)
            steps = np.asarray([int(r[f"{policy_to_key(policy)}_steps"]) for r in rows], dtype=np.float64)
            out.append(
                {
                    "category": cat,
                    "policy": policy,
                    "n": len(rows),
                    "cr_mean": float(cr.mean()),
                    "success_rate": float((cr > threshold).mean()),
                    "steps_mean": float(steps.mean()),
                }
            )
    return out


def policy_to_key(policy: str) -> str:
    return {
        "discrete_s1_det": "discrete",
        "continuous_s1_det": "continuous",
        "fibonacci": "fibonacci",
        "axis6": "axis6",
        "random": "random",
    }[policy]


def aggregate_attributes(cases: Sequence[dict], threshold: float) -> List[dict]:
    attr_keys = sorted(k for k in cases[0].keys() if k.startswith("attr_"))
    out = []
    for key in attr_keys:
        pos = [c for c in cases if int(c.get(key, 0)) == 1]
        neg = [c for c in cases if int(c.get(key, 0)) == 0]
        if len(pos) < 12 or len(neg) < 12:
            continue
        for cohort_name, cohort in [("present", pos), ("absent", neg)]:
            learned_cr = np.asarray([float(c["learned_best_cr"]) for c in cohort], dtype=np.float64)
            disc_cr = np.asarray([float(c["discrete_cr"]) for c in cohort], dtype=np.float64)
            cont_cr = np.asarray([float(c["continuous_cr"]) for c in cohort], dtype=np.float64)
            out.append(
                {
                    "attribute": key.removeprefix("attr_"),
                    "cohort": cohort_name,
                    "n": len(cohort),
                    "learned_best_cr_mean": float(learned_cr.mean()),
                    "learned_best_success_rate": float((learned_cr > threshold).mean()),
                    "discrete_success_rate": float((disc_cr > threshold).mean()),
                    "continuous_success_rate": float((cont_cr > threshold).mean()),
                }
            )
    return out


def _read_steps(root: Path, policy: str, seq: str) -> List[dict]:
    path = root / "policies" / policy / "episodes" / seq / "steps.json"
    if not path.exists():
        return []
    return read_json(path)


def _view_dirs(rows: Sequence[dict]) -> np.ndarray:
    dirs = []
    for row in rows:
        eye = np.asarray(row.get("eye", [0, 0, 0]), dtype=np.float64)
        look_at = np.asarray(row.get("look_at", [0, 0, 0]), dtype=np.float64)
        direction = look_at - eye
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            dirs.append(direction / norm)
    return np.asarray(dirs, dtype=np.float64)


def _unique_direction_count(dirs: np.ndarray, deg: float = 15.0) -> int:
    if len(dirs) == 0:
        return 0
    cos_thr = math.cos(math.radians(deg))
    representatives: List[np.ndarray] = []
    for direction in dirs:
        if not representatives:
            representatives.append(direction)
            continue
        if max(float(np.dot(direction, other)) for other in representatives) < cos_thr:
            representatives.append(direction)
    return len(representatives)


def _mean_pair_angle_deg(dirs: np.ndarray) -> float:
    if len(dirs) < 2:
        return 0.0
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    tri = np.triu_indices(len(dirs), 1)
    return float(np.degrees(np.arccos(dots[tri])).mean())


def _step_metrics(root: Path, policy: str, seq: str) -> dict:
    rows = _read_steps(root, policy, seq)
    dirs = _view_dirs(rows)
    deltas = [float(row.get("coverage_delta", 0.0)) for row in rows]
    cr = [float(row.get("cr", 0.0)) for row in rows]
    tail = deltas[max(0, len(deltas) - 10):]
    return {
        "steps": len(rows),
        "unique_dirs_15deg": _unique_direction_count(dirs, 15.0),
        "mean_pair_angle_deg": _mean_pair_angle_deg(dirs),
        "tail10_gain": float(sum(tail)),
        "tail10_small_gain_steps": int(sum(delta < 1e-3 for delta in tail)),
        "cr_at_10": float(cr[min(9, len(cr) - 1)]) if cr else 0.0,
        "final_cr": float(cr[-1]) if cr else 0.0,
    }


def aggregate_step_diagnostics(root: Path, cases: Sequence[dict]) -> List[dict]:
    out = []
    for case in cases:
        seq = str(case["seq_name"])
        learned = _step_metrics(root, str(case["learned_best_policy"]), seq)
        heuristic = _step_metrics(root, str(case["heuristic_best_policy"]), seq)
        out.append(
            {
                "seq_name": seq,
                "category": case["category"],
                "learned_success": case["learned_success"],
                "heuristic_success": case["heuristic_success"],
                "learned_best_policy": case["learned_best_policy"],
                "heuristic_best_policy": case["heuristic_best_policy"],
                "learned_best_cr": case["learned_best_cr"],
                "heuristic_best_cr": case["heuristic_best_cr"],
                "learned_unique_dirs_15deg": learned["unique_dirs_15deg"],
                "heuristic_unique_dirs_15deg": heuristic["unique_dirs_15deg"],
                "learned_mean_pair_angle_deg": learned["mean_pair_angle_deg"],
                "heuristic_mean_pair_angle_deg": heuristic["mean_pair_angle_deg"],
                "learned_tail10_gain": learned["tail10_gain"],
                "heuristic_tail10_gain": heuristic["tail10_gain"],
                "learned_tail10_small_gain_steps": learned["tail10_small_gain_steps"],
                "heuristic_tail10_small_gain_steps": heuristic["tail10_small_gain_steps"],
                "learned_cr_at_10": learned["cr_at_10"],
                "heuristic_cr_at_10": heuristic["cr_at_10"],
            }
        )
    return out


def case_score_for_hardness(case: dict) -> tuple:
    # Low learned CR dominates; for equal CR, worse if heuristic did much better.
    return (float(case["learned_best_cr"]), -float(case["heuristic_minus_learned_cr"]), int(case["learned_best_steps"]))


def write_summary_plot(out_path: Path, policy_stats: Sequence[dict], cat_stats: Sequence[dict], cases: Sequence[dict], threshold: float) -> None:
    labels = [POLICY_LABELS[s["policy"]] for s in policy_stats]
    colors = [POLICY_COLORS[s["policy"]] for s in policy_stats]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=170)
    x = np.arange(len(policy_stats))
    axes[0, 0].bar(x, [s["success_rate"] for s in policy_stats], color=colors)
    axes[0, 0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_title("success rate (CR > 0.99)", loc="left", fontweight="bold")
    axes[0, 0].grid(axis="y", alpha=0.22)

    axes[0, 1].bar(x, [s["steps_mean"] for s in policy_stats], color=colors)
    axes[0, 1].set_xticks(x, labels, rotation=22, ha="right")
    axes[0, 1].set_title("mean episode length", loc="left", fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.22)

    cats = sorted({s["category"] for s in cat_stats})
    learned_success = []
    heuristic_success = []
    for cat in cats:
        rows = [c for c in cases if c["category"] == cat]
        learned_success.append(np.mean([c["learned_success"] for c in rows]))
        heuristic_success.append(np.mean([c["heuristic_success"] for c in rows]))
    w = 0.35
    cx = np.arange(len(cats))
    axes[1, 0].bar(cx - w / 2, learned_success, width=w, color="#2563eb", label="best learned")
    axes[1, 0].bar(cx + w / 2, heuristic_success, width=w, color="#f97316", label="best heuristic")
    axes[1, 0].set_xticks(cx, cats, rotation=15, ha="right")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("success by category", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.22)

    hard = sorted(cases, key=case_score_for_hardness)[:30]
    axes[1, 1].scatter(
        [c["learned_best_cr"] for c in cases],
        [c["heuristic_best_cr"] for c in cases],
        s=14,
        alpha=0.45,
        color="#64748b",
        label="all samples",
    )
    axes[1, 1].scatter(
        [c["learned_best_cr"] for c in hard],
        [c["heuristic_best_cr"] for c in hard],
        s=28,
        color="#ef4444",
        label="30 hardest learned cases",
    )
    axes[1, 1].axvline(threshold, color="#2563eb", linestyle="--", linewidth=0.9)
    axes[1, 1].axhline(threshold, color="#f97316", linestyle="--", linewidth=0.9)
    axes[1, 1].plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", linewidth=1.0)
    axes[1, 1].set_xlim(0.75, 1.005)
    axes[1, 1].set_ylim(0.75, 1.005)
    axes[1, 1].set_xlabel("best learned final CR")
    axes[1, 1].set_ylabel("best heuristic final CR")
    axes[1, 1].legend(fontsize=8, loc="lower right")
    axes[1, 1].set_title("where learned policy underperforms", loc="left", fontweight="bold")
    axes[1, 1].grid(alpha=0.22)

    fig.suptitle("ShapeNBV 600-sample eval: evidence summary", x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fit_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    tw, th = size
    scale = max(tw / max(w, 1), th / max(h, 1))
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return img.crop((left, top, left + tw, top + th))


def write_hard_gallery(root: Path, out_path: Path, hard_cases: Sequence[dict], n: int = 24) -> None:
    tile_w, tile_h = 410, 310
    cols = 4
    rows = int(math.ceil(min(n, len(hard_cases)) / cols))
    pad = 18
    header = 95
    img = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * tile_h + (rows + 1) * pad + header), "#f8fafc")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("Arial Bold.ttf", 28)
        text_font = ImageFont.truetype("Arial.ttf", 15)
        small_font = ImageFont.truetype("Arial.ttf", 13)
    except Exception:
        title_font = text_font = small_font = ImageFont.load_default()
    draw.text((pad, 22), "Hard learned-policy cases: object view + verifiable metrics", fill="#0f172a", font=title_font)
    draw.text(
        (pad, 60),
        "Sorted by low best-learned CR, with heuristic comparison shown for each sample.",
        fill="#475569",
        font=text_font,
    )
    for idx, case in enumerate(hard_cases[:n]):
        r, c = divmod(idx, cols)
        x0 = pad + c * (tile_w + pad)
        y0 = header + pad + r * (tile_h + pad)
        draw.rounded_rectangle((x0, y0, x0 + tile_w, y0 + tile_h), radius=8, fill="white", outline="#cbd5e1", width=1)
        seq = case["seq_name"]
        case_img_path = root / "cases" / seq / "case_summary.png"
        if case_img_path.exists():
            case_img = fit_crop(Image.open(case_img_path), (tile_w - 24, 175))
            img.paste(case_img, (x0 + 12, y0 + 12))
        text_y = y0 + 197
        draw.text((x0 + 12, text_y), f"#{idx + 1} {case['category']} | {seq.split('_', 1)[1][:18]}", fill="#0f172a", font=text_font)
        text_y += 24
        draw.text(
            (x0 + 12, text_y),
            f"learned: {case['learned_best_policy']} CR {fmt(case['learned_best_cr'])}, {case['learned_best_steps']} steps",
            fill="#1d4ed8",
            font=small_font,
        )
        text_y += 20
        draw.text(
            (x0 + 12, text_y),
            f"heuristic: {case['heuristic_best_policy']} CR {fmt(case['heuristic_best_cr'])}, {case['heuristic_best_steps']} steps",
            fill="#c2410c",
            font=small_font,
        )
        text_y += 20
        draw.text(
            (x0 + 12, text_y),
            f"discrete {fmt(case['discrete_cr'])}/{case['discrete_steps']} | continuous {fmt(case['continuous_cr'])}/{case['continuous_steps']}",
            fill="#334155",
            font=small_font,
        )
        text_y += 20
        cap = str(case.get("caption", ""))
        if len(cap) > 75:
            cap = cap[:72] + "..."
        draw.text((x0 + 12, text_y), cap, fill="#64748b", font=small_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def markdown_table(rows: Sequence[dict], columns: Sequence[str], limit: int | None = None) -> str:
    rows = list(rows[:limit] if limit else rows)
    out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(fmt(v))
            else:
                vals.append(str(v))
        out.append("|" + "|".join(vals) + "|")
    return "\n".join(out)


def _mean(rows: Sequence[dict], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")


def write_markdown(
    root: Path,
    out_path: Path,
    policy_stats: Sequence[dict],
    cat_stats: Sequence[dict],
    attr_stats: Sequence[dict],
    step_diag: Sequence[dict],
    cases: Sequence[dict],
    threshold: float,
) -> None:
    hard = sorted(cases, key=case_score_for_hardness)
    learned_fail = [c for c in cases if not c["learned_success"]]
    heuristic_saves = [c for c in learned_fail if c["heuristic_success"]]
    discrete_only_fail = [c for c in cases if not c["discrete_success"] and c["continuous_success"]]
    continuous_only_fail = [c for c in cases if c["discrete_success"] and not c["continuous_success"]]
    both_learned_fail = [c for c in cases if not c["discrete_success"] and not c["continuous_success"]]
    fail_step_diag = [d for d in step_diag if str(d["learned_success"]) == "False"]
    save_step_diag = [d for d in fail_step_diag if str(d["heuristic_success"]) == "True"]
    lines = [
        "# ShapeNBV 600-Sample Eval Failure Analysis",
        "",
        "This report is generated from the downloaded rollout JSON files under `policies/*/episodes.json`.",
        "Every ranked sample links back to the generated `case_summary.png` and policy trajectory sheets in the same eval folder.",
        "",
        "## Global Policy Metrics",
        "",
        markdown_table(
            policy_stats,
            ["policy", "n", "cr_mean", "cr_p10", "cr_p50", "success_rate", "steps_mean", "steps_p50"],
        ),
        "",
        "## Evidence-Based Takeaways",
        "",
        f"- Learned policies miss the CR>{threshold:.2f} threshold on {len(learned_fail)}/{len(cases)} samples when we allow the better of discrete/continuous PPO.",
        f"- Both learned policies fail on {len(both_learned_fail)} samples; discrete-only failures are {len(discrete_only_fail)}, continuous-only failures are {len(continuous_only_fail)}.",
        f"- A heuristic policy, usually Fibonacci or Axis6, succeeds on {len(heuristic_saves)} of the learned-fail samples. These are the most useful policy-improvement cases because the environment can solve them with the current renderer/action space.",
        f"- On learned-fail samples, best learned has {_mean(fail_step_diag, 'learned_unique_dirs_15deg'):.2f} unique 15-degree view directions on average, versus {_mean(fail_step_diag, 'heuristic_unique_dirs_15deg'):.2f} for the best heuristic.",
        f"- On those same learned-fail samples, best learned gains {_mean(fail_step_diag, 'learned_tail10_gain'):.5f} CR in its last 10 steps on average, versus {_mean(fail_step_diag, 'heuristic_tail10_gain'):.5f} for the best heuristic.",
        "- The claims above are computed directly from `final_cr` and `steps`; inspect `hard_cases.csv` for the exact rows.",
        "- The view-direction and tail-gain diagnostics are computed from the per-episode `steps.json`; inspect `trajectory_diagnostics.csv` for exact values.",
        "",
        "## Category Breakdown",
        "",
        markdown_table(
            [
                {
                    "category": cat,
                    "n": len([c for c in cases if c["category"] == cat]),
                    "learned_success": float(np.mean([c["learned_success"] for c in cases if c["category"] == cat])),
                    "heuristic_success": float(np.mean([c["heuristic_success"] for c in cases if c["category"] == cat])),
                    "learned_cr_mean": float(np.mean([c["learned_best_cr"] for c in cases if c["category"] == cat])),
                    "heuristic_cr_mean": float(np.mean([c["heuristic_best_cr"] for c in cases if c["category"] == cat])),
                }
                for cat in sorted({c["category"] for c in cases})
            ],
            ["category", "n", "learned_success", "heuristic_success", "learned_cr_mean", "heuristic_cr_mean"],
        ),
        "",
        "## Top Hard Cases",
        "",
        markdown_table(
            [
                {
                    "rank": i + 1,
                    "seq_name": f"[{c['seq_name']}]({c['case_summary']})",
                    "category": c["category"],
                    "learned_best": f"{c['learned_best_policy']} {fmt(c['learned_best_cr'])}/{c['learned_best_steps']}",
                    "heuristic_best": f"{c['heuristic_best_policy']} {fmt(c['heuristic_best_cr'])}/{c['heuristic_best_steps']}",
                    "discrete": f"[{fmt(c['discrete_cr'])}/{c['discrete_steps']}]({c['discrete_sheet']})",
                    "continuous": f"[{fmt(c['continuous_cr'])}/{c['continuous_steps']}]({c['continuous_sheet']})",
                    "caption": c.get("caption", ""),
                }
                for i, c in enumerate(hard[:30])
            ],
            ["rank", "seq_name", "category", "learned_best", "heuristic_best", "discrete", "continuous", "caption"],
        ),
        "",
        "## Step-Level Diagnostics For Learned-Fail Cases",
        "",
        markdown_table(
            [
                {
                    "cohort": "all learned fail",
                    "n": len(fail_step_diag),
                    "learned_unique_dirs_15deg": _mean(fail_step_diag, "learned_unique_dirs_15deg"),
                    "heuristic_unique_dirs_15deg": _mean(fail_step_diag, "heuristic_unique_dirs_15deg"),
                    "learned_tail10_gain": _mean(fail_step_diag, "learned_tail10_gain"),
                    "heuristic_tail10_gain": _mean(fail_step_diag, "heuristic_tail10_gain"),
                    "learned_tail10_small_gain_steps": _mean(fail_step_diag, "learned_tail10_small_gain_steps"),
                    "heuristic_tail10_small_gain_steps": _mean(fail_step_diag, "heuristic_tail10_small_gain_steps"),
                },
                {
                    "cohort": "learned fail, heuristic success",
                    "n": len(save_step_diag),
                    "learned_unique_dirs_15deg": _mean(save_step_diag, "learned_unique_dirs_15deg"),
                    "heuristic_unique_dirs_15deg": _mean(save_step_diag, "heuristic_unique_dirs_15deg"),
                    "learned_tail10_gain": _mean(save_step_diag, "learned_tail10_gain"),
                    "heuristic_tail10_gain": _mean(save_step_diag, "heuristic_tail10_gain"),
                    "learned_tail10_small_gain_steps": _mean(save_step_diag, "learned_tail10_small_gain_steps"),
                    "heuristic_tail10_small_gain_steps": _mean(save_step_diag, "heuristic_tail10_small_gain_steps"),
                },
            ],
            [
                "cohort",
                "n",
                "learned_unique_dirs_15deg",
                "heuristic_unique_dirs_15deg",
                "learned_tail10_gain",
                "heuristic_tail10_gain",
                "learned_tail10_small_gain_steps",
                "heuristic_tail10_small_gain_steps",
            ],
        ),
        "",
        "## Attribute Cohort Signals",
        "",
        "Rows compare samples where a caption attribute is present vs absent. These are correlational diagnostics only; use the linked hard cases to verify visual causes.",
        "",
        markdown_table(
            sorted(attr_stats, key=lambda r: (r["attribute"], r["cohort"])),
            ["attribute", "cohort", "n", "learned_best_cr_mean", "learned_best_success_rate", "discrete_success_rate", "continuous_success_rate"],
        ),
    ]
    out_path.write_text("\n".join(lines))


def write_html(root: Path, out_path: Path, policy_stats: Sequence[dict], cases: Sequence[dict], threshold: float) -> None:
    hard = sorted(cases, key=case_score_for_hardness)
    rows = []
    for i, c in enumerate(hard):
        row_class = "bad" if not c["learned_success"] else "ok"
        rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{i + 1}</td>"
            f"<td><a href='{html.escape(c['case_summary'])}'>{html.escape(c['seq_name'])}</a><br><span>{html.escape(c['category'])}</span></td>"
            f"<td>{html.escape(c['caption'])}</td>"
            f"<td>{html.escape(c['learned_best_policy'])}<br>{fmt(c['learned_best_cr'])} / {c['learned_best_steps']}</td>"
            f"<td>{html.escape(c['heuristic_best_policy'])}<br>{fmt(c['heuristic_best_cr'])} / {c['heuristic_best_steps']}</td>"
            f"<td><a href='{html.escape(c['discrete_sheet'])}'>D {fmt(c['discrete_cr'])}/{c['discrete_steps']}</a><br>"
            f"<a href='{html.escape(c['continuous_sheet'])}'>C {fmt(c['continuous_cr'])}/{c['continuous_steps']}</a><br>"
            f"<a href='{html.escape(c['fibonacci_sheet'])}'>F {fmt(c['fibonacci_cr'])}/{c['fibonacci_steps']}</a><br>"
            f"<a href='{html.escape(c['axis6_sheet'])}'>A {fmt(c['axis6_cr'])}/{c['axis6_steps']}</a></td>"
            "</tr>"
        )
    cards = []
    for s in policy_stats:
        cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(POLICY_LABELS[s['policy']])}</h3>"
            f"<p>CR mean <b>{fmt(s['cr_mean'])}</b></p>"
            f"<p>success <b>{fmt(s['success_rate'], 3)}</b></p>"
            f"<p>steps mean <b>{fmt(s['steps_mean'], 2)}</b></p>"
            "</div>"
        )
    out_path.write_text(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ShapeNBV Failure Analysis</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #0f172a; background: #f8fafc; }
h1 { margin-bottom: 4px; }
.muted { color: #64748b; }
.cards { display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }
.card { background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 14px; }
.card h3 { margin: 0 0 8px 0; font-size: 16px; }
.card p { margin: 5px 0; color: #334155; }
img.summary { max-width: 100%; border: 1px solid #cbd5e1; background: white; }
table { width: 100%; border-collapse: collapse; background: white; margin-top: 16px; font-size: 13px; }
th, td { border: 1px solid #cbd5e1; padding: 7px 8px; vertical-align: top; }
th { background: #e2e8f0; position: sticky; top: 0; z-index: 1; }
tr.bad { background: #fff7ed; }
td span { color: #64748b; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
</head>
<body>
"""
        + f"<h1>ShapeNBV 600-Sample Failure Analysis</h1><p class='muted'>Sorted by learned-policy hardness. Threshold: CR &gt; {threshold:.2f}. All links are local artifacts.</p>"
        + "<div class='cards'>"
        + "\n".join(cards)
        + "</div>"
        + "<p><a href='full_eval_atlas.html'>Open full eval atlas</a> | <a href='reports/failure_analysis.md'>Open markdown report</a> | <a href='reports/hard_cases.csv'>Open hard_cases.csv</a></p>"
        + "<img class='summary' src='reports/failure_analysis_summary.png'>"
        + "<h2>Hard-Case Gallery</h2><img class='summary' src='reports/hard_case_gallery.png'>"
        + "<h2>Ranked Cases</h2><table><thead><tr><th>rank</th><th>sample</th><th>caption</th><th>best learned</th><th>best heuristic</th><th>policy links</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></body></html>"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.99)
    args = p.parse_args()

    root = Path(args.eval_dir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(root)
    cases = summarize_cases(root, episodes, args.threshold)
    policy_stats = aggregate_policy(episodes, args.threshold)
    cat_stats = aggregate_by_category(cases, args.threshold)
    attr_stats = aggregate_attributes(cases, args.threshold)
    step_diag = aggregate_step_diagnostics(root, cases)
    hard = sorted(cases, key=case_score_for_hardness)

    write_csv(reports / "policy_stats.csv", policy_stats)
    write_csv(reports / "category_stats.csv", cat_stats)
    write_csv(reports / "attribute_stats.csv", attr_stats)
    write_csv(reports / "case_policy_matrix.csv", cases)
    write_csv(reports / "hard_cases.csv", hard)
    write_csv(reports / "trajectory_diagnostics.csv", step_diag)
    write_summary_plot(reports / "failure_analysis_summary.png", policy_stats, cat_stats, cases, args.threshold)
    write_hard_gallery(root, reports / "hard_case_gallery.png", hard, n=24)
    write_markdown(root, reports / "failure_analysis.md", policy_stats, cat_stats, attr_stats, step_diag, cases, args.threshold)
    write_html(root, root / "failure_analysis.html", policy_stats, cases, args.threshold)
    print(f"[analysis] episodes={len(episodes)} cases={len(cases)}")
    print(f"[analysis] wrote {reports / 'failure_analysis.md'}")
    print(f"[analysis] wrote {root / 'failure_analysis.html'}")


if __name__ == "__main__":
    main()
