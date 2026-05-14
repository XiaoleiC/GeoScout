#!/usr/bin/env python3
"""Attach object-level caption embeddings to existing ShapeNBV preproc files.

This is intentionally separate from mesh voxelization. Once a `.pt` already
contains `grid_gt`, changing captions should only rewrite metadata and
`caption_emb`, not re-sample or re-voxelize the mesh.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def read_jsonl_object_ids(path: Path) -> list[str]:
    object_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            object_id = str(row.get("object_id", "")).strip()
            if not object_id:
                raise SystemExit(f"{path}: row without object_id")
            object_ids.append(object_id)
    if len(object_ids) != len(set(object_ids)):
        raise SystemExit(f"{path}: duplicate object_id values")
    return object_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc-dir", required=True, type=Path)
    parser.add_argument("--embedding-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--caption-jsonl", type=Path, default=None,
                        help="If set, attach only object IDs listed here.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-without-caption", action="store_true",
                        help="Copy input files that have no matching caption embedding.")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    emb_payload = torch.load(args.embedding_path, weights_only=False, map_location="cpu")
    object_id_to_emb: dict[str, torch.Tensor] = emb_payload.get("object_id_to_emb", {})
    object_id_to_text: dict[str, str] = emb_payload.get("object_id_to_text", {})
    if not object_id_to_emb:
        raise SystemExit(f"{args.embedding_path} does not contain object_id_to_emb")

    object_ids = read_jsonl_object_ids(args.caption_jsonl) if args.caption_jsonl else sorted(object_id_to_emb)
    missing_embeddings = sorted(set(object_ids) - set(object_id_to_emb))
    if missing_embeddings:
        raise SystemExit(f"Missing embeddings for {len(missing_embeddings)} object IDs: {missing_embeddings[:10]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = missing_preproc = copied_without_caption = 0
    bad_dims: list[str] = []
    dim = int(emb_payload.get("embedding_dim") or 0)
    norm_values: list[float] = []

    for object_id in tqdm(object_ids, desc="attach caption_emb"):
        src = args.preproc_dir / f"{object_id}.pt"
        dst = args.out_dir / f"{object_id}.pt"
        if not src.exists():
            missing_preproc += 1
            continue
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        emb = object_id_to_emb[object_id].detach().cpu().float()
        if dim and emb.numel() != dim:
            bad_dims.append(object_id)
            continue
        norm_values.append(float(emb.norm().item()))

        data = torch.load(src, weights_only=False, map_location="cpu")
        data["caption_emb"] = emb
        data["caption_text"] = object_id_to_text.get(object_id, "")
        data["caption_emb_model"] = emb_payload.get("model_name", "")
        data["caption_emb_schema_version"] = emb_payload.get("schema_version", "")
        data["caption_emb_source_jsonl"] = emb_payload.get("source_jsonl", "")
        torch.save(data, dst)
        done += 1

    if args.copy_without_caption:
        wanted = {f"{object_id}.pt" for object_id in object_ids}
        for src in args.preproc_dir.glob("*.pt"):
            if src.name in wanted:
                continue
            dst = args.out_dir / src.name
            if dst.exists() and not args.overwrite:
                continue
            shutil.copy2(src, dst)
            copied_without_caption += 1

    report: dict[str, Any] = {
        "schema_version": "shapenbv_attach_caption_embeddings_report_v1",
        "preproc_dir": str(args.preproc_dir),
        "embedding_path": str(args.embedding_path),
        "out_dir": str(args.out_dir),
        "caption_jsonl": str(args.caption_jsonl) if args.caption_jsonl else "",
        "embedding_dim": dim,
        "num_requested": len(object_ids),
        "num_attached": done,
        "num_skipped_existing": skipped,
        "num_missing_preproc": missing_preproc,
        "num_bad_dim": len(bad_dims),
        "bad_dim_object_ids": bad_dims[:50],
        "copied_without_caption": copied_without_caption,
    }
    if norm_values:
        report.update({
            "caption_emb_norm_min": min(norm_values),
            "caption_emb_norm_mean": sum(norm_values) / len(norm_values),
            "caption_emb_norm_max": max(norm_values),
        })
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if missing_preproc or bad_dims:
        raise SystemExit("Attach completed with missing files or bad embedding dimensions")


if __name__ == "__main__":
    main()
