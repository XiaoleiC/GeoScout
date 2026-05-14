#!/usr/bin/env python3
"""Verify that preprocessed ShapeNBV `.pt` files contain caption embeddings."""

from __future__ import annotations

import argparse
import json
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
            object_ids.append(str(row["object_id"]))
    if len(object_ids) != len(set(object_ids)):
        raise SystemExit(f"{path}: duplicate object_id values")
    return object_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc-dir", required=True, type=Path)
    parser.add_argument("--caption-jsonl", required=True, type=Path)
    parser.add_argument("--embedding-path", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    object_ids = read_jsonl_object_ids(args.caption_jsonl)
    emb_payload = torch.load(args.embedding_path, weights_only=False, map_location="cpu")
    object_id_to_emb: dict[str, torch.Tensor] = emb_payload.get("object_id_to_emb", {})
    object_id_to_text: dict[str, str] = emb_payload.get("object_id_to_text", {})
    dim = int(emb_payload.get("embedding_dim") or 0)

    missing_files: list[str] = []
    missing_caption_emb: list[str] = []
    bad_dim: list[str] = []
    mismatched_emb: list[str] = []
    mismatched_text: list[str] = []
    norms: list[float] = []
    examples: list[dict[str, Any]] = []

    for object_id in tqdm(object_ids, desc="verify caption_emb"):
        path = args.preproc_dir / f"{object_id}.pt"
        if not path.exists():
            missing_files.append(object_id)
            continue
        data = torch.load(path, weights_only=False, map_location="cpu")
        if "caption_emb" not in data:
            missing_caption_emb.append(object_id)
            continue
        emb = data["caption_emb"].detach().cpu().float()
        expected = object_id_to_emb.get(object_id)
        if dim and emb.numel() != dim:
            bad_dim.append(object_id)
            continue
        if expected is None or not torch.allclose(emb, expected.detach().cpu().float(), atol=args.atol, rtol=0.0):
            mismatched_emb.append(object_id)
        expected_text = object_id_to_text.get(object_id, "")
        if str(data.get("caption_text", "")) != expected_text:
            mismatched_text.append(object_id)
        norm = float(emb.norm().item())
        norms.append(norm)
        if len(examples) < 5:
            examples.append({
                "object_id": object_id,
                "caption_emb_shape": list(emb.shape),
                "caption_emb_norm": norm,
                "caption_text": str(data.get("caption_text", "")),
            })

    report: dict[str, Any] = {
        "schema_version": "shapenbv_verify_preproc_caption_embeddings_v1",
        "preproc_dir": str(args.preproc_dir),
        "caption_jsonl": str(args.caption_jsonl),
        "embedding_path": str(args.embedding_path),
        "embedding_dim": dim,
        "num_requested": len(object_ids),
        "num_checked_with_caption_emb": len(norms),
        "num_missing_files": len(missing_files),
        "num_missing_caption_emb": len(missing_caption_emb),
        "num_bad_dim": len(bad_dim),
        "num_mismatched_emb": len(mismatched_emb),
        "num_mismatched_text": len(mismatched_text),
        "missing_files": missing_files[:50],
        "missing_caption_emb": missing_caption_emb[:50],
        "bad_dim": bad_dim[:50],
        "mismatched_emb": mismatched_emb[:50],
        "mismatched_text": mismatched_text[:50],
        "examples": examples,
    }
    if norms:
        report.update({
            "caption_emb_norm_min": min(norms),
            "caption_emb_norm_mean": sum(norms) / len(norms),
            "caption_emb_norm_max": max(norms),
        })
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if any(report[key] for key in [
        "num_missing_files",
        "num_missing_caption_emb",
        "num_bad_dim",
        "num_mismatched_emb",
        "num_mismatched_text",
    ]):
        raise SystemExit("Caption embedding verification failed")


if __name__ == "__main__":
    main()
