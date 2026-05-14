#!/usr/bin/env python3
"""Encode per-object GeoScout captions into a reusable torch payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--caption-field", default="embedding_caption")
    args = parser.parse_args()

    rows = read_jsonl(args.caption_jsonl)
    object_ids: list[str] = []
    texts: list[str] = []
    categories: dict[str, str] = {}
    synsets: dict[str, str] = {}
    for row in rows:
        object_id = str(row.get("object_id", "")).strip()
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
        text = str(caption.get(args.caption_field, "")).strip()
        if not object_id:
            raise SystemExit("Encountered row without object_id")
        if not text:
            raise SystemExit(f"{object_id}: missing caption.{args.caption_field}")
        if object_id in categories:
            raise SystemExit(f"Duplicate object_id in caption JSONL: {object_id}")
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        object_ids.append(object_id)
        texts.append(text)
        categories[object_id] = str(caption.get("category") or source.get("category_hint") or "")
        synsets[object_id] = str(source.get("synset") or object_id.split("_", 1)[0])

    print(f"[caption-emb] loading {args.model} on {args.device}", flush=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"[caption-emb] encoding {len(texts)} captions; dim={dim}", flush=True)
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=args.device,
    ).detach().cpu().float()

    object_id_to_emb = {object_id: embs[i] for i, object_id in enumerate(object_ids)}
    object_id_to_text = {object_id: texts[i] for i, object_id in enumerate(object_ids)}
    payload = {
        "schema_version": "geoscout_object_caption_embeddings_v1",
        "source_jsonl": str(args.caption_jsonl),
        "model_name": args.model,
        "embedding_dim": dim,
        "caption_field": args.caption_field,
        "normalized": True,
        "num_objects": len(object_ids),
        "object_ids": object_ids,
        "object_id_to_emb": object_id_to_emb,
        "object_id_to_text": object_id_to_text,
        "object_id_to_category": categories,
        "object_id_to_synset": synsets,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    norms = embs.norm(dim=1)
    print(
        f"[caption-emb] wrote {args.out}; "
        f"norm min/mean/max={norms.min().item():.6f}/"
        f"{norms.mean().item():.6f}/{norms.max().item():.6f}",
        flush=True,
    )
    for object_id in object_ids[:5]:
        print(f"  {object_id}: {object_id_to_text[object_id]}", flush=True)


if __name__ == "__main__":
    main()
