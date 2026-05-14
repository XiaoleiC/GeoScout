"""Encode ShapeNet's 55 category names with sentence-transformers
(MiniLM L6), then save as one shared `.pt` file consumed by
`preprocess.py` to attach a per-object `caption_emb` to each preproc.

Usage:
    python -m scripts.precompute_category_embeddings \
        --out /data/geoscout_preproc_g128/category_embeddings.pt

Schema (saved):
    {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "synset_to_emb": { "03001627": tensor[384], ... }   # 55 entries
        "synset_to_text": { "03001627": "a 3d model of a chair", ... }
    }

Phase 1 design choice — class label as caption:
    1. ShapeNet has no native captions.
    2. We synthesize a caption per category by templating the human-readable
       category name into "a 3d model of a <category>" (Standard CLIP-style
       prompt that gives clean semantic embeddings).
    3. All ~5.5k objects in a class share the SAME caption_emb. The policy
       can use it as a "category hint" early in the episode (the occupancy
       grid is empty) and then largely ignore it once geometry is observed.
    4. Embeddings are CONSTANT across training, so we precompute once and
       attach as a tensor lookup. No live sentence-transformer call during
       training.

Upgrade path (Phase 2) — switch `synset_to_emb` to a per-OBJECT lookup
keyed by `<synset>_<model_id>`. Drop in Text2Shape captions or Cap3D
auto-captions; everything downstream stays the same.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geoscout.data import SYNSET_TO_CATEGORY


CAPTION_TEMPLATE = "a 3d model of a {}"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True,
                   help="Output .pt path.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="HuggingFace sentence-transformers model id.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for encoding (cpu is fine — only 55 strings).")
    args = p.parse_args()

    print(f"[caption] loading model {args.model} on {args.device}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    print(f"[caption] embedding dim = {dim}")

    synset_to_text = {}
    texts = []
    synsets = []
    for synset, category in SYNSET_TO_CATEGORY.items():
        cat_human = category.replace("_", " ")
        text = CAPTION_TEMPLATE.format(cat_human)
        synset_to_text[synset] = text
        texts.append(text)
        synsets.append(synset)

    print(f"[caption] encoding {len(texts)} category captions...")
    embs = model.encode(
        texts, convert_to_tensor=True, normalize_embeddings=True, device=args.device,
    ).cpu()
    synset_to_emb = {syn: embs[i] for i, syn in enumerate(synsets)}

    payload = {
        "model_name": args.model,
        "embedding_dim": int(dim),
        "synset_to_emb": synset_to_emb,
        "synset_to_text": synset_to_text,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[caption] wrote {out_path}  ({len(synset_to_emb)} synsets, {dim}-d)")
    for syn, txt in list(synset_to_text.items())[:5]:
        print(f"  {syn} → {txt!r}")


if __name__ == "__main__":
    main()
