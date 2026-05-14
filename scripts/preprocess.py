"""Batch ShapeNet preprocessing.

Walks the ShapeNetCore.v2 root, samples 100k surface points per mesh,
voxelizes into a 128³ GT occupancy grid, and saves
`<synset>_<model_id>.pt` under `--out_dir`. Idempotent: existing
`.pt` files are skipped. Multiprocess via `--n_workers`.

Usage:
    python -m scripts.preprocess \
        --shapenet_root /data/ShapeNetCore.v2 \
        --out_dir /data/ShapeNet_preproc \
        --categories chair,car,sofa \
        --n_workers 8
"""
from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from geoscout.data import list_shapenet
from geoscout.preprocess import preproc_file_matches_config, preprocess_mesh, save_preproc


_CAPTION_DICT_CACHE = {}


def _load_caption_dict(path: str):
    cached = _CAPTION_DICT_CACHE.get(path)
    if cached is not None:
        return cached
    import torch as _t
    data = _t.load(Path(path), weights_only=False, map_location="cpu")
    _CAPTION_DICT_CACHE[path] = data
    return data


def _process_one(args) -> Tuple[str, str, str]:
    """args = (name, mesh_path, out_path, grid_size, n_pts,
               synset, category, caption_emb_path, grid_storage_dtype)"""
    (name, mesh_path, out_path, grid_size, n_pts,
     synset, category, caption_emb_path, grid_storage_dtype) = args
    try:
        # Look up either a per-object caption embedding (preferred) or the
        # older shared category embedding.  The object-level schema is keyed
        # by the same `<synset>_<model_id>` name used for preproc files.
        caption_emb = None
        if caption_emb_path:
            cdict = _load_caption_dict(str(caption_emb_path))
            object_lookup = cdict.get("object_id_to_emb", {})
            if object_lookup:
                caption_emb = object_lookup.get(name)
            else:
                caption_emb = cdict.get("synset_to_emb", {}).get(synset)

        out_path = Path(out_path)
        if out_path.exists():
            matches, reason = preproc_file_matches_config(
                out_path,
                grid_size=grid_size,
                grid_storage_dtype=grid_storage_dtype,
                n_surface_points=n_pts,
                require_caption_emb=caption_emb is not None,
            )
            if matches:
                return (name, "skip", "")
            print(f"[preprocess] regenerate stale {name}: {reason}", flush=True)

        data = preprocess_mesh(
            Path(mesh_path), grid_size=grid_size, n_surface_points=n_pts,
            caption_emb=caption_emb, synset=synset, category=category,
            grid_storage_dtype=grid_storage_dtype,
        )
        save_preproc(out_path, data)
        return (name, "ok", "")
    except Exception as e:
        return (name, "err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapenet_root", type=str, required=True,
                   help="ShapeNetCore.v2 root (contains synset_id directories).")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--synsets", type=str, default="",
                   help="Comma-separated synset IDs. Empty = all on disk.")
    p.add_argument("--categories", type=str, default="",
                   help="Comma-separated category names (alternative to --synsets).")
    p.add_argument("--limit_per_synset", type=int, default=0,
                   help="Cap N per synset (0 = no cap).")
    p.add_argument("--grid_size", type=int, default=128)
    p.add_argument("--grid_storage_dtype", choices=["float32", "uint8"],
                   default="uint8",
                   help="Storage dtype for binary grid_gt. uint8 is "
                        "recommended for 128^3 grids; the env casts back "
                        "to float on load.")
    p.add_argument("--n_surface_points", type=int, default=0,
                   help="If > 0, save N surface samples to `points_canon` "
                        "for high-fidelity validate.py viz. Default 0 "
                        "saves ~35× space (51k×1.2MB → compact grid-only files). "
                        "Voxelization uses 100k samples regardless.")
    p.add_argument("--caption_emb_path", type=str, default="",
                   help="Path to category-embeddings .pt produced by "
                        "scripts.precompute_category_embeddings. Empty "
                        "skips per-object caption_emb attachment.")
    p.add_argument("--n_workers", type=int, default=8)
    args = p.parse_args()

    synsets = [s.strip() for s in args.synsets.split(",") if s.strip()] or None
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None

    entries = list_shapenet(
        root=Path(args.shapenet_root),
        synsets=synsets,
        categories=categories,
        limit_per_synset=args.limit_per_synset,
    )
    print(f"[preprocess] found {len(entries)} ShapeNet entries.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    caption_emb_path = args.caption_emb_path or None
    if caption_emb_path and not Path(caption_emb_path).exists():
        print(f"[preprocess] WARN: caption_emb_path {caption_emb_path} missing; "
              f"objects will be saved without caption_emb.")
        caption_emb_path = None

    work = []
    for e in entries:
        out_path = out_dir / f"{e.name}.pt"
        work.append((e.name, str(e.mesh_path), str(out_path),
                     args.grid_size, args.n_surface_points,
                     e.synset, e.category, caption_emb_path,
                     args.grid_storage_dtype))

    n_done = n_skip = n_err = 0
    if args.n_workers <= 1:
        for w in work:
            name, status, msg = _process_one(w)
            if status == "ok":     n_done += 1
            elif status == "skip": n_skip += 1
            else:                  n_err += 1; print(f"[ERR] {name}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futs = {ex.submit(_process_one, w): w[0] for w in work}
            for i, fut in enumerate(as_completed(futs)):
                name, status, msg = fut.result()
                if status == "ok":     n_done += 1
                elif status == "skip": n_skip += 1
                else:                  n_err += 1; print(f"[ERR] {name}: {msg}")
                if (i + 1) % 100 == 0:
                    print(f"[preprocess] {i+1}/{len(work)} ok={n_done} skip={n_skip} err={n_err}")

    print(f"[preprocess] DONE: ok={n_done} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
