"""Amazon Berkeley Objects (ABO) 3D dataset loader.

ABO is the **ungated** stand-in for ShapeNet+Text2Shape: ~7,953 GLB
meshes from Amazon product catalog with **per-object natural-language
captions** (product titles, bullet-points, materials, colors, etc.) —
exactly what we need for caption-conditioned NBV, with no manual
license process.

License: CC-BY-4.0 (research + commercial OK with attribution).
Hosted at: s3://amazon-berkeley-objects/ (us-east-1, anonymous read).

Layout on disk after `tar -xf abo-3dmodels.tar`:

    <root>/
        3dmodels/
            metadata/
                3dmodels.csv.gz                  # manifest (model_id, glb path, vertices, ...)
            original/
                <C>/<3dmodel_id>.glb             # C = last char of ASIN — hash bucket
        listings/
            metadata/
                listings_0.json.gz ... listings_15.json.gz
                                                  # NDJSON: one product per line; has captions
        README.md, LICENSE-CC-BY-4.0.txt

Coordinate / scale conventions (from the dataset's own README):
    - +Y is up
    - +Z toward "natural front" of the product
    - Real-world METERS (not normalized!) — caller must rescale
    - Floor-standing items rest on Y=0 plane (Y_min = 0)

Usage parallels `shapenbv.data.list_shapenet`:

    entries = list_abo(root, categories=["CHAIR", "LAMP"], limit=200)
    for e in entries:
        e.glb_path        # → trimesh.load(e.glb_path, force="mesh")
        e.caption_text    # full natural-language caption (200-500 chars)
        e.product_type    # "CHAIR", "LAMP", ...
        e.name            # stable id, mirrors ShapeNetEntry.name
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ABOEntry:
    """One ABO 3D object record."""
    model_id: str            # = ASIN (e.g. "B07RDKQX1V")
    product_type: str        # category, e.g. "CHAIR"
    glb_path: Path           # absolute path to .glb mesh
    caption_text: str        # full natural-language caption
    item_name: str           # short title only
    extent_m: tuple          # (x, y, z) in METERS

    @property
    def name(self) -> str:
        """Stable identifier matching `ShapeNetEntry.name` style."""
        return f"{self.product_type}_{self.model_id}"

    @property
    def mesh_path(self) -> Path:
        """Alias for `glb_path` so train.py can treat ABOEntry and
        ShapeNetEntry uniformly via duck-typing."""
        return self.glb_path


def _read_3dmodels_manifest(root: Path) -> Dict[str, dict]:
    """Read `3dmodels/metadata/3dmodels.csv.gz` → {model_id: row}.

    We intentionally avoid pandas (keeps the loader's deps light); the
    file is small (~7k rows × 13 cols).
    """
    import csv
    path = root / "3dmodels" / "metadata" / "3dmodels.csv.gz"
    if not path.exists():
        # Some unpacks put 3dmodels.csv.gz directly under root.
        alt = root / "3dmodels.csv.gz"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"3dmodels manifest not found at {path}. Did you extract "
                f"abo-3dmodels.tar?"
            )
    out: Dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["3dmodel_id"]] = row
    return out


def _read_listings(root: Path) -> Dict[str, dict]:
    """Stream the 16 NDJSON shards under `listings/metadata/` and keep
    only entries that have a 3dmodel_id. Returns {model_id: listing}.
    """
    out: Dict[str, dict] = {}
    base = root / "listings" / "metadata"
    if not base.exists():
        # Some unpacks expose listings_*.json.gz at top level.
        candidates = sorted(root.glob("listings_*.json.gz"))
    else:
        candidates = sorted(base.glob("listings_*.json.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No listings_*.json.gz under {root}. Did you extract abo-listings.tar?"
        )
    for shard in candidates:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                mid = rec.get("3dmodel_id")
                if mid:
                    out[mid] = rec
    return out


def _english_value(field) -> str:
    """ABO multilingual fields are lists of {language_tag, value}.
    Pull the first English entry; fall back to first if none."""
    if not field:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for entry in field:
            tag = entry.get("language_tag", "") if isinstance(entry, dict) else ""
            if tag.startswith("en"):
                return entry.get("value", "") if isinstance(entry, dict) else str(entry)
        # No English match — first entry's value.
        first = field[0]
        if isinstance(first, dict):
            return first.get("value", "")
        return str(first)
    return str(field)


def _bullet_points_en(field) -> List[str]:
    if not field:
        return []
    if isinstance(field, str):
        return [field]
    return [_english_value([e]) for e in field if e]


def _build_caption(listing: dict, manifest_row: dict) -> str:
    """Compose a natural-language caption from ABO metadata.

    Ingredients (joined by spaces):
        product_type
        item_name (en)
        bullet_point[*] (en, joined)
        + optional: color, material, style
    """
    parts: List[str] = []
    pt = listing.get("product_type", "")
    if isinstance(pt, list):
        pt = pt[0].get("value", "") if pt and isinstance(pt[0], dict) else ""
    pt_human = str(pt).lower().replace("_", " ")
    parts.append(f"a 3d model of a {pt_human}")

    name = _english_value(listing.get("item_name"))
    if name:
        parts.append(name)

    for bp in _bullet_points_en(listing.get("bullet_point")):
        if bp:
            parts.append(bp)

    for attr in ("color", "material", "style", "finish_type", "item_shape"):
        val = _english_value(listing.get(attr))
        if val:
            parts.append(val)

    return ". ".join(parts)


def list_abo(
    root: Path,
    categories: Optional[List[str]] = None,
    limit: int = 0,
    require_glb: bool = True,
) -> List[ABOEntry]:
    """Enumerate ABO 3D models with attached captions.

    Args:
        root: ABO unpacked root (contains `3dmodels/` and `listings/`).
        categories: filter to these product_type values (e.g.
            ["CHAIR", "LAMP", "SOFA"]). None = all.
        limit: cap N total entries (0 = no cap).
        require_glb: skip entries whose `.glb` is missing on disk.

    Returns: list of ABOEntry, sorted by name for reproducibility.
    """
    root = Path(root)
    manifest = _read_3dmodels_manifest(root)
    listings = _read_listings(root)

    cat_set = {c.upper() for c in categories} if categories else None
    out: List[ABOEntry] = []
    for mid, mrow in manifest.items():
        listing = listings.get(mid)
        if listing is None:
            continue
        product_type = listing.get("product_type", "")
        if isinstance(product_type, list):
            product_type = product_type[0].get("value", "") if product_type and isinstance(product_type[0], dict) else ""
        product_type = str(product_type).upper()
        if cat_set is not None and product_type not in cat_set:
            continue

        rel = mrow.get("path", "")
        glb_path = root / "3dmodels" / "original" / rel if rel else (root / "3dmodels" / "original" / mid[-1] / f"{mid}.glb")
        if require_glb and not glb_path.exists():
            continue

        try:
            extent = (
                float(mrow.get("extent_x", 0)),
                float(mrow.get("extent_y", 0)),
                float(mrow.get("extent_z", 0)),
            )
        except ValueError:
            extent = (0.0, 0.0, 0.0)

        out.append(ABOEntry(
            model_id=mid,
            product_type=product_type,
            glb_path=glb_path,
            caption_text=_build_caption(listing, mrow),
            item_name=_english_value(listing.get("item_name")),
            extent_m=extent,
        ))
        if limit and len(out) >= limit:
            break
    out.sort(key=lambda e: e.name)
    return out


def list_categories(root: Path) -> Dict[str, int]:
    """Return {product_type: count_with_3d_model} sorted desc."""
    listings = _read_listings(root)
    counts: Dict[str, int] = {}
    for rec in listings.values():
        if "3dmodel_id" not in rec:
            continue
        pt = rec.get("product_type", "")
        if isinstance(pt, list):
            pt = pt[0].get("value", "") if pt and isinstance(pt[0], dict) else ""
        pt = str(pt).upper()
        counts[pt] = counts.get(pt, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
