"""ShapeNet dataset loader.

Layout (ShapeNetCore.v2):
    <root>/
        <synset_id>/                      # e.g. "03001627" = chair
            <model_id>/                   # SHA-style hash
                models/
                    model_normalized.obj  # mesh, pre-normalized to unit cube
                    model_normalized.json # bbox info
                    ...

We enumerate (synset, model_id) pairs and resolve the absolute path to
`model_normalized.obj`. The "normalized" version centers the mesh at the
origin and rescales the longest axis to 1, so the bbox is roughly
[-0.5, 0.5]³ — directly compatible with our normalized Cube Mode
action box.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# Mapping of frequently-used synset IDs to human-readable category names.
# 55 ShapeNetCore categories — full list available in <root>/synsetoffset2category.txt.
SYNSET_TO_CATEGORY = {
    "02691156": "airplane",
    "02747177": "trash_bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "computer_keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file_cabinet",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "flowerpot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "watercraft",
    "04554684": "washer",
}
CATEGORY_TO_SYNSET = {v: k for k, v in SYNSET_TO_CATEGORY.items()}


@dataclass(frozen=True)
class ShapeNetEntry:
    synset: str
    model_id: str
    mesh_path: Path

    @property
    def category(self) -> str:
        return SYNSET_TO_CATEGORY.get(self.synset, self.synset)

    @property
    def name(self) -> str:
        """Stable identifier matching uCO3D's `<cat>/<seq>` style."""
        return f"{self.synset}_{self.model_id}"


def list_shapenet(
    root: Path,
    synsets: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    limit_per_synset: int = 0,
    require_obj: bool = True,
) -> List[ShapeNetEntry]:
    """Enumerate ShapeNetCore.v2 entries on disk.

    Args:
        root: ShapeNetCore.v2 root (contains synset_id directories).
        synsets: filter to these synset IDs (e.g., ["03001627"] for chairs).
        categories: alternative to `synsets` — pass human-readable
            ("chair"). Translated via CATEGORY_TO_SYNSET.
        limit_per_synset: cap N per synset (0 = no cap).
        require_obj: skip model_ids without `model_normalized.obj`.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ShapeNet root not found: {root}")

    if synsets is None and categories is not None:
        normalized_categories = [c.strip().lower() for c in categories]
        unknown = [c for c in normalized_categories if c not in CATEGORY_TO_SYNSET]
        if unknown:
            known = ", ".join(sorted(CATEGORY_TO_SYNSET.keys()))
            raise ValueError(f"Unknown ShapeNet categories {unknown}. Known: {known}")
        synsets = [CATEGORY_TO_SYNSET[c] for c in normalized_categories]

    if synsets is None:
        synsets = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])

    out: List[ShapeNetEntry] = []
    for sid in synsets:
        sdir = root / sid
        if not sdir.is_dir():
            continue
        models = sorted([p for p in sdir.iterdir() if p.is_dir()])
        if limit_per_synset > 0:
            models = models[: limit_per_synset]
        for mdir in models:
            obj = mdir / "models" / "model_normalized.obj"
            if require_obj and not obj.exists():
                continue
            out.append(ShapeNetEntry(synset=sid, model_id=mdir.name, mesh_path=obj))
    return out


def load_normalization_info(entry: ShapeNetEntry) -> Optional[dict]:
    """Read `model_normalized.json` if present (bbox info).

    Schema (ShapeNetCore.v2):
        {"min": [x, y, z], "max": [x, y, z],
         "centroid": [...], "id": ..., "numVertices": ..., "numFaces": ...}

    None if missing.
    """
    j = entry.mesh_path.parent / "model_normalized.json"
    if not j.exists():
        return None
    with open(j) as f:
        return json.load(f)


def iter_synsets(root: Path) -> Iterator[Tuple[str, str, int]]:
    """Yield (synset_id, category_name, num_models) for each synset on disk."""
    root = Path(root)
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.isdigit():
            continue
        n = sum(1 for _ in p.iterdir() if _.is_dir())
        cat = SYNSET_TO_CATEGORY.get(p.name, p.name)
        yield (p.name, cat, n)
