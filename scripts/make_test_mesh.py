"""Generate a synthetic test mesh for the smoke test.

We don't need ShapeNet to verify the PPO + env + render loop converges —
a simple sphere or torus mesh is enough to answer "can the policy raise
cr to 1 on a single object?". Outputs are written into a directory that
mirrors ShapeNet's layout so `geoscout.data.list_shapenet` works
unchanged:

    <out>/synthetic/sphere_test/models/model_normalized.obj

Usage:
    python -m scripts.make_test_mesh --out /tmp/geoscout_smoke_data \
        --shape sphere --n_subdiv 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_sphere(n_subdiv: int = 3, radius: float = 0.4):
    """UV-sphere via icosphere subdivision. Returns (verts, faces)."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=n_subdiv, radius=radius)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def make_cube(half_side: float = 0.4):
    import trimesh
    mesh = trimesh.creation.box(extents=[2 * half_side] * 3)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def make_torus(major_radius: float = 0.35, minor_radius: float = 0.12,
               major_sections: int = 32, minor_sections: int = 16):
    import trimesh
    mesh = trimesh.creation.torus(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_sections=major_sections,
        minor_sections=minor_sections,
    )
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def write_obj(verts, faces, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for fa in faces:
            f.write(f"f {int(fa[0]) + 1} {int(fa[1]) + 1} {int(fa[2]) + 1}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True,
                   help="Output dataset root (will look like ShapeNet).")
    p.add_argument("--shape", type=str, default="sphere",
                   choices=["sphere", "cube", "torus", "all"],
                   help="`all` writes one mesh per shape into different "
                        "synset dirs — useful for the multi-mesh + "
                        "caption smoke test.")
    p.add_argument("--name", type=str, default="",
                   help="Model id; defaults to '<shape>_test'.")
    p.add_argument("--synset", type=str, default="00000001",
                   help="Numeric folder id (must be digits — see "
                        "geoscout.data.list_shapenet's isdigit() filter). "
                        "Default 00000001 acts as a synthetic placeholder.")
    p.add_argument("--n_subdiv", type=int, default=3)
    args = p.parse_args()

    # Multi-shape: write a small zoo with distinct synset ids that map
    # to distinct ShapeNet category names so caption_emb is non-trivial
    # (sphere → "bowl", cube → "cabinet", torus → "donut/jar"
    # idea — pick real categories whose canonical shape is closest).
    if args.shape == "all":
        zoo = [
            ("sphere", "02880940", make_sphere, {"n_subdiv": 3}),       # bowl-shaped
            ("cube", "02933112", make_cube, {}),                         # cabinet
            ("torus", "03593526", make_torus, {}),                       # jar (close enough)
        ]
        for shape, syn, fn, kw in zoo:
            verts, faces = fn(**kw)
            name = f"{shape}_test"
            out_path = Path(args.out) / syn / name / "models" / "model_normalized.obj"
            write_obj(verts, faces, out_path)
            print(f"[make_test_mesh] wrote {out_path}  "
                  f"({len(verts)} verts, {len(faces)} faces)")
        return

    if args.shape == "sphere":
        verts, faces = make_sphere(n_subdiv=args.n_subdiv)
    elif args.shape == "cube":
        verts, faces = make_cube()
    else:
        verts, faces = make_torus()

    name = args.name or f"{args.shape}_test"
    out_path = Path(args.out) / args.synset / name / "models" / "model_normalized.obj"
    write_obj(verts, faces, out_path)
    print(f"[make_test_mesh] wrote {out_path}  ({len(verts)} verts, {len(faces)} faces)")
    print(f"[make_test_mesh] bbox = "
          f"min={verts.min(0).tolist()}  max={verts.max(0).tolist()}")


if __name__ == "__main__":
    main()
