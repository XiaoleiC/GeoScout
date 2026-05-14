"""Pure-PyTorch mesh renderer smoke test."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


def _write_cube_obj(path: Path):
    verts = [
        (-0.5, -0.5, -0.5), (+0.5, -0.5, -0.5), (+0.5, +0.5, -0.5), (-0.5, +0.5, -0.5),
        (-0.5, -0.5, +0.5), (+0.5, -0.5, +0.5), (+0.5, +0.5, +0.5), (-0.5, +0.5, +0.5),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 7, 6), (5, 8, 7),
        (1, 5, 6), (1, 6, 2),
        (3, 7, 8), (3, 8, 4),
        (2, 6, 7), (2, 7, 3),
        (1, 4, 8), (1, 8, 5),
    ]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def test_smoke():
    """Render one view of a unit cube and check basic invariants."""
    from shapenbv.mesh_renderer import MeshSequenceRenderer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = Path(tmp) / "cube.obj"
        _write_cube_obj(mesh_path)
        renderer = MeshSequenceRenderer(
            mesh_path=mesh_path,
            sequence_name="cube",
            device=device,
            render_size=(64, 64),
            fov_deg=60.0,
        )
        out = renderer.render(
            position_canon=torch.tensor([2.0, 0.0, 0.0], device=device),
            look_at_canon=torch.tensor([0.0, 0.0, 0.0], device=device),
        )

    alpha = out.alpha
    depth = out.depth
    n_hit = int(alpha.sum().item())
    assert n_hit > 0
    fg_depth = depth[alpha > 0.5]
    print(f"[ok] hit_px={n_hit}/{alpha.numel()}  "
          f"depth: min={float(fg_depth.min()):.3f} "
          f"med={float(fg_depth.median()):.3f} "
          f"max={float(fg_depth.max()):.3f}  "
          f"(camera 2 units away from cube of half-extent 0.5 → ~1.5–2.5 expected)")
    assert 1.0 <= float(fg_depth.min()) <= 3.0


if __name__ == "__main__":
    test_smoke()
    sys.exit(0)
