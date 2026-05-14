from __future__ import annotations

import math

import pytest


def test_nvdiffrast_points_align_with_tensor_env_pixel_rays(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("nvdiffrast renderer alignment test requires CUDA")
    pytest.importorskip("nvdiffrast.torch")

    from geoscout.mesh_renderer import NvdiffrastMeshSequenceRenderer, _build_camera_basis

    mesh_path = tmp_path / "asymmetric_plane.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v -0.55 -0.35 0.0",
                "v 0.45 -0.25 0.0",
                "v 0.35 0.60 0.0",
                "v -0.45 0.50 0.0",
                "f 1 2 3",
                "f 1 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    device = torch.device("cuda")
    render_size = 64
    renderer = NvdiffrastMeshSequenceRenderer(
        mesh_path=mesh_path,
        device=device,
        render_size=(render_size, render_size),
        fov_deg=60.0,
        max_faces=0,
    )
    eye = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    at = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    out = renderer.render_batch(eye, at)
    hit = out.alpha[0].reshape(-1) > 0.5
    assert int(hit.sum().item()) > 0

    H = W = render_size
    f = 0.5 * H / math.tan(math.radians(60.0) * 0.5)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rays_cam = torch.stack(
        [(xs - 0.5 * W) / f, (ys - 0.5 * H) / f, torch.ones_like(xs)],
        dim=-1,
    )
    rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
    R = _build_camera_basis(eye[0], at[0])
    rays_world = torch.einsum("ij,hwj->hwi", R, rays_cam).reshape(-1, 3)

    pts = out.points[0].reshape(-1, 3)
    dirs = pts - eye[0].view(1, 3)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    cos = (dirs[hit] * rays_world[hit]).sum(dim=-1)

    assert float(cos.mean().item()) > 0.999
    assert float(torch.quantile(cos, 0.01).item()) > 0.995
