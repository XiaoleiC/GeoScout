"""Lightweight visualization helpers (no IsaacGym deps).

Functions are robust to absent matplotlib / open3d — they silently
no-op so headless training never crashes on a viz line.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _y_up_to_z_up(arr):
    """Rotate Y-up world coordinates → Z-up (matplotlib 3D default).

    ABO/ShapeNet meshes are Y-up; without this transform mpl renders
    them on their side (seat vertical, back horizontal). Maps
    (x, y, z) → (x, -z, y), preserving handedness.
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    out = np.empty_like(arr, dtype=np.float32)
    out[..., 0] = arr[..., 0]
    out[..., 1] = -arr[..., 2]
    out[..., 2] = arr[..., 1]
    return out


def _voxel_density_render(ax, points, bbox_min=None, bbox_max=None, *,
                          grid_size: int = 60,
                          min_count: int = 2,
                          pad_frac: float = 0.05,
                          cmap_name: str = "viridis"):
    """Render a [N,3] point cloud as a 3D voxel-density grid.

    By default uses the pcd's OWN tight bbox (with `pad_frac` padding)
    rather than the action grid. This keeps the visualization fine
    enough to show chair backs / table tops. Low-resolution action-grid
    views look coarse; never use them for surface viz.

    `bbox_min`/`bbox_max` are honoured if provided; otherwise auto-
    derived from the pcd extent.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return False
    if bbox_min is None or bbox_max is None:
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        span = np.maximum(pmax - pmin, 1e-6)
        bb_min = pmin - span * pad_frac
        bb_max = pmax + span * pad_frac
    else:
        bb_min = np.asarray(bbox_min, dtype=np.float32)
        bb_max = np.asarray(bbox_max, dtype=np.float32)
    # Cube the bbox so voxels are isotropic (matplotlib's ax.voxels
    # axes-aspect would otherwise stretch them along the longer dim).
    centre = 0.5 * (bb_min + bb_max)
    half = 0.5 * float(np.max(bb_max - bb_min))
    bb_min = centre - half
    bb_max = centre + half
    extent = np.maximum(bb_max - bb_min, 1e-6)

    idx = np.floor((pts - bb_min) / extent * grid_size).astype(np.int64)
    valid = (idx >= 0).all(axis=1) & (idx < grid_size).all(axis=1)
    idx = idx[valid]
    if idx.size == 0:
        return False
    counts = np.zeros((grid_size, grid_size, grid_size), dtype=np.int64)
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    occ = counts >= min_count
    if not occ.any():
        return False
    log_c = np.log1p(counts.astype(np.float32))
    log_c /= max(log_c.max(), 1e-6)
    try:
        import matplotlib.cm as cm
    except ImportError:
        return False
    cmap = cm.get_cmap(cmap_name)
    facecolors = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
    rgba = cmap(log_c)
    facecolors[..., :3] = rgba[..., :3]
    facecolors[..., 3] = np.where(occ, 0.65 + 0.3 * log_c, 0.0)
    # Build edge-aligned voxel coordinates so cubes appear in world space
    # (instead of integer indices).
    xs = np.linspace(bb_min[0], bb_max[0], grid_size + 1)
    ys = np.linspace(bb_min[1], bb_max[1], grid_size + 1)
    zs = np.linspace(bb_min[2], bb_max[2], grid_size + 1)
    Xc, Yc, Zc = np.meshgrid(xs, ys, zs, indexing="ij")
    # `edgecolor` thin black lines make cubes look CRISP (matplotlib's
    # default no-edge + transparent voxels alias to fuzzy spheres at
    # small sizes — exactly the "balls" complaint).
    ax.voxels(Xc, Yc, Zc, occ,
              facecolors=facecolors,
              edgecolor=(0, 0, 0, 0.45),
              linewidth=0.15)
    return True


def save_pointcloud_ply(points, path: Union[str, Path]) -> None:
    """Save [N, 3] points as ASCII PLY (open3d preferred, manual fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = _to_numpy(points).astype(np.float32)

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(path), pcd)
        return
    except ImportError:
        pass

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_trajectory_plot(
    eye_positions: Sequence,
    look_ats: Sequence,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    path: Union[str, Path],
    object_pointcloud: Optional[np.ndarray] = None,
) -> None:
    """3D scatter of camera positions + bbox wireframe + (optional) pcd.

    No-op if matplotlib unavailable (headless install OK).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Y-up → Z-up so the chair stands upright in mpl 3D.
    eyes = _y_up_to_z_up(np.asarray(eye_positions))
    ats = _y_up_to_z_up(np.asarray(look_ats))
    bb_min_v = _y_up_to_z_up(np.asarray(bbox_min))
    bb_max_v = _y_up_to_z_up(np.asarray(bbox_max))
    bb_min_v, bb_max_v = np.minimum(bb_min_v, bb_max_v), np.maximum(bb_min_v, bb_max_v)
    T = len(eyes)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if object_pointcloud is not None and len(object_pointcloud) > 0:
        pc = _y_up_to_z_up(np.asarray(object_pointcloud))
        # Voxel-density rendering — replaces the gray "dust" scatter.
        _voxel_density_render(ax, pc)

    sc = ax.scatter(
        eyes[:, 0], eyes[:, 1], eyes[:, 2],
        c=np.arange(T), cmap="viridis", s=60, edgecolors="k", label="camera",
    )
    ax.plot(eyes[:, 0], eyes[:, 1], eyes[:, 2], "k-", alpha=0.4)

    for eye, at in zip(eyes, ats):
        ax.plot([eye[0], at[0]], [eye[1], at[1]], [eye[2], at[2]],
                "r-", alpha=0.25, linewidth=0.6)

    x_min, y_min, z_min = bb_min_v
    x_max, y_max, z_max = bb_max_v
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), "b-", alpha=0.3)

    plt.colorbar(sc, ax=ax, shrink=0.6, label="step index")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
    ax.set_title(f"Camera trajectory ({T} steps)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_coverage_curve(cr_history: Sequence[float], path: Union[str, Path]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(cr_history)), cr_history, "-o", lw=1.5, ms=4)
    ax.fill_between(range(len(cr_history)), 0, cr_history, alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("Coverage Ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CR trajectory (final = {cr_history[-1]:.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_occupancy_grid_slices(
    grid_tri_cls,
    path: Union[str, Path],
    gt=None,
    caption: str = "",
) -> None:
    """Three mid-slice views of the tri-class occupancy grid.

    grid_tri_cls: [G, G, G] in {-1, 0, +1} (or [1, G, G, G]).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = _to_numpy(grid_tri_cls)
    if g.ndim == 4 and g.shape[0] == 1:
        g = g[0]
    G = g.shape[0]
    mid = G // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, slc, title in [
        (axes[0], g[mid, :, :], f"X={mid}"),
        (axes[1], g[:, mid, :], f"Y={mid}"),
        (axes[2], g[:, :, mid], f"Z={mid}"),
    ]:
        ax.imshow(slc.T, origin="lower", cmap="seismic", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(caption or "Occupancy grid (red=occ / blue=free / white=unknown)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_action_history_plot(
    action_indices: Sequence,        # [T, 6] int (per-step MultiDiscrete idx)
    cr_history: Sequence[float],     # [T] float
    nvec: Sequence[int],             # bin counts per action dim, len=6
    path,
    title: str = "Action history",
) -> None:
    """6-panel time-series of the discrete action indices (+ CR overlay).

    Panels (one per action dim): x, y, z, roll(unused), pitch, yaw.
    The y-axis on each panel spans [0, nvec[i]-1] so the policy's
    distribution across the discrete grid is visually obvious.

    A bottom panel overlays cr(t) so reward dynamics line up with action
    choices.  Used in validate.py to inspect what the trained policy is
    actually doing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    actions = _to_numpy(action_indices)
    cr = _to_numpy(cr_history)
    if actions.ndim != 2 or actions.shape[1] != 6:
        return
    T = actions.shape[0]
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7, 1, figsize=(10, 13), sharex=True)
    for i, lbl in enumerate(labels):
        ax = axes[i]
        ax.plot(actions[:, i], "o-", markersize=3, linewidth=0.8)
        ax.set_ylim(-0.5, nvec[i] - 0.5)
        ax.set_ylabel(f"{lbl}\n[0..{nvec[i] - 1}]")
        ax.grid(alpha=0.2)
    axes[6].plot(cr, "g-", linewidth=1.2)
    axes[6].set_ylabel("CR\n[0..1]")
    axes[6].set_xlabel("step")
    axes[6].set_ylim(0, 1)
    axes[6].grid(alpha=0.2)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_combined_dashboard(
    *,
    eye_positions: Sequence,
    look_ats: Sequence,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    cr_history: Sequence[float],
    action_indices: Sequence,
    nvec: Sequence[int],
    gt_pointcloud=None,
    pred_pointcloud=None,
    mesh_path=None,                  # NEW: shaded mesh background
    path=None,
    title: str = "GeoScout episode dashboard",
) -> None:
    """One-page composite: GT pcd vs predicted pcd, trajectory, CR curve,
    action history. Best for at-a-glance inspection of one episode.

    Layout:
        +---------------+---------------+
        |   3D viz      |  CR curve     |
        |   (GT + pred  |               |
        |    + traj)    +---------------+
        |               |  action       |
        |               |  heatmap      |
        +---------------+---------------+
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return
    if path is None:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eyes = _y_up_to_z_up(_to_numpy(eye_positions))
    ats = _y_up_to_z_up(_to_numpy(look_ats))
    bb_min_v = _y_up_to_z_up(np.asarray(bbox_min))
    bb_max_v = _y_up_to_z_up(np.asarray(bbox_max))
    bb_min_v, bb_max_v = np.minimum(bb_min_v, bb_max_v), np.maximum(bb_min_v, bb_max_v)
    cr = _to_numpy(cr_history)
    actions = _to_numpy(action_indices)
    T = len(eyes)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0])

    # 3D panel: shaded mesh (preferred) + voxel-density predicted pcd + trajectory.
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    drew_mesh = False
    if mesh_path is not None:
        drew_mesh = _draw_shaded_mesh(ax3d, mesh_path)
    if not drew_mesh and gt_pointcloud is not None:
        gt = _y_up_to_z_up(_to_numpy(gt_pointcloud))
        if gt.shape[0] > 5000:
            gt = gt[np.random.choice(gt.shape[0], 5000, replace=False)]
        ax3d.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=1.5, c="tab:gray",
                     alpha=0.6, label=f"GT ({len(gt)})")
    if pred_pointcloud is not None and len(pred_pointcloud) > 0:
        pp = _y_up_to_z_up(_to_numpy(pred_pointcloud))
        # Voxel-density on pcd's own tight bbox (auto-derived) at 50³ →
        # ~2cm voxels for a 1m chair, so backs / legs are recognizable.
        _voxel_density_render(ax3d, pp)
    sc = ax3d.scatter(eyes[:, 0], eyes[:, 1], eyes[:, 2],
                      c=np.arange(T), cmap="plasma", s=40, edgecolors="k",
                      label="camera")
    ax3d.plot(eyes[:, 0], eyes[:, 1], eyes[:, 2], "k-", alpha=0.3)
    for eye, at in zip(eyes, ats):
        ax3d.plot([eye[0], at[0]], [eye[1], at[1]], [eye[2], at[2]],
                  "r-", alpha=0.15, linewidth=0.4)
    x_min, y_min, z_min = bb_min_v
    x_max, y_max, z_max = bb_max_v
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax3d.plot(*zip(corners[i], corners[j]), "b-", alpha=0.25)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z (up)")
    ax3d.set_title(f"GT vs predicted pcd (cr={float(cr[-1]) if len(cr) > 0 else 0:.3f}, "
                   f"T={T})")
    ax3d.legend(loc="upper right", fontsize=8)

    # CR curve.
    ax_cr = fig.add_subplot(gs[0, 1])
    ax_cr.plot(cr, "g-o", markersize=3)
    ax_cr.set_xlabel("step")
    ax_cr.set_ylabel("Coverage Ratio")
    ax_cr.set_ylim(0, 1)
    ax_cr.grid(alpha=0.3)
    ax_cr.set_title(f"CR trajectory (final={float(cr[-1]) if len(cr) > 0 else 0:.3f})")

    # Action heatmap (one row per active action dim, normalized).
    ax_act = fig.add_subplot(gs[1, 1])
    if actions.ndim == 2 and actions.shape[1] == 6:
        # Skip the frozen roll dim (idx 3) for clarity.
        active_dims = [0, 1, 2, 4, 5]
        labels = ["x", "y", "z", "pitch", "yaw"]
        norm_idx = np.stack([
            actions[:, d] / max(1, nvec[d] - 1) for d in active_dims
        ], axis=0)
        ax_act.imshow(norm_idx, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_act.set_yticks(range(len(active_dims)))
        ax_act.set_yticklabels(labels)
        ax_act.set_xlabel("step")
        ax_act.set_title("Action indices (normalized to [0,1])")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ============================================================================
# Professional-grade 3D-vision viz (camera frustums, filmstrip, coverage map)
# ============================================================================

# Default frustum size as fraction of action-box max-extent. Large enough
# to read at-a-glance, small enough not to clutter.
_FRUSTUM_DEPTH_FRAC = 0.10


def _camera_basis_from_eye_at(eye: np.ndarray, at: np.ndarray):
    """Standard look-at basis (right-handed) — used in PLOT frame
    (Z-up). Inputs eye/at are already rotated from world Y-up to plot
    Z-up by the caller. Falls back to +Y if look ≈ ±Z to avoid a
    degenerate cross product.
    """
    forward = at - eye
    fn = np.linalg.norm(forward)
    forward = forward / max(fn, 1e-8)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right = right / max(np.linalg.norm(right), 1e-8)
    true_up = np.cross(forward, right)
    return right, true_up, forward


def _draw_camera_frustum(
    ax,
    eye: np.ndarray,
    at: np.ndarray,
    color="black",
    depth: float = 0.1,
    aspect: float = 1.0,
    fov_deg: float = 60.0,
    linewidth: float = 0.8,
    alpha: float = 0.9,
) -> None:
    """Draw a single open camera-frustum (8 line segments) on a 3D
    matplotlib axis.

    Frustum geometry:
        apex = eye
        far-plane corners at distance `depth` along forward, sized by
        the camera's FoV. The 4 apex→corner edges + 4 corner-corner
        rectangle edges = 8 line segments. Optional red top-edge
        marker disambiguates orientation (which way is "up").
    """
    right, up, forward = _camera_basis_from_eye_at(eye, at)
    half_h = depth * np.tan(np.deg2rad(fov_deg) * 0.5)
    half_w = half_h * aspect

    centre = eye + forward * depth
    tl = centre - right * half_w + up * half_h
    tr = centre + right * half_w + up * half_h
    bl = centre - right * half_w - up * half_h
    br = centre + right * half_w - up * half_h

    # 4 apex→corner rays.
    for corner in (tl, tr, br, bl):
        ax.plot([eye[0], corner[0]], [eye[1], corner[1]], [eye[2], corner[2]],
                color=color, linewidth=linewidth, alpha=alpha)
    # Far-plane rectangle (clockwise tl→tr→br→bl→tl).
    quad = [tl, tr, br, bl, tl]
    for a, b in zip(quad[:-1], quad[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, linewidth=linewidth, alpha=alpha)
    # Red "up" edge (top-left → top-right) marks orientation.
    ax.plot([tl[0], tr[0]], [tl[1], tr[1]], [tl[2], tr[2]],
            color="crimson", linewidth=linewidth * 1.4, alpha=min(1.0, alpha + 0.1))


def _draw_shaded_mesh(ax, mesh_path, *, alpha: float = 0.65,
                      base_color=(0.78, 0.78, 0.84)):
    """Render a triangulated mesh on a 3D matplotlib axis with
    Lambertian shading from a fixed light direction. This replaces the
    pure-pcd "gray scatter" with a proper shaded surface — much more
    legible for understanding which faces of the object the cameras
    actually see.

    No-ops silently when trimesh / mpl_toolkits aren't available."""
    try:
        import trimesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return False
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if hasattr(mesh, "geometry") and not hasattr(mesh, "vertices"):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        return False
    if faces.size == 0:
        return False

    # Y-up world → Z-up plot frame (so the chair stands upright in mpl 3D).
    verts = _y_up_to_z_up(verts)

    # Per-face triangle vertices [F, 3, 3].
    tri = verts[faces]
    # Lambertian shading: face brightness = max(0, n·L).
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    n = n / n_norm
    light_dir = np.array([0.4, 0.6, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shade = np.clip(np.abs(n @ light_dir), 0.25, 1.0)         # double-sided
    base = np.array(base_color)
    face_colors = (shade[:, None] * base[None, :]).clip(0, 1)
    face_colors = np.concatenate([face_colors, np.full((len(face_colors), 1), alpha)], axis=1)

    poly = Poly3DCollection(tri, facecolors=face_colors,
                             edgecolors="none", linewidths=0.0)
    ax.add_collection3d(poly)
    return True


def save_camera_trajectory_pro(
    eye_positions: Sequence,
    look_ats: Sequence,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    path,
    object_pointcloud=None,
    mesh_path=None,                  # NEW: render a shaded mesh as background
    fov_deg: float = 60.0,
    title: str = "Camera trajectory",
) -> None:
    """Camera-frustum trajectory plot — replacement for
    `save_trajectory_plot`. Each view is drawn as an open 8-line
    frustum with the top edge highlighted in crimson; centres are
    connected by a thin polyline; the colormap (viridis) tracks step
    index. Plus EITHER a shaded mesh (preferred, when `mesh_path`
    given) OR a GT-pcd overlay (fallback) for the object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return
    eyes = _y_up_to_z_up(_to_numpy(eye_positions))
    ats = _y_up_to_z_up(_to_numpy(look_ats))
    bb_min_v = _y_up_to_z_up(np.asarray(bbox_min))
    bb_max_v = _y_up_to_z_up(np.asarray(bbox_max))
    bb_min_v, bb_max_v = np.minimum(bb_min_v, bb_max_v), np.maximum(bb_min_v, bb_max_v)
    T = len(eyes)
    if T == 0:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = matplotlib.colormaps["viridis"]

    # Object visualization: shaded mesh preferred; pcd as voxel-density fallback.
    drew_mesh = False
    if mesh_path is not None:
        drew_mesh = _draw_shaded_mesh(ax, mesh_path)
    if not drew_mesh and object_pointcloud is not None:
        pc = _y_up_to_z_up(_to_numpy(object_pointcloud))
        _voxel_density_render(ax, pc)

    # Centres polyline.
    ax.plot(eyes[:, 0], eyes[:, 1], eyes[:, 2], color="0.4", linewidth=0.8, alpha=0.75)

    # Frustum-depth: 10% of the larger action-box span — readable but not bloated.
    box_span = float(np.max(bb_max_v - bb_min_v))
    frustum_depth = box_span * _FRUSTUM_DEPTH_FRAC

    for t, (eye, at) in enumerate(zip(eyes, ats)):
        col = cmap(t / max(T - 1, 1))
        _draw_camera_frustum(
            ax, eye=eye, at=at, color=col, depth=frustum_depth,
            fov_deg=fov_deg, linewidth=0.85, alpha=0.9,
        )

    # Bbox wireframe.
    x_min, y_min, z_min = bb_min_v
    x_max, y_max, z_max = bb_max_v
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="steelblue",
                alpha=0.25, linewidth=0.7)

    # Colorbar for step index.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.07)
    cb.set_label("view index")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
    ax.set_title(f"{title} ({T} views)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_filmstrip(
    render_paths,
    path,
    n_cols: int = 8,
    thumbnail_px: int = 120,
    captions=None,
    title: str = "Per-view renders",
) -> None:
    """Lay out per-step rendered images as a grid filmstrip.

    `render_paths`: list of PNGs (env writes
    `step_NNN_render.png`). The composite already has rgb/depth/alpha
    side-by-side; we just thumbnail it and tile.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return
    paths = [Path(p) for p in render_paths if Path(p).exists()]
    if not paths:
        return
    N = len(paths)
    n_cols = max(1, min(n_cols, N))
    n_rows = int(np.ceil(N / n_cols))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 1.6), squeeze=False,
    )
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if k >= N:
            ax.axis("off")
            continue
        try:
            img = Image.open(paths[k])
            img.thumbnail((thumbnail_px * 4, thumbnail_px))
            ax.imshow(np.asarray(img))
        except Exception:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", fontsize=6)
        cap = captions[k] if captions is not None and k < len(captions) else f"step {k+1}"
        ax.set_title(cap, fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_coverage_heatmap(
    grid_gt,             # [G, G, G] binary GT occupancy
    scanned_history,     # [T, G, G, G] cumulative scanned grids per step,
                          # OR [G, G, G] final scanned mask
    range_gt,            # [6] (xmax_centre, xmin_c, ymax_c, ymin_c, zmax_c, zmin_c)
    voxel_size,          # [3]
    path,
    title: str = "Coverage map",
) -> None:
    """3D scatter of GT voxel centres colored by VISIT TIME (first step
    each voxel was observed). Voxels never seen are drawn pale gray;
    voxels seen at step k are colored by k via viridis.

    `scanned_history` may be either:
      - [T, G, G, G] sticky mask per step (preferred — gives per-voxel
        first-seen step); or
      - [G, G, G] final mask (we just use 0/1 colors).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return
    gg = _to_numpy(grid_gt) > 0.5
    sh = _to_numpy(scanned_history)
    rg = np.asarray(range_gt).ravel()
    vs = np.asarray(voxel_size).ravel()
    G = gg.shape[0]
    # Voxel centres in world frame.
    ii, jj, kk = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    mins = np.array([rg[1], rg[3], rg[5]], dtype=np.float32)   # voxel-CENTRE convention
    cx_w = mins[0] + ii * vs[0]
    cy_w = mins[1] + jj * vs[1]
    cz_w = mins[2] + kk * vs[2]
    # Y-up world → Z-up plot frame.
    cx = cx_w
    cy = -cz_w
    cz = cy_w

    # First-seen step per voxel (T * gt mask not yet seen → 0; else step+1).
    if sh.ndim == 4:
        T = sh.shape[0]
        seen = sh > 0.5                                              # [T, G, G, G]
        any_seen = seen.any(axis=0)                                  # [G, G, G]
        first_step = np.argmax(seen, axis=0)                         # [G, G, G]
        first_step = np.where(any_seen, first_step, T)               # T = "never seen"
    else:
        T = 1
        any_seen = sh > 0.5
        first_step = np.where(any_seen, 0, 1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = matplotlib.colormaps["viridis"]

    # Unseen GT voxels — pale gray (so user sees what's still missing).
    unseen_mask = gg & ~any_seen
    if unseen_mask.any():
        ax.scatter(cx[unseen_mask], cy[unseen_mask], cz[unseen_mask],
                   s=18, c="0.7", marker="s", alpha=0.55, depthshade=False,
                   label=f"unseen ({int(unseen_mask.sum())})")
    # Seen GT voxels — colored by first-seen step.
    seen_mask = gg & any_seen
    if seen_mask.any():
        c = first_step[seen_mask] / max(T - 1, 1) if T > 1 else np.zeros(int(seen_mask.sum()))
        sc = ax.scatter(cx[seen_mask], cy[seen_mask], cz[seen_mask],
                        s=22, c=c, cmap=cmap, vmin=0, vmax=1, marker="s",
                        alpha=0.95, depthshade=True,
                        label=f"seen ({int(seen_mask.sum())})")
        cb = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.06)
        cb.set_label("first-seen step (norm.)")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
    ratio = float(seen_mask.sum() / max(gg.sum(), 1))
    ax.set_title(f"{title} — covered {ratio:.1%}")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
