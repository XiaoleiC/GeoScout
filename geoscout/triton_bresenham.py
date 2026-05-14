"""Triton kernels for exact 3D Bresenham free-space scatter.

The PyTorch fallback in voxel_utils.bresenham3D_strict returns a large
``[num_path_voxels, 3]`` tensor and then deduplicates it. For 400x400
GeoScout renders this can mean hundreds of thousands of rays and tens of
millions of intermediate path rows per step.

This module keeps the same integer Bresenham update order but scatters
visited voxels directly into a flat uint8 mask. Duplicate rays therefore
collapse naturally as set-union writes of the value 1.
"""
from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - availability depends on the CUDA image.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:  # pragma: no cover - exercised on Modal CUDA.

    @triton.jit
    def _bresenham_scatter_kernel_blocked(
        env_ids,
        sources,
        targets,
        out_mask,
        num_rays: tl.constexpr,
        grid_size: tl.constexpr,
        max_steps: tl.constexpr,
        include_source: tl.constexpr,
        include_target: tl.constexpr,
        block_rays: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_rays + tl.arange(0, block_rays)
        active_ray = offsets < num_rays

        env = tl.load(env_ids + offsets, mask=active_ray, other=0)
        base = offsets * 3
        x = tl.load(sources + base + 0, mask=active_ray, other=0)
        y = tl.load(sources + base + 1, mask=active_ray, other=0)
        z = tl.load(sources + base + 2, mask=active_ray, other=0)
        tx = tl.load(targets + base + 0, mask=active_ray, other=0)
        ty = tl.load(targets + base + 1, mask=active_ray, other=0)
        tz = tl.load(targets + base + 2, mask=active_ray, other=0)

        dx0 = tx - x
        dy0 = ty - y
        dz0 = tz - z
        dx = tl.abs(dx0)
        dy = tl.abs(dy0)
        dz = tl.abs(dz0)
        sx = tl.where(dx0 > 0, 1, tl.where(dx0 < 0, -1, 0))
        sy = tl.where(dy0 > 0, 1, tl.where(dy0 < 0, -1, 0))
        sz = tl.where(dz0 > 0, 1, tl.where(dz0 < 0, -1, 0))

        drive_x = (dx >= dy) & (dx >= dz)
        drive_y = (~drive_x) & (dy >= dz)
        drive_z = (~drive_x) & (~drive_y)

        da = tl.where(drive_x, dx, tl.where(drive_y, dy, dz))
        sa = tl.where(drive_x, sx, tl.where(drive_y, sy, sz))

        b_is_x = drive_z
        b_is_y = drive_x
        b_is_z = drive_y
        c_is_x = drive_y
        c_is_y = drive_z
        c_is_z = drive_x

        db = tl.where(b_is_x, dx, tl.where(b_is_y, dy, dz))
        dc = tl.where(c_is_x, dx, tl.where(c_is_y, dy, dz))
        sb = tl.where(b_is_x, sx, tl.where(b_is_y, sy, sz))
        sc = tl.where(c_is_x, sx, tl.where(c_is_y, sy, sz))
        p1 = 2 * db - da
        p2 = 2 * dc - da

        same_as_target = (x == tx) & (y == ty) & (z == tz)
        in_bounds = (
            active_ray
            & (x >= 0)
            & (x < grid_size)
            & (y >= 0)
            & (y < grid_size)
            & (z >= 0)
            & (z < grid_size)
        )
        if include_source:
            source_store = in_bounds
            if not include_target:
                source_store = source_store & (~same_as_target)
            flat = env * grid_size * grid_size * grid_size + x * grid_size * grid_size + y * grid_size + z
            tl.store(out_mask + flat, 1, mask=source_store)

        for step in range(max_steps):
            step_active = active_ray & (step < da)
            mask1 = (p1 >= 0) & step_active & (da > 0)
            mask2 = (p2 >= 0) & step_active & (da > 0)

            x += tl.where(mask1 & b_is_x, sb, 0)
            y += tl.where(mask1 & b_is_y, sb, 0)
            z += tl.where(mask1 & b_is_z, sb, 0)
            p1 = tl.where(mask1, p1 - 2 * da, p1)

            x += tl.where(mask2 & c_is_x, sc, 0)
            y += tl.where(mask2 & c_is_y, sc, 0)
            z += tl.where(mask2 & c_is_z, sc, 0)
            p2 = tl.where(mask2, p2 - 2 * da, p2)

            x += tl.where(step_active & drive_x, sa, 0)
            y += tl.where(step_active & drive_y, sa, 0)
            z += tl.where(step_active & drive_z, sa, 0)
            p1 = tl.where(step_active, p1 + 2 * db, p1)
            p2 = tl.where(step_active, p2 + 2 * dc, p2)

            same_as_target = (x == tx) & (y == ty) & (z == tz)
            store_mask = (
                step_active
                & (x >= 0)
                & (x < grid_size)
                & (y >= 0)
                & (y < grid_size)
                & (z >= 0)
                & (z < grid_size)
            )
            if not include_target:
                store_mask = store_mask & (~same_as_target)
            flat = env * grid_size * grid_size * grid_size + x * grid_size * grid_size + y * grid_size + z
            tl.store(out_mask + flat, 1, mask=store_mask)

    @triton.jit
    def _bresenham_scatter_kernel(
        env_ids,
        sources,
        targets,
        out_mask,
        num_rays: tl.constexpr,
        grid_size: tl.constexpr,
        max_steps: tl.constexpr,
        include_source: tl.constexpr,
        include_target: tl.constexpr,
    ):
        pid = tl.program_id(0)
        active_ray = pid < num_rays

        env = tl.load(env_ids + pid, mask=active_ray, other=0)
        base = pid * 3
        x = tl.load(sources + base + 0, mask=active_ray, other=0)
        y = tl.load(sources + base + 1, mask=active_ray, other=0)
        z = tl.load(sources + base + 2, mask=active_ray, other=0)
        tx = tl.load(targets + base + 0, mask=active_ray, other=0)
        ty = tl.load(targets + base + 1, mask=active_ray, other=0)
        tz = tl.load(targets + base + 2, mask=active_ray, other=0)

        dx0 = tx - x
        dy0 = ty - y
        dz0 = tz - z
        dx = tl.abs(dx0)
        dy = tl.abs(dy0)
        dz = tl.abs(dz0)
        sx = tl.where(dx0 > 0, 1, tl.where(dx0 < 0, -1, 0))
        sy = tl.where(dy0 > 0, 1, tl.where(dy0 < 0, -1, 0))
        sz = tl.where(dz0 > 0, 1, tl.where(dz0 < 0, -1, 0))

        drive_x = (dx >= dy) & (dx >= dz)
        drive_y = (~drive_x) & (dy >= dz)
        drive_z = (~drive_x) & (~drive_y)

        da = tl.where(drive_x, dx, tl.where(drive_y, dy, dz))
        sa = tl.where(drive_x, sx, tl.where(drive_y, sy, sz))

        # other_axes = [(drive_axis + 1) % 3, (drive_axis + 2) % 3]
        b_is_x = drive_z
        b_is_y = drive_x
        b_is_z = drive_y
        c_is_x = drive_y
        c_is_y = drive_z
        c_is_z = drive_x

        db = tl.where(b_is_x, dx, tl.where(b_is_y, dy, dz))
        dc = tl.where(c_is_x, dx, tl.where(c_is_y, dy, dz))
        sb = tl.where(b_is_x, sx, tl.where(b_is_y, sy, sz))
        sc = tl.where(c_is_x, sx, tl.where(c_is_y, sy, sz))
        p1 = 2 * db - da
        p2 = 2 * dc - da

        same_as_target = (x == tx) & (y == ty) & (z == tz)
        in_bounds = (
            active_ray
            & (x >= 0)
            & (x < grid_size)
            & (y >= 0)
            & (y < grid_size)
            & (z >= 0)
            & (z < grid_size)
        )
        if include_source:
            source_store = in_bounds
            if not include_target:
                source_store = source_store & (~same_as_target)
            flat = env * grid_size * grid_size * grid_size + x * grid_size * grid_size + y * grid_size + z
            tl.store(out_mask + flat, 1, mask=source_store)

        for step in range(max_steps):
            step_active = active_ray & (step < da)
            mask1 = (p1 >= 0) & step_active & (da > 0)
            mask2 = (p2 >= 0) & step_active & (da > 0)

            x += tl.where(mask1 & b_is_x, sb, 0)
            y += tl.where(mask1 & b_is_y, sb, 0)
            z += tl.where(mask1 & b_is_z, sb, 0)
            p1 = tl.where(mask1, p1 - 2 * da, p1)

            x += tl.where(mask2 & c_is_x, sc, 0)
            y += tl.where(mask2 & c_is_y, sc, 0)
            z += tl.where(mask2 & c_is_z, sc, 0)
            p2 = tl.where(mask2, p2 - 2 * da, p2)

            x += tl.where(step_active & drive_x, sa, 0)
            y += tl.where(step_active & drive_y, sa, 0)
            z += tl.where(step_active & drive_z, sa, 0)
            p1 = tl.where(step_active, p1 + 2 * db, p1)
            p2 = tl.where(step_active, p2 + 2 * dc, p2)

            same_as_target = (x == tx) & (y == ty) & (z == tz)
            store_mask = (
                step_active
                & (x >= 0)
                & (x < grid_size)
                & (y >= 0)
                & (y < grid_size)
                & (z >= 0)
                & (z < grid_size)
            )
            if not include_target:
                store_mask = store_mask & (~same_as_target)
            flat = env * grid_size * grid_size * grid_size + x * grid_size * grid_size + y * grid_size + z
            tl.store(out_mask + flat, 1, mask=store_mask)

    @triton.jit
    def _apply_free_mask_kernel(
        prob_grid,
        free_mask,
        numel: tl.constexpr,
        delta: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        active = offsets < numel
        mask_val = tl.load(free_mask + offsets, mask=active, other=0)
        prob_val = tl.load(prob_grid + offsets, mask=active, other=0.0)
        prob_val += tl.where(mask_val != 0, delta, 0.0)
        tl.store(prob_grid + offsets, prob_val, mask=active)


def scatter_bresenham3d_to_mask(
    env_ids: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_envs: int,
    grid_size: int,
    include_source: bool,
    include_target: bool,
    out_mask: Optional[torch.Tensor] = None,
    max_steps: Optional[int] = None,
    block_rays: int = 64,
) -> Optional[torch.Tensor]:
    """Scatter exact Bresenham paths into ``[num_envs, G, G, G]`` uint8 mask.

    Returns ``None`` when Triton/CUDA is unavailable so callers can fall
    back to the pure PyTorch implementation.
    """
    if (
        not _TRITON_AVAILABLE
        or not env_ids.is_cuda
        or not sources.is_cuda
        or not targets.is_cuda
    ):
        return None

    if targets.numel() == 0:
        if out_mask is None:
            return torch.zeros(
                (int(num_envs), int(grid_size), int(grid_size), int(grid_size)),
                dtype=torch.uint8,
                device=targets.device,
            )
        return out_mask

    env_ids = env_ids.to(dtype=torch.int64, device=targets.device).contiguous().view(-1)
    sources = sources.to(dtype=torch.int64, device=targets.device).contiguous().view(-1, 3)
    targets = targets.to(dtype=torch.int64, device=targets.device).contiguous().view(-1, 3)
    if sources.shape != targets.shape:
        raise ValueError(f"sources {tuple(sources.shape)} != targets {tuple(targets.shape)}")
    if env_ids.shape[0] != targets.shape[0]:
        raise ValueError(f"env_ids {tuple(env_ids.shape)} incompatible with targets {tuple(targets.shape)}")

    G = int(grid_size)
    if out_mask is None:
        out_mask = torch.zeros((int(num_envs), G, G, G), dtype=torch.uint8, device=targets.device)
    else:
        if out_mask.dtype != torch.uint8 or not out_mask.is_cuda:
            raise ValueError("out_mask must be a CUDA uint8 tensor")

    n_rays = int(targets.shape[0])
    steps = int(max_steps if max_steps is not None else 3 * G)
    br = max(1, int(block_rays))
    _bresenham_scatter_kernel_blocked[(triton.cdiv(n_rays, br),)](
        env_ids,
        sources,
        targets,
        out_mask.view(-1),
        num_rays=n_rays,
        grid_size=G,
        max_steps=steps,
        include_source=bool(include_source),
        include_target=bool(include_target),
        block_rays=br,
    )
    return out_mask


def apply_free_mask_to_grid(
    prob_grid: torch.Tensor,
    free_mask: torch.Tensor,
    *,
    delta: float,
    block_size: int = 256,
) -> bool:
    """Apply one set-union free-space decrement from ``free_mask``.

    ``free_mask`` is uint8 and already represents the Section 7 set
    union: each free voxel is marked once, regardless of how many rays
    traversed it. This kernel only replaces the expensive
    ``nonzero(mask) -> index_put_(..., accumulate=True)`` writeback with
    a dense in-place pass over the grid.
    """
    if (
        not _TRITON_AVAILABLE
        or not prob_grid.is_cuda
        or not free_mask.is_cuda
        or free_mask.dtype != torch.uint8
        or prob_grid.dtype != torch.float32
        or prob_grid.numel() != free_mask.numel()
        or not prob_grid.is_contiguous()
        or not free_mask.is_contiguous()
    ):
        return False

    n = int(prob_grid.numel())
    if n == 0:
        return True
    bs = int(block_size)
    _apply_free_mask_kernel[(triton.cdiv(n, bs),)](
        prob_grid.view(-1),
        free_mask.view(-1),
        numel=n,
        delta=float(delta),
        block_size=bs,
    )
    return True
