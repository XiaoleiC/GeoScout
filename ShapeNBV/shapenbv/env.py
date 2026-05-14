"""ShapeNBVEnv — single-object NBV gym env for ShapeNBV.

Algorithm lineage from `gennbv/env/env_train_gennbv.py`:
  - Action is `MultiDiscrete([81, 81, 81, 1, 13, 13])` over (x, y, z, roll,
    pitch, yaw). Roll frozen. Position bounds rescaled from House3K (16m
    room) to ShapeNet's canonical Cube Mode bbox (≈ unit cube).
  - Per step: render depth+alpha, back-project to world points, voxelize
    to indices, ray-cast from camera voxel to each target voxel via
    Bresenham, update log-odds prob_grid (free = -0.05, occupied = +1.0).
  - Reward = (Δ coverage × 20) + short_path_penalty(×0.1) + success_bonus(1)
    - collision_penalty(10), with `only_positive_rewards=True` clipping the
    per-step total before terminal success/collision terms are applied.
  - Termination on collision OR step ≥ 100 OR cr > 0.99.

Renderer: pure-PyTorch mesh raycaster (depth + alpha only — no shading).
Collision: mesh-level point-inside test plus GT surface-voxel fallback.
"""
from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from .mesh_renderer import MeshSequenceRenderer, ShapeNetIndex
from .preprocess import load_preproc
from .voxel_utils import (
    bresenham3D_strict,
    grid_occupancy_tri_cls,
    pose_coord_to_idx_3D,
    scanned_pts_to_idx_3D,
)
from .viz import (
    save_action_history_plot,
    save_camera_trajectory_pro,
    save_combined_dashboard,
    save_coverage_curve,
    save_coverage_heatmap,
    save_filmstrip,
    save_occupancy_grid_slices,
    save_pointcloud_ply,
    save_trajectory_plot,
)


POSE_DIM = 6   # (x, y, z, roll, pitch, yaw)

# ShapeNet rescaling of GenNBV's House3K action grid.
# GenNBV (House3K, 16m × 16m × 10m room):
#   action_low_world = [-8., -8., 0.1, 0., -π/2, 0.]
#   action_unit      = [0.2, 0.2, 0.2, 0., π/12, π/6]
#   clip_pose_idx_up = [80, 80, 50, 0, 12, 12]
# ShapeNBV (ShapeNet model_normalized.obj, ≈ [-0.5, 0.5]³ canonical):
#   action_low_world = [-1.0, -1.0, -1.0, 0., -π/2, 0.]
#   action_unit      = [0.025, 0.025, 0.025, 0., π/12, π/6]
#   clip_pose_idx_up = [80, 80, 80, 0, 12, 12]    ← symmetric in z
DEFAULT_ACTION_LOW_WORLD = np.array(
    [-1.0, -1.0, -1.0, 0.0, -math.pi / 2, 0.0], dtype=np.float32
)
DEFAULT_ACTION_UNIT = np.array(
    [0.025, 0.025, 0.025, 0.0, math.pi / 12, math.pi / 6], dtype=np.float32
)
DEFAULT_CLIP_POSE_IDX_UP = np.array([80, 80, 80, 0, 12, 12], dtype=np.int64)
DEFAULT_CLIP_POSE_IDX_LOW = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
# MultiDiscrete `nvec` = (idx_up - idx_low + 1).
NVEC = (DEFAULT_CLIP_POSE_IDX_UP - DEFAULT_CLIP_POSE_IDX_LOW + 1).astype(np.int64)

# Log-odds rule (GenNBV `update_occ_grid`).
LOG_ODDS_FREE = -0.05
LOG_ODDS_OCCUPIED = 1.0
LOG_ODDS_CLAMP = 5.0

# Reward scales — GenNBV config values after the IsaacGym dt multiplier.
REWARD_SCALE_SURFACE_COVERAGE = 20.0
REWARD_SCALE_SHORT_PATH = 0.1
REWARD_SCALE_TERMINATION = 1.0
REWARD_SCALE_COLLISION = 10.0

DEFAULT_GRID_SIZE = 128
DEFAULT_OBS_GRID_SIZE = 32
GRID_SIZE = DEFAULT_GRID_SIZE
DEFAULT_BUFFER_SIZE = 100
DEFAULT_EPISODE_LEN = 100
DEFAULT_CR_SUCCESS = 0.99
DEFAULT_SHORT_PATH_GRACE = 30
DEFAULT_SHORT_PATH_CLIP = 2.0


class ShapeNBVEnv(gym.Env):
    """One-object NBV environment.

    Each episode draws a (synset, model_id) from the index, loads its
    pre-computed `grid_gt`, and runs up to `episode_len` discrete-action
    steps. ShapeNBV keeps GenNBV's action/reward structure while adding
    ShapeNet-specific high-resolution GT, observation downsampling,
    miss-ray free-space updates, and mesh-based collision checks.

    Args:
        index: a `ShapeNetIndex` with mesh paths + render config.
        preproc_dir: directory of `<name>.pt` files produced by
            `scripts/preprocess.py`.
        sequence_names: optional restriction on entries this env draws
            from at reset. Defaults to all entries in `index`.
        device: torch device for renderer + voxel ops.
        buffer_size: pose history length (matches GenNBV's `stack=100`).
        episode_len: max steps per episode.
        grid_size: reward / GT voxel grid resolution.
        obs_grid_size: policy observation grid resolution. Defaults to
            32 so coverage can stay high-resolution without feeding a
            128^3 dense grid into the policy network.
        action_low_world / action_unit / clip_pose_idx_*: per-dim grid
            params; rescaled defaults provided for ShapeNet.
        cr_success_threshold: cr above this triggers the success bonus.
        coverage_reward_scale / short_path_scale / termination_bonus:
            effective reward scales aligned to GenNBV after its dt
            multiplier.
        collision_penalty: penalty for invalid camera centers.
        short_path_grace / short_path_clip: GenNBV's short-path penalty.
        only_positive_rewards: clip reward total at 0 BEFORE adding the
            termination bonus.
        seed: rng seed for sequence picking.
        debug_dir: when set, dumps per-step + per-episode viz.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        index: ShapeNetIndex,
        preproc_dir: str,
        sequence_names: Optional[List[str]] = None,
        device: str = "cuda",
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        episode_len: int = DEFAULT_EPISODE_LEN,
        grid_size: int = DEFAULT_GRID_SIZE,
        obs_grid_size: int = DEFAULT_OBS_GRID_SIZE,
        action_low_world: np.ndarray = DEFAULT_ACTION_LOW_WORLD,
        action_unit: np.ndarray = DEFAULT_ACTION_UNIT,
        clip_pose_idx_up: np.ndarray = DEFAULT_CLIP_POSE_IDX_UP,
        clip_pose_idx_low: np.ndarray = DEFAULT_CLIP_POSE_IDX_LOW,
        cr_success_threshold: float = DEFAULT_CR_SUCCESS,
        coverage_reward_scale: float = REWARD_SCALE_SURFACE_COVERAGE,
        short_path_grace: int = DEFAULT_SHORT_PATH_GRACE,
        short_path_clip: float = DEFAULT_SHORT_PATH_CLIP,
        short_path_scale: float = REWARD_SCALE_SHORT_PATH,
        termination_bonus: float = REWARD_SCALE_TERMINATION,
        collision_penalty: float = REWARD_SCALE_COLLISION,
        only_positive_rewards: bool = True,
        # Caption channel: dimension of the per-episode `caption_emb` that
        # gets concatenated to the observation. 0 = disabled (Phase 0
        # baseline). 384 = sentence-transformers MiniLM-L6 (Phase 1).
        # Source of `caption_emb`: per-object preproc dict's `caption_emb`
        # field (precomputed from `precompute_category_embeddings.py`),
        # falling back to zeros if missing.
        caption_dim: int = 0,
        # Force the camera to look at the object centre (origin in
        # canonical frame) every step, ignoring the policy's
        # roll/pitch/yaw indices. Mirrors `tensor_env.TensorBatchEnv`'s
        # flag of the same name; required at validation time when
        # training used auto_lookat_center, otherwise the policy's
        # untrained pitch/yaw heads point the camera at random.
        auto_lookat_center: bool = False,
        # When True, skip the free-voxel bresenham raycast on every step.
        # The policy still gets the occupied / unknown signal in
        # `prob_grid`, just no "free" decrement along the ray. Use for
        # smoke tests / single-object debugging where speed matters and
        # the explored-vs-unexplored distinction is less informative.
        skip_free_raycast: bool = False,
        # When free raycasting is enabled, also traverse alpha-miss rays
        # through the current grid AABB. This captures the information
        # in empty images: rays that pass through the ROI without a hit
        # are free along their in-grid segment.
        update_empty_rays: bool = True,
        coverage_hit_dilate_radius: int = 1,
        seed: Optional[int] = None,
        debug_dir: Optional[str] = None,
        # Skip per-step .ply / occupancy-slice .png dumps. Per-episode
        # finalize artefacts (cr_curve, trajectory_pro, dashboard, ...)
        # still run. Validate dumps 360 small files per 12-ep run on
        # Modal volume — disabling these is ~5-10× faster, keeps the
        # dashboards intact.
        skip_step_dumps: bool = False,
    ):
        super().__init__()

        self.skip_free_raycast = bool(skip_free_raycast)
        self.update_empty_rays = bool(update_empty_rays)
        self.auto_lookat_center = bool(auto_lookat_center)
        self.caption_dim = int(caption_dim)
        self.skip_step_dumps = bool(skip_step_dumps)
        self.index = index
        self.preproc_dir = Path(preproc_dir)
        self.sequences: List[str] = sequence_names or list(index.sequence_names)
        if not self.sequences:
            raise ValueError("ShapeNBVEnv: no sequences provided.")
        self.device = torch.device(device)
        self.buffer_size = int(buffer_size)
        self.episode_len = int(episode_len)
        self.grid_size = int(grid_size)
        self.obs_grid_size = int(obs_grid_size) if int(obs_grid_size) > 0 else self.grid_size
        if self.obs_grid_size <= 0:
            raise ValueError(f"obs_grid_size must be positive, got {self.obs_grid_size}")
        if self.obs_grid_size > self.grid_size:
            raise ValueError(
                f"obs_grid_size={self.obs_grid_size} cannot exceed grid_size={self.grid_size}"
            )
        self.coverage_hit_dilate_radius = max(0, int(coverage_hit_dilate_radius))

        self.action_low_world = np.asarray(action_low_world, dtype=np.float32)
        self.action_unit = np.asarray(action_unit, dtype=np.float32)
        self.clip_pose_idx_up = np.asarray(clip_pose_idx_up, dtype=np.int64)
        self.clip_pose_idx_low = np.asarray(clip_pose_idx_low, dtype=np.int64)

        self.cr_success_threshold = float(cr_success_threshold)
        self.coverage_reward_scale = float(coverage_reward_scale)
        self.short_path_grace = int(short_path_grace)
        self.short_path_clip = float(short_path_clip)
        self.short_path_scale = float(short_path_scale)
        self.termination_bonus = float(termination_bonus)
        self.collision_penalty = float(collision_penalty)
        self.only_positive_rewards = bool(only_positive_rewards)

        self.action_space = spaces.MultiDiscrete(
            (self.clip_pose_idx_up - self.clip_pose_idx_low + 1).tolist()
        )
        obs_dim = self.buffer_size * POSE_DIM + self.obs_grid_size ** 3 + self.caption_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state.
        self._renderer: Optional[MeshSequenceRenderer] = None
        self._preproc: Optional[dict] = None
        self._action_history: deque = deque(maxlen=self.buffer_size)
        self._step_idx: int = 0
        self._cr_prev: float = 0.0

        # Voxel grids — created on reset.
        self._prob_grid: Optional[torch.Tensor] = None
        self._scanned_gt_grid: Optional[torch.Tensor] = None
        self._grid_gt: Optional[torch.Tensor] = None
        self._range_gt: Optional[torch.Tensor] = None
        self._voxel_size: Optional[torch.Tensor] = None
        self._num_valid_gt: float = 1.0

        # Per-episode book-keeping.
        self._eye_history: List[np.ndarray] = []
        self._at_history: List[np.ndarray] = []
        self._cr_history: List[float] = []
        # Discrete action indices per step (for viz / debugging).  We keep
        # this in addition to the decoded pose history so the action-dim
        # heatmap in `viz.save_action_history_plot` can be drawn directly.
        self._action_idx_history: List[np.ndarray] = []
        # Per-step copies of `scanned_gt_grid` so the coverage-heatmap
        # viz can colour each voxel by FIRST-SEEN step. Size cap: at
        # At 128³ this can become large; only enable debug dumps for
        # short diagnostic rollouts.
        self._scanned_gt_history: List[np.ndarray] = []
        # Cached numpy views of the per-episode GT grid (set on reset)
        # so viz helpers don't repeatedly round-trip through GPU.
        self._range_gt_np: Optional[np.ndarray] = None
        self._voxel_size_np: Optional[np.ndarray] = None
        self._action_box_min: Optional[np.ndarray] = None
        self._action_box_max: Optional[np.ndarray] = None

        self._rng = random.Random(seed)

        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._episode_idx: int = -1
        self._cur_episode_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)

        if self.debug_dir is not None and self._cur_episode_dir is not None and self._cr_history:
            self._finalize_episode_dump()

        forced = (options or {}).get("seq_name") if options else None
        seq_name = forced if forced is not None else self._rng.choice(self.sequences)

        preproc_path = self.preproc_dir / f"{seq_name}.pt"
        if not preproc_path.exists():
            raise FileNotFoundError(
                f"ShapeNBVEnv: missing preproc {preproc_path}. "
                f"Run shapenbv.scripts.preprocess first."
            )
        self._preproc = load_preproc(preproc_path, map_location=self.device)
        self._renderer = self.index.get_or_build_renderer(
            sequence_name=seq_name,
            T_canon=self._preproc.get("T_canon"),
            device=self.device,
        )

        # Phase 1 caption: load per-object caption_emb (or zeros).
        if self.caption_dim > 0:
            ce = self._preproc.get("caption_emb")
            if ce is None:
                self._current_caption_emb = np.zeros(self.caption_dim, dtype=np.float32)
            else:
                ce = ce.detach().cpu().numpy().astype(np.float32).ravel()
                if ce.shape[0] != self.caption_dim:
                    raise ValueError(
                        f"caption_emb dim {ce.shape[0]} != env caption_dim "
                        f"{self.caption_dim} for {seq_name}"
                    )
                self._current_caption_emb = ce
        else:
            self._current_caption_emb = None

        self._episode_idx += 1
        if self.debug_dir is not None:
            self._cur_episode_dir = self.debug_dir / f"episode_{self._episode_idx:04d}"
            self._cur_episode_dir.mkdir(parents=True, exist_ok=True)

        grid_gt = self._preproc["grid_gt"].to(self.device).float()
        if tuple(grid_gt.shape) != (self.grid_size, self.grid_size, self.grid_size):
            raise ValueError(
                f"preproc grid shape {tuple(grid_gt.shape)} does not match "
                f"env grid_size={self.grid_size} for {preproc_path}. "
                f"Re-run preprocessing with --grid_size {self.grid_size}."
            )
        self._grid_gt = grid_gt.unsqueeze(0)
        self._range_gt = self._preproc["range_gt"].to(self.device).unsqueeze(0)
        self._voxel_size = self._preproc["voxel_size_gt"].to(self.device).unsqueeze(0)
        # Cache the numpy views (used by viz finalize, not the hot path).
        self._range_gt_np = self._preproc["range_gt"].detach().cpu().numpy().ravel()
        self._voxel_size_np = self._preproc["voxel_size_gt"].detach().cpu().numpy().ravel()
        self._num_valid_gt = max(1.0, float(self._preproc["num_valid_voxel_gt"].item()))

        G = self.grid_size
        self._prob_grid = torch.zeros(1, G, G, G, dtype=torch.float32, device=self.device)
        self._scanned_gt_grid = torch.zeros(1, G, G, G, dtype=torch.float32, device=self.device)

        self._action_history.clear()
        for _ in range(self.buffer_size):
            self._action_history.append(np.zeros(POSE_DIM, dtype=np.float32))

        self._step_idx = 0
        self._cr_prev = 0.0
        self._eye_history = []
        self._at_history = []
        self._cr_history = []
        self._action_idx_history = []
        self._scanned_gt_history = []
        # In-memory accumulated predicted points (used by validate when
        # skip_step_dumps=True; otherwise we glob the per-step .ply files).
        self._accumulated_pred: List[np.ndarray] = []

        idx_up = self.clip_pose_idx_up.astype(np.float32)
        idx_low = self.clip_pose_idx_low.astype(np.float32)
        self._action_box_min = (idx_low * self.action_unit + self.action_low_world)[:3]
        self._action_box_max = (idx_up * self.action_unit + self.action_low_world)[:3]

        # Initial pose: deterministic top-centre view when valid, then
        # random valid fallback. This matches TensorBatchEnv and keeps
        # validation observations aligned with training.
        init_action = self._sample_valid_init_action()
        init_pose6 = self._decode_action(init_action)
        self._push_action(init_pose6)
        self._action_idx_history.append(np.asarray(init_action, dtype=np.int64).copy())
        self._cr_history.append(0.0)
        self._eye_history.append(init_pose6[:3].copy())
        self._at_history.append(self._lookat_for_pose6(init_pose6))

        return self._build_observation(), {"seq_name": seq_name}

    def step(self, action: np.ndarray):
        assert self._renderer is not None, "Call reset() first."
        action = np.asarray(action, dtype=np.int64).clip(
            self.clip_pose_idx_low, self.clip_pose_idx_up
        )
        pose6 = self._decode_action(action)
        eye = pose6[:3]
        at = self._lookat_for_pose6(pose6)
        self._eye_history.append(eye.copy())
        self._at_history.append(at)

        collision = self._camera_collides_object(eye)
        terminated = False
        truncated = False
        self._step_idx += 1
        cr_now = self._cr_prev

        cov_delta = 0.0
        if collision:
            terminated = True
        else:
            position = torch.from_numpy(eye.astype(np.float32)).to(self.device)
            target = torch.from_numpy(at.astype(np.float32)).to(self.device)
            out = self._renderer.render(position_canon=position, look_at_canon=target)

            depth = out.depth
            alpha = out.alpha
            mask = alpha > 0.5
            empty_free_paths = None
            if not self.skip_free_raycast and self.update_empty_rays:
                empty_free_paths = self._empty_ray_free_voxel_paths(
                    alpha=alpha,
                    eye=position,
                    at=target,
                    fov_deg=self._renderer.fov_deg,
                    width=self._renderer.intrinsics.width,
                    height=self._renderer.intrinsics.height,
                )
            if mask.sum() > 0:
                pts_world = self._backproject_depth(
                    depth=depth, mask=mask, eye=position, at=target,
                    fov_deg=self._renderer.fov_deg,
                    width=self._renderer.intrinsics.width,
                    height=self._renderer.intrinsics.height,
                )
                self._update_grids(
                    pts_world=pts_world,
                    eye=position,
                    extra_free_paths=empty_free_paths,
                )
                if not self.skip_step_dumps:
                    self._dump_step_viz(pts_world)
                else:
                    # Cache points in memory so validate can build the
                    # predicted pcd without per-step .ply writes.
                    if pts_world.numel() > 0:
                        N = pts_world.shape[0]
                        if N > 5000:
                            sub = pts_world[torch.randperm(N, device=pts_world.device)[:5000]]
                        else:
                            sub = pts_world
                        self._accumulated_pred.append(sub.detach().cpu().numpy())
                cr_now = self._coverage_ratio()
                cov_delta = cr_now - self._cr_prev
                self._cr_prev = cr_now
            else:
                self._apply_free_paths(empty_free_paths)

            if cr_now > self.cr_success_threshold:
                terminated = True
            elif self._step_idx >= self.episode_len:
                truncated = True

        short_path = self._short_path_penalty()

        reward = (
            cov_delta * self.coverage_reward_scale
            + short_path * self.short_path_scale
        )
        if self.only_positive_rewards:
            reward = max(0.0, reward)
        success = (cr_now > self.cr_success_threshold) and not collision
        if success:
            reward += self.termination_bonus
        if collision:
            reward -= self.collision_penalty

        self._push_action(pose6)
        self._action_idx_history.append(np.asarray(action, dtype=np.int64).copy())
        self._cr_history.append(cr_now)
        # Snapshot the cumulative GT-coverage mask for the heatmap viz.
        if self.debug_dir is not None:
            self._scanned_gt_history.append(
                self._scanned_gt_grid[0].detach().cpu().numpy().astype(np.float32)
            )

        info = {
            "cr": cr_now,
            "covered_voxels": float(self._scanned_gt_grid.sum().item())
            if self._scanned_gt_grid is not None else 0.0,
            "collision": collision,
            "success_truncation": success,
            "early_stopped": success,
            "TimeLimit.truncated": truncated,
            "step_idx": self._step_idx,
            "reward_terms": {
                "surface_coverage": cov_delta,
                "short_path": short_path,
                "success_bonus": self.termination_bonus if success else 0.0,
                "collision_penalty": self.collision_penalty if collision else 0.0,
            },
        }
        if terminated or truncated:
            info["cr_history"] = list(self._cr_history)

        return self._build_observation(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # action / observation helpers
    # ------------------------------------------------------------------
    def _decode_action(self, action_idx: np.ndarray) -> np.ndarray:
        """[6] discrete indices → [6] (x, y, z, roll, pitch, yaw)."""
        action_idx = np.asarray(action_idx, dtype=np.float32)
        return action_idx * self.action_unit + self.action_low_world

    def _push_action(self, pose6: np.ndarray) -> None:
        self._action_history.append(pose6.astype(np.float32))

    def _build_observation(self) -> np.ndarray:
        action_arr = np.stack(list(self._action_history), axis=0).reshape(-1)
        tri_cls = grid_occupancy_tri_cls(self._prob_grid, return_tri_cls_only=True)
        if self.obs_grid_size != self.grid_size:
            tri_cls = self._downsample_tri_grid_for_obs(tri_cls)
        tri_cls_flat = tri_cls.reshape(-1).detach().cpu().numpy().astype(np.float32)
        parts = [action_arr.astype(np.float32), tri_cls_flat]
        if self.caption_dim > 0:
            parts.append(self._current_caption_emb)
        return np.concatenate(parts, axis=0)

    def _downsample_tri_grid_for_obs(self, tri: torch.Tensor) -> torch.Tensor:
        """Downsample {-1,0,+1} tri-class grid for the policy branch.

        Max pooling is intentionally conservative: occupied beats
        unknown/free, and unknown beats free. A coarse cell becomes free
        only when all fine cells inside it are free.
        """
        x = tri.unsqueeze(1)
        if self.grid_size % self.obs_grid_size == 0:
            k = self.grid_size // self.obs_grid_size
            return F.max_pool3d(x, kernel_size=k, stride=k).squeeze(1)
        return F.interpolate(
            x,
            size=(self.obs_grid_size, self.obs_grid_size, self.obs_grid_size),
            mode="nearest",
        ).squeeze(1)

    def _coverage_ratio(self) -> float:
        if self._scanned_gt_grid is None:
            return 0.0
        return float(self._scanned_gt_grid.sum().item()) / self._num_valid_gt

    def _short_path_penalty(self) -> float:
        """GenNBV `_reward_short_path`: -clip(step - grace, 0, clip)."""
        extra = max(0.0, self._step_idx - self.short_path_grace)
        return -min(extra, self.short_path_clip)

    # ------------------------------------------------------------------
    # render → grid update
    # ------------------------------------------------------------------
    def _backproject_depth(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor,
        eye: torch.Tensor,
        at: torch.Tensor,
        fov_deg: float,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """FoV-pinhole back-projection of depth pixels to world points.

        Returns [N, 3] world points (foreground only).
        """
        device = depth.device
        fy = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
        fx = fy * (width / height)
        cx = 0.5 * width
        cy = 0.5 * height

        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x_cam = (xs - cx) / fx
        y_cam = (ys - cy) / fy
        rays_cam = torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1)  # [H, W, 3]
        pts_cam = rays_cam * depth.unsqueeze(-1)                                # [H, W, 3]

        # World basis from look-at (camera at `eye`, looking at `at`).
        forward = (at - eye)
        forward = forward / (forward.norm() + 1e-8)
        up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        if torch.abs((forward * up_world).sum()) > 0.95:
            up_world = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        right = torch.cross(forward, up_world, dim=0)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(right, forward, dim=0)
        R = torch.stack([right, up, forward], dim=1)  # [3, 3], cam→world

        pts_world = eye + (pts_cam.reshape(-1, 3) @ R.T)
        pts_world = pts_world.reshape(*depth.shape, 3)
        return pts_world[mask]

    def _world_rays_for_camera(
        self,
        eye: torch.Tensor,
        at: torch.Tensor,
        fov_deg: float,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Return [H*W, 3] unit world rays matching the mesh renderer."""
        device = eye.device
        fy = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
        fx = fy * (width / height)
        cx = 0.5 * width
        cy = 0.5 * height
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        rays_cam = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], dim=-1)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)

        look = at - eye
        look = look / (look.norm() + 1e-8)
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        if torch.abs((look * up).sum()) > 0.999:
            up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        right = torch.linalg.cross(up, look, dim=0)
        right = right / (right.norm() + 1e-8)
        true_up = torch.linalg.cross(look, right, dim=0)
        R_cw = torch.stack([right, true_up, look], dim=1)
        return (rays_cam.reshape(-1, 3) @ R_cw.T)

    def _grid_bbox_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rg = self._range_gt[0]
        vs = self._voxel_size[0]
        bbox_min = torch.stack([rg[1] - 0.5 * vs[0], rg[3] - 0.5 * vs[1], rg[5] - 0.5 * vs[2]])
        bbox_max = bbox_min + float(self.grid_size) * vs
        return bbox_min, bbox_max

    def _empty_ray_free_voxel_paths(
        self,
        alpha: torch.Tensor,
        eye: torch.Tensor,
        at: torch.Tensor,
        fov_deg: float,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Return grid-AABB segments of alpha-miss rays as free-path voxels."""
        if self._range_gt is None or self._voxel_size is None or self._prob_grid is None:
            return torch.empty((0, 3), dtype=torch.long, device=eye.device)
        miss_mask = (alpha.reshape(-1) <= 0.5)
        if not miss_mask.any():
            return torch.empty((0, 3), dtype=torch.long, device=eye.device)

        rays = self._world_rays_for_camera(eye, at, fov_deg, width, height)[miss_mask]
        bbox_min, bbox_max = self._grid_bbox_min_max()
        voxel_size = self._voxel_size[0]

        eps_dir = 1e-9
        dir_safe = torch.where(
            rays.abs() < eps_dir,
            torch.where(rays >= 0.0, torch.full_like(rays, eps_dir), torch.full_like(rays, -eps_dir)),
            rays,
        )
        t0 = (bbox_min.unsqueeze(0) - eye.unsqueeze(0)) / dir_safe
        t1 = (bbox_max.unsqueeze(0) - eye.unsqueeze(0)) / dir_safe
        t_near = torch.minimum(t0, t1).amax(dim=-1)
        t_far = torch.maximum(t0, t1).amin(dim=-1)
        t_start = t_near.clamp(min=0.0)
        intersects = t_far > (t_start + 1e-6)
        if not intersects.any():
            return torch.empty((0, 3), dtype=torch.long, device=eye.device)

        rays = rays[intersects]
        t_near = t_near[intersects]
        t_start = t_start[intersects]
        t_far = t_far[intersects]
        voxel_eps = float(voxel_size.min().detach().item()) * 0.25
        t_entry = torch.where(t_near <= 0.0, torch.zeros_like(t_start), t_start + voxel_eps)
        t_exit = torch.maximum(t_far - voxel_eps, t_entry)
        start_pts = eye.unsqueeze(0) + t_entry.unsqueeze(1) * rays
        end_pts = eye.unsqueeze(0) + t_exit.unsqueeze(1) * rays

        start_idx = torch.floor((start_pts - bbox_min.unsqueeze(0)) / voxel_size.unsqueeze(0)).long()
        end_idx = torch.floor((end_pts - bbox_min.unsqueeze(0)) / voxel_size.unsqueeze(0)).long()
        start_idx = start_idx.clamp(0, self.grid_size - 1)
        end_idx = end_idx.clamp(0, self.grid_size - 1)
        return bresenham3D_strict(
            pts_source=start_idx,
            pts_target=end_idx,
            map_size=self.grid_size,
            include_source=True,
            include_target=True,
        )

    def _apply_free_paths(self, paths: Optional[torch.Tensor]) -> None:
        if paths is None or paths.numel() == 0:
            return
        paths = torch.unique(paths, dim=0)
        self._prob_grid[0, paths[:, 0], paths[:, 1], paths[:, 2]] += LOG_ODDS_FREE
        self._prob_grid.clamp_(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)

    def _update_grids(
        self,
        pts_world: torch.Tensor,
        eye: torch.Tensor,
        extra_free_paths: Optional[torch.Tensor] = None,
    ) -> None:
        """Voxelize back-projected pts, ray-cast, update prob_grid + scanned_gt_grid."""
        if pts_world.numel() == 0:
            self._apply_free_paths(extra_free_paths)
            return

        pts_idxs = scanned_pts_to_idx_3D(
            pts_target=[pts_world],
            range_gt=self._range_gt,
            voxel_size_gt=self._voxel_size,
            map_size=self.grid_size,
        )
        target_idx = pts_idxs[0]
        if isinstance(target_idx, list) or target_idx.shape[0] == 0:
            self._apply_free_paths(extra_free_paths)
            return

        cam_pos = eye.view(1, 3)
        source_idx = pose_coord_to_idx_3D(
            poses=cam_pos,
            range_gt=self._range_gt,
            voxel_size_gt=self._voxel_size,
            map_size=self.grid_size,
            if_col=False,
        )

        if not self.skip_free_raycast:
            free_parts = []
            paths = bresenham3D_strict(
                pts_source=source_idx,
                pts_target=target_idx,
                map_size=self.grid_size,
                include_source=True,
                include_target=False,
            )
            if paths.shape[0] > 0:
                G = self.grid_size
                target_ids = target_idx[:, 0] * (G * G) + target_idx[:, 1] * G + target_idx[:, 2]
                path_ids = paths[:, 0] * (G * G) + paths[:, 1] * G + paths[:, 2]
                target_ids_sorted = torch.sort(target_ids).values
                pos = torch.searchsorted(target_ids_sorted, path_ids)
                pos_clamped = pos.clamp(max=target_ids_sorted.numel() - 1)
                is_hit_endpoint = (
                    (pos < target_ids_sorted.numel())
                    & (target_ids_sorted[pos_clamped] == path_ids)
                )
                paths = paths[~is_hit_endpoint]
            if paths.shape[0] > 0:
                free_parts.append(paths)
            if extra_free_paths is not None and extra_free_paths.numel() > 0:
                G = self.grid_size
                target_ids = target_idx[:, 0] * (G * G) + target_idx[:, 1] * G + target_idx[:, 2]
                path_ids = (
                    extra_free_paths[:, 0] * (G * G)
                    + extra_free_paths[:, 1] * G
                    + extra_free_paths[:, 2]
                )
                target_ids_sorted = torch.sort(target_ids).values
                pos = torch.searchsorted(target_ids_sorted, path_ids)
                pos_clamped = pos.clamp(max=target_ids_sorted.numel() - 1)
                is_hit_endpoint = (
                    (pos < target_ids_sorted.numel())
                    & (target_ids_sorted[pos_clamped] == path_ids)
                )
                extra_free_paths = extra_free_paths[~is_hit_endpoint]
                if extra_free_paths.shape[0] > 0:
                    free_parts.append(extra_free_paths)
            if free_parts:
                self._apply_free_paths(torch.cat(free_parts, dim=0))
        self._prob_grid[0, target_idx[:, 0], target_idx[:, 1], target_idx[:, 2]] = LOG_ODDS_OCCUPIED
        self._prob_grid.clamp_(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)

        step_cov = torch.zeros_like(self._scanned_gt_grid)
        step_cov[0, target_idx[:, 0], target_idx[:, 1], target_idx[:, 2]] = 1.0
        coverage_step = step_cov
        if self.coverage_hit_dilate_radius > 0:
            r = self.coverage_hit_dilate_radius
            coverage_step = F.max_pool3d(
                step_cov.unsqueeze(1),
                kernel_size=2 * r + 1,
                stride=1,
                padding=r,
            ).squeeze(1)
        self._scanned_gt_grid = torch.clamp(
            self._scanned_gt_grid + coverage_step * self._grid_gt,
            min=0.0,
            max=1.0,
        )

    def _camera_collides_object(self, eye_np: np.ndarray) -> bool:
        if self._grid_gt is None or self._range_gt is None or self._voxel_size is None:
            return False
        eye_t = torch.from_numpy(eye_np.astype(np.float32)).to(self.device).view(1, 3)
        mesh_inside = False
        if self._renderer is not None:
            mesh_inside = bool(self._renderer.points_inside_mesh(eye_t)[0].item())

        idx = pose_coord_to_idx_3D(
            poses=eye_t,
            range_gt=self._range_gt,
            voxel_size_gt=self._voxel_size,
            map_size=self.grid_size,
            if_col=True,
        )
        if (idx < 0).any().item():
            return mesh_inside
        idx0 = idx[0].clamp_(0, self.grid_size - 1)
        surface_hit = bool(self._grid_gt[0, idx0[0], idx0[1], idx0[2]].item() > 0.5)
        return mesh_inside or surface_hit

    def _sample_valid_init_action(self, max_tries: int = 64) -> np.ndarray:
        preferred = np.array([40, 40, 80, 0, 0, 0], dtype=np.int64)
        if not self._camera_collides_object(self._decode_action(preferred)[:3]):
            return preferred
        for _ in range(max_tries):
            ai = np.array(
                [self._rng.randint(int(self.clip_pose_idx_low[k]), int(self.clip_pose_idx_up[k]))
                 for k in range(POSE_DIM)],
                dtype=np.int64,
            )
            pose6 = self._decode_action(ai)
            if not self._camera_collides_object(pose6[:3]):
                return ai
        return preferred

    def _lookat_for_pose6(self, pose6: np.ndarray) -> np.ndarray:
        x, y, z = pose6[:3]
        if self.auto_lookat_center:
            # Always aim at the object centre (origin in canonical frame).
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, pitch, yaw = pose6[3], pose6[4], pose6[5]
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        look_dir = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
        return np.array([x + look_dir[0], y + look_dir[1], z + look_dir[2]], dtype=np.float32)

    # ------------------------------------------------------------------
    # debug viz
    # ------------------------------------------------------------------
    def _dump_step_viz(self, pts_world: torch.Tensor) -> None:
        if self._cur_episode_dir is None or pts_world.numel() == 0:
            return
        idx = self._step_idx
        N = pts_world.shape[0]
        pts_dump = (pts_world[torch.randperm(N, device=pts_world.device)[:50_000]]
                    if N > 50_000 else pts_world)
        try:
            save_pointcloud_ply(pts_dump, self._cur_episode_dir / f"step_{idx:03d}_backproject.ply")
            tri = grid_occupancy_tri_cls(self._prob_grid, return_tri_cls_only=True)
            save_occupancy_grid_slices(
                tri,
                self._cur_episode_dir / f"step_{idx:03d}_grid.png",
                caption=f"step {idx}  CR={self._coverage_ratio():.3f}",
            )
        except Exception as e:
            print(f"[shapenbv.env] step viz failed: {type(e).__name__}: {e}")

    def _finalize_episode_dump(self) -> None:
        """Dump per-episode viz artefacts (pro 3D-vision style).

        Writes (under `<debug_dir>/episode_NNNN/`):
            cr_curve.png            — coverage vs step
            trajectory.png          — bare matplotlib trajectory (legacy)
            trajectory_pro.png      — frustum-style camera trajectory
                                      with viridis-step colormap + GT pcd
            coverage_heatmap.png    — GT voxels colored by first-seen step
            filmstrip.png           — N×1 grid of step renders
            action_history.png      — 6-panel action-dim time series
            dashboard.png           — combined GT/pred/traj/CR/action grid
            gt.ply / predicted.ply  — point clouds (pred from validate.py)
        """
        if self._cur_episode_dir is None:
            return
        ep_dir = self._cur_episode_dir
        try:
            save_coverage_curve(self._cr_history, ep_dir / "cr_curve.png")

            # Bare-bones legacy plot (kept for backward compat).
            save_trajectory_plot(
                eye_positions=self._eye_history,
                look_ats=self._at_history,
                bbox_min=self._action_box_min,
                bbox_max=self._action_box_max,
                path=ep_dir / "trajectory.png",
            )

            # GT surface points (preferred) — falls back to grid_gt voxel
            # centres when `points_canon` was skipped at preproc time
            # (Phase 1 default for the 35× space win).
            gt_pts = None
            if self._preproc is not None and "points_canon" in self._preproc:
                gt_pts = self._preproc["points_canon"].detach().cpu().numpy()
                save_pointcloud_ply(gt_pts, ep_dir / "gt.ply")
            elif self._preproc is not None and "grid_gt" in self._preproc:
                gt_pts = _grid_gt_to_voxel_centres(
                    self._preproc["grid_gt"].detach().cpu().numpy(),
                    self._range_gt_np,
                    self._voxel_size_np,
                )
                save_pointcloud_ply(gt_pts, ep_dir / "gt_voxels.ply")

            # Pro frustum-trajectory plot. Pass `mesh_path` so viz can
            # render the actual triangulated surface with Lambertian
            # shading instead of a low-info gray pcd scatter.
            mesh_path_for_viz = (
                getattr(self._renderer, "mesh_path", None)
                if self._renderer is not None else None
            )
            save_camera_trajectory_pro(
                eye_positions=self._eye_history,
                look_ats=self._at_history,
                bbox_min=self._action_box_min,
                bbox_max=self._action_box_max,
                path=ep_dir / "trajectory_pro.png",
                object_pointcloud=gt_pts,
                mesh_path=mesh_path_for_viz,
                fov_deg=self.fov_deg if hasattr(self, "fov_deg") else 60.0,
                title=f"Episode {self._episode_idx} trajectory",
            )

            # Coverage heatmap (GT voxels colored by first-seen step).
            if self._scanned_gt_history:
                sh = np.stack(self._scanned_gt_history, axis=0)        # [T, G, G, G]
                save_coverage_heatmap(
                    grid_gt=self._preproc["grid_gt"].detach().cpu().numpy(),
                    scanned_history=sh,
                    range_gt=self._range_gt_np,
                    voxel_size=self._voxel_size_np,
                    path=ep_dir / "coverage_heatmap.png",
                    title=f"Episode {self._episode_idx} coverage",
                )

            # Filmstrip of per-step diagnostics. Prefer the
            # `step_*_render.png` (RGB-D-α composite) when present;
            # fall back to `step_*_grid.png` (occupancy slices) which
            # the env does dump by default.
            render_paths = sorted(ep_dir.glob("step_*_render.png"))
            if not render_paths:
                render_paths = sorted(ep_dir.glob("step_*_grid.png"))
            if render_paths:
                captions = [f"step {i+1}  cr={self._cr_history[i+1]:.2f}"
                            if (i + 1) < len(self._cr_history) else f"step {i+1}"
                            for i in range(len(render_paths))]
                save_filmstrip(
                    render_paths=render_paths,
                    path=ep_dir / "filmstrip.png",
                    n_cols=8,
                    captions=captions,
                    title=f"Episode {self._episode_idx} renders",
                )

            if self._action_idx_history:
                save_action_history_plot(
                    action_indices=np.stack(self._action_idx_history, axis=0),
                    cr_history=self._cr_history,
                    nvec=NVEC,
                    path=ep_dir / "action_history.png",
                    title=f"Action history ({len(self._action_idx_history)} steps)",
                )
                save_combined_dashboard(
                    eye_positions=self._eye_history,
                    look_ats=self._at_history,
                    bbox_min=self._action_box_min,
                    bbox_max=self._action_box_max,
                    cr_history=self._cr_history,
                    action_indices=np.stack(self._action_idx_history, axis=0),
                    nvec=NVEC,
                    gt_pointcloud=gt_pts,
                    pred_pointcloud=None,
                    mesh_path=mesh_path_for_viz,
                    path=ep_dir / "dashboard.png",
                    title=f"Episode {self._episode_idx} (final cr="
                          f"{self._cr_history[-1]:.3f})",
                )
        except Exception as e:
            import traceback
            print(f"[shapenbv.env] viz finalize failed: {type(e).__name__}: {e}")
            traceback.print_exc()


def _grid_gt_to_voxel_centres(grid_gt, range_gt, voxel_size):
    """Recover a coarse GT pcd from `grid_gt` when `points_canon` was
    skipped at preproc time.  Each occupied voxel emits one point at
    its centre — matches the GenNBV voxel-CENTRE convention used by
    `range_gt`.  Memory: 51^3 = ~133 KB max.
    """
    rg = np.asarray(range_gt).ravel()
    vs = np.asarray(voxel_size).ravel()
    G = grid_gt.shape[0]
    occ = np.argwhere(grid_gt > 0.5).astype(np.float32)
    if occ.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    mins = np.array([rg[1], rg[3], rg[5]], dtype=np.float32)
    return mins[None, :] + occ * vs[None, :]
