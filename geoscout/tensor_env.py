"""TensorBatchEnv — IsaacGym-style fully tensor-batched VecEnv for GeoScout.

Replaces SB3's DummyVecEnv (Python loop over N envs) with a single
process / single tensor batch: all N envs evolve together as [N, ...]
GPU tensors, every op is one CUDA kernel launch instead of N. Empirically
this gives 5–20× higher fps than DummyVecEnv on small/mid envs that
spend their time on GPU ops (mesh raycast + grid update).

Design notes:

* Mesh pool — all meshes, GT grids, and bboxes are loaded once on
  construction. Each env samples a mesh id at reset; the step path groups
  envs by mesh id and runs one render_batch per active mesh.

* Action / observation spaces — default to the GenNBV-faithful discrete
  Cube Mode (`MultiDiscrete([81, 81, 81, 1, 13, 13])`).  For policy
  ablations, `action_space_type="continuous_tanh"` exposes a Gaussian
  Box policy over finite raw actions and maps them through tanh
  to the same cube/rpy pose domain.  The policy observation contains
  `Box(buffer*6 + obs_grid_size**3 + caption_dim,)`; coverage/reward can
  still run on the higher-resolution `grid_size`.

* Async/sync interface — implements `step_async` / `step_wait` (SB3
  VecEnv contract) so PPO's `collect_rollouts` is happy. Internally
  step() does ONE big batched op over all N envs.

* Reset semantics — episodes that finish mid-rollout are reset in-place
  on the next step (mask-driven), so the buffer indices stay aligned
  with N.

* Free-voxel raycast — uses explicit voxel-index 3D Bresenham traversal
  per env, then combines paths by env_id for one scatter into
  [N, G, G, G] grids.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .preprocess import load_preproc
from .mesh_renderer import (
    MeshSequenceRenderer,
    NvdiffrastMeshSequenceRenderer,
    Open3DMeshSequenceRenderer,
)
from .voxel_utils import bresenham3D_strict, grid_occupancy_tri_cls
try:
    from .cuda_bresenham import (
        empty_ray_pairs_cuda,
        scatter_bresenham3d_to_mask_cuda,
        warmup_cuda_bresenham,
    )
except Exception:  # pragma: no cover - fallback for non-CUDA/local envs.
    empty_ray_pairs_cuda = None
    scatter_bresenham3d_to_mask_cuda = None
    warmup_cuda_bresenham = None
try:
    from .cuda_voxel_renderer import (
        last_extension_error as voxel_renderer_extension_error,
        voxel_first_hit_cuda,
        voxel_first_hit_reference,
        warmup_cuda_voxel_renderer,
    )
except Exception:  # pragma: no cover - fallback for non-CUDA/local envs.
    voxel_renderer_extension_error = None
    voxel_first_hit_cuda = None
    voxel_first_hit_reference = None
    warmup_cuda_voxel_renderer = None
try:
    from .triton_bresenham import apply_free_mask_to_grid, scatter_bresenham3d_to_mask
except Exception:  # pragma: no cover - fallback for non-CUDA/local envs.
    apply_free_mask_to_grid = None
    scatter_bresenham3d_to_mask = None


# Mirror env.py's GenNBV-faithful action grid + effective reward scales.
DEFAULT_ACTION_LOW_WORLD = np.array(
    [-1.0, -1.0, -1.0, 0.0, -math.pi / 2, 0.0], dtype=np.float32
)
DEFAULT_ACTION_UNIT = np.array(
    [0.025, 0.025, 0.025, 0.0, math.pi / 12, math.pi / 6], dtype=np.float32
)
DEFAULT_CLIP_POSE_IDX_UP = np.array([80, 80, 80, 0, 12, 12], dtype=np.int64)
DEFAULT_CLIP_POSE_IDX_LOW = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
NVEC = (DEFAULT_CLIP_POSE_IDX_UP - DEFAULT_CLIP_POSE_IDX_LOW + 1).astype(np.int64)
POSE_DIM = 6

REWARD_SCALE_SURFACE_COVERAGE = 20.0
REWARD_SCALE_SHORT_PATH = 0.1
REWARD_SCALE_TERMINATION = 1.0
REWARD_SCALE_COLLISION = 10.0
LOG_ODDS_FREE = -0.05
LOG_ODDS_OCCUPIED = 1.0
LOG_ODDS_CLAMP = 5.0


class TensorBatchEnv(VecEnv):
    """N parallel GeoScout envs as one tensor batch.

    Compatible with SB3's `VecEnv` interface so PPO can drop in. All
    state lives on GPU as [N, ...] tensors; step / reset / observation
    construction are batched ops with no Python per-env loop.

    Args:
        num_envs: N
        mesh_path: ShapeNet `.obj` (all N envs share this mesh).
        preproc_path: `.pt` from `geoscout.preprocess.preprocess_mesh`.
        device: "cuda" or "cpu".
        buffer_size: pose history length (Hybrid_Encoder input).
        grid_size: reward / GT occupancy grid resolution.
        obs_grid_size: policy observation grid resolution.
        episode_len: max steps per episode.
        render_size: square (H, W) for the raycaster.
        fov_deg: camera fov.
        cr_success_threshold: episode terminates with bonus when cr > this.
        short_path_grace / short_path_clip: GenNBV `_reward_short_path`.
        only_positive_rewards: clip per-step reward at 0 before bonus.
        skip_free_raycast: skip free-voxel update for speed (smoke test).
        update_empty_rays: when free raycasting is enabled, mark
            alpha-miss rays through the current grid AABB as free.
        coverage_hit_dilate_radius: dilate rendered hit voxels by this
            many cells before matching against GT coverage. 0 preserves
            exact voxel matching.
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        num_envs: int,
        mesh_paths,                            # List[Path] OR single Path (back-compat)
        preproc_paths,                          # List[Path] OR single Path
        device: str = "cuda",
        buffer_size: int = 30,
        grid_size: int = 128,
        obs_grid_size: int = 32,
        episode_len: int = 100,
        render_size: int = 400,
        fov_deg: float = 60.0,
        cr_success_threshold: float = 0.99,
        coverage_reward_scale: float = REWARD_SCALE_SURFACE_COVERAGE,
        short_path_grace: int = 30,
        short_path_clip: float = 2.0,
        short_path_scale: float = REWARD_SCALE_SHORT_PATH,
        only_positive_rewards: bool = True,
        skip_free_raycast: bool = False,
        n_free_samples_per_ray: int = 16,
        update_empty_rays: bool = True,
        coverage_hit_dilate_radius: int = 1,
        # Phase 1 caption channel: dimension of per-episode caption_emb.
        # 0 = disabled. 384 = sentence-transformer MiniLM-L6.
        caption_dim: int = 0,
        # Action representation. "discrete" is the original GenNBV-style
        # MultiDiscrete Cube Mode. "continuous_tanh" is an ablation that
        # uses a Gaussian Box policy over raw actions in [-8, 8]^5:
        #   tanh(a[:3]) -> x/y/z in (-1, 1)
        #   tanh(a[3])  -> pitch in (-pi/2, pi/2)
        #   tanh(a[4])  -> yaw in (0, 2pi)
        # Roll stays fixed at 0, matching the discrete action grid where
        # the roll dimension has only one valid value.
        action_space_type: str = "discrete",
        # When True, ignore the (roll, pitch, yaw) action dims and force
        # the camera to look at the object center every step. Reduces
        # the effective action space to the 3 position dims, which is
        # what made Run E (object_nbv_zgr discrete) actually learn on
        # uCO3D — random 6D pitch/yaw means ~95% of viewpoints face
        # empty space, so PPO with sparse coverage reward stays at cr=0
        # for the entire smoke budget. Off by default (GenNBV-faithful);
        # turn on for single-object smoke tests.
        auto_lookat_center: bool = False,
        # Cap on per-mesh triangle count. When > 0, MeshSequenceRenderer
        # quadric-decimates each mesh to ~max_faces faces at load time.
        # Crucial for ABO furniture (often 50K-100K tris) — without this
        # the pure-PyTorch Möller-Trumbore drops fps from ~500 (smoke) to
        # ~20 (ABO). 5000 keeps coarse silhouette intact for depth.
        max_faces: int = 0,
        # Renderer-side exact acceleration: rays that miss the mesh AABB
        # cannot hit any triangle, so skip ray-triangle tests for them.
        renderer_bbox_ray_cull: bool = True,
        renderer_backend: str = "open3d",
        # Free-space exact acceleration: use a Triton CUDA kernel to
        # scatter 3D Bresenham paths directly into the per-step free mask.
        # Falls back to the pure PyTorch path when CUDA/Triton is absent.
        use_triton_free_raycast: bool = True,
        free_raycast_backend: str = "auto",
        triton_bresenham_block_rays: int = 64,
        # How to apply the per-step uint8 free mask to the log-odds grid.
        # "index" preserves the older nonzero/index_put path for A/B
        # profiling; "dense" uses a dense PyTorch add; "triton" uses a
        # dense Triton pass and falls back to "dense" if unavailable.
        free_mask_apply_mode: str = "triton",
        # Reward redesign knobs (ShapeNet's random-baseline cr is already
        # ~0.83; the original linear (cr_t - cr_{t-1}) gradient is too
        # flat in the high-cr regime where learning matters).
        # `coverage_reward_type`:
        #   "linear"             GenNBV delta coverage.
        #   "log"                steeper near high CR.
        #   "remaining"          new voxels normalized by remaining voxels.
        #   "information_gain"   dense NBV shaping: new coverage + novelty
        #                        bonus - overlap/revisit penalties.
        coverage_reward_type: str = "linear",
        termination_bonus: float = REWARD_SCALE_TERMINATION,
        novelty_reward_scale: float = 0.0,
        remaining_reward_scale: float = 0.0,
        redundancy_penalty_scale: float = 0.0,
        view_revisit_penalty_scale: float = 0.0,
        view_revisit_angle_deg: float = 12.0,
        # Negative reward applied at collision (camera inside/intersecting
        # the object mesh). Bypasses `only_positive_rewards` clamp so the
        # agent actually feels the cost.
        collision_penalty: float = REWARD_SCALE_COLLISION,
        # Infrastructure-only performance knobs. These do not alter the
        # reward/action/observation definitions. Duplicate miss rays with
        # the same grid entry/exit voxels are harmless because the downstream
        # free-space scatter writes into a binary mask before the log-odds
        # update. Keep dedupe off by default; it only adds a costly unique.
        dedupe_empty_ray_pairs: bool = False,
        # Optional profiling for smoke tests. When enabled, the hot path
        # synchronizes CUDA between sections so wall-time attribution is
        # meaningful. Keep disabled during PPO training.
        profile_timing: bool = False,
        # 128³ preproc pools cannot be cached as one giant GPU tensor
        # (600 meshes would be ~5GB just for GT grids). None = automatic:
        # keep small pools on GPU, large/high-res pools on CPU and copy
        # the selected per-env grids on reset.
        cache_pool_grid_on_device: Optional[bool] = None,
        seed: int = 0,
    ):
        # Normalize to lists. `mesh_paths` and `preproc_paths` define a
        # POOL of unique meshes; per-env mesh assignment is randomly
        # sampled from this pool at every reset. Length-1 pool collapses
        # to "single-mesh shared across N envs" (the original
        # TensorBatchEnv behaviour).
        from pathlib import Path as _P
        if isinstance(mesh_paths, (str, _P)):
            mesh_paths = [mesh_paths]
        if isinstance(preproc_paths, (str, _P)):
            preproc_paths = [preproc_paths]
        mesh_paths = [Path(p) for p in mesh_paths]
        preproc_paths = [Path(p) for p in preproc_paths]
        self._pool_names = [p.stem for p in preproc_paths]
        if len(mesh_paths) != len(preproc_paths):
            raise ValueError(
                f"mesh_paths ({len(mesh_paths)}) and preproc_paths "
                f"({len(preproc_paths)}) must match in length."
            )
        self._pool_size = len(mesh_paths)

        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.buffer_size = int(buffer_size)
        self.grid_size = int(grid_size)
        self.obs_grid_size = int(obs_grid_size) if int(obs_grid_size) > 0 else self.grid_size
        if self.obs_grid_size <= 0:
            raise ValueError(f"obs_grid_size must be positive, got {self.obs_grid_size}")
        if self.obs_grid_size > self.grid_size:
            raise ValueError(
                f"obs_grid_size={self.obs_grid_size} cannot exceed grid_size={self.grid_size}"
            )
        self.episode_len = int(episode_len)
        self.render_size = int(render_size)
        self.fov_deg = float(fov_deg)
        self.cr_success_threshold = float(cr_success_threshold)
        self.coverage_reward_scale = float(coverage_reward_scale)
        self.short_path_scale = float(short_path_scale)
        self.coverage_reward_type = str(coverage_reward_type)
        valid_reward_types = ("linear", "log", "remaining", "information_gain")
        if self.coverage_reward_type not in valid_reward_types:
            raise ValueError(
                f"coverage_reward_type must be one of {valid_reward_types}, "
                f"got {self.coverage_reward_type}"
            )
        self.termination_bonus = float(termination_bonus)
        self.novelty_reward_scale = float(novelty_reward_scale)
        self.remaining_reward_scale = float(remaining_reward_scale)
        self.redundancy_penalty_scale = float(redundancy_penalty_scale)
        self.view_revisit_penalty_scale = float(view_revisit_penalty_scale)
        self.view_revisit_angle_deg = float(view_revisit_angle_deg)
        self.collision_penalty = float(collision_penalty)
        self.dedupe_empty_ray_pairs = bool(dedupe_empty_ray_pairs)
        self.profile_timing = bool(profile_timing)
        self.last_step_profile: Dict[str, Any] = {}
        self._last_free_update_stats: Dict[str, Any] = {}
        self.short_path_grace = int(short_path_grace)
        self.short_path_clip = float(short_path_clip)
        self.only_positive_rewards = bool(only_positive_rewards)
        self.skip_free_raycast = bool(skip_free_raycast)
        self.n_free_samples_per_ray = int(n_free_samples_per_ray)
        self.update_empty_rays = bool(update_empty_rays)
        self.coverage_hit_dilate_radius = max(0, int(coverage_hit_dilate_radius))
        self.action_space_type = str(action_space_type)
        if self.action_space_type not in ("discrete", "continuous_tanh"):
            raise ValueError(
                "action_space_type must be 'discrete' or 'continuous_tanh', "
                f"got {self.action_space_type!r}"
            )
        self.auto_lookat_center = bool(auto_lookat_center)
        self.caption_dim = int(caption_dim)
        self.renderer_bbox_ray_cull = bool(renderer_bbox_ray_cull)
        self.renderer_backend = str(renderer_backend)
        if self.renderer_backend == "voxel":
            self.renderer_backend = "voxel_cuda"
        if self.renderer_backend not in ("torch", "open3d", "nvdiffrast", "voxel_cuda"):
            raise ValueError(
                "renderer_backend must be 'torch', 'open3d', 'nvdiffrast', or 'voxel_cuda', "
                f"got {self.renderer_backend!r}"
            )
        self.use_triton_free_raycast = bool(use_triton_free_raycast)
        self.free_raycast_backend = str(free_raycast_backend)
        valid_free_backends = ("auto", "cuda", "triton", "torch")
        if self.free_raycast_backend not in valid_free_backends:
            raise ValueError(
                f"free_raycast_backend must be one of {valid_free_backends}, "
                f"got {self.free_raycast_backend!r}"
            )
        self.triton_bresenham_block_rays = max(1, int(triton_bresenham_block_rays))
        self.free_mask_apply_mode = str(free_mask_apply_mode)
        valid_apply_modes = ("index", "dense", "triton")
        if self.free_mask_apply_mode not in valid_apply_modes:
            raise ValueError(
                f"free_mask_apply_mode must be one of {valid_apply_modes}, "
                f"got {self.free_mask_apply_mode!r}"
            )
        self.cuda_bresenham_warmed = False
        if (
            self.device.type == "cuda"
            and self.use_triton_free_raycast
            and self.free_raycast_backend in ("auto", "cuda")
            and warmup_cuda_bresenham is not None
        ):
            # Keep the one-time torch CUDA extension compile out of the
            # first environment step.  Failure is non-fatal because auto
            # mode can still fall back to Triton/PyTorch.
            try:
                self.cuda_bresenham_warmed = bool(warmup_cuda_bresenham(self.device))
            except Exception:
                self.cuda_bresenham_warmed = False
        self.free_mask_apply_warmed = False
        if (
            self.device.type == "cuda"
            and self.free_mask_apply_mode == "triton"
            and apply_free_mask_to_grid is not None
        ):
            try:
                warm_grid = torch.zeros(1, 2, 2, 2, dtype=torch.float32, device=self.device)
                warm_mask = torch.ones(1, 2, 2, 2, dtype=torch.uint8, device=self.device)
                self.free_mask_apply_warmed = bool(
                    apply_free_mask_to_grid(warm_grid, warm_mask, delta=LOG_ODDS_FREE)
                )
            except Exception:
                self.free_mask_apply_warmed = False

        # Action / observation spaces. GeoScout uses the normalized Cube
        # Mode action domain; this can be represented either as GenNBV's
        # dense MultiDiscrete grid or as a tanh-squashed continuous policy.
        nvec = NVEC
        self._nvec = nvec
        if self.action_space_type == "discrete":
            action_space = spaces.MultiDiscrete(nvec)
        else:
            # SB3 PPO requires finite continuous-action bounds.  The env
            # still applies tanh(raw), and tanh(±8) is effectively at the
            # open interval boundary without introducing infinities.
            action_space = spaces.Box(
                low=-8.0,
                high=8.0,
                shape=(5,),
                dtype=np.float32,
            )
        obs_dim = self.buffer_size * POSE_DIM + self.obs_grid_size ** 3 + self.caption_dim
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        super().__init__(
            num_envs=self.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self._action_unit = torch.tensor(DEFAULT_ACTION_UNIT, dtype=torch.float32, device=self.device)
        self._action_low = torch.tensor(DEFAULT_ACTION_LOW_WORLD, dtype=torch.float32, device=self.device)
        self._idx_up = torch.tensor(DEFAULT_CLIP_POSE_IDX_UP, dtype=torch.long, device=self.device)
        self._idx_low = torch.tensor(DEFAULT_CLIP_POSE_IDX_LOW, dtype=torch.long, device=self.device)
        self._rays_cam = self._build_camera_rays()

        # ----- Build the mesh pool -----
        # For each mesh in the pool, load its preproc dict and stash the
        # GT grid + bbox + caption_emb as a [P, ...] tensor. At reset
        # time each env picks an index into this pool (random uniform);
        # per-env [N, ...] tensors are built by `index_select`.
        G = self.grid_size
        P = len(preproc_paths)
        pool_grid_elems = int(P) * int(G) ** 3
        if cache_pool_grid_on_device is None:
            # 64M float elements is ~256MB; above that, the reset-time CPU
            # copy is far cheaper than burning GPU memory on idle meshes.
            cache_pool_grid_on_device = (G <= 64 and pool_grid_elems <= 64_000_000)
        self._pool_grid_on_device = bool(cache_pool_grid_on_device)
        load_location = self.device if self._pool_grid_on_device else "cpu"
        cache_where = "GPU" if self._pool_grid_on_device else "CPU(uint8)"
        print(
            f"[tensor_env] grid_size={self.grid_size} obs_grid_size={self.obs_grid_size} "
            f"coverage_hit_dilate_radius={self.coverage_hit_dilate_radius} "
            f"pool_grid_cache={cache_where} pool_grid_elems={pool_grid_elems}",
            flush=True,
        )
        pool_grid_gt = []
        pool_range_gt = []
        pool_voxel_size = []
        pool_num_valid = []
        pool_bbox_min = []
        pool_caption_emb = []
        pool_T_canon = []
        print(f"[tensor_env] loading preproc for {P}-mesh pool...", flush=True)
        for i, pp in enumerate(preproc_paths):
            preproc = load_preproc(pp, map_location=load_location)
            grid_raw = preproc["grid_gt"]
            if self._pool_grid_on_device:
                grid_gt = grid_raw.to(self.device).float()
            else:
                grid_gt = (grid_raw.detach().cpu() > 0.5).to(torch.uint8)
            if tuple(grid_gt.shape) != (G, G, G):
                raise ValueError(
                    f"preproc grid shape {tuple(grid_gt.shape)} does not match "
                    f"env grid_size={G} for {pp}. Re-run preprocessing with "
                    f"--grid_size {G}."
                )
            pool_grid_gt.append(grid_gt)
            rg = preproc["range_gt"].cpu().numpy().ravel()    # voxel-CENTRE conv
            vs = preproc["voxel_size_gt"].cpu().numpy().ravel()
            pool_range_gt.append(torch.tensor(rg, dtype=torch.float32, device=self.device))
            pool_voxel_size.append(torch.tensor(vs, dtype=torch.float32, device=self.device))
            # Voxel grid origin = (rg[1,3,5] - 0.5 * voxel_size) per axis.
            pool_bbox_min.append(torch.tensor(
                [rg[1] - 0.5 * vs[0], rg[3] - 0.5 * vs[1], rg[5] - 0.5 * vs[2]],
                dtype=torch.float32, device=self.device,
            ))
            pool_num_valid.append(float(preproc["num_valid_voxel_gt"].item()))
            if self.caption_dim > 0:
                ce = preproc.get("caption_emb")
                if ce is None:
                    pool_caption_emb.append(
                        torch.zeros(self.caption_dim, dtype=torch.float32, device=self.device))
                else:
                    cev = ce.to(self.device).float().view(-1)
                    if cev.shape[0] != self.caption_dim:
                        raise ValueError(
                            f"caption_emb dim {cev.shape[0]} != env caption_dim "
                            f"{self.caption_dim} for {pp}"
                        )
                    pool_caption_emb.append(cev)
            pool_T_canon.append(preproc.get("T_canon"))
            if (i + 1) % 50 == 0 or i + 1 == P:
                print(f"[tensor_env] preproc loaded {i + 1}/{P}", flush=True)
        self._pool_grid_gt = torch.stack(pool_grid_gt, dim=0)         # [P, G, G, G]
        self._pool_range_gt = torch.stack(pool_range_gt, dim=0)       # [P, 6]
        self._pool_voxel_size = torch.stack(pool_voxel_size, dim=0)   # [P, 3]
        self._pool_bbox_min = torch.stack(pool_bbox_min, dim=0)       # [P, 3]
        self._pool_num_valid = torch.tensor(pool_num_valid, dtype=torch.float32, device=self.device)  # [P]
        if self.caption_dim > 0:
            self._pool_caption_emb = torch.stack(pool_caption_emb, dim=0)  # [P, caption_dim]
        else:
            self._pool_caption_emb = None

        # Per-env mesh assignment (random uniform sampled at reset). The
        # actual values are written by `_reset_envs`.
        self._env_mesh_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # `_grid_gt`, `_bbox_min`, `_voxel_size` and `_caption_emb` are
        # always derived from `_env_mesh_id`. They're rebuilt on every
        # `_reset_envs` call. Initialise here as zeros — `_reset_all`
        # at the end of __init__ will populate them.
        self._grid_gt = torch.zeros(self.num_envs, G, G, G, dtype=torch.float32, device=self.device)
        self._bbox_min = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._voxel_size = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._num_valid_gt_per_env = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        if self.caption_dim > 0:
            self._caption_emb = torch.zeros(self.num_envs, self.caption_dim,
                                            dtype=torch.float32, device=self.device)
        else:
            self._caption_emb = None
        self.reset_continuous_action_stats()

        # Build one renderer per mesh in the pool. At render time we
        # group envs by mesh_id and call each renderer's `render_batch`
        # over its slice of the cameras (still 1 GPU op per group).
        # Length-1 pool degenerates to the original single-mesh path.
        self._renderers = []
        self.max_faces = int(max_faces)
        print(
            f"[tensor_env] building {P} mesh renderers "
            f"(backend={self.renderer_backend}, max_faces={self.max_faces}) ...",
            flush=True,
        )
        face_counts = []
        for i, (mp, T_canon) in enumerate(zip(mesh_paths, pool_T_canon)):
            # voxel_cuda bypasses mesh rendering during step(), but it still
            # uses the mesh-level point-in-solid test for camera collision.
            # MeshSequenceRenderer is enough for that and avoids building an
            # unused Open3D CPU BVH.
            if self.renderer_backend == "open3d":
                renderer_cls = Open3DMeshSequenceRenderer
            elif self.renderer_backend == "nvdiffrast":
                renderer_cls = NvdiffrastMeshSequenceRenderer
            else:
                renderer_cls = MeshSequenceRenderer
            kwargs = dict(
                mesh_path=mp,
                sequence_name=str(mp.name),
                device=self.device,
                render_size=(self.render_size, self.render_size),
                fov_deg=self.fov_deg,
                T_canon=T_canon,
                max_faces=self.max_faces,
            )
            if renderer_cls is MeshSequenceRenderer:
                kwargs["bbox_ray_cull"] = self.renderer_bbox_ray_cull
            r = renderer_cls(**kwargs)
            self._renderers.append(r)
            face_counts.append(r._faces.shape[0])
            if (i + 1) % 25 == 0 or i + 1 == P:
                print(f"[tensor_env] renderer {i + 1}/{P} built", flush=True)
        import numpy as _np
        fc = _np.asarray(face_counts)
        print(f"[tensor_env] face counts: min={fc.min()} max={fc.max()} "
              f"mean={fc.mean():.0f} median={int(_np.median(fc))}", flush=True)
        self.voxel_renderer_warmed = False
        if self.renderer_backend == "voxel_cuda" and warmup_cuda_voxel_renderer is not None:
            try:
                self.voxel_renderer_warmed = bool(warmup_cuda_voxel_renderer(self.device))
            except Exception:
                self.voxel_renderer_warmed = False
        # Bbox of the Cube Mode action grid for viz / collision-bound checks.
        idx_up = self._idx_up.float()
        idx_lo = self._idx_low.float()
        self._action_box_min = (idx_lo * self._action_unit + self._action_low)[:3].cpu().numpy()
        self._action_box_max = (idx_up * self._action_unit + self._action_low)[:3].cpu().numpy()

        # Per-env state tensors (all [N, ...] on device).
        N, G, B = self.num_envs, self.grid_size, self.buffer_size
        self._prob_grid = torch.zeros(N, G, G, G, dtype=torch.float32, device=self.device)
        self._scanned_gt_grid = torch.zeros(N, G, G, G, dtype=torch.float32, device=self.device)
        # Reused per-step workspace for the exact free-space set union.
        # Keeping this buffer hot avoids a large [N, G, G, G] allocation
        # on every env step. It is overwritten before every use.
        self._free_mask_buffer = torch.empty(N, G, G, G, dtype=torch.uint8, device=self.device)
        if self.free_mask_apply_warmed and apply_free_mask_to_grid is not None:
            try:
                self._free_mask_buffer.zero_()
                apply_free_mask_to_grid(
                    self._prob_grid,
                    self._free_mask_buffer,
                    delta=LOG_ODDS_FREE,
                )
            except Exception:
                self.free_mask_apply_warmed = False
        self._step_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self._cr_prev = torch.zeros(N, dtype=torch.float32, device=self.device)
        # Pose history: ring buffer [N, B, 6] of decoded float poses.
        self._action_history = torch.zeros(N, B, POSE_DIM, dtype=torch.float32, device=self.device)
        # Last action indices [N, 6] — used for viz dump.
        self._last_action_idx = torch.zeros(N, POSE_DIM, dtype=torch.long, device=self.device)

        # Per-env coverage-ratio denominator (set by `_reset_envs`
        # alongside `_num_valid_gt_per_env`).

        # Pending step-async actions buffer.
        self._async_actions: Optional[torch.Tensor] = None

        # Tracker for SB3 / Monitor: no per-env episode wrapper, so we
        # roll our own basic episode-info dict at done time.
        self._ep_step_count = torch.zeros(N, dtype=torch.long, device=self.device)
        self._ep_reward_sum = torch.zeros(N, dtype=torch.float32, device=self.device)
        self._ep_new_gt_sum = torch.zeros(N, dtype=torch.float32, device=self.device)
        self._ep_visible_gt_sum = torch.zeros(N, dtype=torch.float32, device=self.device)
        self._ep_redundant_gt_sum = torch.zeros(N, dtype=torch.float32, device=self.device)
        self._ep_revisit_sum = torch.zeros(N, dtype=torch.float32, device=self.device)

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        # Reset all envs to seed initial poses.
        self._reset_all()

    # ------------------------------------------------------------------
    # SB3 VecEnv API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._reset_all()
        return self._build_observation_np()

    def reset_to_mesh_ids(self, mesh_ids: torch.Tensor | np.ndarray | Sequence[int]) -> np.ndarray:
        """Reset envs to a caller-specified list of pool mesh ids.

        Training uses random mesh assignment on reset.  Evaluation sometimes
        needs exact one-pass coverage of a fixed sample list; this helper keeps
        that path deterministic without changing the default training reset
        semantics.  The number of ids must match ``num_envs``.
        """
        mesh_ids_t = torch.as_tensor(mesh_ids, dtype=torch.long, device=self.device).flatten()
        if int(mesh_ids_t.numel()) != int(self.num_envs):
            raise ValueError(
                f"reset_to_mesh_ids expected {self.num_envs} ids, got {mesh_ids_t.numel()}"
            )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs_to_mesh_ids(env_ids, mesh_ids_t)
        return self._build_observation_np()

    def reset_continuous_action_stats(self) -> None:
        """Reset rollout-level diagnostics for raw continuous actions."""
        self._cont_action_stat_steps = 0
        self._cont_action_abs_mean_sum = 0.0
        self._cont_action_abs_gt3_frac_sum = 0.0
        self._cont_action_clip_frac_sum = 0.0
        self._cont_action_abs_max = 0.0

    def get_continuous_action_stats(self) -> Dict[str, float]:
        """Return averaged diagnostics for `continuous_tanh` action health."""
        steps = max(int(getattr(self, "_cont_action_stat_steps", 0)), 1)
        return {
            "raw_abs_mean": float(getattr(self, "_cont_action_abs_mean_sum", 0.0) / steps),
            "raw_abs_gt3_frac": float(getattr(self, "_cont_action_abs_gt3_frac_sum", 0.0) / steps),
            "raw_clip_frac": float(getattr(self, "_cont_action_clip_frac_sum", 0.0) / steps),
            "raw_abs_max": float(getattr(self, "_cont_action_abs_max", 0.0)),
            "stat_steps": float(getattr(self, "_cont_action_stat_steps", 0)),
        }

    def step_async(self, actions: np.ndarray) -> None:
        if self.action_space_type == "discrete":
            # actions: [N, 6] int64
            arr = np.asarray(actions, dtype=np.int64)
        else:
            # actions: [N, 5] float32 raw Gaussian samples. SB3 clips to
            # the finite Box bounds; `_decode_actions` then applies tanh
            # before mapping into the physical camera-pose domain.
            arr = np.asarray(actions, dtype=np.float32)
        self._async_actions = torch.from_numpy(arr).to(self.device)

    def step_wait(self):
        actions = self._async_actions
        self._async_actions = None
        return self._step_batch(actions)

    def close(self) -> None:
        # Free large per-env tensors.
        self._prob_grid = None
        self._scanned_gt_grid = None
        self._action_history = None

    # SB3 VecEnv has a few more abstract methods. Provide minimal stubs.
    def get_attr(self, attr_name, indices=None):
        idx = self._indices(indices)
        return [getattr(self, attr_name) for _ in idx]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(
            f"TensorBatchEnv has no per-env Python instance — env_method({method_name}) unsupported."
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        idx = self._indices(indices)
        return [False for _ in idx]

    def _indices(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    # ------------------------------------------------------------------
    # Step (the hot path, all batched)
    # ------------------------------------------------------------------
    def _step_batch(self, actions: torch.Tensor):
        """One batched env step.

        Args:
            actions: [N, 6] long indices.

        Returns:
            obs:    [N, obs_dim] float32 numpy
            reward: [N] float32 numpy
            done:   [N] bool numpy
            infos:  list[dict] of length N
        """
        profile_enabled = bool(getattr(self, "profile_timing", False))
        profile: Dict[str, Any] = {}
        if profile_enabled:
            def _sync_for_profile() -> None:
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)

            _sync_for_profile()
            t_prev = time.perf_counter()

            def mark(name: str) -> None:
                nonlocal t_prev
                _sync_for_profile()
                now = time.perf_counter()
                profile[f"{name}_s"] = now - t_prev
                t_prev = now
        else:
            def mark(name: str) -> None:
                return

        N = self.num_envs
        device = self.device
        if self.action_space_type == "discrete":
            actions = actions.long().clamp(self._idx_low, self._idx_up)  # [N, 6]
            self._last_action_idx = actions
        else:
            actions = actions.float()                                    # [N, 5]
            with torch.no_grad():
                raw_abs = actions.abs()
                self._cont_action_stat_steps += 1
                self._cont_action_abs_mean_sum += float(raw_abs.mean().detach().cpu().item())
                self._cont_action_abs_gt3_frac_sum += float((raw_abs > 3.0).float().mean().detach().cpu().item())
                self._cont_action_clip_frac_sum += float((raw_abs >= 7.99).float().mean().detach().cpu().item())
                self._cont_action_abs_max = max(
                    self._cont_action_abs_max,
                    float(raw_abs.max().detach().cpu().item()),
                )
        pose6, eyes, ats = self._decode_actions(actions)
        if self.action_space_type == "continuous_tanh":
            self._last_action_idx = self._pose6_to_action_idx(pose6)
        unique_mesh_ids = torch.unique(self._env_mesh_id).tolist()
        mark("decode_actions")

        # ---- Collision: mesh-level inside test plus surface-voxel fallback
        eye_idx = torch.floor((eyes - self._bbox_min) / self._voxel_size).long()  # [N, 3]
        in_box = ((eye_idx >= 0) & (eye_idx < self.grid_size)).all(dim=-1)        # [N]
        eye_idx_clamp = eye_idx.clamp(0, self.grid_size - 1)
        env_arange = torch.arange(N, device=device)
        eye_on_gt_surface = self._grid_gt[
            env_arange, eye_idx_clamp[:, 0], eye_idx_clamp[:, 1], eye_idx_clamp[:, 2]
        ] > 0.5
        surface_collision = in_box & eye_on_gt_surface
        mesh_inside = torch.zeros(N, dtype=torch.bool, device=device)
        for mid in unique_mesh_ids:
            sel = (self._env_mesh_id == mid).nonzero().flatten()
            mesh_inside[sel] = self._renderers[int(mid)].points_inside_mesh(eyes[sel])
        collision = surface_collision | mesh_inside                       # [N]
        mark("collision")

        H = W = self.render_size
        active_env = ~collision                                             # [N]
        if self.renderer_backend == "voxel_cuda":
            # The voxel renderer is already in reward-grid coordinates: one
            # CUDA thread traverses one pixel-center ray through grid_gt and
            # returns the first occupied voxel. This removes CPU Open3D and
            # also skips dense depth backprojection.
            rays_world = self._world_rays(eyes, ats)                       # [N, H*W, 3]
            mark("world_rays")
            target_idx_clamp, depth_flat, voxel_hit_mask = self._voxel_first_hit_render(
                eyes,
                rays_world,
            )
            hit_pixel_mask = voxel_hit_mask & active_env.view(N, 1)
            in_grid = hit_pixel_mask
            if profile_enabled:
                total_rays = float(N * H * W)
                hit_rays = float(voxel_hit_mask.sum().detach().cpu().item())
                profile["render/active_rays"] = total_rays
                profile["render/total_rays"] = total_rays
                profile["render/active_ratio"] = 1.0
                profile["render/bbox_ray_cull"] = 0.0
                profile["render/voxel_cuda"] = 1.0
                profile["render/hit_rays"] = hit_rays
                profile["render/hit_ratio"] = hit_rays / max(total_rays, 1.0)
            mark("render")
            mark("backproject")
        else:
            # ---- Render: group envs by mesh_id, one render_batch per group.
            # Single-mesh pool collapses to one render_batch covering all
            # N envs (same speed as before). Multi-mesh: still 1 GPU/CPU op
            # per unique mesh, with envs scattered back to per-env tensors.
            depth = torch.zeros(N, H, W, dtype=torch.float32, device=device)
            alpha = torch.zeros(N, H, W, dtype=torch.float32, device=device)
            direct_points: Optional[torch.Tensor] = None
            render_active_rays = 0.0
            render_total_rays = 0.0
            render_bbox_cull = 0.0
            render_hit_rays = 0.0
            for mid in unique_mesh_ids:
                sel = (self._env_mesh_id == mid).nonzero().flatten()       # [k] indices
                sub_eyes = eyes[sel]                                        # [k, 3]
                sub_ats = ats[sel]                                          # [k, 3]
                renderer = self._renderers[int(mid)]
                out = renderer.render_batch(sub_eyes, sub_ats)
                depth[sel] = out.depth
                alpha[sel] = out.alpha
                if out.points is not None:
                    if direct_points is None:
                        direct_points = torch.zeros(
                            N, H, W, 3, dtype=torch.float32, device=device
                        )
                    direct_points[sel] = out.points.to(device=device, dtype=torch.float32)
                if profile_enabled:
                    rstats = getattr(renderer, "last_render_stats", {}) or {}
                    render_active_rays += float(rstats.get("active_rays", 0.0))
                    render_total_rays += float(rstats.get("total_rays", 0.0))
                    render_bbox_cull = max(render_bbox_cull, float(rstats.get("bbox_ray_cull", 0.0)))
                    render_hit_rays += float(rstats.get("hit_rays", 0.0))
            if profile_enabled:
                profile["render/active_rays"] = render_active_rays
                profile["render/total_rays"] = render_total_rays
                profile["render/active_ratio"] = (
                    render_active_rays / max(render_total_rays, 1.0)
                )
                profile["render/bbox_ray_cull"] = render_bbox_cull
                profile["render/hit_rays"] = render_hit_rays
                profile["render/hit_ratio"] = render_hit_rays / max(render_total_rays, 1.0)
                profile["render/direct_points"] = 1.0 if direct_points is not None else 0.0
            mark("render")

            rays_world = self._world_rays(eyes, ats)                       # [N, H*W, 3]
            mark("world_rays")

            # Backproject every pixel to 3D world coords (only foreground used).
            depth_flat = depth.view(N, H * W)                               # [N, R]
            alpha_flat = alpha.view(N, H * W)
            if direct_points is None:
                pts_world = eyes.unsqueeze(1) + depth_flat.unsqueeze(-1) * rays_world
            else:
                # nvdiffrast can interpolate the visible surface xyz directly.
                # Use that point rather than converting it to scalar depth and
                # re-backprojecting through a second pixel-convention path.
                pts_world = direct_points.view(N, H * W, 3)
            hit_pixel_mask = (alpha_flat > 0.5) & active_env.view(N, 1)     # [N, R]
            mark("backproject")

            # Voxelize: which (env, voxel) cells got hit by any ray?
            target_idx = torch.floor(
                (pts_world - self._bbox_min.unsqueeze(1)) / self._voxel_size.unsqueeze(1)
            ).long()                                                       # [N, R, 3]
            in_grid = ((target_idx >= 0) & (target_idx < self.grid_size)).all(dim=-1) & hit_pixel_mask
            target_idx_clamp = target_idx.clamp(0, self.grid_size - 1)     # [N, R, 3]

        # `hit_grid` [N, G, G, G] = 1 where any pixel hit that voxel.
        hit_grid = torch.zeros_like(self._grid_gt, dtype=torch.float32)
        env_ids_per_pixel = env_arange.view(N, 1).expand(N, H * W)        # [N, R]
        env_ids_flat = env_ids_per_pixel[in_grid]                          # [M]
        flat_idx = target_idx_clamp[in_grid]                               # [M, 3]
        hit_grid[env_ids_flat, flat_idx[:, 0], flat_idx[:, 1], flat_idx[:, 2]] = 1.0
        mark("hit_voxelize")
        coverage_hit_grid = hit_grid
        if self.coverage_hit_dilate_radius > 0:
            r = self.coverage_hit_dilate_radius
            coverage_hit_grid = F.max_pool3d(
                hit_grid.unsqueeze(1),
                kernel_size=2 * r + 1,
                stride=1,
                padding=r,
            ).squeeze(1)
        mark("hit_dilate")

        # ---- Update prob_grid + scanned_gt_grid --------------------
        # Endpoint hard-assign (hit voxels → +LOG_ODDS_OCCUPIED in prob_grid).
        self._prob_grid = torch.where(
            hit_grid > 0.5,
            torch.full_like(self._prob_grid, LOG_ODDS_OCCUPIED),
            self._prob_grid,
        )
        mark("occupied_update")
        if not self.skip_free_raycast:
            self._update_free_voxels(
                eyes,
                target_idx_clamp,
                in_grid,
                rays_world=rays_world,
                hit_pixel_mask=hit_pixel_mask | collision.view(N, 1),
            )
        mark("free_update")
        self._prob_grid = self._prob_grid.clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)
        mark("prob_clamp")

        # Sticky GT-hit grid (used for CR and reward). Keep the previous
        # coverage mask around so the reward can explicitly separate
        # truly new information from overlap with already-scanned surface.
        prev_scanned_gt = self._scanned_gt_grid
        hit_gt_grid = coverage_hit_grid * self._grid_gt
        new_gt_grid = hit_gt_grid * (1.0 - prev_scanned_gt)
        visible_gt_voxels = hit_gt_grid.sum(dim=(1, 2, 3))
        new_gt_voxels = new_gt_grid.sum(dim=(1, 2, 3))
        redundant_gt_voxels = (hit_gt_grid * prev_scanned_gt).sum(dim=(1, 2, 3))
        prev_covered_voxels = prev_scanned_gt.sum(dim=(1, 2, 3))
        remaining_gt_voxels = (self._num_valid_gt_per_env - prev_covered_voxels).clamp(min=1.0)
        linear_delta = new_gt_voxels / self._num_valid_gt_per_env
        novelty_ratio = new_gt_voxels / visible_gt_voxels.clamp(min=1.0)
        redundancy_ratio = redundant_gt_voxels / visible_gt_voxels.clamp(min=1.0)
        remaining_gain = new_gt_voxels / remaining_gt_voxels

        self._scanned_gt_grid = torch.clamp(
            prev_scanned_gt + hit_gt_grid, min=0.0, max=1.0,
        )
        mark("coverage_update")

        # ---- Coverage + reward -------------------------------------
        cr = self._scanned_gt_grid.sum(dim=(1, 2, 3)) / self._num_valid_gt_per_env  # [N]
        if self.coverage_reward_type == "log":
            # log-coverage: reward ∝ log(1 - cr_prev) - log(1 - cr_now).
            # Steeper near cr=1 — so going 0.84 → 0.99 gives ~2.77 vs
            # linear 0.15. Critical for ShapeNet where random already
            # achieves cr=0.83 and linear delta has ~no headroom.
            eps = 1e-3
            cov_delta = (
                torch.log((1.0 - self._cr_prev).clamp(min=eps))
                - torch.log((1.0 - cr).clamp(min=eps))
            )
        elif self.coverage_reward_type == "remaining":
            cov_delta = remaining_gain
        elif self.coverage_reward_type == "information_gain":
            cov_delta = linear_delta
        else:
            cov_delta = linear_delta

        revisit_penalty = torch.zeros(N, dtype=torch.float32, device=device)
        if self.view_revisit_penalty_scale != 0.0:
            past_pos = self._action_history[:, :, :3]                         # [N, B, 3]
            past_norm = past_pos.norm(dim=-1)                                  # [N, B]
            cur_norm = eyes.norm(dim=-1)                                       # [N]
            valid = (past_norm > 1e-6) & (cur_norm[:, None] > 1e-6)
            past_dir = past_pos / past_norm.clamp(min=1e-6).unsqueeze(-1)
            cur_dir = eyes / cur_norm.clamp(min=1e-6).unsqueeze(-1)
            cos = (past_dir * cur_dir[:, None, :]).sum(dim=-1)
            cos = torch.where(valid, cos, torch.full_like(cos, -1.0))
            max_cos = cos.max(dim=1).values
            cos_thresh = math.cos(math.radians(self.view_revisit_angle_deg))
            revisit_penalty = ((max_cos - cos_thresh) / max(1.0 - cos_thresh, 1e-6)).clamp(0.0, 1.0)
        self._cr_prev = cr
        self._step_idx = self._step_idx + 1

        # Short-path penalty: clip(step - grace, 0, clip)
        extra = (self._step_idx.float() - self.short_path_grace).clamp(min=0.0)
        extra = extra.clamp(max=self.short_path_clip)
        reward = self.coverage_reward_scale * cov_delta - self.short_path_scale * extra
        if self.coverage_reward_type == "information_gain":
            reward = (
                reward
                + self.novelty_reward_scale * novelty_ratio
                + self.remaining_reward_scale * remaining_gain
                - self.redundancy_penalty_scale * redundancy_ratio
            )
        if self.view_revisit_penalty_scale != 0.0:
            reward = reward - self.view_revisit_penalty_scale * revisit_penalty
        if self.only_positive_rewards:
            reward = reward.clamp(min=0.0)

        timeout = self._step_idx >= self.episode_len                       # [N]
        coverage_done = cr > self.cr_success_threshold                     # [N]
        terminated = coverage_done | collision                             # [N]
        truncated = timeout & ~terminated                                  # [N]
        success = coverage_done & ~collision
        reward = reward + self.termination_bonus * success.float()
        # Apply collision penalty AFTER the only_positive clamp + bonus,
        # so it actually pushes reward negative and gives PPO a real
        # gradient to avoid colliding viewpoints.
        if self.collision_penalty != 0.0:
            reward = reward - self.collision_penalty * collision.float()
        mark("reward")

        # Push into history.
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._action_history[:, -1, :] = pose6

        # Track ep stats.
        self._ep_step_count = self._ep_step_count + 1
        self._ep_reward_sum = self._ep_reward_sum + reward
        self._ep_new_gt_sum = self._ep_new_gt_sum + new_gt_voxels
        self._ep_visible_gt_sum = self._ep_visible_gt_sum + visible_gt_voxels
        self._ep_redundant_gt_sum = self._ep_redundant_gt_sum + redundant_gt_voxels
        self._ep_revisit_sum = self._ep_revisit_sum + revisit_penalty

        done = terminated | truncated                                      # [N]

        # Build info dicts BEFORE reset (for terminal_observation).
        infos: List[Dict[str, Any]] = []
        cr_cpu = cr.detach().cpu().numpy()
        col_cpu = collision.detach().cpu().numpy()
        term_cpu = terminated.detach().cpu().numpy()
        trunc_cpu = truncated.detach().cpu().numpy()
        ep_len_cpu = self._ep_step_count.detach().cpu().numpy()
        ep_rew_cpu = self._ep_reward_sum.detach().cpu().numpy()
        mesh_id_cpu = self._env_mesh_id.detach().cpu().numpy()
        cov_delta_cpu = linear_delta.detach().cpu().numpy()
        novelty_cpu = novelty_ratio.detach().cpu().numpy()
        redundancy_cpu = redundancy_ratio.detach().cpu().numpy()
        remaining_cpu = remaining_gain.detach().cpu().numpy()
        visible_cpu = visible_gt_voxels.detach().cpu().numpy()
        new_gt_cpu = new_gt_voxels.detach().cpu().numpy()
        revisit_cpu = revisit_penalty.detach().cpu().numpy()
        ep_new_cpu = self._ep_new_gt_sum.detach().cpu().numpy()
        ep_visible_cpu = self._ep_visible_gt_sum.detach().cpu().numpy()
        ep_redundant_cpu = self._ep_redundant_gt_sum.detach().cpu().numpy()
        ep_revisit_cpu = self._ep_revisit_sum.detach().cpu().numpy()
        mark("info_cpu_transfer")
        for i in range(N):
            mesh_id = int(mesh_id_cpu[i])
            ep_visible = float(ep_visible_cpu[i])
            d: Dict[str, Any] = {
                "cr": float(cr_cpu[i]),
                "collision": bool(col_cpu[i]),
                "step_idx": int(ep_len_cpu[i]),
                "early_stopped": bool(term_cpu[i] and not col_cpu[i]),
                "mesh_id": mesh_id,
                "seq_name": self._pool_names[mesh_id] if mesh_id < len(self._pool_names) else "",
                "coverage_delta": float(cov_delta_cpu[i]),
                "new_gt_voxels": float(new_gt_cpu[i]),
                "visible_gt_voxels": float(visible_cpu[i]),
                "novelty_ratio": float(novelty_cpu[i]),
                "redundancy_ratio": float(redundancy_cpu[i]),
                "remaining_gain": float(remaining_cpu[i]),
                "view_revisit_penalty": float(revisit_cpu[i]),
            }
            if bool(term_cpu[i] or trunc_cpu[i]):
                d["episode"] = {
                    "r": float(ep_rew_cpu[i]),
                    "l": int(ep_len_cpu[i]),
                }
                d["episode_new_gt_voxels"] = float(ep_new_cpu[i])
                d["episode_visible_gt_voxels"] = ep_visible
                d["episode_redundant_gt_voxels"] = float(ep_redundant_cpu[i])
                d["episode_novelty_ratio"] = (
                    float(ep_new_cpu[i]) / max(ep_visible, 1.0)
                )
                d["episode_redundancy_ratio"] = (
                    float(ep_redundant_cpu[i]) / max(ep_visible, 1.0)
                )
                d["episode_revisit_penalty_mean"] = (
                    float(ep_revisit_cpu[i]) / max(float(ep_len_cpu[i]), 1.0)
                )
                d["TimeLimit.truncated"] = bool(trunc_cpu[i] and not term_cpu[i])
            infos.append(d)

        # Build obs (uses current grids — pre-reset).
        obs = self._build_observation_np()
        mark("obs_build")
        obs_terminal = obs.copy()
        done_cpu = done.detach().cpu().numpy()
        for i, is_done in enumerate(done_cpu):
            if is_done:
                infos[i]["terminal_observation"] = obs_terminal[i]

        # In-place reset of done envs.
        done_idx = torch.nonzero(done).flatten()
        if done_idx.numel() > 0:
            self._reset_envs(done_idx)
        mark("reset_done_envs")

        rew_np = reward.detach().cpu().numpy()
        done_np = done.detach().cpu().numpy()
        mark("return_cpu_transfer")

        # Return obs that reflects the FRESH state of just-reset envs.
        if done_idx.numel() > 0:
            obs = self._build_observation_np()
            mark("obs_rebuild_after_reset")

        if profile_enabled:
            timing_total = sum(
                v for k, v in profile.items()
                if k.endswith("_s") and isinstance(v, (float, int))
            )
            profile["timed_total_s"] = float(timing_total)
            profile.update({
                f"free/{k}": v for k, v in getattr(self, "_last_free_update_stats", {}).items()
            })
            self.last_step_profile = profile
        else:
            self.last_step_profile = {}

        return obs, rew_np, done_np, infos

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------
    def _decode_actions(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actions -> decoded pose history, eye positions and look-at.

        Discrete mode consumes GenNBV-style action indices. Continuous
        mode consumes raw Gaussian policy samples and maps them through a
        tanh transform to keep bounded physical variables strictly inside
        their intended Cube Mode ranges.
        """
        if self.action_space_type == "discrete":
            pose6 = actions.float() * self._action_unit + self._action_low  # [N, 6]
        else:
            raw = actions.float()
            if raw.ndim != 2 or raw.shape[-1] != 5:
                raise ValueError(
                    "continuous_tanh actions must have shape [N, 5] "
                    f"(x,y,z,pitch,yaw raw), got {tuple(raw.shape)}"
                )
            squashed = torch.tanh(raw)
            pose6 = torch.zeros(raw.shape[0], POSE_DIM, dtype=torch.float32, device=self.device)
            pose6[:, 0:3] = squashed[:, 0:3]
            pose6[:, 3] = 0.0
            pose6[:, 4] = squashed[:, 3] * (math.pi / 2.0)
            pose6[:, 5] = (squashed[:, 4] + 1.0) * math.pi
        eyes = pose6[:, :3]                                             # [N, 3]
        if self.auto_lookat_center:
            ats = torch.zeros_like(eyes)
        else:
            pitches = pose6[:, 4]
            yaws = pose6[:, 5]
            cp, sp = torch.cos(pitches), torch.sin(pitches)
            cy, sy = torch.cos(yaws), torch.sin(yaws)
            look_dirs = torch.stack([cp * cy, cp * sy, sp], dim=-1)     # [N, 3]
            ats = eyes + look_dirs
        return pose6, eyes, ats

    def _pose6_to_action_idx(self, pose6: torch.Tensor) -> torch.Tensor:
        """Approximate a decoded continuous pose on the discrete grid.

        This is used only for diagnostics / visualization fields that
        historically expected integer action indices. The environment
        dynamics keep using the continuous `pose6` above.
        """
        idx = torch.round((pose6 - self._action_low) / self._action_unit.clamp(min=1e-12)).long()
        idx[:, 3] = 0
        return idx.clamp(self._idx_low, self._idx_up)

    # ------------------------------------------------------------------
    # World-ray construction: per-camera rays in world frame.
    # ------------------------------------------------------------------
    def _build_camera_rays(self) -> torch.Tensor:
        """Build fixed camera-frame pixel-center rays for this render setup."""
        H, W = self.render_size, self.render_size
        fov_rad = math.radians(self.fov_deg)
        f = 0.5 * H / math.tan(0.5 * fov_rad)
        cx, cy = 0.5 * W, 0.5 * H
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        ray_x = (xs - cx) / f
        ray_y = (ys - cy) / f
        rays_cam = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
        return rays_cam.reshape(-1, 3).contiguous()

    def _world_rays(self, eyes: torch.Tensor, ats: torch.Tensor) -> torch.Tensor:
        """[N, 3] eye + [N, 3] at → [N, H*W, 3] unit world rays."""
        N = eyes.shape[0]
        device = eyes.device
        rays_cam = self._rays_cam.to(device=device)                              # [HW, 3]

        # Per-camera basis (right, true_up, look) → R_cw [N, 3, 3].
        look = (ats - eyes)
        look = look / (look.norm(dim=-1, keepdim=True) + 1e-8)
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32).expand(N, 3).clone()
        # Avoid degenerate up // look.
        deg = (look * up).sum(dim=-1).abs() > 0.999
        if deg.any():
            up_alt = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            up[deg] = up_alt
        right = torch.linalg.cross(up, look, dim=-1)
        right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
        true_up = torch.linalg.cross(look, right, dim=-1)
        # R_cw cols = (right, true_up, look).
        R_cw = torch.stack([right, true_up, look], dim=-1)  # [N, 3, 3]
        # rays_world = R_cw @ rays_cam^T  → [N, 3, HW] → [N, HW, 3]
        rays_world = torch.einsum("nij,hj->nhi", R_cw, rays_cam)
        return rays_world

    def _voxel_first_hit_render(
        self,
        eyes: torch.Tensor,
        rays_world: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render first occupied reward-grid voxel for every pixel ray.

        Returns:
            target_idx: [N, H*W, 3] long. Meaningful only where hit_mask is true.
            depth: [N, H*W] float ray distance to the entered occupied voxel.
            hit_mask: [N, H*W] bool.
        """
        if voxel_first_hit_cuda is not None:
            out = voxel_first_hit_cuda(
                self._grid_gt,
                self._bbox_min,
                self._voxel_size,
                eyes,
                rays_world,
                occ_threshold=0.5,
            )
            if out is not None:
                return out

        if self.device.type == "cuda":
            reason = (
                voxel_renderer_extension_error()
                if callable(voxel_renderer_extension_error)
                else "extension import failed"
            )
            raise RuntimeError(
                "renderer_backend='voxel_cuda' requires the custom CUDA voxel "
                f"renderer, but it is unavailable: {reason}"
            )
        if voxel_first_hit_reference is None:
            raise RuntimeError("voxel renderer reference fallback is unavailable")
        return voxel_first_hit_reference(
            self._grid_gt,
            self._bbox_min,
            self._voxel_size,
            eyes,
            rays_world,
            occ_threshold=0.5,
        )

    # ------------------------------------------------------------------
    # Free-voxel raycast: scatter voxel-traversal paths along
    # (eye voxel, target voxel) into _prob_grid as -LOG_ODDS_FREE.
    # ------------------------------------------------------------------
    def _update_free_voxels(
        self,
        eyes: torch.Tensor,         # [N, 3] world
        target_idx: torch.Tensor,   # [N, R, 3] long voxel idx
        valid_mask: torch.Tensor,   # [N, R] bool
        rays_world: Optional[torch.Tensor] = None,      # [N, R, 3] unit rays
        hit_pixel_mask: Optional[torch.Tensor] = None,  # [N, R] bool
    ) -> None:
        """Run voxel-index Bresenham paths and decrement free cells.

        GenNBV's CUDA raycast includes the camera/source voxel when it
        lies in the map and then hard-assigns hit targets to occupied.
        GeoScout applies the hit assignment before this free update, so
        we keep the source voxel for GenNBV-like semantics but explicitly
        remove all hit endpoints from the free set.  If
        `update_empty_rays` is enabled, alpha-miss rays are also traversed
        from their grid-box entry voxel to their grid-box exit voxel.
        """
        profile_enabled = bool(getattr(self, "profile_timing", False))
        stats: Dict[str, Any] = {}
        if profile_enabled:
            stats = {
                "hit_valid_pixels": int(valid_mask.sum().detach().cpu().item()),
                "hit_unique_targets": 0,
                "hit_path_voxels": 0,
                "miss_pixels": 0,
                "miss_intersect_rays": 0,
                "empty_ray_pairs_before_unique": 0,
                "empty_ray_pairs_after_unique": 0,
                "empty_path_voxels": 0,
                "union_free_voxels": 0,
                "cuda_empty_pair_builder": 0,
                "dedupe_empty_ray_pairs": int(bool(getattr(self, "dedupe_empty_ray_pairs", False))),
                "free_raycast_backend": str(getattr(self, "free_raycast_backend", "auto")),
                "free_raycast_backend_used": "",
                "free_mask_apply_mode": str(getattr(self, "free_mask_apply_mode", "index")),
                "triton_bresenham_block_rays": int(getattr(self, "triton_bresenham_block_rays", 1)),
            }
        self._last_free_update_stats = stats

        # Historical config gate: <= 0 still disables free raycast, but
        # positive values no longer control path resolution.
        if self.n_free_samples_per_ray <= 0:
            return
        N, _ = valid_mask.shape
        device = eyes.device
        G = self.grid_size

        source_idx = torch.floor((eyes - self._bbox_min) / self._voxel_size).long()  # [N, 3]
        if (
            bool(getattr(self, "use_triton_free_raycast", True))
            and str(getattr(self, "free_raycast_backend", "auto")) != "torch"
            and (
                scatter_bresenham3d_to_mask_cuda is not None
                or scatter_bresenham3d_to_mask is not None
            )
            and device.type == "cuda"
        ):
            used_triton = self._update_free_voxels_triton(
                eyes=eyes,
                source_idx=source_idx,
                target_idx=target_idx,
                valid_mask=valid_mask,
                rays_world=rays_world,
                hit_pixel_mask=hit_pixel_mask,
                stats=stats if profile_enabled else None,
            )
            if used_triton:
                return

        flat_env_parts = []
        flat_idx_parts = []

        # Strict traversal intentionally trades the old one-shot K-sample
        # approximation for official Bresenham semantics. The Python
        # env loop is the main performance risk when N or render_size is
        # large; use skip_free_raycast for smoke-speed runs.
        for env_idx in range(N):
            targets = target_idx[env_idx, valid_mask[env_idx]]                    # [Ri, 3]
            if targets.numel() > 0:
                targets = torch.unique(targets, dim=0, sorted=False)
                if profile_enabled:
                    stats["hit_unique_targets"] += int(targets.shape[0])

                paths = bresenham3D_strict(
                    pts_source=source_idx[env_idx: env_idx + 1],
                    pts_target=targets,
                    map_size=G,
                    include_source=True,
                    include_target=False,
                )
                paths = self._remove_hit_endpoints(paths, targets)
                if paths.numel() > 0:
                    if profile_enabled:
                        stats["hit_path_voxels"] += int(paths.shape[0])
                    flat_env_parts.append(torch.full(
                        (paths.shape[0],), env_idx, dtype=torch.long, device=device,
                    ))
                    flat_idx_parts.append(paths)

            if (
                getattr(self, "update_empty_rays", False)
                and rays_world is not None
                and hit_pixel_mask is not None
            ):
                miss_mask = ~hit_pixel_mask[env_idx]
                if profile_enabled:
                    stats["miss_pixels"] += int(miss_mask.sum().detach().cpu().item())
                paths = self._empty_ray_paths_for_env(
                    env_idx,
                    eyes[env_idx],
                    rays_world[env_idx],
                    miss_mask,
                    stats=stats if profile_enabled else None,
                )
                if targets.numel() > 0:
                    paths = self._remove_hit_endpoints(paths, targets)
                if paths.numel() > 0:
                    if profile_enabled:
                        stats["empty_path_voxels"] += int(paths.shape[0])
                    flat_env_parts.append(torch.full(
                        (paths.shape[0],), env_idx, dtype=torch.long, device=device,
                    ))
                    flat_idx_parts.append(paths)

        if not flat_idx_parts:
            return

        flat_env = torch.cat(flat_env_parts, dim=0)
        flat_idx = torch.cat(flat_idx_parts, dim=0)
        # Section 7 defines F_t as a set union.  Hit-ray and miss-ray
        # traversals can overlap, so deduplicate across both sources
        # before applying the per-step free log-odds decrement.
        env_and_idx = torch.cat([flat_env.unsqueeze(1), flat_idx], dim=1)
        env_and_idx = torch.unique(env_and_idx, dim=0)
        if profile_enabled:
            stats["union_free_voxels"] = int(env_and_idx.shape[0])
        flat_env = env_and_idx[:, 0]
        flat_idx = env_and_idx[:, 1:]
        self._prob_grid.index_put_(
            (flat_env, flat_idx[:, 0], flat_idx[:, 1], flat_idx[:, 2]),
            values=torch.full((flat_env.shape[0],), LOG_ODDS_FREE,
                              dtype=torch.float32, device=device),
            accumulate=True,
        )

    def _update_free_voxels_triton(
        self,
        *,
        eyes: torch.Tensor,              # [N, 3] world
        source_idx: torch.Tensor,        # [N, 3]
        target_idx: torch.Tensor,        # [N, R, 3]
        valid_mask: torch.Tensor,        # [N, R]
        rays_world: Optional[torch.Tensor],
        hit_pixel_mask: Optional[torch.Tensor],
        stats: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """CUDA/Triton implementation of the same free-space set-union update."""
        if (
            source_idx.device.type != "cuda"
            or (
                scatter_bresenham3d_to_mask_cuda is None
                and scatter_bresenham3d_to_mask is None
            )
        ):
            return False

        N = self.num_envs
        G = self.grid_size
        device = source_idx.device
        env_arange = torch.arange(N, device=device)

        if stats is not None:
            def _sync_free_profile() -> None:
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)

            _sync_free_profile()
            t_free_prev = time.perf_counter()

            def free_mark(name: str) -> None:
                nonlocal t_free_prev
                _sync_free_profile()
                now = time.perf_counter()
                stats[f"{name}_s"] = now - t_free_prev
                t_free_prev = now
        else:
            def free_mark(name: str) -> None:
                return

        backend_pref = str(getattr(self, "free_raycast_backend", "auto"))

        def scatter_paths(
            env_ids: torch.Tensor,
            sources: torch.Tensor,
            targets: torch.Tensor,
            *,
            include_source: bool,
            include_target: bool,
            out_mask: torch.Tensor,
            max_steps: int,
            kind: str,
        ) -> Optional[torch.Tensor]:
            """Try the requested accelerated scatter backend."""
            if backend_pref in ("auto", "cuda") and scatter_bresenham3d_to_mask_cuda is not None:
                out = scatter_bresenham3d_to_mask_cuda(
                    env_ids,
                    sources,
                    targets,
                    num_envs=N,
                    grid_size=G,
                    include_source=include_source,
                    include_target=include_target,
                    out_mask=out_mask,
                    max_steps=max_steps,
                )
                if out is not None:
                    if stats is not None:
                        stats[f"cuda_{kind}_raycast"] = 1
                        stats["free_raycast_backend_used"] = "cuda"
                    return out
                if backend_pref == "cuda":
                    if stats is not None:
                        stats[f"cuda_{kind}_raycast"] = 0
                    return None

            if backend_pref in ("auto", "triton") and scatter_bresenham3d_to_mask is not None:
                out = scatter_bresenham3d_to_mask(
                    env_ids,
                    sources,
                    targets,
                    num_envs=N,
                    grid_size=G,
                    include_source=include_source,
                    include_target=include_target,
                    out_mask=out_mask,
                    max_steps=max_steps,
                    block_rays=self.triton_bresenham_block_rays,
                )
                if out is not None:
                    if stats is not None:
                        stats[f"triton_{kind}_raycast"] = 1
                        if not stats.get("free_raycast_backend_used"):
                            stats["free_raycast_backend_used"] = "triton"
                    return out

            return None

        free_mask = getattr(self, "_free_mask_buffer", None)
        if (
            free_mask is None
            or free_mask.shape != (N, G, G, G)
            or free_mask.device != device
            or free_mask.dtype != torch.uint8
        ):
            free_mask = torch.empty((N, G, G, G), dtype=torch.uint8, device=device)
            self._free_mask_buffer = free_mask
        free_mask.zero_()
        free_mark("mask_alloc")
        has_free_paths = False

        hit_env = env_arange.view(N, 1).expand_as(valid_mask)[valid_mask]
        hit_targets = target_idx[valid_mask]
        if hit_targets.numel() > 0:
            hit_pairs = hit_targets.new_empty((0, 4))
            hit_pairs = torch.cat([hit_env.view(-1, 1), hit_targets], dim=1)
            hit_pairs = torch.unique(hit_pairs, dim=0, sorted=False)
            hit_env = hit_pairs[:, 0]
            hit_targets = hit_pairs[:, 1:]
            if stats is not None:
                stats["hit_unique_targets"] = int(hit_targets.shape[0])
            hit_sources = source_idx[hit_env]
            hit_max_delta = int((hit_targets - hit_sources).abs().max().detach().cpu().item())
            if stats is not None:
                stats["hit_max_delta"] = hit_max_delta
            free_mark("hit_unique")
            can_try_fast_hit = (
                hit_max_delta <= 3 * G
                or (
                    backend_pref in ("auto", "cuda")
                    and scatter_bresenham3d_to_mask_cuda is not None
                )
            )
            if can_try_fast_hit:
                fast_hit_mask = scatter_paths(
                    hit_env,
                    hit_sources,
                    hit_targets,
                    include_source=True,
                    include_target=False,
                    out_mask=free_mask,
                    max_steps=max(1, hit_max_delta),
                    kind="hit",
                )
                if fast_hit_mask is not None:
                    free_mask = fast_hit_mask
                    if stats is not None:
                        stats["triton_hit_raycast"] = int(stats.get("triton_hit_raycast", 0))
                        stats["cuda_hit_raycast"] = int(stats.get("cuda_hit_raycast", 0))
                    has_free_paths = True
                    free_mark("hit_scatter")
                elif hit_max_delta <= 3 * G:
                    return False
                else:
                    can_try_fast_hit = False
            else:
                free_mark("hit_scatter")

            if not can_try_fast_hit:
                # Thin-bbox safety valve: if a camera/source voxel is very
                # far outside the object grid and the selected backend is
                # Triton-only, use the original PyTorch strict traversal
                # rather than compiling a very long static Triton loop.
                # The custom CUDA backend uses a runtime loop and therefore
                # normally avoids this slow path.
                if stats is not None:
                    stats["triton_hit_raycast"] = 0
                    stats["cuda_hit_raycast"] = 0
                for env_idx in range(N):
                    env_targets = hit_targets[hit_env == env_idx]
                    if env_targets.numel() == 0:
                        continue
                    paths = bresenham3D_strict(
                        pts_source=source_idx[env_idx: env_idx + 1],
                        pts_target=env_targets,
                        map_size=G,
                        include_source=True,
                        include_target=False,
                    )
                    paths = self._remove_hit_endpoints(paths, env_targets)
                    if paths.numel() == 0:
                        continue
                    if stats is not None:
                        stats["hit_path_voxels"] += int(paths.shape[0])
                    free_mask[env_idx, paths[:, 0], paths[:, 1], paths[:, 2]] = 1
                    has_free_paths = True
                free_mark("hit_fallback")
        else:
            hit_env = torch.empty((0,), dtype=torch.long, device=device)
            hit_targets = torch.empty((0, 3), dtype=torch.long, device=device)
            free_mark("hit_unique")

        if (
            getattr(self, "update_empty_rays", False)
            and rays_world is not None
            and hit_pixel_mask is not None
        ):
            empty_result = None
            if (
                empty_ray_pairs_cuda is not None
                and eyes.is_cuda
                and rays_world.is_cuda
                and hit_pixel_mask.is_cuda
                and os.environ.get("SHAPENBV_DISABLE_CUDA_EMPTY_PAIRS", "0") != "1"
            ):
                empty_result = empty_ray_pairs_cuda(
                    eyes,
                    rays_world,
                    hit_pixel_mask,
                    self._bbox_min,
                    self._voxel_size,
                    grid_size=G,
                    dedupe=bool(getattr(self, "dedupe_empty_ray_pairs", False)),
                )
            if empty_result is not None:
                empty_env, empty_sources, empty_targets, pairs_before, pairs_after = empty_result
                if stats is not None:
                    stats["miss_pixels"] += int((~hit_pixel_mask).sum().detach().cpu().item())
                    stats["miss_intersect_rays"] += int(pairs_before)
                    stats["empty_ray_pairs_before_unique"] += int(pairs_before)
                    stats["empty_ray_pairs_after_unique"] += int(pairs_after)
                    stats["cuda_empty_pair_builder"] = 1
            else:
                empty_env_parts = []
                empty_src_parts = []
                empty_tgt_parts = []
                for env_idx in range(N):
                    miss_mask = ~hit_pixel_mask[env_idx]
                    if stats is not None:
                        stats["miss_pixels"] += int(miss_mask.sum().detach().cpu().item())
                    start_idx, end_idx = self._empty_ray_pairs_for_env(
                        env_idx,
                        eyes[env_idx],
                        rays_world[env_idx],
                        miss_mask,
                        stats=stats,
                    )
                    if start_idx.numel() == 0:
                        continue
                    empty_env_parts.append(torch.full(
                        (start_idx.shape[0],), env_idx, dtype=torch.long, device=device,
                    ))
                    empty_src_parts.append(start_idx)
                    empty_tgt_parts.append(end_idx)
                if empty_src_parts:
                    empty_env = torch.cat(empty_env_parts, dim=0)
                    empty_sources = torch.cat(empty_src_parts, dim=0)
                    empty_targets = torch.cat(empty_tgt_parts, dim=0)
                else:
                    empty_env = torch.empty((0,), dtype=torch.long, device=device)
                    empty_sources = torch.empty((0, 3), dtype=torch.long, device=device)
                    empty_targets = torch.empty((0, 3), dtype=torch.long, device=device)
            free_mark("empty_pairs")
            if empty_sources.numel() > 0:
                free_mask = scatter_paths(
                    empty_env,
                    empty_sources,
                    empty_targets,
                    include_source=True,
                    include_target=True,
                    out_mask=free_mask,
                    max_steps=G,
                    kind="free",
                )
                if free_mask is None:
                    return False
                has_free_paths = True
                free_mark("empty_scatter")
            else:
                free_mark("empty_scatter")
        else:
            free_mark("empty_pairs")

        # Hit endpoints are occupied, never free, even if an empty ray or
        # a hit-source path traversed the same voxel.
        if hit_targets.numel() > 0:
            free_mask[
                hit_env,
                hit_targets[:, 0],
                hit_targets[:, 1],
                hit_targets[:, 2],
            ] = 0
        free_mark("clear_hit_endpoints")

        if not has_free_paths:
            if stats is not None:
                stats["union_free_voxels"] = 0
                stats["triton_free_raycast"] = 1
                stats["cuda_free_raycast"] = int(stats.get("cuda_free_raycast", 0))
            return True

        apply_mode = str(getattr(self, "free_mask_apply_mode", "index"))
        if apply_mode == "index":
            flat_free = torch.nonzero(free_mask.view(-1), as_tuple=False).flatten()
            free_mark("mask_nonzero")
            if stats is not None:
                stats["union_free_voxels"] = int(flat_free.shape[0])
                stats["triton_free_raycast"] = int(
                    stats.get("triton_free_raycast", 0)
                    or stats.get("triton_hit_raycast", 0)
                )
                stats["cuda_free_raycast"] = int(
                    stats.get("cuda_free_raycast", 0)
                    or stats.get("cuda_hit_raycast", 0)
                )
                stats["free_mask_apply_index"] = 1
            if flat_free.numel() == 0:
                return True

            self._prob_grid.view(-1).index_put_(
                (flat_free,),
                values=torch.full(
                    (flat_free.shape[0],), LOG_ODDS_FREE,
                    dtype=torch.float32,
                    device=device,
                ),
                accumulate=True,
            )
            free_mark("mask_apply")
            return True

        union_free = None
        if stats is not None:
            union_free = int(free_mask.sum().detach().cpu().item())
            stats["union_free_voxels"] = union_free
            stats["triton_free_raycast"] = int(
                stats.get("triton_free_raycast", 0)
                or stats.get("triton_hit_raycast", 0)
            )
            stats["cuda_free_raycast"] = int(
                stats.get("cuda_free_raycast", 0)
                or stats.get("cuda_hit_raycast", 0)
            )
            free_mark("mask_count")
        if union_free == 0:
            return True

        used_triton_apply = False
        if apply_mode == "triton" and apply_free_mask_to_grid is not None:
            used_triton_apply = apply_free_mask_to_grid(
                self._prob_grid,
                free_mask,
                delta=LOG_ODDS_FREE,
            )
        if used_triton_apply:
            if stats is not None:
                stats["free_mask_apply_triton"] = 1
            free_mark("mask_apply")
            return True

        # Dense PyTorch fallback: same set-union semantics as the Triton
        # mask application, but with an intermediate float mask allocation.
        self._prob_grid.add_(free_mask.to(dtype=self._prob_grid.dtype) * LOG_ODDS_FREE)
        if stats is not None:
            stats["free_mask_apply_dense"] = 1
        free_mark("mask_apply")
        return True

    def _remove_hit_endpoints(self, paths: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Remove occupied endpoint voxels from a candidate free path."""
        if paths.numel() == 0 or targets.numel() == 0:
            return paths
        G = self.grid_size
        target_ids = targets[:, 0] * (G * G) + targets[:, 1] * G + targets[:, 2]
        path_ids = paths[:, 0] * (G * G) + paths[:, 1] * G + paths[:, 2]
        target_ids_sorted = torch.sort(target_ids).values
        pos = torch.searchsorted(target_ids_sorted, path_ids)
        pos_clamped = pos.clamp(max=target_ids_sorted.numel() - 1)
        is_hit_endpoint = (
            (pos < target_ids_sorted.numel())
            & (target_ids_sorted[pos_clamped] == path_ids)
        )
        return paths[~is_hit_endpoint]

    def _empty_ray_pairs_for_env(
        self,
        env_idx: int,
        eye: torch.Tensor,
        rays_world: torch.Tensor,
        miss_mask: torch.Tensor,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return grid-entry/grid-exit voxel pairs for alpha-miss rays."""
        if miss_mask.numel() == 0 or not miss_mask.any():
            empty = torch.empty((0, 3), dtype=torch.long, device=eye.device)
            return empty, empty

        G = self.grid_size
        bbox_min = self._bbox_min[env_idx]
        voxel_size = self._voxel_size[env_idx]
        bbox_max = bbox_min + float(G) * voxel_size
        rays = rays_world[miss_mask]
        if rays.numel() == 0:
            empty = torch.empty((0, 3), dtype=torch.long, device=eye.device)
            return empty, empty

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
            empty = torch.empty((0, 3), dtype=torch.long, device=eye.device)
            return empty, empty

        rays = rays[intersects]
        if stats is not None:
            stats["miss_intersect_rays"] = stats.get("miss_intersect_rays", 0) + int(rays.shape[0])
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
        start_idx = start_idx.clamp(0, G - 1)
        end_idx = end_idx.clamp(0, G - 1)
        if stats is not None:
            stats["empty_ray_pairs_before_unique"] = (
                stats.get("empty_ray_pairs_before_unique", 0) + int(start_idx.shape[0])
            )
        if getattr(self, "dedupe_empty_ray_pairs", False) and start_idx.numel() > 0:
            start_end = torch.cat([start_idx, end_idx], dim=1)
            start_end = torch.unique(start_end, dim=0, sorted=False)
            start_idx = start_end[:, :3]
            end_idx = start_end[:, 3:]
        if stats is not None:
            stats["empty_ray_pairs_after_unique"] = (
                stats.get("empty_ray_pairs_after_unique", 0) + int(start_idx.shape[0])
            )
        return start_idx, end_idx

    def _empty_ray_paths_for_env(
        self,
        env_idx: int,
        eye: torch.Tensor,
        rays_world: torch.Tensor,
        miss_mask: torch.Tensor,
        stats: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Traverse alpha-miss rays through this env's current grid AABB."""
        start_idx, end_idx = self._empty_ray_pairs_for_env(
            env_idx,
            eye,
            rays_world,
            miss_mask,
            stats=stats,
        )
        if start_idx.numel() == 0:
            return start_idx
        return bresenham3D_strict(
            pts_source=start_idx,
            pts_target=end_idx,
            map_size=self.grid_size,
            include_source=True,
            include_target=True,
        )

    # ------------------------------------------------------------------
    # Observation: [pose_history flat, grid_tri_cls flat]
    # ------------------------------------------------------------------
    def _build_observation_np(self) -> np.ndarray:
        # action_history: [N, B, 6] → [N, B*6]
        ah = self._action_history.reshape(self.num_envs, -1)
        tri = grid_occupancy_tri_cls(self._prob_grid, return_tri_cls_only=True)  # [N, G, G, G]
        if self.obs_grid_size != self.grid_size:
            tri = self._downsample_tri_grid_for_obs(tri)
        tri_flat = tri.reshape(self.num_envs, -1)
        parts = [ah, tri_flat]
        if self.caption_dim > 0 and self._caption_emb is not None:
            parts.append(self._caption_emb)
        obs = torch.cat(parts, dim=-1)                                            # [N, obs_dim]
        return obs.detach().cpu().numpy().astype(np.float32)

    def _downsample_tri_grid_for_obs(self, tri: torch.Tensor) -> torch.Tensor:
        """Downsample reward-resolution {-1,0,+1} tri-grid to policy obs.

        Max pooling preserves occupied hits when any fine cell in a coarse
        bin is occupied; unknown beats free for mixed free/unknown bins.
        This is intentionally conservative for NBV: a coarse cell only
        becomes free when all contributing fine cells are free.
        """
        G = self.grid_size
        O = self.obs_grid_size
        x = tri.unsqueeze(1)                                                      # [N,1,G,G,G]
        if G % O == 0:
            k = G // O
            return F.max_pool3d(x, kernel_size=k, stride=k).squeeze(1)
        return F.interpolate(x, size=(O, O, O), mode="nearest").squeeze(1)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_all(self) -> None:
        idx = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(idx)

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """In-place reset for the listed env indices.

        - Random uniform mesh_id sample from the pool (uniform over P).
        - Per-env GT tensors gathered from the pool by `index_select`.
        - All other state zero'd.
        """
        N = env_ids.shape[0]
        if N == 0:
            return
        self._prob_grid[env_ids] = 0.0
        self._scanned_gt_grid[env_ids] = 0.0
        self._step_idx[env_ids] = 0
        self._cr_prev[env_ids] = 0.0
        self._action_history[env_ids] = 0.0
        self._ep_step_count[env_ids] = 0
        self._ep_reward_sum[env_ids] = 0.0
        self._ep_new_gt_sum[env_ids] = 0.0
        self._ep_visible_gt_sum[env_ids] = 0.0
        self._ep_redundant_gt_sum[env_ids] = 0.0
        self._ep_revisit_sum[env_ids] = 0.0

        # Random mesh assignment from the pool. With pool_size==1 this
        # is the single shared mesh — the original behaviour.
        new_mesh_ids = torch.randint(
            0, self._pool_size, (N,), dtype=torch.long, device=self.device,
        )
        self._reset_envs_to_mesh_ids(env_ids, new_mesh_ids)

    def _reset_envs_to_mesh_ids(self, env_ids: torch.Tensor, new_mesh_ids: torch.Tensor) -> None:
        """Reset env state using explicit pool mesh ids."""
        N = env_ids.shape[0]
        if N == 0:
            return
        new_mesh_ids = new_mesh_ids.to(self.device, dtype=torch.long).flatten()
        if int(new_mesh_ids.numel()) != int(N):
            raise ValueError(
                f"_reset_envs_to_mesh_ids expected {N} mesh ids, got {new_mesh_ids.numel()}"
            )
        if bool(((new_mesh_ids < 0) | (new_mesh_ids >= self._pool_size)).any().item()):
            raise ValueError(f"mesh ids must be in [0, {self._pool_size})")

        self._prob_grid[env_ids] = 0.0
        self._scanned_gt_grid[env_ids] = 0.0
        self._step_idx[env_ids] = 0
        self._cr_prev[env_ids] = 0.0
        self._action_history[env_ids] = 0.0
        self._ep_step_count[env_ids] = 0
        self._ep_reward_sum[env_ids] = 0.0
        self._ep_new_gt_sum[env_ids] = 0.0
        self._ep_visible_gt_sum[env_ids] = 0.0
        self._ep_redundant_gt_sum[env_ids] = 0.0
        self._ep_revisit_sum[env_ids] = 0.0

        self._env_mesh_id[env_ids] = new_mesh_ids
        # Pull per-env GT slices.
        if self._pool_grid_on_device:
            selected_grid = self._pool_grid_gt[new_mesh_ids]
        else:
            selected_grid = self._pool_grid_gt[
                new_mesh_ids.detach().cpu()
            ].to(self.device, dtype=torch.float32, non_blocking=True)
        self._grid_gt[env_ids] = selected_grid
        self._bbox_min[env_ids] = self._pool_bbox_min[new_mesh_ids]
        self._voxel_size[env_ids] = self._pool_voxel_size[new_mesh_ids]
        self._num_valid_gt_per_env[env_ids] = self._pool_num_valid[new_mesh_ids].clamp(min=1.0)
        if self._caption_emb is not None:
            self._caption_emb[env_ids] = self._pool_caption_emb[new_mesh_ids]

        # Initial pose: middle xy, top z, looking down. Same as env.py.
        ix0 = self._idx_up[0] // 2
        iy0 = self._idx_up[1] // 2
        iz0 = self._idx_up[2]
        ip0 = 0          # pitch = -π/2 (down)
        iyaw0 = 0
        init_idx = torch.stack([
            ix0.expand(N), iy0.expand(N), iz0.expand(N),
            torch.zeros(N, dtype=torch.long, device=self.device),
            torch.tensor(ip0, dtype=torch.long, device=self.device).expand(N),
            torch.tensor(iyaw0, dtype=torch.long, device=self.device).expand(N),
        ], dim=-1)                                                            # [N, 6]
        init_pose = init_idx.float() * self._action_unit + self._action_low  # [N, 6]
        self._action_history[env_ids, -1, :] = init_pose
        self._last_action_idx[env_ids] = init_idx
