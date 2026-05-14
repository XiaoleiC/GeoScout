"""GenNBV's Hybrid_Encoder, ported to gymnasium / SB3.

Verbatim algorithm match to `gennbv/network/hybrid_encoder.py`:
  - Pose history → sin/cos positional embedding (freqs=2) → 2-layer MLP (256d).
  - tri-class occupancy grid → 2× Conv3d(16, k=3, s=2) + BN + ReLU → 256.
  - Concat (action 256, grid 256) → Linear 512 → 256.
Output: 256-d feature vector consumed by SB3's PPO actor / critic heads.

The only adjustment: this version flat-indexes the observation
`[buffer_size×6 (pose) + grid_size³ (grid)]` rather than splitting on
`+ k×H×W` (image stream). ShapeNBV runs depth-only, no image input.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import gymnasium as gym
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Hybrid_Encoder(BaseFeaturesExtractor):
    """Feature extractor for [pose_history, occupancy_grid] flat obs.

    Args:
        observation_space: SB3 wraps the env's Box observation_space.
        encoder_param: kept for API compat — currently unused (the
            internal sizes are hardcoded to GenNBV's defaults).
        net_param: dict with key "append_hidden_shapes" — last element
            is the output feature dim (default 256). Kept for API compat.
        visual_input_shape: not used (no image stream); kept for compat.
        state_input_shape: tuple/list. `state_input_shape[0]` is
            `buffer_size × pose_dim` (= buffer_size × 6).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        encoder_param=None,
        net_param=None,
        visual_input_shape: Union[List, Tuple] = (1, 64, 64),
        state_input_shape: Union[List, Tuple] = (600,),
        grid_size: int = 32,
        caption_dim: int = 0,
        **_unused,    # tolerate extra kwargs (state_input_only, etc.)
    ):
        assert net_param is not None, "Hybrid_Encoder needs net_param"
        feature_dim = net_param["append_hidden_shapes"][-1]
        super().__init__(observation_space, feature_dim)
        self.state_input_shape = state_input_shape
        self.grid_size = int(grid_size)

        # ---- 3D CNN over occupancy grid -------------------------------
        # Input: [num_env, 1, G, G, G]. GenNBV used G=20; ShapeNBV
        # defaults to a 32³ policy observation obtained by conservative
        # downsampling from the 128³ reward grid. The flatten dim is
        # derived so tests and ablations can use other resolutions.
        self.naive_encoder_grid = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        g1 = (self.grid_size - 3) // 2 + 1
        g2 = (g1 - 3) // 2 + 1
        if g2 <= 0:
            raise ValueError(f"grid_size={self.grid_size} is too small for the 3D conv stack.")
        grid_flat_dim = 16 * g2 ** 3
        self.output_layer_grid = nn.Sequential(
            nn.Linear(grid_flat_dim, 256, bias=True),
            nn.ReLU(inplace=True),
        )

        # ---- MLP over positional-encoded pose history ----------------
        # state_input_shape[0] = buffer_size × 6 (pre-encoding).
        # After positional_encoding (freqs=2 → sin/cos pairs): ×4 multiplier
        # so encoded dim = state_input_shape[0] × 4.
        # GenNBV expects 2400 = 100 × 6 × 4 (buffer=100). We tolerate any
        # buffer size and adapt the first Linear layer at __init__ time.
        encoded_state_dim = state_input_shape[0] * 4
        self.naive_encoder_action = nn.Sequential(
            nn.Linear(encoded_state_dim, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
        )

        # ---- Optional caption stream ---------------------------------
        # Phase 1: per-episode sentence-transformer caption_emb (e.g. 384d
        # for MiniLM-L6) projected to 128d, then fused alongside action +
        # grid features. Disabled when caption_dim == 0.
        self.caption_dim = int(caption_dim)
        fusion_in = 256 + 256
        if self.caption_dim > 0:
            self.naive_encoder_caption = nn.Sequential(
                nn.Linear(self.caption_dim, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128, bias=True),
                nn.ReLU(inplace=True),
            )
            fusion_in += 128
        else:
            self.naive_encoder_caption = None

        # ---- Fusion --------------------------------------------------
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_in, 256, bias=True),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def positional_encoding(positions: torch.Tensor, freqs: int = 2) -> torch.Tensor:
        """[num_env, buffer_size, 6] → [num_env, buffer_size, 4×6]."""
        freq_bands = (2.0 ** torch.arange(freqs, device=positions.device)).float()
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1],)
        )
        return torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Observation layout (must match env._build_observation):
            [pose_history (state_input_shape[0]),
             grid_tri_cls (grid_size³),
             caption_emb (caption_dim, optional)]
        All flat float32.
        Total = state_input_shape[0] + grid_size³ + caption_dim.
        """
        num_env = observations.shape[0]
        state_dim = self.state_input_shape[0]
        grid_dim = self.grid_size ** 3

        # Pose history → positional encoding → MLP.
        action_input = observations[:, :state_dim].view(num_env, -1, 6)
        action_input = self.positional_encoding(action_input).view(num_env, -1)
        feature_action = self.naive_encoder_action(action_input)

        # Occupancy grid → 3D CNN → MLP.
        grid_input = observations[:, state_dim : state_dim + grid_dim]
        grid_input = grid_input.reshape(num_env, 1, self.grid_size, self.grid_size, self.grid_size)
        feature_grid = self.naive_encoder_grid(grid_input).reshape(num_env, -1)
        feature_grid = self.output_layer_grid(feature_grid)

        feats = [feature_action, feature_grid]
        if self.caption_dim > 0 and self.naive_encoder_caption is not None:
            cap = observations[:, state_dim + grid_dim : state_dim + grid_dim + self.caption_dim]
            feats.append(self.naive_encoder_caption(cap))
        return self.output_layer(torch.cat(feats, dim=-1))
