"""
Video Reward Model for preference learning.

Architecture:
  1. CLIP ViT-B/32 encodes each frame → frame-level features (D=512).
  2. Temporal aggregator: mean-pool frame features → video-level feature.
  3. Motion feature: scalar (pre-computed optical flow variance).
  4. Temporal consistency feature: scalar (pre-computed LPIPS mean).
  5. Concatenate → 512+2 = 514 → MLP projection → scalar reward.

The Bradley-Terry loss (same as text reward model in prior repo) is:
    L = -log σ(r_chosen - r_rejected)

This architecture deliberately mirrors GPT2RewardModel from the prior RLHF
repo (src/models/reward_model.py) but operates on video frames instead of
token hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VideoRewardModelOutput:
    rewards: torch.Tensor           # (B,) scalar rewards
    frame_features: torch.Tensor    # (B, T, D) per-frame CLIP features


class TemporalAttentionPooling(nn.Module):
    """
    Attention-weighted pooling over frame features.
    Learns which frames are most informative for reward prediction.
    Better than simple mean-pooling when temporal structure matters.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) frame features.
        Returns:
            (B, D) pooled features.
        """
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)               # (B, D)


class VideoRewardModel(nn.Module):
    """
    Multi-signal video reward model with Bradley-Terry preference head.

    Args:
        clip_model_name: CLIP variant for frame encoding.
        hidden_dim: intermediate MLP dimension.
        dropout: dropout rate.
        use_attention_pooling: use attention-weighted temporal pooling (vs mean).
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_attention_pooling: bool = True,
    ):
        super().__init__()

        # CLIP encoder (frozen)
        import clip
        self.clip, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        for p in self.clip.parameters():
            p.requires_grad_(False)

        clip_dim = self.clip.visual.output_dim  # 512 for ViT-B/32

        # Temporal aggregator
        self.temporal_pool = (
            TemporalAttentionPooling(clip_dim)
            if use_attention_pooling
            else lambda x: x.mean(dim=1)
        )

        # Extra scalar signals (motion smoothness, temporal consistency) — 2 dims
        extra_dim = 2

        # MLP reward head
        input_dim = clip_dim + extra_dim
        self.reward_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames with CLIP.

        Args:
            frames: (B, T, C, H, W) in [-1, 1].

        Returns:
            (B, T, D) normalized CLIP features.
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)

        with torch.no_grad():
            features = self.clip.encode_image(flat)  # (B*T, D)
        features = F.normalize(features, dim=-1)
        return features.view(B, T, -1)              # (B, T, D)

    def forward(
        self,
        frames: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None,
        temporal_scores: Optional[torch.Tensor] = None,
    ) -> VideoRewardModelOutput:
        """
        Args:
            frames: (B, T, C, H, W) in [-1, 1].
            motion_scores: (B,) pre-computed motion smoothness (optional).
            temporal_scores: (B,) pre-computed temporal consistency (optional).

        Returns:
            VideoRewardModelOutput with (B,) rewards.
        """
        B = frames.shape[0]
        device = frames.device

        frame_features = self.encode_frames(frames)         # (B, T, D)
        video_features = self.temporal_pool(frame_features)  # (B, D)

        # Extra scalar signals — zeros if not provided
        if motion_scores is None:
            motion_scores = torch.zeros(B, device=device)
        if temporal_scores is None:
            temporal_scores = torch.zeros(B, device=device)

        extras = torch.stack([motion_scores, temporal_scores], dim=1)  # (B, 2)
        combined = torch.cat([video_features, extras], dim=1)          # (B, D+2)

        rewards = self.reward_head(combined).squeeze(-1)  # (B,)

        return VideoRewardModelOutput(rewards=rewards, frame_features=frame_features)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def preference_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bradley-Terry preference loss.

    Identical to the text RLHF reward model loss in:
        rlhf-and-reward-modelling-alt/src/models/reward_model.py

    L = -log σ(r_chosen - r_rejected)

    Args:
        chosen_rewards: (B,) rewards for preferred videos.
        rejected_rewards: (B,) rewards for rejected videos.

    Returns:
        (loss, accuracy) scalars.
    """
    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()
    accuracy = (margin > 0).float().mean()
    return loss, accuracy
