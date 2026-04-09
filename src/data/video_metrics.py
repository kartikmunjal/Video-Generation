"""
Automated video quality metrics used to build the preference dataset and
train the reward model without human labels.

Three signals (each in [0, 1], higher = better):
  1. motion_smoothness  — low variance in optical flow magnitude over time
  2. temporal_consistency — high frame-to-frame perceptual similarity (low LPIPS)
  3. prompt_adherence  — high CLIP similarity between prompt and frames

Composite reward is a weighted sum:
  R = w_motion * motion + w_temporal * temporal + w_clip * clip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Motion smoothness via optical flow
# ---------------------------------------------------------------------------

def compute_motion_smoothness(frames: torch.Tensor) -> float:
    """
    Compute motion smoothness from a video tensor.

    Args:
        frames: (T, C, H, W) float tensor in [-1, 1].

    Returns:
        Smoothness score in [0, 1]. Higher = smoother.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for motion smoothness computation.")

    T = frames.shape[0]
    if T < 2:
        return 1.0

    # Convert to uint8 grayscale numpy arrays
    frames_np = ((frames.cpu().float() + 1.0) * 127.5).clamp(0, 255).byte().numpy()
    # frames_np: (T, C, H, W) uint8

    flow_magnitudes = []
    for t in range(T - 1):
        f1 = frames_np[t].transpose(1, 2, 0)  # (H, W, C)
        f2 = frames_np[t + 1].transpose(1, 2, 0)
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        flow_magnitudes.append(mag)

    flow_arr = np.array(flow_magnitudes)
    if flow_arr.max() < 1e-6:
        return 1.0  # static video — perfectly smooth

    # Coefficient of variation: low CV = smooth
    cv = flow_arr.std() / (flow_arr.mean() + 1e-8)
    # Map: CV=0 → score=1, CV>=2 → score=0
    score = float(np.clip(1.0 - cv / 2.0, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# 2. Temporal consistency via LPIPS
# ---------------------------------------------------------------------------

def compute_temporal_consistency(
    frames: torch.Tensor,
    device: str = "cpu",
) -> float:
    """
    Compute temporal consistency using LPIPS between consecutive frames.

    Args:
        frames: (T, C, H, W) float tensor in [-1, 1].
        device: device string.

    Returns:
        Consistency score in [0, 1]. Higher = more consistent.
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips is required: pip install lpips")

    T = frames.shape[0]
    if T < 2:
        return 1.0

    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    frames = frames.to(device)
    scores = []
    with torch.no_grad():
        for t in range(T - 1):
            f1 = frames[t].unsqueeze(0)   # (1, C, H, W)
            f2 = frames[t + 1].unsqueeze(0)
            d = loss_fn(f1, f2).item()    # perceptual distance
            scores.append(d)

    mean_lpips = float(np.mean(scores))
    # Map: lpips=0 → score=1 (identical), lpips=1 → score=0
    score = float(np.clip(1.0 - mean_lpips, 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# 3. Prompt adherence via CLIP score
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip(device: str = "cpu"):
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None or _clip_device != device:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_device = device
    return _clip_model, _clip_preprocess


def compute_clip_score(
    frames: torch.Tensor,
    prompt: str,
    device: str = "cpu",
    sample_frames: int = 4,
) -> float:
    """
    Compute average CLIP similarity between prompt and sampled frames.

    Args:
        frames: (T, C, H, W) float tensor in [-1, 1].
        prompt: text prompt.
        device: device string.
        sample_frames: number of frames to sample for speed.

    Returns:
        CLIP score in [0, 1].
    """
    import clip
    from PIL import Image

    model, preprocess = _get_clip(device)

    T = frames.shape[0]
    indices = np.linspace(0, T - 1, min(sample_frames, T), dtype=int)

    # Convert frames to PIL images for CLIP preprocessing
    frames_uint8 = ((frames.cpu().float() + 1.0) * 127.5).clamp(0, 255).byte()
    pil_images = [
        Image.fromarray(frames_uint8[i].permute(1, 2, 0).numpy())
        for i in indices
    ]
    image_inputs = torch.stack([preprocess(img) for img in pil_images]).to(device)

    text_input = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)       # (N, D)
        text_features = model.encode_text(text_input)           # (1, D)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = (image_features @ text_features.T).mean().item()  # in [-1, 1]

    # Shift to [0, 1]
    return float((similarity + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

@dataclass
class VideoMetrics:
    motion_smoothness: float
    temporal_consistency: float
    prompt_adherence: float
    composite: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "motion_smoothness": self.motion_smoothness,
            "temporal_consistency": self.temporal_consistency,
            "prompt_adherence": self.prompt_adherence,
            "composite": self.composite,
        }


def compute_video_metrics(
    frames: torch.Tensor,
    prompt: str,
    device: str = "cpu",
    weights: Optional[Dict[str, float]] = None,
) -> VideoMetrics:
    """
    Compute all three quality signals and aggregate into a composite reward.

    Default weights:  motion=0.2, temporal=0.3, clip=0.5.
    The clip signal is weighted highest because it captures both semantics
    and visual quality implicitly.

    Args:
        frames: (T, C, H, W) tensor in [-1, 1].
        prompt: text prompt.
        device: torch device string.
        weights: optional dict with keys "motion", "temporal", "clip".

    Returns:
        VideoMetrics dataclass.
    """
    if weights is None:
        weights = {"motion": 0.2, "temporal": 0.3, "clip": 0.5}

    w_m = weights.get("motion", 0.2)
    w_t = weights.get("temporal", 0.3)
    w_c = weights.get("clip", 0.5)

    m = compute_motion_smoothness(frames)
    t = compute_temporal_consistency(frames, device=device)
    c = compute_clip_score(frames, prompt, device=device)

    composite = w_m * m + w_t * t + w_c * c

    return VideoMetrics(
        motion_smoothness=m,
        temporal_consistency=t,
        prompt_adherence=c,
        composite=composite,
    )
