"""
Evaluation utilities for video generation quality.

Computes all three automated metrics across a set of generated videos
and produces a summary report. Used by run_ablation.py and as the final
eval step in the iterative DPO loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.video_dataset import load_video_frames
from ..data.video_metrics import compute_video_metrics, VideoMetrics


class VideoEvaluator:
    """
    Evaluates a set of generated videos against their prompts.

    Args:
        device: torch device for CLIP and LPIPS computation.
        weights: composite reward weights (motion, temporal, clip).
    """

    def __init__(
        self,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
        num_frames: int = 16,
        height: int = 480,
        width: int = 720,
    ):
        self.device = device
        self.weights = weights or {"motion": 0.2, "temporal": 0.3, "clip": 0.5}
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def evaluate_single(self, video_path: str, prompt: str) -> VideoMetrics:
        frames = load_video_frames(video_path, self.num_frames, self.height, self.width)
        return compute_video_metrics(frames, prompt, device=self.device, weights=self.weights)

    def evaluate_batch(
        self,
        pairs: List[Dict],
        desc: str = "Evaluating",
    ) -> pd.DataFrame:
        """
        Evaluate a list of {prompt, video_path} dicts.

        Args:
            pairs: list of dicts with keys "prompt" and "video_path".
            desc: tqdm description.

        Returns:
            DataFrame with one row per video.
        """
        rows = []
        for item in tqdm(pairs, desc=desc):
            try:
                metrics = self.evaluate_single(item["video_path"], item["prompt"])
                rows.append({
                    "video_path": item["video_path"],
                    "prompt": item["prompt"],
                    **metrics.to_dict(),
                    "condition": item.get("condition", "unknown"),
                })
            except Exception as e:
                print(f"  Warning: failed to evaluate {item['video_path']}: {e}")

        return pd.DataFrame(rows)

    def summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return mean ± std for each metric."""
        metrics = ["motion_smoothness", "temporal_consistency", "prompt_adherence", "composite"]
        result = {}
        for m in metrics:
            if m in df.columns:
                result[f"{m}_mean"] = df[m].mean()
                result[f"{m}_std"] = df[m].std()
        return result


def run_evaluation(
    video_dir: str,
    prompts_path: str,
    output_path: str,
    condition_name: str = "model",
    device: str = "cuda",
    num_frames: int = 16,
    height: int = 480,
    width: int = 720,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Evaluate all videos in video_dir against prompts from prompts_path.

    Assumes filenames match: gen_{i:04d}.mp4 ↔ i-th line in prompts_path.
    Or a manifest.json from generate_videos.py.

    Args:
        video_dir: directory containing generated .mp4 files.
        prompts_path: text file with one prompt per line, or manifest.json.
        output_path: where to save the results CSV.
        condition_name: label for this experimental condition (e.g., "lora_r16").

    Returns:
        DataFrame with evaluation results.
    """
    video_dir = Path(video_dir)
    evaluator = VideoEvaluator(device=device, weights=weights,
                               num_frames=num_frames, height=height, width=width)

    # Build (prompt, video_path) pairs
    manifest_path = video_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        pairs = []
        for prompt, paths in manifest.items():
            for p in paths:
                if os.path.exists(p):
                    pairs.append({"prompt": prompt, "video_path": p, "condition": condition_name})
    else:
        with open(prompts_path) as f:
            prompts = [l.strip() for l in f if l.strip()]
        pairs = []
        for i, prompt in enumerate(prompts):
            video_path = video_dir / f"gen_{i:04d}_A.mp4"
            if video_path.exists():
                pairs.append({"prompt": prompt, "video_path": str(video_path), "condition": condition_name})

    print(f"Evaluating {len(pairs)} videos for condition: {condition_name}")
    df = evaluator.evaluate_batch(pairs)
    summary = evaluator.summary(df)

    print(f"\n--- {condition_name} ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return df
