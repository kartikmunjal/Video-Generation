"""
Video dataset for CogVideoX-2B fine-tuning.

Expects a directory layout:
    data/videos/
        clip_0001.mp4
        clip_0002.mp4
        ...
    data/captions.json  {"clip_0001": "a cat jumping over a fence", ...}

Or a CSV/TSV in WebVid format with columns: videoid, caption, page_dir.
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_video_frames(
    path: str,
    num_frames: int = 16,
    height: int = 480,
    width: int = 720,
) -> torch.Tensor:
    """
    Decode a video file and return uniformly sampled frames as a float tensor.

    Returns:
        Tensor of shape (T, C, H, W) in [-1, 1].
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path, width=width, height=height)
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).float()  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)     # (T, C, H, W)
        frames = frames / 127.5 - 1.0           # → [-1, 1]
        return frames
    except ImportError:
        # Fallback: opencv
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        frames = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            idx += 1
        cap.release()
        arr = np.stack(frames[:num_frames], axis=0).astype(np.float32)  # (T,H,W,C)
        arr = torch.from_numpy(arr).permute(0, 3, 1, 2) / 127.5 - 1.0
        return arr


class VideoDataset(Dataset):
    """
    Dataset that yields (video_tensor, caption) pairs for fine-tuning.

    Supports two loading modes:
      - "directory": scans `video_dir` for *.mp4 files, reads captions from a JSON.
      - "webvid": reads a CSV with columns [videoid, caption, page_dir].
    """

    def __init__(
        self,
        video_dir: str,
        captions_path: str,
        num_frames: int = 16,
        height: int = 480,
        width: int = 720,
        mode: str = "directory",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.transform = transform

        if mode == "directory":
            with open(captions_path) as f:
                captions = json.load(f)
            self.samples = [
                (str(self.video_dir / f"{stem}.mp4"), cap)
                for stem, cap in captions.items()
                if (self.video_dir / f"{stem}.mp4").exists()
            ]
        elif mode == "webvid":
            import pandas as pd
            df = pd.read_csv(captions_path, sep="\t")
            self.samples = [
                (str(self.video_dir / row["page_dir"] / f"{row['videoid']}.mp4"), row["caption"])
                for _, row in df.iterrows()
                if (self.video_dir / row["page_dir"] / f"{row['videoid']}.mp4").exists()
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'directory' or 'webvid'.")

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        video_path, caption = self.samples[idx]
        frames = load_video_frames(video_path, self.num_frames, self.height, self.width)
        if self.transform is not None:
            frames = self.transform(frames)
        return {"pixel_values": frames, "caption": caption}


class VideoDataCollator:
    """
    Collates variable-length captions into a batch with padding.
    Used with DataLoader's collate_fn.
    """

    def __init__(self, tokenizer, max_text_len: int = 226):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __call__(self, batch: List[Dict]) -> Dict:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])  # (B, T, C, H, W)
        captions = [b["caption"] for b in batch]

        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }
