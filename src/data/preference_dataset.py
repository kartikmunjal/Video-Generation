"""
Preference dataset for DiffusionDPO training.

Each sample is a (prompt, chosen_video, rejected_video) triplet. Labels can
come from automated metrics (video_metrics.py) or the Gradio annotation UI.

Storage format (JSONL, one record per line):
{
    "prompt": "a dog running on a beach",
    "chosen":  "data/videos/gen_0001_A.mp4",
    "rejected": "data/videos/gen_0001_B.mp4",
    "source": "human" | "automated",
    "scores": {"chosen": 0.82, "rejected": 0.61}   # optional, for analysis
}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .video_dataset import load_video_frames


def load_preference_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_preference_jsonl(records: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class VideoPreferenceDataset(Dataset):
    """
    Loads (prompt, chosen_video, rejected_video) triplets for DPO training.

    If `preload_frames=True` (default False), frames are decoded at init time
    and kept in memory — faster training but high RAM usage.
    """

    def __init__(
        self,
        records: List[Dict],
        num_frames: int = 16,
        height: int = 480,
        width: int = 720,
        preload_frames: bool = False,
        source_filter: Optional[str] = None,
    ):
        if source_filter is not None:
            records = [r for r in records if r.get("source") == source_filter]

        self.records = records
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self._cache: Optional[List] = None
        if preload_frames:
            print(f"Preloading {len(records)} preference pairs...")
            self._cache = [self._load(r) for r in records]

    def _load(self, record: Dict) -> Dict:
        chosen_frames = load_video_frames(
            record["chosen"], self.num_frames, self.height, self.width
        )
        rejected_frames = load_video_frames(
            record["rejected"], self.num_frames, self.height, self.width
        )
        return {
            "prompt": record["prompt"],
            "chosen_frames": chosen_frames,
            "rejected_frames": rejected_frames,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        if self._cache is not None:
            return self._cache[idx]
        return self._load(self.records[idx])


def collate_preference_batch(batch: List[Dict]) -> Dict:
    """
    Collate a list of preference samples into batched tensors.
    Used as DataLoader collate_fn for DPO training.
    """
    prompts = [b["prompt"] for b in batch]
    chosen = torch.stack([b["chosen_frames"] for b in batch])    # (B, T, C, H, W)
    rejected = torch.stack([b["rejected_frames"] for b in batch])
    return {"prompts": prompts, "chosen_frames": chosen, "rejected_frames": rejected}
