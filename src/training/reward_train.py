"""
Stage 2: Reward model training on video preference pairs.

Uses Bradley-Terry loss (same as text reward model in prior repo) with the
VideoRewardModel as backbone. Supports both automated preference pairs
(from collect_preferences.py) and human-labeled pairs (from gradio_app.py).

After training, the reward model is used in the iterative DPO loop to score
newly generated videos and label them as (chosen, rejected) pairs.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..data.preference_dataset import VideoPreferenceDataset, collate_preference_batch, load_preference_jsonl
from ..data.video_metrics import compute_motion_smoothness, compute_temporal_consistency
from ..models.video_reward_model import VideoRewardModel, preference_loss


class RewardTrainer:
    """
    Trains the VideoRewardModel on (chosen, rejected) video preference pairs.

    Args:
        config: dict loaded from reward_config.yaml.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def setup(self):
        cfg = self.config
        tr_cfg = cfg["training"]
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        # --- Model ---
        self.model = VideoRewardModel(
            clip_model_name=model_cfg.get("clip_model", "ViT-B/32"),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(self.device)
        print(f"Reward model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # --- Data ---
        records = load_preference_jsonl(data_cfg["preference_path"])
        source_filter = data_cfg.get("source_filter")
        dataset = VideoPreferenceDataset(
            records=records,
            num_frames=data_cfg.get("num_frames", 16),
            height=data_cfg.get("height", 480),
            width=data_cfg.get("width", 720),
            preload_frames=data_cfg.get("preload_frames", False),
            source_filter=source_filter,
        )

        train_frac = data_cfg.get("train_split", 0.9)
        n_train = int(len(dataset) * train_frac)
        n_val = len(dataset) - n_train
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(tr_cfg.get("seed", 42)),
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=tr_cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_preference_batch,
            num_workers=2,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=tr_cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_preference_batch,
            num_workers=2,
        )
        print(f"Dataset: {n_train} train, {n_val} val preference pairs")

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=tr_cfg["learning_rate"],
            weight_decay=1e-2,
        )

        from transformers import get_cosine_schedule_with_warmup
        total_steps = tr_cfg["num_epochs"] * len(self.train_loader)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=tr_cfg.get("lr_warmup_steps", 50),
            num_training_steps=total_steps,
        )

        self.output_dir = tr_cfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_val_acc = 0.0

    def _precompute_extra_signals(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute motion and temporal signals for a batch on CPU (no GPU needed).
        Returns dicts with (B,) tensors.
        """
        B = frames.shape[0]
        motion = torch.tensor(
            [compute_motion_smoothness(frames[i].cpu()) for i in range(B)],
            dtype=torch.float32,
        )
        temporal = torch.tensor(
            [compute_temporal_consistency(frames[i].cpu()) for i in range(B)],
            dtype=torch.float32,
        )
        return {"motion": motion.to(self.device), "temporal": temporal.to(self.device)}

    def _forward_batch(self, chosen_frames, rejected_frames):
        chosen_sigs = self._precompute_extra_signals(chosen_frames)
        rejected_sigs = self._precompute_extra_signals(rejected_frames)

        chosen_out = self.model(
            chosen_frames.to(self.device),
            motion_scores=chosen_sigs["motion"],
            temporal_scores=chosen_sigs["temporal"],
        )
        rejected_out = self.model(
            rejected_frames.to(self.device),
            motion_scores=rejected_sigs["motion"],
            temporal_scores=rejected_sigs["temporal"],
        )
        return chosen_out.rewards, rejected_out.rewards

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss, total_acc, n = 0.0, 0.0, 0

        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, desc="train" if train else "val", leave=False):
                chosen_rewards, rejected_rewards = self._forward_batch(
                    batch["chosen_frames"], batch["rejected_frames"]
                )
                loss, acc = preference_loss(chosen_rewards, rejected_rewards)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                total_loss += loss.item() * len(batch["prompts"])
                total_acc += acc.item() * len(batch["prompts"])
                n += len(batch["prompts"])

        return {"loss": total_loss / n, "accuracy": total_acc / n}

    def train(self):
        cfg = self.config["training"]

        for epoch in range(cfg["num_epochs"]):
            t0 = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch+1}/{cfg['num_epochs']} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
                f"time={elapsed:.0f}s"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                best_path = os.path.join(self.output_dir, "best.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"  New best val accuracy: {self.best_val_acc:.4f}. Saved to {best_path}")

        final_path = os.path.join(self.output_dir, "final.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"\nTraining complete. Final model: {final_path}")
        print(f"Best val accuracy: {self.best_val_acc:.4f} (random baseline = 0.50)")
        return final_path
