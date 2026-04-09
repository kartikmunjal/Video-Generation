"""
Stage 1: LoRA fine-tuning of CogVideoX-2B on domain-specific videos.

Training objective: standard diffusion denoising loss
    L = E_{t, v, ε} [ ||ε - ε_θ(v_t, t, c)||² ]

where:
  - v is a video from the domain dataset
  - c is the text caption
  - t ~ U[0, T] is a uniformly sampled diffusion timestep
  - v_t is v noised to timestep t
  - ε_θ is the transformer's noise prediction (only LoRA params are updated)

This is analogous to SFT in the text RLHF pipeline — it adapts the base model
to the target domain before the preference alignment stages.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.video_dataset import VideoDataset, VideoDataCollator
from ..models.lora_adapter import CogVideoXLoRA, count_lora_params


class LoRAFinetuner:
    """
    Manages the LoRA fine-tuning loop for CogVideoX-2B.

    Args:
        config: dict loaded from cogvideox_lora.yaml.
    """

    def __init__(self, config: dict):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            log_with="wandb" if config.get("wandb", {}).get("enabled", False) else None,
        )
        self.device = self.accelerator.device

    def setup(self):
        """Load model, data, optimizer, scheduler."""
        cfg = self.config
        tr_cfg = cfg["training"]
        lora_cfg = cfg["lora"]
        data_cfg = cfg["data"]

        # --- Model ---
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[cfg["model"].get("dtype", "bfloat16")]

        self.lora_model = CogVideoXLoRA.from_pretrained(
            model_name=cfg["model"]["pretrained_model_name_or_path"],
            lora_r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg.get("target_modules"),
            dtype=dtype,
            device=str(self.device),
        )
        stats = count_lora_params(self.lora_model.transformer)
        print(f"LoRA ratio: {stats['lora_ratio']:.4%} "
              f"({stats['trainable_params']:,} / {stats['total_params']:,} params)")

        # Freeze everything except LoRA
        for name, param in self.lora_model.transformer.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

        # --- Data ---
        pipe = self.lora_model.pipe
        dataset = VideoDataset(
            video_dir=data_cfg["video_dir"],
            captions_path=data_cfg["captions_path"],
            num_frames=data_cfg.get("num_frames", 16),
            height=data_cfg.get("height", 480),
            width=data_cfg.get("width", 720),
            max_samples=data_cfg.get("max_samples"),
        )
        collator = VideoDataCollator(pipe.tokenizer, max_text_len=226)
        self.dataloader = DataLoader(
            dataset,
            batch_size=tr_cfg["batch_size"],
            shuffle=True,
            num_workers=tr_cfg.get("dataloader_num_workers", 4),
            collate_fn=collator,
        )

        # --- Optimizer ---
        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.lora_model.trainable_parameters(),
            lr=tr_cfg["learning_rate"],
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
        )

        # --- Scheduler ---
        from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
        total_steps = tr_cfg["num_epochs"] * len(self.dataloader) // tr_cfg["gradient_accumulation_steps"]
        warmup = tr_cfg.get("lr_warmup_steps", 100)

        if tr_cfg.get("lr_scheduler", "cosine") == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup, total_steps)
        else:
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, warmup)

        # Accelerate wraps
        (
            self.lora_model.transformer,
            self.optimizer,
            self.dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.lora_model.transformer,
            self.optimizer,
            self.dataloader,
            self.scheduler,
        )

        self.vae = self.lora_model.pipe.vae.to(self.device)
        self.text_encoder = self.lora_model.pipe.text_encoder.to(self.device)
        self.noise_scheduler = self.lora_model.pipe.scheduler

        self.output_dir = tr_cfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def _encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to latent space using the frozen VAE.

        Args:
            pixel_values: (B, T, C, H, W) in [-1, 1].

        Returns:
            Latent tensor (B, C_lat, T_lat, H_lat, W_lat).
        """
        B, T, C, H, W = pixel_values.shape
        frames = pixel_values.view(B * T, C, H, W)
        with torch.no_grad():
            latents = self.vae.encode(frames).latent_dist.sample()
            latents = latents.view(B, T, *latents.shape[1:]).permute(0, 2, 1, 3, 4)
            latents = latents * self.vae.config.scaling_factor
        return latents

    def _encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        with torch.no_grad():
            enc = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return enc.last_hidden_state

    def train(self):
        """Main training loop."""
        cfg = self.config["training"]
        global_step = 0
        best_loss = float("inf")

        for epoch in range(cfg["num_epochs"]):
            self.lora_model.transformer.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}"):
                pixel_values = batch["pixel_values"].to(self.device)   # (B, T, C, H, W)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Encode to latent space
                latents = self._encode_video(pixel_values)
                encoder_hidden_states = self._encode_text(input_ids, attention_mask)

                # Sample noise and timestep
                noise = torch.randn_like(latents)
                B = latents.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
                ).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise with LoRA transformer
                with self.accelerator.accumulate(self.lora_model.transformer):
                    noise_pred = self.lora_model.transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample

                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.lora_model.trainable_parameters(),
                            cfg.get("max_grad_norm", 1.0),
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.detach().item()

                if self.accelerator.sync_gradients:
                    global_step += 1

                    if global_step % cfg.get("logging_steps", 50) == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"  step={global_step}, loss={loss.item():.4f}, lr={lr:.2e}")

                    if global_step % cfg.get("checkpointing_steps", 500) == 0:
                        ckpt_path = os.path.join(self.output_dir, f"step_{global_step}")
                        self.accelerator.unwrap_model(self.lora_model.transformer).save_pretrained(ckpt_path)
                        print(f"  Checkpoint saved: {ckpt_path}")

            mean_loss = epoch_loss / len(self.dataloader)
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1}: loss={mean_loss:.4f}, time={elapsed:.0f}s")

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_path = os.path.join(self.output_dir, "best")
                self.accelerator.unwrap_model(self.lora_model.transformer).save_pretrained(best_path)

        # Final save
        final_path = os.path.join(self.output_dir, "final")
        self.accelerator.unwrap_model(self.lora_model.transformer).save_pretrained(final_path)
        print(f"Training complete. Final model saved to {final_path}")
        return final_path
