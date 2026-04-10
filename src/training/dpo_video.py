"""
DiffusionDPO for video generation.

Adapts the Direct Preference Optimization objective to diffusion models.
Instead of token log-probabilities, we use the negative diffusion ELBO
(denoising error sum) as a proxy for the log-likelihood of a video:

    log p_θ(v|c) ≈ -E_t [ ||ε - ε_θ(v_t, t, c)||² ]

The DPO loss is then:
    L_DPO = -log σ( β · (log p_θ(v_w|c) - log p_ref(v_w|c))
                  - β · (log p_θ(v_l|c) - log p_ref(v_l|c)) )

where:
  - v_w = chosen (preferred) video
  - v_l = rejected video
  - p_θ = current policy (fine-tuned LoRA model)
  - p_ref = reference policy (frozen LoRA checkpoint)
  - β = KL penalty (higher = closer to reference, conservative)

Reference: Wallace et al. (2023), "Diffusion Model Alignment Using DPO"

This is the video analogue of dpo.py in rlhf-and-reward-modelling-alt.
The math is identical; we just swap sequence_logprob with diffusion_elbo.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.preference_dataset import VideoPreferenceDataset, collate_preference_batch, load_preference_jsonl
from ..models.lora_adapter import CogVideoXLoRA, load_lora_weights


def diffusion_elbo(
    transformer,
    vae,
    text_encoder,
    noise_scheduler,
    frames: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_timesteps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Estimate log p_θ(v|c) via the diffusion ELBO on uniformly sampled timesteps.

    L_ELBO(v) = -E_t [ ||ε - ε_θ(v_t, t, c)||² ]

    We sample `num_timesteps` values of t uniformly and average the MSE.
    This is a Monte Carlo estimate of the ELBO — more timesteps = lower variance
    but slower gradient steps.

    Args:
        transformer: noise-prediction model (CogVideoX transformer).
        vae: frozen VAE encoder.
        text_encoder: frozen T5 text encoder.
        noise_scheduler: diffusion noise schedule.
        frames: (B, T, C, H, W) video frames in [-1, 1].
        input_ids: (B, L) tokenized prompt.
        attention_mask: (B, L) attention mask.
        num_timesteps: number of timesteps to sample for ELBO estimate.
        device: torch device string.

    Returns:
        (B,) tensor of ELBO estimates (higher = more likely under the model).
    """
    B, T, C, H, W = frames.shape

    # Encode video to latent space (frozen VAE)
    with torch.no_grad():
        flat_frames = frames.view(B * T, C, H, W)
        latents = vae.encode(flat_frames).latent_dist.sample()
        latents = latents.view(B, T, *latents.shape[1:]).permute(0, 2, 1, 3, 4)
        latents = latents * vae.config.scaling_factor

        # Encode text (frozen T5)
        enc = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = enc.last_hidden_state

    # Sample num_timesteps uniformly and compute average denoising error
    total_mse = torch.zeros(B, device=device)
    max_t = noise_scheduler.config.num_train_timesteps

    for _ in range(num_timesteps):
        t = torch.randint(0, max_t, (B,), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)

        noise_pred = transformer(
            hidden_states=noisy_latents,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        mse = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        # Mean over all dims except batch
        mse = mse.view(B, -1).mean(dim=1)
        total_mse = total_mse + mse

    avg_mse = total_mse / num_timesteps
    # ELBO = -E[MSE] — higher ELBO = model assigns higher likelihood to video
    return -avg_mse


def dpo_loss(
    policy_chosen_elbo: torch.Tensor,
    policy_rejected_elbo: torch.Tensor,
    ref_chosen_elbo: torch.Tensor,
    ref_rejected_elbo: torch.Tensor,
    beta: float = 0.5,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DiffusionDPO loss.

    Implicit rewards:
        r_θ(v, c) = β · (log p_θ(v|c) - log p_ref(v|c))
                  = β · (ELBO_θ(v) - ELBO_ref(v))

    Loss:
        L = -log σ(r_θ(v_w) - r_θ(v_l))

    Args:
        policy_chosen_elbo: (B,) ELBO of preferred video under current policy.
        policy_rejected_elbo: (B,) ELBO of rejected video under current policy.
        ref_chosen_elbo: (B,) ELBO under reference (frozen) policy.
        ref_rejected_elbo: (B,) ELBO under reference (frozen) policy.
        beta: KL regularization strength.
        label_smoothing: smooth labels to prevent overconfidence.

    Returns:
        (loss, chosen_implicit_reward, rejected_implicit_reward).
    """
    chosen_reward = beta * (policy_chosen_elbo - ref_chosen_elbo)
    rejected_reward = beta * (policy_rejected_elbo - ref_rejected_elbo)

    margin = chosen_reward - rejected_reward

    if label_smoothing > 0:
        loss = -(
            (1 - label_smoothing) * F.logsigmoid(margin)
            + label_smoothing * F.logsigmoid(-margin)
        ).mean()
    else:
        loss = -F.logsigmoid(margin).mean()

    return loss, chosen_reward.detach(), rejected_reward.detach()


class DiffusionDPOTrainer:
    """
    Trains a LoRA-adapted CogVideoX-2B using DiffusionDPO.

    Requires:
      - A LoRA fine-tuned checkpoint (policy)
      - A frozen copy of the same checkpoint (reference policy)
      - A preference dataset (chosen_video, rejected_video, prompt)
    """

    def __init__(self, config: dict):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config["training"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        )
        self.device = self.accelerator.device

    def setup(self):
        cfg = self.config
        tr_cfg = cfg["training"]
        dpo_cfg = cfg["dpo"]
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        lora_cfg = cfg["lora"]

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[model_cfg.get("dtype", "bfloat16")]

        # Policy (trainable LoRA)
        print("Loading policy model (trainable)...")
        self.policy = CogVideoXLoRA.from_lora_checkpoint(
            base_model=model_cfg["pretrained_model_name_or_path"],
            lora_checkpoint=model_cfg["lora_checkpoint"],
            dtype=dtype,
            device=str(self.device),
        )

        # Reference policy (frozen — same checkpoint)
        print("Loading reference model (frozen)...")
        self.ref_policy = CogVideoXLoRA.from_lora_checkpoint(
            base_model=model_cfg["pretrained_model_name_or_path"],
            lora_checkpoint=model_cfg["lora_checkpoint"],
            dtype=dtype,
            device=str(self.device),
        )
        for p in self.ref_policy.transformer.parameters():
            p.requires_grad_(False)

        # Shared components from policy pipeline
        self.vae = self.policy.pipe.vae.to(self.device)
        self.text_encoder = self.policy.pipe.text_encoder.to(self.device)
        self.noise_scheduler = self.policy.pipe.scheduler
        self.tokenizer = self.policy.pipe.tokenizer

        # Freeze VAE and text encoder
        for model in [self.vae, self.text_encoder]:
            for p in model.parameters():
                p.requires_grad_(False)

        # Dataset
        records = load_preference_jsonl(data_cfg["preference_path"])
        dataset = VideoPreferenceDataset(
            records=records,
            num_frames=data_cfg.get("num_frames", 16),
            height=data_cfg.get("height", 480),
            width=data_cfg.get("width", 720),
        )

        train_n = int(len(dataset) * data_cfg.get("train_split", 0.9))
        val_n = len(dataset) - train_n
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [train_n, val_n])

        self.train_loader = DataLoader(
            train_ds, batch_size=tr_cfg["batch_size"], shuffle=True,
            collate_fn=collate_preference_batch, num_workers=2,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=tr_cfg["batch_size"], shuffle=False,
            collate_fn=collate_preference_batch, num_workers=2,
        )
        print(f"DPO dataset: {train_n} train, {val_n} val pairs")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=tr_cfg["learning_rate"],
            weight_decay=1e-2,
        )
        from transformers import get_constant_schedule_with_warmup
        self.scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=tr_cfg.get("lr_warmup_steps", 50),
        )

        (
            self.policy.transformer,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.policy.transformer,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

        self.output_dir = tr_cfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.beta = dpo_cfg["beta"]
        self.num_timesteps = dpo_cfg.get("num_timesteps", 50)
        self.label_smoothing = dpo_cfg.get("label_smoothing", 0.0)

    def _tokenize_prompts(self, prompts):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=226,
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)

    def _step(self, batch: dict, train: bool) -> Dict[str, float]:
        chosen_frames = batch["chosen_frames"].to(self.device)
        rejected_frames = batch["rejected_frames"].to(self.device)
        input_ids, attention_mask = self._tokenize_prompts(batch["prompts"])

        kwargs = dict(
            vae=self.vae,
            text_encoder=self.text_encoder,
            noise_scheduler=self.noise_scheduler,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_timesteps=self.num_timesteps,
            device=str(self.device),
        )

        # Reference ELBOs (no gradient)
        with torch.no_grad():
            ref_chosen_elbo = diffusion_elbo(
                self.ref_policy.transformer, frames=chosen_frames, **kwargs
            )
            ref_rejected_elbo = diffusion_elbo(
                self.ref_policy.transformer, frames=rejected_frames, **kwargs
            )

        # Policy ELBOs (with gradient for policy, not reference)
        with self.accelerator.accumulate(self.policy.transformer):
            policy_chosen_elbo = diffusion_elbo(
                self.policy.transformer, frames=chosen_frames, **kwargs
            )
            policy_rejected_elbo = diffusion_elbo(
                self.policy.transformer, frames=rejected_frames, **kwargs
            )

            loss, r_chosen, r_rejected = dpo_loss(
                policy_chosen_elbo,
                policy_rejected_elbo,
                ref_chosen_elbo,
                ref_rejected_elbo,
                beta=self.beta,
                label_smoothing=self.label_smoothing,
            )

            if train:
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.policy.trainable_parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        acc = (r_chosen > r_rejected).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc,
                "chosen_reward": r_chosen.mean().item(),
                "rejected_reward": r_rejected.mean().item()}

    def train(self):
        cfg = self.config["training"]
        global_step = 0

        for epoch in range(cfg["num_epochs"]):
            self.policy.transformer.train()
            epoch_losses = []

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                metrics = self._step(batch, train=True)
                epoch_losses.append(metrics["loss"])

                if self.accelerator.sync_gradients:
                    global_step += 1
                    if global_step % cfg.get("logging_steps", 25) == 0:
                        print(f"  step={global_step} loss={metrics['loss']:.4f} "
                              f"acc={metrics['accuracy']:.4f} "
                              f"r_chosen={metrics['chosen_reward']:.3f} "
                              f"r_rejected={metrics['rejected_reward']:.3f}")

                    if global_step % cfg.get("save_steps", 200) == 0:
                        ckpt = os.path.join(self.output_dir, f"step_{global_step}")
                        self.accelerator.unwrap_model(self.policy.transformer).save_pretrained(ckpt)

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}: avg_loss={mean_loss:.4f}")

        final_path = os.path.join(self.output_dir, "final")
        self.accelerator.unwrap_model(self.policy.transformer).save_pretrained(final_path)
        print(f"DPO training complete. Saved to {final_path}")
        return final_path
