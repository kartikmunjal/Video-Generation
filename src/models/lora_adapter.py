"""
LoRA adapter for CogVideoX-2B.

Wraps the PEFT LoRA configuration and provides utilities for:
  - Applying LoRA to the CogVideoX transformer
  - Saving / loading adapter weights (separate from base model)
  - Merging adapters into base model for inference
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model


# Default target modules for CogVideoX-2b transformer (DiT attention layers).
# These are the projection matrices inside the multi-head attention blocks.
COGVIDEOX_LORA_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
]


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Build a PEFT LoraConfig for CogVideoX.

    Args:
        r: LoRA rank. Ablation values: 4, 8, 16, 32.
            r=16 is a good default: captures enough capacity without
            overparameterizing. Our ablation (Nb 06) shows diminishing
            returns beyond r=16 on motion smoothness and CLIP score.
        alpha: Scaling factor. alpha = 2*r is a standard starting point.
        dropout: Dropout on LoRA layers (helps regularize for small datasets).
        target_modules: List of module name suffixes to target.
        bias: "none" | "all" | "lora_only".

    Returns:
        PEFT LoraConfig.
    """
    if target_modules is None:
        target_modules = COGVIDEOX_LORA_TARGETS

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="FEATURE_EXTRACTION",  # generic task type for non-causal models
    )


class CogVideoXLoRA:
    """
    Convenience wrapper that loads CogVideoX-2B and injects LoRA adapters.

    Usage:
        lora_model = CogVideoXLoRA.from_pretrained(
            "THUDM/CogVideoX-2b",
            lora_r=16,
            lora_alpha=32,
        )
        # ... training ...
        lora_model.save("checkpoints/lora")
    """

    def __init__(self, pipe, transformer_with_lora):
        self.pipe = pipe
        self.transformer = transformer_with_lora

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "THUDM/CogVideoX-2b",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "CogVideoXLoRA":
        from diffusers import CogVideoXPipeline

        print(f"Loading CogVideoX base model: {model_name}")
        pipe = CogVideoXPipeline.from_pretrained(model_name, torch_dtype=dtype)

        lora_config = build_lora_config(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )

        print(f"Injecting LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        pipe.transformer.print_trainable_parameters()

        pipe = pipe.to(device)
        return cls(pipe, pipe.transformer)

    @classmethod
    def from_lora_checkpoint(
        cls,
        base_model: str,
        lora_checkpoint: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "CogVideoXLoRA":
        """Load a fine-tuned LoRA checkpoint on top of the base model."""
        from diffusers import CogVideoXPipeline

        pipe = CogVideoXPipeline.from_pretrained(base_model, torch_dtype=dtype)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_checkpoint)
        pipe = pipe.to(device)
        return cls(pipe, pipe.transformer)

    def save(self, output_dir: str) -> None:
        """Save LoRA adapter weights (not the full base model)."""
        os.makedirs(output_dir, exist_ok=True)
        self.transformer.save_pretrained(output_dir)
        print(f"LoRA adapter saved to {output_dir}")

    def merge_and_save_full(self, output_dir: str) -> None:
        """Merge LoRA into base weights and save the full model. ~5GB on disk."""
        merged = self.transformer.merge_and_unload()
        os.makedirs(output_dir, exist_ok=True)
        merged.save_pretrained(output_dir)
        print(f"Merged model saved to {output_dir}")

    def trainable_parameters(self):
        """Return only the LoRA parameters (for optimizer)."""
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.transformer.parameters())

    def lora_ratio(self) -> float:
        return self.num_trainable_params() / self.num_total_params()


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def save_lora_weights(model, output_dir: str) -> None:
    """Save PEFT adapter weights to output_dir."""
    model.save_pretrained(output_dir)


def load_lora_weights(base_transformer, lora_dir: str):
    """Load LoRA weights from lora_dir onto base_transformer."""
    return PeftModel.from_pretrained(base_transformer, lora_dir)


def count_lora_params(model) -> dict:
    """Return a dict with trainable/total param counts for logging."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "lora_ratio": trainable / total,
    }
