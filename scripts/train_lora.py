"""
Train a LoRA adapter on CogVideoX-2B.

Usage:
    python scripts/train_lora.py --config configs/cogvideox_lora.yaml
    python scripts/train_lora.py --config configs/cogvideox_lora.yaml --lora_r 32  # override rank

All hyperparams live in the YAML; CLI flags can override individual values.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.lora_finetune import LoRAFinetuner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cogvideox_lora.yaml")
    p.add_argument("--lora_r", type=int, default=None, help="Override lora.r")
    p.add_argument("--lora_alpha", type=int, default=None, help="Override lora.alpha")
    p.add_argument("--learning_rate", type=float, default=None, help="Override training.learning_rate")
    p.add_argument("--num_epochs", type=int, default=None, help="Override training.num_epochs")
    p.add_argument("--output_dir", default=None, help="Override training.output_dir")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.lora_r is not None:
        config["lora"]["r"] = args.lora_r
        if config["lora"]["alpha"] == 2 * 16:  # only auto-update if using default
            config["lora"]["alpha"] = 2 * args.lora_r
    if args.lora_alpha is not None:
        config["lora"]["alpha"] = args.lora_alpha
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    if args.wandb_run_name is not None:
        config.setdefault("wandb", {})["run_name"] = args.wandb_run_name

    print("Configuration:")
    print(f"  Model: {config['model']['pretrained_model_name_or_path']}")
    print(f"  LoRA r={config['lora']['r']}, alpha={config['lora']['alpha']}")
    print(f"  LR: {config['training']['learning_rate']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Output: {config['training']['output_dir']}")

    finetuner = LoRAFinetuner(config)
    finetuner.setup()
    finetuner.train()


if __name__ == "__main__":
    main()
