"""
Run DiffusionDPO on a preference dataset.

Usage:
    python scripts/train_dpo.py --config configs/dpo_config.yaml
    python scripts/train_dpo.py --config configs/dpo_config.yaml --beta 1.0
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.dpo_video import DiffusionDPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dpo_config.yaml")
    p.add_argument("--beta", type=float, default=None, help="Override dpo.beta")
    p.add_argument("--lora_checkpoint", default=None, help="Override model.lora_checkpoint")
    p.add_argument("--preference_path", default=None, help="Override data.preference_path")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.beta is not None:
        config["dpo"]["beta"] = args.beta
    if args.lora_checkpoint is not None:
        config["model"]["lora_checkpoint"] = args.lora_checkpoint
    if args.preference_path is not None:
        config["data"]["preference_path"] = args.preference_path
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir

    print(f"DiffusionDPO | β={config['dpo']['beta']} | "
          f"data={config['data']['preference_path']}")

    trainer = DiffusionDPOTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
