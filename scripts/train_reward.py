"""
Train the video reward model on preference pairs.

Usage:
    # Train on automated pairs only
    python scripts/train_reward.py --config configs/reward_config.yaml

    # Train on human-labeled pairs only
    python scripts/train_reward.py --config configs/reward_config.yaml --source_filter human

    # Merge auto and human pairs first, then train
    python scripts/train_reward.py \
        --config configs/reward_config.yaml \
        --merge_prefs data/prefs_auto.jsonl data/prefs_human.jsonl \
        --merged_output data/prefs_merged.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.preference_dataset import load_preference_jsonl, save_preference_jsonl
from training.reward_train import RewardTrainer


def merge_preference_files(paths, output_path):
    all_records = []
    for p in paths:
        records = load_preference_jsonl(p)
        print(f"  Loaded {len(records)} records from {p}")
        all_records.extend(records)
    save_preference_jsonl(all_records, output_path)
    print(f"Merged {len(all_records)} records → {output_path}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reward_config.yaml")
    p.add_argument("--source_filter", default=None, choices=["human", "automated"],
                   help="Only use preferences from this source")
    p.add_argument("--merge_prefs", nargs="+", default=None,
                   help="List of JSONL preference files to merge before training")
    p.add_argument("--merged_output", default="data/prefs_merged.jsonl")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.merge_prefs:
        merged_path = merge_preference_files(args.merge_prefs, args.merged_output)
        config["data"]["preference_path"] = merged_path

    if args.source_filter:
        config["data"]["source_filter"] = args.source_filter

    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    print(f"Training reward model on: {config['data']['preference_path']}")
    trainer = RewardTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
