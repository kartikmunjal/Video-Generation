"""
Ablation sweep over LoRA rank, reward signal design, DPO beta, and more.

Runs all conditions defined in ablation_config.yaml, generates videos for
each, evaluates quality metrics, and saves a summary CSV.

Usage:
    python scripts/run_ablation.py --config configs/ablation_config.yaml
    python scripts/run_ablation.py --config configs/ablation_config.yaml \
        --ablation lora_rank   # run only one ablation
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.evaluate import run_evaluation


def set_nested(d: dict, key_path: str, value: Any) -> None:
    """Set a nested dict value using dot-notation key path."""
    parts = key_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def get_nested(d: dict, key_path: str) -> Any:
    parts = key_path.split(".")
    for part in parts:
        d = d[part]
    return d


def run_lora_rank_ablation(cfg: dict, ablation_cfg: dict, output_dir: str) -> pd.DataFrame:
    """Train LoRA with different ranks and evaluate."""
    from training.lora_finetune import LoRAFinetuner
    import subprocess

    results = []
    for r in ablation_cfg["values"]:
        print(f"\n{'='*50}")
        print(f"LoRA rank = {r}")
        print(f"{'='*50}")

        condition = f"lora_r{r}"
        ckpt_dir = os.path.join(output_dir, "lora_rank", condition)
        video_dir = os.path.join(output_dir, "lora_rank", condition + "_videos")
        eval_path = os.path.join(output_dir, "lora_rank", condition + "_eval.csv")

        # Train LoRA with this rank
        lora_config = copy.deepcopy(cfg.get("base_lora_config", {}))
        lora_config["lora"]["r"] = r
        lora_config["lora"]["alpha"] = 2 * r
        lora_config["training"]["output_dir"] = ckpt_dir

        if not os.path.exists(os.path.join(ckpt_dir, "final")):
            trainer = LoRAFinetuner(lora_config)
            trainer.setup()
            trainer.train()
        else:
            print(f"  Checkpoint exists, skipping training: {ckpt_dir}")

        # Generate evaluation videos
        subprocess.run([
            sys.executable, "scripts/generate_videos.py",
            "--prompts", cfg["eval_prompts"],
            "--output_dir", video_dir,
            "--lora_checkpoint", os.path.join(ckpt_dir, "final"),
            "--num_videos_per_prompt", "1",
        ], check=True)

        # Evaluate
        df = run_evaluation(
            video_dir=video_dir,
            prompts_path=cfg["eval_prompts"],
            output_path=eval_path,
            condition_name=condition,
        )
        summary = {
            "condition": condition,
            "lora_r": r,
            **{k: v for k, v in df.mean(numeric_only=True).items()},
        }
        results.append(summary)

    return pd.DataFrame(results)


def run_reward_signal_ablation(cfg: dict, ablation_cfg: dict, output_dir: str) -> pd.DataFrame:
    """Evaluate videos with different reward weight combinations."""
    results = []
    base_video_dir = cfg.get("base_video_dir", os.path.join(output_dir, "base_videos"))

    for weights in ablation_cfg["values"]:
        condition = "clip{:.0f}_motion{:.0f}_temporal{:.0f}".format(
            weights["clip"] * 10, weights["motion"] * 10, weights["temporal"] * 10
        )
        eval_path = os.path.join(output_dir, "reward_signal", condition + "_eval.csv")
        print(f"\n  Reward weights: {weights} → condition: {condition}")

        df = run_evaluation(
            video_dir=base_video_dir,
            prompts_path=cfg["eval_prompts"],
            output_path=eval_path,
            condition_name=condition,
            weights=weights,
        )
        results.append({"condition": condition, **weights,
                         **{k: v for k, v in df.mean(numeric_only=True).items()}})

    return pd.DataFrame(results)


def run_dpo_beta_ablation(cfg: dict, ablation_cfg: dict, output_dir: str) -> pd.DataFrame:
    """Run DPO with different β values and evaluate."""
    import subprocess
    results = []

    for beta in ablation_cfg["values"]:
        condition = f"dpo_beta{beta}"
        ckpt_dir = os.path.join(output_dir, "dpo_beta", condition)
        video_dir = os.path.join(output_dir, "dpo_beta", condition + "_videos")
        eval_path = os.path.join(output_dir, "dpo_beta", condition + "_eval.csv")

        print(f"\n  DPO β = {beta}")

        # Run DPO with this beta
        subprocess.run([
            sys.executable, "scripts/train_dpo.py",
            "--config", "configs/dpo_config.yaml",
            "--beta", str(beta),
            "--output_dir", ckpt_dir,
        ], check=True)

        # Generate videos from DPO checkpoint
        subprocess.run([
            sys.executable, "scripts/generate_videos.py",
            "--prompts", cfg["eval_prompts"],
            "--output_dir", video_dir,
            "--lora_checkpoint", os.path.join(ckpt_dir, "final"),
        ], check=True)

        df = run_evaluation(video_dir, cfg["eval_prompts"], eval_path, condition)
        results.append({"condition": condition, "beta": beta,
                         **{k: v for k, v in df.mean(numeric_only=True).items()}})

    return pd.DataFrame(results)


ABLATION_FNS = {
    "lora_rank": run_lora_rank_ablation,
    "reward_weights": run_reward_signal_ablation,
    "dpo_beta": run_dpo_beta_ablation,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ablation_config.yaml")
    p.add_argument("--ablation", default=None, help="Run only this ablation (e.g., lora_rank)")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = args.output_dir or cfg.get("output_dir", "checkpoints/ablations")
    os.makedirs(output_dir, exist_ok=True)

    ablations_to_run = (
        {args.ablation: cfg["ablations"][args.ablation]}
        if args.ablation
        else cfg["ablations"]
    )

    all_results = {}
    for name, ablation_cfg in ablations_to_run.items():
        if name not in ABLATION_FNS:
            print(f"Warning: no handler for ablation '{name}'. Skipping.")
            continue
        print(f"\n{'#'*60}")
        print(f"# Running ablation: {name}")
        print(f"{'#'*60}")
        df = ABLATION_FNS[name](cfg, ablation_cfg, output_dir)
        all_results[name] = df
        summary_path = os.path.join(output_dir, f"{name}_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\n{name} summary saved to {summary_path}")
        print(df.to_string(index=False))

    # Combined summary
    combined_path = os.path.join(output_dir, "ablation_summary.json")
    with open(combined_path, "w") as f:
        json.dump(
            {k: v.to_dict(orient="records") for k, v in all_results.items()},
            f, indent=2, default=str
        )
    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
