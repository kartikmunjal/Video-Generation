"""
Generate a metrics summary report from ablation results.

Usage:
    python eval/metrics_report.py --results_dir checkpoints/ablations
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_lora_rank(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["prompt_adherence_mean", "motion_smoothness_mean", "temporal_consistency_mean"]
    labels = ["CLIP Score (↑)", "Motion Smoothness (↑)", "Temporal Consistency (↑)"]

    for ax, metric, label in zip(axes, metrics, labels):
        if metric in df.columns:
            ax.bar(df["lora_r"].astype(str), df[metric], color="#4C72B0", alpha=0.85)
            ax.set_xlabel("LoRA Rank (r)")
            ax.set_ylabel(label)
            ax.set_title(f"LoRA Rank vs {label.split('(')[0].strip()}")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Ablation: LoRA Rank", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_lora_rank.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_dpo_beta(df: pd.DataFrame, out_dir: str) -> None:
    if "beta" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    ax.bar(x, df["composite_mean"], color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"β={b}" for b in df["beta"]])
    ax.set_xlabel("DPO β (KL penalty)")
    ax.set_ylabel("Composite Reward (↑)")
    ax.set_title("Ablation: DPO β")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_dpo_beta.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary_table(summary: dict) -> None:
    for ablation_name, records in summary.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"{'='*60}")
        cols = [c for c in df.columns if any(
            m in c for m in ["condition", "lora_r", "beta", "clip", "motion", "temporal", "composite", "mean"]
        )]
        print(df[cols].to_string(index=False, float_format="{:.4f}".format))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="checkpoints/ablations")
    p.add_argument("--output_dir", default="checkpoints/ablations/figures")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.results_dir, "ablation_summary.json")
    if not os.path.exists(summary_path):
        print(f"No summary file found at {summary_path}. Run run_ablation.py first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print_summary_table(summary)

    if "lora_rank" in summary:
        df = pd.DataFrame(summary["lora_rank"])
        plot_lora_rank(df, args.output_dir)

    if "dpo_beta" in summary:
        df = pd.DataFrame(summary["dpo_beta"])
        plot_dpo_beta(df, args.output_dir)

    print(f"\nFigures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
