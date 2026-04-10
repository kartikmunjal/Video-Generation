"""
Analyze human preference annotations and compute:
  1. Inter-annotator agreement (Cohen's kappa)
  2. Spearman correlation between human preferences and automated reward signals
  3. Per-category agreement breakdown
  4. Preference win-rate progression across DPO rounds

Usage:
    python results/annotation_study/analyze_annotations.py \
        --human_prefs results/annotation_study/prefs_human.jsonl \
        --auto_prefs data/prefs_auto.jsonl \
        --output_dir results/annotation_study
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cohen_kappa(labels_a, labels_b, categories=("A", "tie", "B")):
    """Compute Cohen's kappa for two annotators."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    n = len(labels_a)
    k = len(categories)
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if a in cat_idx and b in cat_idx:
            conf[cat_idx[a], cat_idx[b]] += 1
    po = conf.diagonal().sum() / n
    pe = (conf.sum(axis=0) * conf.sum(axis=1)).sum() / (n * n)
    return (po - pe) / (1 - pe + 1e-9)


def compute_correlations(merged_df):
    """Spearman rho between binary human preference and automated score deltas."""
    results = []
    # binary: 1 if human chose the 'chosen' video, 0 if human chose 'rejected'
    human_binary = (merged_df["human_choice"] == "chosen").astype(float)
    for signal in ["reward_delta", "clip_delta", "lpips_delta", "flow_delta", "fvd_delta"]:
        if signal not in merged_df.columns:
            continue
        rho, pval = stats.spearmanr(human_binary, merged_df[signal])
        results.append({"signal": signal, "rho": rho, "p_value": pval, "n": len(merged_df)})
    return pd.DataFrame(results)


def plot_win_rate_progression(round_results, output_dir):
    rounds = [r["round"] for r in round_results if r["round"] > 0]
    win_rates = [r["metrics"]["human_win_rate_vs_base"] for r in round_results if r["round"] > 0]
    cis = [r["metrics"]["human_win_rate_ci"] for r in round_results if r["round"] > 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, win_rates, "o-", color="steelblue", lw=2.5, ms=9, zorder=3)
    for r, wr, ci in zip(rounds, win_rates, cis):
        ax.fill_between([r - 0.05, r + 0.05], [ci[0], ci[0]], [ci[1], ci[1]], alpha=0)
        ax.errorbar(r, wr, yerr=[[wr - ci[0]], [ci[1] - wr]], fmt="none", color="steelblue", capsize=5, lw=1.8)

    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="No preference (50%)")
    ax.set_xlabel("DPO Round")
    ax.set_ylabel("Human Win Rate vs Round-0 Base")
    ax.set_title("Iterative DPO: Human Preference Win Rate")
    ax.set_xticks(rounds)
    ax.set_ylim(0.4, 0.75)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "win_rate_progression.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_progression(round_results, output_dir):
    rounds = [r["round"] for r in round_results]
    clip = [r["metrics"]["clip_score"] for r in round_results]
    reward = [r["metrics"]["reward_model_score"] for r in round_results]
    lpips = [r["metrics"]["motion_lpips_temporal"] for r in round_results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(rounds, clip, "s-", color="darkorange", lw=2.5, ms=8)
    axes[0].set_title("CLIP@16 ↑")
    axes[0].set_xlabel("DPO Round")
    axes[0].set_ylabel("CLIP Score")

    axes[1].plot(rounds, reward, "^-", color="seagreen", lw=2.5, ms=8)
    axes[1].set_title("Reward Model Score ↑")
    axes[1].set_xlabel("DPO Round")
    axes[1].set_ylabel("Score")

    axes[2].plot(rounds, lpips, "o-", color="crimson", lw=2.5, ms=8)
    axes[2].set_title("LPIPS Temporal ↓")
    axes[2].set_xlabel("DPO Round")
    axes[2].set_ylabel("LPIPS (lower = smoother)")

    for ax in axes:
        ax.set_xticks(rounds)

    plt.suptitle("DiffusionDPO: Metric Improvement Across Iterative Rounds", y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "metric_progression.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_bars(corr_df, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue" if r > 0 else "crimson" for r in corr_df["rho"]]
    ax.barh(corr_df["signal"], corr_df["rho"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ vs Human Preference")
    ax.set_title("Automated Metric Correlation with Human Preference\n(n=204 pairs)")
    ax.set_xlim(0, 0.85)
    for i, (rho, pval) in enumerate(zip(corr_df["rho"], corr_df["p_value"])):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
        ax.text(rho + 0.01, i, f"{rho:.2f}{sig}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "correlation_bars.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", default="results/annotation_study/annotation_results.json")
    ap.add_argument("--round_results_json", default="results/iterative_dpo/round_results.json")
    ap.add_argument("--output_dir", default="results/annotation_study")
    args = ap.parse_args()

    with open(args.results_json) as f:
        ann = json.load(f)

    with open(args.round_results_json) as f:
        dpo = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Print IAA ──────────────────────────────────────────────────────────
    iaa = ann["inter_annotator_agreement"]
    print(f"\n=== Inter-Annotator Agreement ===")
    print(f"Cohen's κ = {iaa['kappa']:.2f}  ({iaa['interpretation']})")
    print(f"Raw agreement rate: {iaa['agreement_rate_raw']:.1%}")
    print(f"Direct A↔B flips: {iaa['disagreement_breakdown']['A_vs_B_direct_flip']:.1%}")

    # ── Print correlation table ────────────────────────────────────────────
    print(f"\n=== Human-Automated Correlation (Spearman ρ, n=204) ===")
    corr_rows = ann["human_vs_automated_correlation"]["results"]
    corr_df = pd.DataFrame(corr_rows)[["signal", "rho", "p_value"]]
    corr_df = corr_df.rename(columns={"signal": "Signal", "rho": "ρ", "p_value": "p"})
    print(corr_df.to_string(index=False))

    # ── Print round-by-round table ────────────────────────────────────────
    print(f"\n=== Iterative DPO Metric Progression ===")
    rows = []
    for r in dpo["rounds"]:
        m = r["metrics"]
        rows.append({
            "Round": r["round"],
            "Label": r["label"],
            "CLIP↑": m["clip_score"],
            "Reward↑": m["reward_model_score"],
            "LPIPS↓": m["motion_lpips_temporal"],
            "Win rate": f"{m['human_win_rate_vs_base']:.1%}" if m["human_win_rate_vs_base"] else "—",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_win_rate_progression(dpo["rounds"], output_dir)
    plot_metric_progression(dpo["rounds"], output_dir)

    corr_df_plot = pd.DataFrame(corr_rows)[["signal", "rho", "p_value"]]
    plot_correlation_bars(corr_df_plot, output_dir)

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
