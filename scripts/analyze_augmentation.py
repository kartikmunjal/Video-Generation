#!/usr/bin/env python3
"""Paired hierarchical analysis for the frozen augmentation-policy study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def paired_table(predictions: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    keys = ["official_split", "seed", "relative_path", "class_index"]
    a = predictions[predictions.arm == left][keys + ["correct"]]
    b = predictions[predictions.arm == right][keys + ["correct"]]
    return a.merge(b, on=keys, suffixes=("_left", "_right"), validate="one_to_one")


def draw_outcomes(paired: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    split_draws = rng.choice(sorted(paired.official_split.unique()), 3, replace=True)
    macro, worst = [], []
    for split in split_draws:
        split_rows = paired[paired.official_split == split]
        seed_draws = rng.choice(sorted(split_rows.seed.unique()), 3, replace=True)
        for seed in seed_draws:
            rows = split_rows[split_rows.seed == seed]
            class_results = []
            for cls, class_rows in rows.groupby("class_index"):
                sampled = class_rows.iloc[rng.integers(0, len(class_rows), len(class_rows))]
                left = float(sampled.correct_left.mean())
                right = float(sampled.correct_right.mean())
                class_results.append((int(cls), left, right, left - right))
            class_results.sort(key=lambda value: (value[2], value[0]))
            macro.append(float(np.mean([value[3] for value in class_results])))
            count = max(1, int(np.ceil(0.1 * len(class_results))))
            worst.append(float(np.mean([value[3] for value in class_results[:count]])))
    return float(np.mean(macro)), float(np.mean(worst))


def point_outcomes(paired: pd.DataFrame) -> tuple[float, float]:
    macro, worst = [], []
    for (_, _), rows in paired.groupby(["official_split", "seed"]):
        class_results = []
        for cls, class_rows in rows.groupby("class_index"):
            left = float(class_rows.correct_left.mean())
            right = float(class_rows.correct_right.mean())
            class_results.append((int(cls), left, right, left - right))
        class_results.sort(key=lambda value: (value[2], value[0]))
        macro.append(float(np.mean([value[3] for value in class_results])))
        count = max(1, int(np.ceil(0.1 * len(class_results))))
        worst.append(float(np.mean([value[3] for value in class_results[:count]])))
    return float(np.mean(macro)), float(np.mean(worst))


def interval(estimate: float, draws: np.ndarray) -> dict:
    return {"estimate": estimate, "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])]}


def selected_candidate(record: pd.Series, arm: str) -> dict | None:
    index = int(record.selections[arm])
    return None if index == -1 else record.candidates[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    predictions = pd.read_json(args.predictions, lines=True)
    runs = pd.read_csv(args.run_metrics)
    selections = pd.read_json(args.selections, lines=True)
    required_arms = {"none", "random", "reward_guided", "pixel_fidelity_diagnostic"}
    if set(predictions.arm) != required_arms or set(runs.arm) != required_arms:
        raise SystemExit("arm contract mismatch")
    if predictions.official_split.nunique() != 3 or predictions.seed.nunique() != 3:
        raise SystemExit("trial multiplicity mismatch")

    paired = paired_table(predictions, "reward_guided", "random")
    macro_point, worst_point = point_outcomes(paired)
    rng = np.random.default_rng(2025)
    draws = np.asarray([draw_outcomes(paired, rng) for _ in range(args.bootstrap_replicates)])
    primary = {
        "guided_minus_random_macro_recall": interval(macro_point, draws[:, 0]),
        "guided_minus_random_worst_decile_recall": interval(worst_point, draws[:, 1]),
    }
    raw_ps = []
    for column in range(2):
        raw_ps.append(min(1.0, 2 * min(float((draws[:, column] <= 0).mean()),
                                        float((draws[:, column] >= 0).mean()))))
    order = np.argsort(raw_ps)
    adjusted = [0.0, 0.0]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (2 - rank) * raw_ps[index]))
        adjusted[index] = running
    for result, p_raw, p_holm in zip(primary.values(), raw_ps, adjusted):
        result.update({"p_raw": p_raw, "p_holm": p_holm})

    rejection = selections.selections.map(lambda item: int(item["reward_guided"]) == -1).astype(float)
    rejection_draws = []
    for _ in range(args.bootstrap_replicates):
        values = []
        for split in rng.choice(sorted(selections.official_split.unique()), 3, replace=True):
            rows = selections[selections.official_split == split]
            sampled = rows.iloc[rng.integers(0, len(rows), len(rows))]
            values.extend(sampled.selections.map(
                lambda item: int(item["reward_guided"]) == -1
            ).astype(float))
        rejection_draws.append(float(np.mean(values)))

    selection_frequency = {}
    for arm in ("random", "reward_guided", "pixel_fidelity_diagnostic"):
        names = []
        for row in selections.itertuples(index=False):
            candidate = selected_candidate(pd.Series(row._asdict()), arm)
            names.append("clean_duplicate" if candidate is None else candidate["transformation"])
        selection_frequency[arm] = {
            name: int(count) for name, count in pd.Series(names).value_counts().sort_index().items()
        }

    # Mechanism: does choosing lower predicted damage associate with lower class error?
    selection_cells = []
    for row in selections.itertuples(index=False):
        values = {}
        for arm in ("random", "reward_guided"):
            index = int(row.selections[arm])
            values[arm] = (
                0.0 if index == -1 else
                max(0.0, 2 * float(row.candidates[index]["reward_gap_probability"]) - 1)
            )
        selection_cells.append({"official_split": int(row.official_split),
                                "class_index": int(row.class_index),
                                "reward_gap_delta": values["reward_guided"] - values["random"]})
    selection_cells = pd.DataFrame(selection_cells).groupby(
        ["official_split", "class_index"], as_index=False
    ).reward_gap_delta.mean()
    errors = predictions[predictions.arm.isin(["random", "reward_guided"])].groupby(
        ["official_split", "class_index", "arm"], as_index=False
    ).correct.mean().pivot(index=["official_split", "class_index"], columns="arm", values="correct")
    errors["error_delta"] = (1 - errors.reward_guided) - (1 - errors.random)
    mechanism = selection_cells.merge(errors[["error_delta"]].reset_index(),
                                      on=["official_split", "class_index"], validate="one_to_one")
    rho = float(spearmanr(mechanism.reward_gap_delta, mechanism.error_delta).statistic)
    rho_draws = []
    for _ in range(args.bootstrap_replicates):
        sampled = mechanism.iloc[rng.integers(0, len(mechanism), len(mechanism))]
        rho_draws.append(float(spearmanr(sampled.reward_gap_delta, sampled.error_delta).statistic))

    result = {
        "study_id": "ucf101_reward_guided_augmentation_v1",
        "n_trials": 3,
        "n_optimization_seeds": 3,
        "test_predictions_per_arm": int(len(predictions) / 4),
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary": primary,
        "quality_rejection_rate": interval(float(rejection.mean()), np.asarray(rejection_draws)),
        "selection_frequency": selection_frequency,
        "mechanism": {
            "n_split_class_cells": int(len(mechanism)),
            "spearman_reward_gap_vs_error_delta": interval(rho, np.asarray(rho_draws)),
        },
        "run_summary": runs.groupby("arm").agg(
            fit_seconds_mean=("fit_seconds", "mean"),
            optimizer_iterations_min=("optimizer_iterations", "min"),
            optimizer_iterations_max=("optimizer_iterations", "max"),
            train_examples=("train_examples", "first"),
        ).to_dict(orient="index"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
