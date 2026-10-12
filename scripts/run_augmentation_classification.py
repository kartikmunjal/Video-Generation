#!/usr/bin/env python3
"""Evaluate none/random/reward-guided/oracle augmentation under equal budgets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--clean-features", type=Path, required=True)
    parser.add_argument("--augmentation-features", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/study_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assignments = pd.read_json(args.assignments, lines=True)
    selections = pd.read_json(args.selections, lines=True)
    clean = np.load(args.clean_features)
    clean_paths = clean["relative_paths"].astype(str)
    clean_index = {path: index for index, path in enumerate(clean_paths)}
    augmented = np.load(args.augmentation_features)
    if str(clean["weights"][0]) != str(augmented["weights"][0]):
        raise SystemExit("clean/augmentation feature weight enum mismatch")
    if str(clean["weights_parameter_sha256"][0]) != str(
        augmented["weights_parameter_sha256"][0]
    ):
        raise SystemExit("clean/augmentation R3D-18 parameter hash mismatch")
    if not np.isfinite(clean["features"]).all() or not np.isfinite(augmented["features"]).all():
        raise SystemExit("nonfinite feature values")
    if len(augmented["features"]) != 3 * len(selections):
        raise SystemExit("augmentation feature count mismatch")
    aug_lookup = {
        (int(split), str(path), str(arm)): vector
        for split, path, arm, vector in zip(
            augmented["official_splits"], augmented["relative_paths"],
            augmented["arms"], augmented["features"]
        )
    }
    run_rows, prediction_rows = [], []
    for split in config["official_splits"]:
        sources = selections[selections.official_split == split]
        test = assignments[(assignments.official_split == split) & (assignments.partition == "test")]
        test_X = clean["features"][[clean_index[path] for path in test.relative_path]]
        test_y = test.class_index.to_numpy()
        clean_X = clean["features"][[clean_index[path] for path in sources.relative_path]]
        clean_y = sources.class_index.to_numpy()
        for seed in config["optimization_seeds"]:
            for arm in config["arms"]:
                if arm == "none":
                    # Duplicate clean rows so example count and optimizer exposure
                    # match every augmented arm exactly.
                    train_X = np.concatenate([clean_X, clean_X], axis=0)
                else:
                    arm_X = np.stack([
                        aug_lookup[(split, path, arm)] for path in sources.relative_path
                    ])
                    train_X = np.concatenate([clean_X, arm_X], axis=0)
                train_y = np.concatenate([clean_y, clean_y])
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                       random_state=seed),
                )
                started = time.perf_counter()
                model.fit(train_X, train_y)
                prediction = model.predict(test_X)
                elapsed = time.perf_counter() - started
                run_rows.append({
                    "official_split": split, "seed": seed, "arm": arm,
                    "unique_train_sources": int(len(sources)),
                    "train_examples": int(len(train_y)), "test_examples": int(len(test)),
                    "accuracy": float((prediction == test_y).mean()),
                    "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
                    "optimizer_iterations": int(np.max(model.named_steps["logisticregression"].n_iter_)),
                    "optimizer_max_iterations": 500,
                    "fit_seconds": elapsed,
                })
                for row, label, predicted in zip(test.itertuples(index=False), test_y, prediction):
                    prediction_rows.append({
                        "official_split": split, "seed": seed, "arm": arm,
                        "relative_path": row.relative_path, "class_index": int(label),
                        "prediction": int(predicted), "correct": int(label == predicted),
                    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(args.output_dir / "run_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_json(
        args.output_dir / "predictions.jsonl", orient="records", lines=True
    )
    print(args.output_dir / "run_metrics.csv")


if __name__ == "__main__":
    main()
