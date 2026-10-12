#!/usr/bin/env python3
"""Generate, score, and select fixed augmentation candidates per source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def stable_sources(rows: pd.DataFrame, n: int, split: int, cls: int) -> pd.DataFrame:
    ordered = rows.assign(_key=[
        hashlib.sha256(f"augmentation:{split}:{cls}:{path}".encode()).hexdigest()
        for path in rows.relative_path
    ]).sort_values("_key")
    if len(ordered) < n:
        raise ValueError(f"class {cls} has only {len(ordered)} sources; need {n}")
    return ordered.head(n).drop(columns="_key")


def temporal_candidate(frames: np.ndarray, kind: str, value: float) -> np.ndarray:
    count = len(frames)
    if kind == "speed_factor":
        # Slow-down augmentation: sample a shorter source span into the same
        # output length, avoiding interpolation artifacts.
        end = max(1, round((count - 1) * float(value)))
        indices = np.linspace(0, end, count).astype(int)
        return frames[indices]
    if kind == "temporal_crop_fraction":
        kept = max(2, round(count * float(value)))
        start = (count - kept) // 2
        indices = np.linspace(start, start + kept - 1, count).astype(int)
        return frames[indices]
    raise ValueError(kind)


def aligned_pixel_mse(source: np.ndarray, candidate: np.ndarray) -> float:
    """Pixel MSE on the source grid, including codec-padded candidates."""
    aligned = candidate
    if candidate.shape[1:3] != source.shape[1:3]:
        height, width = source.shape[1:3]
        y_index = np.linspace(0, candidate.shape[1] - 1, height).round().astype(int)
        x_index = np.linspace(0, candidate.shape[2] - 1, width).round().astype(int)
        aligned = candidate[:, y_index][:, :, x_index]
    return float(np.mean((source.astype(np.float32) - aligned.astype(np.float32)) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--reward-repo", type=Path, required=True)
    parser.add_argument("--reward-model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/study_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reward_repo / "scripts"))
    from analyze_interventions import FEATURES
    from score_intervention_pairs import degrade, load_frames, metrics

    config = json.loads(args.config.read_text())
    assignments = pd.read_json(args.assignments, lines=True)
    candidate_specs = list(config["candidate_transformations"].items())
    if len(candidate_specs) != config["candidates_per_source"]:
        raise SystemExit("candidate count does not match frozen config")
    rows = []
    for split in config["official_splits"]:
        train = assignments[
            (assignments.official_split == split) & (assignments.partition == "train")
        ]
        selected_sources = pd.concat([
            stable_sources(class_rows, config["training_sources_per_class"], split, int(cls))
            for cls, class_rows in train.groupby("class_index", sort=True)
        ], ignore_index=True)
        payload = joblib.load(args.reward_model_dir / f"reward_split{split}.joblib")
        model = payload["model"]
        if payload["features"] != FEATURES:
            raise SystemExit("reward feature contract mismatch")
        for source in selected_sources.itertuples(index=False):
            frames, fps = load_frames(args.video_root / Path(source.relative_path))
            clean_metrics = metrics(frames)
            candidate_rows = []
            for candidate_index, (kind, value) in enumerate(candidate_specs):
                if kind in {"speed_factor", "temporal_crop_fraction"}:
                    altered = temporal_candidate(frames, kind, float(value))
                elif kind == "brightness_multiplier":
                    altered = degrade(frames, fps, kind, value, seed=0)
                elif kind == "h264_crf":
                    altered = degrade(frames, fps, kind, value, seed=0)
                else:
                    raise SystemExit(f"unsupported candidate: {kind}")
                altered_metrics = metrics(altered)
                vector = pd.DataFrame([{
                    f"delta_{feature}": clean_metrics[feature] - altered_metrics[feature]
                    for feature in FEATURES
                }])
                probability_clean_better = float(model.predict_proba(vector)[0, 1])
                # H.264 may pad to a codec-compatible macroblock grid. Align only
                # the pixel-fidelity diagnostic to the source grid; reward metrics
                # and the downstream candidate retain the decoded frames unchanged.
                mse = aligned_pixel_mse(frames, altered)
                candidate_rows.append({
                    "candidate_index": candidate_index,
                    "transformation": kind,
                    "value": value,
                    "reward_gap_probability": probability_clean_better,
                    "pixel_mse": mse,
                    "candidate_metrics": altered_metrics,
                })
            best_guided = min(candidate_rows, key=lambda row: row["reward_gap_probability"])
            guided = (
                best_guided["candidate_index"]
                if best_guided["reward_gap_probability"] <= config["reward_gap_rejection_threshold"]
                else -1  # duplicate clean source to preserve the exact example budget
            )
            fidelity = min(candidate_rows, key=lambda row: row["pixel_mse"])["candidate_index"]
            random_key = int(hashlib.sha256(
                f"random:{split}:{source.relative_path}".encode()
            ).hexdigest()[:8], 16)
            random_choice = random_key % len(candidate_rows)
            rows.append({
                "schema_version": 1,
                "official_split": int(split),
                "relative_path": source.relative_path,
                "source_group": source.source_group,
                "source_sha256": source.sha256,
                "class_index": int(source.class_index),
                "clean_metrics": clean_metrics,
                "candidates": candidate_rows,
                "selections": {
                    "random": int(random_choice),
                    "reward_guided": int(guided),
                    "pixel_fidelity_diagnostic": int(fidelity),
                },
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"sources": len(rows), "candidates": len(rows) * len(candidate_specs)}))


if __name__ == "__main__":
    main()
