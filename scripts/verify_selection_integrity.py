#!/usr/bin/env python3
"""Fail closed on selection lineage, hashes, budgets, and leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/study_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selections = pd.read_json(args.selections, lines=True)
    assignments = pd.read_json(args.assignments, lines=True)
    expected = len(config["official_splits"]) * 101 * config["training_sources_per_class"]
    if len(selections) != expected:
        raise SystemExit(f"expected {expected} selection sources, found {len(selections)}")
    if selections.duplicated(["official_split", "relative_path"]).any():
        raise SystemExit("duplicate split/source selection")
    checked = 0
    for split in config["official_splits"]:
        rows = selections[selections.official_split == split]
        counts = rows.groupby("class_index").size()
        if len(counts) != 101 or not (counts == config["training_sources_per_class"]).all():
            raise SystemExit(f"class budget mismatch in split {split}")
        test_paths = set(assignments[(assignments.official_split == split) &
                                     (assignments.partition == "test")].relative_path)
        if set(rows.relative_path) & test_paths:
            raise SystemExit(f"train/test leakage in split {split}")
        for row in rows.itertuples(index=False):
            if len(row.candidates) != config["candidates_per_source"]:
                raise SystemExit("candidate budget mismatch")
            valid = set(range(len(row.candidates)))
            for arm, index in row.selections.items():
                allowed = valid | ({-1} if arm == "reward_guided" else set())
                if int(index) not in allowed:
                    raise SystemExit(f"invalid selection index for {arm}")
            if sha256(args.video_root / Path(row.relative_path)) != row.source_sha256:
                raise SystemExit(f"source hash mismatch: {row.relative_path}")
            checked += 1
    payload = {
        "study_id": config["study_id"],
        "status": "passed",
        "sources_checked": checked,
        "official_splits": len(config["official_splits"]),
        "classes_per_split": 101,
        "sources_per_class": config["training_sources_per_class"],
        "source_hashes_verified": True,
        "train_test_overlap": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
