#!/usr/bin/env python3
"""Freeze a study config and parent artifacts without overwrite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/study_v1.json"))
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("preregistration_manifest.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {args.output}")
    config = json.loads(args.config.read_text())
    missing = [str(path) for path in args.input if not path.is_file()]
    if missing:
        raise SystemExit(f"missing parent artifacts: {missing}")
    manifest = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(args.config),
        "parent_artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.input)
        ],
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
