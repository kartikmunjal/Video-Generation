#!/usr/bin/env python3
"""Capture the executable software/hardware environment as JSON."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }
    try:
        import torch
        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        payload["torch"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
