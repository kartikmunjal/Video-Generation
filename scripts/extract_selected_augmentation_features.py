#!/usr/bin/env python3
"""Extract frozen R3D-18 features for selected augmentation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--reward-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    sys.path.insert(0, str(args.reward_repo / "scripts"))
    sys.path.insert(0, str(Path(__file__).parent))
    from build_augmentation_selections import temporal_candidate
    from score_intervention_pairs import degrade, load_frames

    records = [json.loads(line) for line in args.selections.read_text().splitlines() if line.strip()]
    weights = R3D_18_Weights.KINETICS400_V1
    transform = weights.transforms()
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.eval().cuda()
    output_features, paths, splits, classes, arms, candidate_indices = [], [], [], [], [], []
    with torch.inference_mode():
        for record_index, record in enumerate(records):
            frames, fps = load_frames(args.video_root / Path(record["relative_path"]))
            for arm in ("random", "reward_guided", "pixel_fidelity_diagnostic"):
                candidate_index = int(record["selections"][arm])
                if candidate_index == -1:
                    altered = frames
                else:
                    candidate = record["candidates"][candidate_index]
                    kind, value = candidate["transformation"], candidate["value"]
                    if kind in {"speed_factor", "temporal_crop_fraction"}:
                        altered = temporal_candidate(frames, kind, float(value))
                    else:
                        altered = degrade(frames, fps, kind, value, seed=0)
                tensor = torch.from_numpy(altered).permute(0, 3, 1, 2)
                feature = model(transform(tensor).unsqueeze(0).cuda()).squeeze(0)
                output_features.append(feature.float().cpu().numpy())
                paths.append(record["relative_path"])
                splits.append(record["official_split"])
                classes.append(record["class_index"])
                arms.append(arm)
                candidate_indices.append(candidate_index)
            if (record_index + 1) % 100 == 0:
                print(json.dumps({"completed_sources": record_index + 1,
                                  "total_sources": len(records)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        model_hash.update(name.encode())
        model_hash.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    np.savez_compressed(
        args.output,
        features=np.asarray(output_features, dtype=np.float32),
        relative_paths=np.asarray(paths), official_splits=np.asarray(splits),
        class_indices=np.asarray(classes), arms=np.asarray(arms),
        candidate_indices=np.asarray(candidate_indices), weights=np.asarray([str(weights)]),
        weights_parameter_sha256=np.asarray([model_hash.hexdigest()]),
    )
    print(args.output)


if __name__ == "__main__":
    main()
