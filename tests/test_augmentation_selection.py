import pandas as pd
import numpy as np

from scripts.build_augmentation_selections import aligned_pixel_mse, stable_sources
from scripts.analyze_augmentation import paired_table, point_outcomes


def test_source_selection_is_exact_and_deterministic():
    rows = pd.DataFrame({"relative_path": [f"v{i}" for i in range(20)]})
    first = stable_sources(rows, 5, split=1, cls=2)
    second = stable_sources(rows, 5, split=1, cls=2)
    assert first.relative_path.tolist() == second.relative_path.tolist()
    assert len(first) == first.relative_path.nunique() == 5


def test_pixel_mse_aligns_codec_padded_grid():
    source = np.zeros((2, 9, 11, 3), dtype=np.uint8)
    padded = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    assert aligned_pixel_mse(source, padded) == 0.0


def test_paired_outcomes_preserve_video_alignment():
    rows = []
    for split in (1, 2, 3):
        for seed in (10, 11, 12):
            for cls in range(10):
                for video in range(2):
                    common = {"official_split": split, "seed": seed,
                              "relative_path": f"s{split}_c{cls}_v{video}",
                              "class_index": cls}
                    rows.append({**common, "arm": "reward_guided", "correct": 1})
                    rows.append({**common, "arm": "random", "correct": 0})
    paired = paired_table(pd.DataFrame(rows), "reward_guided", "random")
    macro, worst = point_outcomes(paired)
    assert macro == 1.0
    assert worst == 1.0
