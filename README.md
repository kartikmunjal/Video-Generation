# Reward-Guided Video Augmentation

Research software for testing whether a source-disjoint, intervention-trained
video-quality reward can select better training augmentations than random
selection under an identical data and compute budget.

## Evidence status

The confirmatory study is **complete**. Its generated report, raw selections,
features, predictions, integrity result, environment record, and SHA-256
artifact manifest are in [`results/confirmatory/`](results/confirmatory/).
The frozen comparison did not establish a macro-recall improvement over random
augmentation, and the proposed split×class mechanism was unsupported; those
null results are retained. Earlier
DiffusionDPO, human-win-rate, attention-causality, FVD, and camera-control
numbers were withdrawn because the repository lacks the raw generations,
checkpoints, annotations, and executed run manifests needed to support them.

The frozen design is [RESEARCH_PLAN.md](RESEARCH_PLAN.md); parameters are in
[configs/study_v1.json](configs/study_v1.json).
See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the gated executable
workflow.

## Confirmatory comparison

Four arms—none, random, reward-guided, and a pixel-fidelity diagnostic—receive
the same UCF-101 sources, selected-example counts, and optimizer iteration cap.
Realized iterations are recorded. Evaluation uses
the three official recognition splits and three optimization seeds per split.
Primary outcomes are guided-minus-random macro accuracy and worst-decile recall,
with paired hierarchical-bootstrap 95% confidence intervals.

<!-- BEGIN GENERATED CONFIRMATORY RESULTS -->
## Confirmatory results

The study evaluated **7,575 training sources**, **30,300
candidate augmentations**, four equal-example arms, **`N_trials=3` official
splits**, three deterministic reproducibility repeats, and **2,000 paired
hierarchical-bootstrap replicates**. The classifier produced
**33,639 test predictions per arm** across the
three seed labels; **0 split × arm × video predictions**
varied across seeds.

### Guided minus random

- Macro recall: **0.114 pp [-0.227, 0.518]**, Holm-adjusted
  **p=0.568**.
- Worst-decile recall: **1.528 pp [-0.236, 3.233]**, Holm-adjusted
  **p=0.216**.
- Guided rejection to a clean duplicate: **42.53%
  [41.06%, 44.01%]**.
- Mechanism association across **303
  split × class cells**: Spearman rho **-0.015 [-0.120, 0.097]**.

Both downstream confidence intervals cross zero, Holm-adjusted tests do not
reject the null, and the mechanism interval crosses zero. The confirmatory
study therefore did not establish that reward guidance improves recognition
over random augmentation under this budget.

### Selected augmentation counts

| Arm | Counts |
|---|---|
| `random` | `brightness_multiplier`=1,940, `h264_crf`=1,920, `speed_factor`=1,853, `temporal_crop_fraction`=1,862 |
| `reward_guided` | `clean_duplicate`=3,222, `h264_crf`=458, `speed_factor`=3,340, `temporal_crop_fraction`=555 |
| `pixel_fidelity_diagnostic` | `brightness_multiplier`=1, `h264_crf`=7,533, `temporal_crop_fraction`=41 |

### Equal-budget classifier execution

| Arm | Training examples | Realized iterations | Mean fit time |
|---|---:|---:|---:|
| `none` | 5,050 | 22–27 | 0.432 s |
| `pixel_fidelity_diagnostic` | 5,050 | 23–26 | 0.437 s |
| `random` | 5,050 | 24–24 | 0.421 s |
| `reward_guided` | 5,050 | 23–28 | 0.445 s |

The parent-matched `lbfgs` solver is deterministic, so the seed-labelled runs
are reproducibility repeats rather than independent trials. Inferential
replication comes from the three official dataset splits.
<!-- END GENERATED CONFIRMATORY RESULTS -->

The mechanism test asks whether reward guidance reduces severe corruption
exposure and whether that reduction tracks downstream errors across split ×
class cells. This is an intervention-linked augmentation study, not a claim that
reward maximization generally improves text-to-video generation.

## Legacy CogVideoX code

The LoRA, reward-model, DiffusionDPO, interpretability, and camera-control
modules remain exploratory prototypes. They may become a separate extension
only after a new immutable protocol, real checkpoint hashes, paired random
control, independent evaluator, and executed multi-seed artifacts exist.

## Layout

- `src/`: exploratory model and training components.
- `configs/study_v1.json`: confirmatory augmentation-policy protocol.
- `results/confirmatory/`: complete raw and derived evidence bundle.
- other `results/` entries: legacy illustrative JSON excluded from confirmatory reporting.
- `tests/`: deterministic unit and integrity tests.
