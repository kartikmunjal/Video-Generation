# Reward-guided video augmentation policy study — frozen protocol

**Study ID:** `ucf101_reward_guided_augmentation_v1`  
**Status:** preregistered; no confirmatory results have been run  
**Parents:** `ucf101_curation_v1`, `ucf101_quality_reward_v1`

## Research question

Can a reward model validated on source-grouped controlled interventions select
quality-preserving video augmentations that improve downstream recognition over
random augmentation under identical data and compute budgets?

This is the confirmatory study for this repository. CogVideoX LoRA and
DiffusionDPO remain exploratory implementations until real checkpoints, paired
generation artifacts, and an independently validated evaluator exist.

## Arms

1. no augmentation;
2. uniformly random augmentation from the frozen candidate set;
3. reward-guided selection from the same candidates;
4. minimum pixel-MSE selection, used only as a fidelity diagnostic (not an oracle or upper bound).

Each source receives the same number of candidates. Every arm contains the same
number of unique sources and training examples and receives the same optimizer
iteration cap. Realized optimizer iterations are recorded. Reward-guided selection may use only the reward model fitted on the
training side of the corresponding official split.

## Candidate transformations

Slow temporal resampling, brightness reduction, mild H.264 compression, and a
centered temporal crop are frozen in `configs/study_v1.json`. Candidate generation is
seeded and records source lineage. Any candidate crossing the quality model's
predeclared rejection threshold is retained in raw artifacts but excluded by
the guided policy.

## Evaluation

- Official UCF-101 splits 1–3 are independent trials.
- Classifier and feature-extractor protocol matches the curation parent study.
- Optimization seeds are `[2025, 2026, 2027]` within every split and arm.
- The untouched official test videos are scored once per frozen checkpoint.

Primary outcomes:

- macro top-1 accuracy difference, guided minus random;
- worst-decile recall difference, guided minus random;
- quality-rejection rate among selected candidates.

Secondary outcomes include selection frequencies, per-class effects, training
time, realized optimizer iterations, and reward/downstream disagreement.

Use 2,000 deterministic paired hierarchical-bootstrap replicates over official
splits, seeds, and test videos. Report `N_trials`, per-trial sample counts, and
95% CIs. Holm correction covers the two downstream confirmatory outcomes.

## Mechanism test

An improvement is trusted only if guided selection mechanically reduces severe
temporal/spatial corruption exposure and the reduction predicts downstream
error changes across split × class cells. This supports an intervention-linked
mechanism; it does not establish that reward maximization generally improves
video generation.

## Integrity gates

- Parent study manifests and reward checkpoint hashes match.
- Source lineage is complete and train/test disjoint.
- Candidate counts, selected counts, unique-source counts, and optimizer steps
  match across arms.
- Reward normalization is trained-side only.
- No test metric is used for selection or early stopping.
- Reports regenerate from raw predictions and run manifests.

## Claim policy

All earlier claims about iterative DiffusionDPO, human win rates, attention
causality, and camera-control improvements are withdrawn unless separately
rerun under an immutable extension protocol.
