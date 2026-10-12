# Implementation note 001 — H.264 diagnostic grid alignment

**Recorded:** 2026-09-29, before any augmentation-selection output existed.

The first confirmatory execution terminated on its first H.264 candidate. The
encoder padded a 226-pixel source height to 240 pixels for macroblock
compatibility, and the pixel-MSE diagnostic could not subtract arrays with
different grids. No selection, feature, classifier, or test outcome was
produced.

The implementation now deterministically nearest-resamples the decoded H.264
candidate back to the source grid **only for the explicitly diagnostic
pixel-MSE calculation**. The decoded
candidate used by the reward metrics and downstream R3D-18 feature extractor is
unchanged. The frozen candidate set, reward-guided policy, rejection threshold,
primary outcomes, and statistical plan are unchanged.
