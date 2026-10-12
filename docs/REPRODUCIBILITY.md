# Reproduction guide

This study consumes the parent curation assignments/R3D-18 features, three
reward checkpoints, and locally acquired UCF-101 videos. Parent hashes are in
`preregistration_manifest.json`; the executed environment is in
`results/confirmatory/environment.json`.

Run `PYTHONPATH=. pytest -q`, then:

1. `build_augmentation_selections.py`;
2. `verify_selection_integrity.py` (must pass);
3. `extract_selected_augmentation_features.py`;
4. `run_augmentation_classification.py`;
5. `analyze_augmentation.py` with 2,000 bootstrap replicates;
6. `publish_confirmatory_report.py`.

The executed Windows commands are in `scripts/run_*_norgate.cmd`. Stages fail
closed on parent, hash, finite-value, leakage, or budget violations.
The two `protocol_implementation_note_*.md` files record pre-outcome execution
corrections.
