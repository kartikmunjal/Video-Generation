@echo off
cd /d "C:\Users\Kunal Munjal\Downloads\video_research\video-generation"
if exist results\confirmatory\classification.exitcode del results\confirmatory\classification.exitcode
"C:\Users\Kunal Munjal\Downloads\T1forecasting\.venv311\Scripts\python.exe" scripts\run_augmentation_classification.py --assignments "C:\Users\Kunal Munjal\Downloads\video_research\video-curation\results\confirmatory\raw\official_assignments.jsonl" --clean-features "C:\Users\Kunal Munjal\Downloads\video_research\video-curation\results\confirmatory\raw\r3d18_features.npz" --augmentation-features results\confirmatory\augmentation_features.npz --selections results\confirmatory\selections.jsonl --config configs\study_v1.json --output-dir results\confirmatory\classification 1> results\confirmatory\classification.stdout.log 2> results\confirmatory\classification.stderr.log
echo %errorlevel% > results\confirmatory\classification.exitcode
