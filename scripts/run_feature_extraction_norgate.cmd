@echo off
cd /d "C:\Users\Kunal Munjal\Downloads\video_research\video-generation"
if exist results\confirmatory\features.exitcode del results\confirmatory\features.exitcode
"C:\Users\Kunal Munjal\Downloads\T1forecasting\.venv311\Scripts\python.exe" scripts\verify_selection_integrity.py --selections results\confirmatory\selections.jsonl --assignments "C:\Users\Kunal Munjal\Downloads\video_research\video-curation\results\confirmatory\raw\official_assignments.jsonl" --video-root "C:\Users\Kunal Munjal\Downloads\datasets\ucf101\UCF101\UCF-101" --config configs\study_v1.json --output results\confirmatory\selection_integrity.json 1> results\confirmatory\integrity.stdout.log 2> results\confirmatory\integrity.stderr.log
if errorlevel 1 (
  echo 1 > results\confirmatory\features.exitcode
  exit /b 1
)
"C:\Users\Kunal Munjal\Downloads\T1forecasting\.venv311\Scripts\python.exe" scripts\extract_selected_augmentation_features.py --selections results\confirmatory\selections.jsonl --video-root "C:\Users\Kunal Munjal\Downloads\datasets\ucf101\UCF101\UCF-101" --reward-repo "C:\Users\Kunal Munjal\Downloads\video_research\video-quality-reward-modeling" --output results\confirmatory\augmentation_features.npz 1> results\confirmatory\features.stdout.log 2> results\confirmatory\features.stderr.log
echo %errorlevel% > results\confirmatory\features.exitcode
