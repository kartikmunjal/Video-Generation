@echo off
cd /d "C:\Users\Kunal Munjal\Downloads\video_research\video-generation"
if exist results\confirmatory\selection.exitcode del results\confirmatory\selection.exitcode
"C:\Users\Kunal Munjal\Downloads\T1forecasting\.venv311\Scripts\python.exe" scripts\build_augmentation_selections.py --assignments "C:\Users\Kunal Munjal\Downloads\video_research\video-curation\results\confirmatory\raw\official_assignments.jsonl" --video-root "C:\Users\Kunal Munjal\Downloads\datasets\ucf101\UCF101\UCF-101" --reward-repo "C:\Users\Kunal Munjal\Downloads\video_research\video-quality-reward-modeling" --reward-model-dir "C:\Users\Kunal Munjal\Downloads\video_research\video-quality-reward-modeling\results\confirmatory\models" --config configs\study_v1.json --output results\confirmatory\selections.jsonl 1> results\confirmatory\selection.stdout.log 2> results\confirmatory\selection.stderr.log
echo %errorlevel% > results\confirmatory\selection.exitcode
