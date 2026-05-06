# STSMIRS Fight Detection AI - Folder Guide

Use this file first if you are checking the project.

## Active Folders

| Folder | Purpose |
| --- | --- |
| `src/` | Main AI pipeline code for live camera, tracking, action detection, and backend alerts. |
| `models/` | Final trained model files used by the project. |
| `datasets_final/` | Clean, reviewer-friendly datasets for current action training. |
| `project_videos/` | Original action videos used by the live action/LSTM pipeline. |
| `data/features/` | Extracted LSTM feature sequences from videos. |
| `data/backend_bridge/` | Alert/backend logs generated when AI events are sent to the backend bridge. |
| `tools/` | Dataset preparation and utility scripts. |

## Current Final Dataset

Use this for the clean fighting-vs-normal classifier:

```text
datasets_final/fighting_vs_normal/
  train/
    fighting/
    normal/
  val/
    fighting/
    normal/
  test/
    fighting/
    normal/
```

The reference multi-action dataset is also kept here:

```text
datasets_final/all_actions_reference/
```

It contains `fall`, `fighting`, `loitering`, `running`, and `walking`, but the current clean live model focus is `fighting_vs_normal`.

## Current Trained Models

| Model | Purpose |
| --- | --- |
| `models/lstm_classifier.pth` | Existing live action model used by `src/main.py`. |
| `models/action_fighting_vs_normal_best.pt` | Clean YOLO image classifier used as live Fighting vs Normal confirmation. |
| `yolov8n.pt` | YOLO person detector/tracker base model. |

The live UI displays `Fighting` or `Normal`. It does not show `Walking`.

## Archived / Not Active

Old or confusing dataset experiments were moved to:

```text
_archive_unused_datasets_20260430/
```

They were archived, not permanently deleted, so they can be recovered later if needed.

## Run Live AI Pipeline

```powershell
cd "C:\Users\DELL\Desktop\patent\fight\fight-detection-ai"
$env:YOLO_CONFIG_DIR=(Resolve-Path ".ultralytics").Path
.\.venv\Scripts\python.exe src\main.py --source 0
```

## Train Fighting vs Normal Again

```powershell
cd "C:\Users\DELL\Desktop\patent\fight\fight-detection-ai"
$env:YOLO_CONFIG_DIR=(Resolve-Path ".ultralytics").Path
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model=YOLO(r'_archive_unused_datasets_20260430\yolov8n-cls.pt'); model.train(data=r'datasets_final\fighting_vs_normal', epochs=8, imgsz=224, batch=32, device='cpu', workers=0, project='models', name='action_fighting_vs_normal', exist_ok=True)"
```
