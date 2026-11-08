# Computer-Vision-Project
In the following 3 weeks, you will be building on your previous work with image datasets, and focus on applying the YOLO (You Only Look Once)framework to your custom dataset for your chosen task(classfication, detection, segmentation). In this tutorial we will focus on object detection only, for other tasks you will have to make few changes in notebook as well as labelled files, because each task requires specific lableling format and pre-trained models. You will fine-tune the YOLO model using the dataset you've created or annotated.Then focus on evaluating the YOLO model. You will assess its performance, discuss a potential use-case, and critically analyze its shortcomings and ethical considerations. 

 

 

**Tasks:** 
- Clone this repository
- Integrate your custom dataset into the YOLO framework (transfer learning).
- Fine-tune the YOLO model using your dataset. This involves adjusting the model's parameters to better suit your specific data.
- Conduct evaluation of your fine-tuned YOLO model. Use appropriate metrics to assess its performance (e.g., precision, recall, accuracy, F1 score, etc.).

---

# Group 6 setup: gesture-controlled photo booth

YOLOv8s fine-tuned to detect four hand gestures (palm, fist, like, dislike). The images and labels live in the shared Google Drive folder **ComputerVisionProject-Group6-Dataset**; the code reads them from there through the Google Drive desktop app, so nothing has to be downloaded by hand.

## 1. Get the code
```
git clone https://github.com/Amynguyen2424/Computer-Vision-Project-Group6.git
cd Computer-Vision-Project-Group6
git checkout Ben
```

## 2. Install Google Drive for desktop
1. Download and install it from https://www.google.com/drive/download/ and sign in with the Google account that has access to the group folder.
2. In Google Drive in your browser, open **Shared with me**, right-click **ComputerVisionProject-Group6-Dataset** and choose **Organize → Add shortcut → My Drive**. The desktop app only sees shared folders that have a shortcut in your Drive.
3. Check that the folder shows up: on Windows under the Google Drive drive (usually `G:\My Drive`), on Mac under **Google Drive** in Finder.
4. Optional, makes the first build faster: right-click the folder there and choose **Offline access → Available offline** (about 1.4 GB).

## 3. Set up Python
You need Python 3.10 or newer. In the project folder:

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac
```

Install PyTorch first:
- **NVIDIA GPU (Windows):** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- **No NVIDIA GPU, or Mac:** `pip install torch torchvision`

Then the rest, and register the notebook kernel:
```
pip install -r requirements.txt
python -m ipykernel install --user --name cv-group6 --display-name "Python (CV Group 6)"
```

## 4. Build the dataset from Drive
```
python prepare_dataset.py
```
It finds the Drive folder, checks every image/label pair (and fixes Joseph's class order), and copies everything into `dataset/` (ignored by git). It should end with **train 1280 / val 160 / test 160** and **128** team photos. The first run takes about 10-15 minutes because Drive fetches each image; later runs are much faster. Run it again whenever someone changes photos or labels on Drive.

## 5. Run the notebooks
```
jupyter lab
```
Open a notebook, check that the kernel (top right) is **Python (CV Group 6)**, then **Run → Run All Cells**.

| File | What it does |
|---|---|
| `Gesture_YOLO_Baseline.ipynb` | Trains and evaluates the baseline (about 8.5 min on an RTX 5060 Ti, much longer on a CPU) |
| `Gesture_YOLO_Tuning_imgsz.ipynb` | Ben's hyperparameter, `imgsz` 480 / 640 / 800 / 960. Uses the saved models in `models/` and only re-scores them, unless you delete a model to retrain it |
| `gesture_eval.py` | Baseline hyperparameters (`BASELINE`) and the evaluation every run uses. For your own hyperparameter, train with `{**BASELINE, "<name>": value}` and score with `evaluate()` and `error_analysis()` |
| `prepare_dataset.py` | Builds `dataset/` from the Drive folder |
| `report/Baseline_Evaluation_Report.pdf` | Detailed baseline evaluation. Rebuild it with `python report/build_baseline_report.py` (Windows, needs Microsoft Edge) |
| `models/`, `results/` | Saved weights and the evaluation output of every run |

Our own photos (numbers 369-400 of each class) are the test set, so always compare models on the **team-only test** numbers, not on validation.

## 6. Tune your hyperparameter
Each of us changes **one** hyperparameter to **three** values different from the baseline and compares against it. Everything else must stay at the baseline values, which are defined once in `gesture_eval.py`:

| Key | Baseline | Key | Baseline |
|---|---|---|---|
| `epochs` | 50 | `optimizer` | `"SGD"` |
| `patience` | 15 | `lr0` | 0.01 |
| `imgsz` | 640 (taken by Ben) | `momentum` | 0.937 |
| `batch` | 16 | `weight_decay` | 0.0005 |
| `fliplr` | 0.5 | `flipud` | 0.0 |
| `seed` | 0 | `deterministic` | True |

Any other [Ultralytics training setting](https://docs.ultralytics.com/modes/train/#train-settings) works too (for example `mosaic`, `mixup`, `scale`, `degrees`, `hsv_s`, `cos_lr`, `dropout`), at its default value for the baseline. `copy_paste` only affects segmentation models, and `erasing` and `auto_augment` only affect classification models, so they won't change our detector.

**Steps**
1. Make your own branch: `git checkout -b <your name>`.
2. In JupyterLab, create a new notebook in the project folder named `Gesture_YOLO_Tuning_<parameter>.ipynb` and pick the **Python (CV Group 6)** kernel.
3. Paste this into a cell, set `PARAM` and `VALUES`, and run it. It trains one model per value with everything else at the baseline, then scores the baseline and your models with the same code:

```python
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from gesture_eval import BASELINE, evaluate, error_analysis

PARAM = "lr0"                      # your hyperparameter: one key of BASELINE (or another Ultralytics setting)
VALUES = [0.001, 0.005, 0.02]      # three values, all different from the baseline's

# 1. Train one model per value (skipped if it is already saved; delete the .pt file to retrain)
for v in VALUES:
    name = f"{PARAM}_{v}"
    weights, out = Path("models") / f"{name}_best.pt", Path("results") / name
    if weights.exists():
        continue
    model = YOLO("yolov8s.pt")
    model.train(**{**BASELINE, PARAM: v}, project="runs", name=name, exist_ok=True)
    run = Path(model.trainer.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "weights" / "best.pt", weights)
    for f in ["results.csv", "results.png", "args.yaml"]:
        shutil.copy2(run / f, out / f)

# 2. Score the baseline and your models with the same code
runs = {BASELINE.get(PARAM, "default"): "baseline", **{v: f"{PARAM}_{v}" for v in VALUES}}
rows = []
for v, name in runs.items():
    weights = Path("models") / f"{name}_best.pt"
    imgsz = v if PARAM == "imgsz" else BASELINE["imgsz"]
    overall, per_class = evaluate(weights, imgsz, Path("results") / name)
    ea = error_analysis(weights, imgsz, Path("results") / name)
    team = overall.iloc[2]         # "test, team only": our own photos
    rows.append({PARAM: v, "run": name, "team mAP50": team["mAP50"], "team mAP50-95": team["mAP50-95"],
                 "team F1": team["F1"], "fist mAP50": per_class.loc[("test (team only)", "fist"), "mAP50"],
                 "F1 @ conf 0.25": ea["summary"]["F1"]})
summary = pd.DataFrame(rows).set_index(PARAM)
summary.round(4).to_csv(Path("results") / f"tuning_{PARAM}_summary.csv")
summary.round(3)
```

4. Below it, add your plots and a short discussion: which value is best, by how much, and why. Each run's folder in `results/` has the training curves, confusion matrices and error analysis files to draw from. `Gesture_YOLO_Tuning_imgsz.ipynb` is a full worked example.
5. Commit your notebook, `models/<parameter>_*_best.pt`, `results/<parameter>_*/` and `results/tuning_<parameter>_summary.csv` to your branch. `dataset/` and `runs/` are ignored by git on purpose.

**Things to keep in mind**
- **Don't edit `BASELINE` in `gesture_eval.py`.** Everyone's comparison depends on it.
- **Training time:** each run takes about 5-25 minutes on an RTX 5060 Ti, depending on the setting, and much longer on a laptop CPU. Three runs on a CPU can take hours. If your computer has no NVIDIA GPU, ask Ben to run your training.
- If training stops with a memory error, lower the batch size for your runs only (`{**BASELINE, PARAM: v, "batch": 8}`) and say so in your discussion. If the data loader hangs on Windows, add `"workers": 0`.
- **Noise:** on our 128 photos, a single model's F1 is uncertain by about ±0.06 (see the baseline report). Differences smaller than that are not real improvements. 
