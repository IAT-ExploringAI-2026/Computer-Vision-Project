"""Shared training settings and evaluation for the gesture detector (Group 6).

Used by the baseline analysis and by every hyperparameter-tuning run, so all models
are scored the same way:

    evaluate(weights, imgsz, out_dir)        val / test (all) / test (team only) metrics + plots
    error_analysis(weights, imgsz, out_dir)  team photos at conf 0.25: per person, box size,
                                             confusion matrix, threshold sweep, failure cases

Re-evaluate saved models (e.g. after the dataset is rebuilt with prepare_dataset.py):
    python gesture_eval.py models/*.pt
    python gesture_eval.py models/baseline_best.pt --out-root <dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).parent
CLASSES = ["palm", "fist", "like", "dislike"]

# Baseline hyperparameters (Gesture_YOLO_Baseline.ipynb). Tuning runs change one key.
BASELINE = dict(
    data="data.yaml",
    epochs=50,
    patience=15,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    fliplr=0.5,
    flipud=0.0,
    seed=0,
    deterministic=True,
    workers=4,          # set to 0 if the data loader hangs on Windows
    cache="ram",        # decode the large photos once instead of every epoch
)

# (file prefix, data yaml, split, label, manifest filter)
EVAL_SETS = [
    ("val", "data.yaml", "val", "validation", lambda m: m["split"] == "val"),
    ("test", "data.yaml", "test", "test, all", lambda m: m["split"] == "test"),
    ("test_team", "data_team_test.yaml", "test", "test, team only", lambda m: m["source"] == "team"),
]
PER_CLASS_NAMES = {"val": "validation", "test": "test (all)", "test_team": "test (team only)"}

# Box size buckets, as a fraction of the image area.
SIZE_BINS = [0, 0.01, 0.05, 1.0]
SIZE_LABELS = ["small (<1%)", "medium (1-5%)", "large (>5%)"]

GT_COLOR, PRED_COLOR = (0, 200, 0), (0, 0, 255)   # BGR


def manifest():
    return pd.read_csv(ROOT / "dataset" / "split_manifest.csv")


def train_imgsz(weights):
    """The imgsz a checkpoint was trained at (models are evaluated at their own size)."""
    return int(YOLO(weights).ckpt["train_args"]["imgsz"])


def _write_metrics(out_dir, updates):
    """Merge updates into out_dir/metrics.json, keeping keys written by the training notebook."""
    path = out_dir / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics.update(updates)
    path.write_text(json.dumps(metrics, indent=2))
    return metrics


def evaluate(weights, imgsz, out_dir, batch=16):
    """Ultralytics val on validation, full test and team-only test.

    Writes per_class.csv, the val/test plots and the "overall" block of metrics.json.
    Returns (overall, per_class) DataFrames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    m_all = manifest()
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0

    tables, overall, speed = {}, {}, {}
    for prefix, data, split, label, sel in EVAL_SETS:
        m = model.val(data=str(ROOT / data), split=split, imgsz=imgsz, batch=batch, plots=True,
                      project=str(ROOT / "runs"), name=f"{out_dir.name}_{prefix}", exist_ok=True,
                      verbose=False)
        b = m.box
        rows = [{"class": CLASSES[c], "precision": b.p[i], "recall": b.r[i], "F1": b.f1[i],
                 "mAP50": b.ap50[i], "mAP50-95": b.ap[i]} for i, c in enumerate(b.ap_class_index)]
        rows.append({"class": "all", "precision": b.mp, "recall": b.mr, "F1": f1(b.mp, b.mr),
                     "mAP50": b.map50, "mAP50-95": b.map})
        tables[prefix] = pd.DataFrame(rows).set_index("class")
        overall[f"{label} ({int(sel(m_all).sum())})"] = tables[prefix].loc["all"]
        speed[prefix] = {k: round(v, 2) for k, v in m.speed.items()}
        for f in Path(m.save_dir).glob("*.png"):
            shutil.copy2(f, out_dir / f"{prefix}_{f.name}")

    overall = pd.DataFrame(overall).T
    per_class = pd.concat({PER_CLASS_NAMES[k]: t for k, t in tables.items()})
    per_class.round(4).to_csv(out_dir / "per_class.csv")
    _write_metrics(out_dir, {
        "weights": Path(weights).name,
        "eval_imgsz": imgsz,
        "split_sizes": m_all["split"].value_counts().to_dict(),
        "overall": overall.round(4).to_dict(orient="index"),
        "val_speed_ms_per_image": speed["val"],
    })
    return overall, per_class


# --------------------------------------------------------------------------- error analysis

def _iou(a, b):
    """IoU between box arrays a (N,4) and b (M,4) in xyxy -> (N,M)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def _ground_truth(stem, h, w):
    rows = np.loadtxt(ROOT / "dataset" / "labels" / "test" / f"{stem}.txt", ndmin=2)
    cls, (x, y, bw, bh) = rows[:, 0].astype(int), rows[:, 1:].T
    xyxy = np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], 1)
    return cls, xyxy, bw * bh


def _match(gt_xyxy, pred_xyxy, pred_conf, iou_thr):
    """Greedy class-agnostic matching, highest confidence first (like Ultralytics' confusion
    matrix). Returns {pred index: (gt index, iou)} for matched predictions."""
    if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
        return {}
    ious = _iou(pred_xyxy, gt_xyxy)
    taken, matches = set(), {}
    for p in np.argsort(-pred_conf):
        free = [(ious[p, g], g) for g in range(len(gt_xyxy)) if g not in taken and ious[p, g] >= iou_thr]
        if free:
            best_iou, g = max(free)
            taken.add(g)
            matches[p] = (g, best_iou)
    return matches


def _score(images, conf, iou_thr):
    """Per-hand and per-prediction outcomes for all images at one confidence threshold."""
    hands, preds = [], []
    for im in images:
        keep = im["pred_conf"] >= conf
        p_cls, p_xyxy, p_conf = im["pred_cls"][keep], im["pred_xyxy"][keep], im["pred_conf"][keep]
        matches = _match(im["gt_xyxy"], p_xyxy, p_conf, iou_thr)
        by_gt = {g: (p, iou) for p, (g, iou) in matches.items()}
        for g, c in enumerate(im["gt_cls"]):
            p, iou = by_gt.get(g, (None, np.nan))
            hands.append({"file": im["file"], "person": im["person"], "class": CLASSES[c],
                          "area_frac": im["gt_area"][g], "gt_index": g,
                          "pred_class": CLASSES[p_cls[p]] if p is not None else "missed",
                          "conf": p_conf[p] if p is not None else np.nan, "iou": iou})
        for p in range(len(p_cls)):
            g = matches.get(p, (None,))[0]
            preds.append({"file": im["file"], "person": im["person"], "pred_class": CLASSES[p_cls[p]],
                          "conf": p_conf[p],
                          "true_class": CLASSES[im["gt_cls"][g]] if g is not None else "background"})
    hands = pd.DataFrame(hands)
    preds = pd.DataFrame(preds, columns=["file", "person", "pred_class", "conf", "true_class"])
    hands["correct"] = hands["class"] == hands["pred_class"]
    preds["correct"] = preds["pred_class"] == preds["true_class"]
    return hands, preds


def _prf(hands, preds):
    tp = int(hands["correct"].sum())
    p = tp / len(preds) if len(preds) else 0.0
    r = tp / len(hands) if len(hands) else 0.0
    return {"hands": len(hands), "predictions": len(preds), "correct": tp,
            "precision": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def _confusion_plot(hands, preds, path, conf):
    labels = CLASSES + ["background"]
    cm = pd.DataFrame(0, index=labels, columns=labels)   # rows true, columns predicted
    for t, p in zip(hands["class"], hands["pred_class"].replace("missed", "background")):
        cm.loc[t, p] += 1
    for p in preds.loc[preds["true_class"] == "background", "pred_class"]:
        cm.loc["background", p] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm.values, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if v > cm.values.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Team photos, conf ≥ {conf}, IoU ≥ 0.5 (counts)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return cm


def _failure_grid(images, hands, preds_by_file, path, n=8):
    """Crops around misclassified and missed (smallest first) hands: green = label, red = prediction."""
    fails = hands[~hands["correct"]]
    wrong = fails[fails["pred_class"] != "missed"].sort_values("file").head(n // 2)
    missed = fails[fails["pred_class"] == "missed"].sort_values("area_frac").head(n - len(wrong))
    fails = pd.concat([missed, wrong])
    if fails.empty:
        return
    by_file = {im["file"]: im for im in images}
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(fails) / cols)), cols, figsize=(16, 4.4 * np.ceil(len(fails) / cols)))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, f) in zip(np.ravel(axes), fails.iterrows()):
        im = by_file[f["file"]]
        img = cv2.imread(str(im["path"]))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = im["gt_xyxy"][f["gt_index"]]
        side = max(x2 - x1, y2 - y1) * 3
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx1, cy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        cx2, cy2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        lw = max(2, int(side / 150))
        for c, box in zip(im["gt_cls"], im["gt_xyxy"]):
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), GT_COLOR, lw)
        for c, cf, box in preds_by_file[f["file"]]:
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), PRED_COLOR, lw)
            cv2.putText(img, f"{CLASSES[c]} {cf:.2f}", (int(box[0]), int(box[1]) - lw * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, lw / 3, PRED_COLOR, lw)
        ax.imshow(img[cy1:cy2, cx1:cx2, ::-1])
        what = "missed" if f["pred_class"] == "missed" else f"predicted {f['pred_class']}"
        ax.set_title(f"{Path(f['file']).stem} ({f['person']}): {f['class']} {what}\n"
                     f"hand = {f['area_frac'] * 100:.2f}% of image", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def error_analysis(weights, imgsz, out_dir, conf=0.25, iou_thr=0.5):
    """Detection-level analysis of the team photos at a deployment threshold.

    Writes error_analysis.json, team_by_person.csv, team_by_size.csv, team_by_class.csv,
    conf_sweep.csv, team_hands.csv, team_preds.csv, confusion_conf025.png and failure_cases.jpg.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    team = manifest().query("source == 'team'").sort_values("file")

    images, speeds = [], []
    for _, row in team.iterrows():
        path = ROOT / "dataset" / "images" / "test" / row["file"]
        r = model.predict(str(path), imgsz=imgsz, conf=0.01, verbose=False)[0]
        speeds.append(r.speed)
        h, w = r.orig_shape
        gt_cls, gt_xyxy, gt_area = _ground_truth(path.stem, h, w)
        images.append({"file": row["file"], "person": row["person"], "path": path,
                       "gt_cls": gt_cls, "gt_xyxy": gt_xyxy, "gt_area": gt_area,
                       "pred_cls": r.boxes.cls.cpu().numpy().astype(int),
                       "pred_xyxy": r.boxes.xyxy.cpu().numpy(),
                       "pred_conf": r.boxes.conf.cpu().numpy()})
    speed = pd.DataFrame(speeds[1:] or speeds).mean().round(2).to_dict()   # skip warm-up image

    hands, preds = _score(images, conf, iou_thr)
    hands["size"] = pd.cut(hands["area_frac"], SIZE_BINS, labels=SIZE_LABELS)
    hands.to_csv(out_dir / "team_hands.csv", index=False)
    preds.to_csv(out_dir / "team_preds.csv", index=False)

    by_person = pd.DataFrame({p: _prf(hands[hands["person"] == p], preds[preds["person"] == p])
                              for p in sorted(team["person"].unique())}).T
    mean_iou = hands[hands["correct"]].groupby("person")["iou"].mean()
    by_person["mean_IoU_correct"] = mean_iou
    by_person.round(3).to_csv(out_dir / "team_by_person.csv")

    by_class = pd.DataFrame({c: {"hands": int((hands["class"] == c).sum()),
                                 "recall": hands.loc[hands["class"] == c, "correct"].mean(),
                                 "precision": preds.loc[preds["pred_class"] == c, "correct"].mean(),
                                 "missed": int(((hands["class"] == c) & (hands["pred_class"] == "missed")).sum()),
                                 "wrong_class": int(((hands["class"] == c) & ~hands["correct"]
                                                     & (hands["pred_class"] != "missed")).sum())}
                             for c in CLASSES}).T
    by_class.round(3).to_csv(out_dir / "team_by_class.csv")

    by_size = hands.groupby("size", observed=False).agg(hands=("correct", "size"), recall=("correct", "mean"))
    by_size["missed"] = hands.groupby("size", observed=False)["pred_class"].apply(lambda s: int((s == "missed").sum()))
    by_size.round(3).to_csv(out_dir / "team_by_size.csv")

    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        h_t, p_t = _score(images, t, iou_thr)
        sweep.append({"conf": t, **_prf(h_t, p_t)})
    sweep = pd.DataFrame(sweep)
    sweep.round(4).to_csv(out_dir / "conf_sweep.csv", index=False)
    best = sweep.loc[sweep["F1"].idxmax()]

    cm = _confusion_plot(hands, preds, out_dir / "confusion_conf025.png", conf)
    preds_by_file = {im["file"]: [(c, cf, b) for c, cf, b in zip(im["pred_cls"], im["pred_conf"], im["pred_xyxy"])
                                  if cf >= conf] for im in images}
    _failure_grid(images, hands, preds_by_file, out_dir / "failure_cases.jpg")

    summary = {
        "conf": conf, "iou": iou_thr, "images": len(images),
        **_prf(hands, preds),
        "missed_hands": int((hands["pred_class"] == "missed").sum()),
        "wrong_class": int(((hands["pred_class"] != "missed") & ~hands["correct"]).sum()),
        "background_false_positives": int((preds["true_class"] == "background").sum()),
        "mean_IoU_correct": round(float(hands.loc[hands["correct"], "iou"].mean()), 4),
        "best_F1_conf": {"conf": float(best["conf"]), "precision": round(best["precision"], 4),
                         "recall": round(best["recall"], 4), "F1": round(best["F1"], 4)},
        "predict_speed_ms_per_image": speed,
    }
    summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}
    (out_dir / "error_analysis.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "by_person": by_person, "by_class": by_class, "by_size": by_size,
            "sweep": sweep, "confusion": cm, "hands": hands, "preds": preds}


def main():
    plt.switch_backend("Agg")   # figures are only saved when run as a script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+", type=Path, help="checkpoints, e.g. models/*.pt")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results",
                    help="results go in <out-root>/<weights name without _best>")
    args = ap.parse_args()
    for w in args.weights:
        imgsz = train_imgsz(w)
        out = args.out_root / w.stem.removesuffix("_best")
        print(f"\n=== {w.name} (imgsz {imgsz}) -> {out}")
        overall, _ = evaluate(w, imgsz, out)
        print(overall.round(3).to_string())
        ea = error_analysis(w, imgsz, out)
        print(json.dumps(ea["summary"], indent=2))


if __name__ == "__main__":
    main()
