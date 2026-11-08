"""Build report/Baseline_Evaluation_Report.pdf from results/baseline/ (Group 6 gesture detector).

Every number in the report is read from the saved results, so after re-evaluating the
baseline (python gesture_eval.py models/baseline_best.pt) just run this again:

    python report/build_baseline_report.py

Figures and the HTML source are written to report/build/ and printed to PDF with
headless Microsoft Edge (or Chrome).
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "baseline"
BUILD = Path(__file__).resolve().parent / "build"
PDF = Path(__file__).resolve().parent / "Baseline_Evaluation_Report.pdf"
CLASSES = ["palm", "fist", "like", "dislike"]

# Label mistakes found by checking every team photo against the models' predictions (2026-11-05).
# The group decided to keep them as they are and report them as human error.
LOOSE_BOXES = ["fist_380.png", "fist_381.png", "fist_384.png"]          # Ben: box much larger than the fist
UNLABELLED = {"fist_376.jpg": "2 raised fists", "fist_382.png": "2 raised fists in the crowd",
              "palm_388.jpg": "palms of the people behind the main subject"}
AMBIGUOUS = {"palm_376.jpg": "hand held across the face, back of the hand to the camera"}

# Categorical slots 1-4 (validated default palette) and recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- data

def load():
    d = {"metrics": json.loads((RES / "metrics.json").read_text()),
         "ea": json.loads((RES / "error_analysis.json").read_text()),
         "per_class": pd.read_csv(RES / "per_class.csv", index_col=[0, 1]),
         "person": pd.read_csv(RES / "team_by_person.csv", index_col=0),
         "cls": pd.read_csv(RES / "team_by_class.csv", index_col=0),
         "size": pd.read_csv(RES / "team_by_size.csv", index_col=0),
         "sweep": pd.read_csv(RES / "conf_sweep.csv"),
         "hands": pd.read_csv(RES / "team_hands.csv"),
         "preds": pd.read_csv(RES / "team_preds.csv"),
         "history": pd.read_csv(RES / "results.csv"),
         "manifest": pd.read_csv(ROOT / "dataset" / "split_manifest.csv")}
    d["history"].columns = d["history"].columns.str.strip()
    return d


def bootstrap(hands, preds, n=2000, seed=0):
    """95% CI for precision / recall / F1 at conf 0.25, resampling team photos."""
    files = hands["file"].unique()
    h = hands.groupby("file")["correct"].agg(["sum", "size"])
    p = preds.groupby("file")["correct"].agg(["sum", "size"]).reindex(files, fill_value=0)
    h = h.reindex(files)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(files), len(files))
        r = h["sum"].values[idx].sum() / h["size"].values[idx].sum()
        pr = p["sum"].values[idx].sum() / max(p["size"].values[idx].sum(), 1)
        out.append((pr, r, 2 * pr * r / (pr + r) if pr + r else 0))
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    return {k: (lo[i], hi[i]) for i, k in enumerate(["precision", "recall", "F1"])}


def duplicate_pairs(manifest):
    """Team test photos that are the same picture (16x16 average hash, <= 12 bits apart)."""
    team = manifest[manifest["source"] == "team"].sort_values("file")
    hashes = []
    for f in team["file"]:
        g = cv2.resize(cv2.imread(str(ROOT / "dataset" / "images" / "test" / f), 0), (16, 16),
                       interpolation=cv2.INTER_AREA)
        hashes.append((g > g.mean()).ravel())
    H = np.array(hashes)
    rows = team.reset_index(drop=True)
    pairs = []
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            if (H[i] != H[j]).sum() <= 12:
                pairs.append((rows.loc[i, "file"], rows.loc[i, "person"], rows.loc[j, "file"], rows.loc[j, "person"]))
    return pairs


# --------------------------------------------------------------------------- figures

def fig_training(h, best_epoch):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for col, lbl, c, ls in [("train/box_loss", "train box loss", SERIES[0], "-"),
                            ("val/box_loss", "val box loss", SERIES[0], "--"),
                            ("train/cls_loss", "train class loss", SERIES[1], "-"),
                            ("val/cls_loss", "val class loss", SERIES[1], "--")]:
        a.plot(h["epoch"], h[col], color=c, ls=ls, label=lbl)
    a.set_ylim(0, min(2.0, h[["train/cls_loss", "val/cls_loss"]].max().max() * 1.05))
    a.set_title("Loss per epoch"); a.set_xlabel("epoch"); a.legend(ncol=2, fontsize=8.5)
    for col, lbl, c in [("metrics/mAP50(B)", "val mAP50", SERIES[0]), ("metrics/mAP50-95(B)", "val mAP50-95", SERIES[2])]:
        b.plot(h["epoch"], h[col], color=c, label=lbl)
    b.axvline(best_epoch, color=INK_2, ls=":", lw=1.2)
    b.text(best_epoch - 0.5, 0.95, f"best epoch {best_epoch}", color=INK_2, fontsize=8.5, ha="right")
    b.set_ylim(0.5, 1.0); b.set_title("Validation accuracy per epoch"); b.set_xlabel("epoch"); b.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(BUILD / "fig_training.png"); plt.close(fig)


def fig_per_class(pc):
    sets = ["validation", "test (all)", "test (team only)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    x, w = np.arange(len(CLASSES)), 0.26
    for ax, metric in zip(axes, ["mAP50", "mAP50-95"]):
        for i, s in enumerate(sets):
            vals = [pc.loc[(s, c), metric] for c in CLASSES]
            ax.bar(x + (i - 1) * w, vals, w - 0.02, color=SERIES[i], label=s)
        team = [pc.loc[("test (team only)", c), metric] for c in CLASSES]
        for xi, v in zip(x, team):
            ax.text(xi + w, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, CLASSES); ax.set_ylim(0, 1.08); ax.set_title(metric); ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); fig.savefig(BUILD / "fig_per_class.png"); plt.close(fig)


def fig_sweep(sweep, deploy, best):
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for col, c in [("precision", SERIES[0]), ("recall", SERIES[1]), ("F1", SERIES[2])]:
        ax.plot(sweep["conf"], sweep[col], color=c, label=col, marker="o", ms=3)
    for t, lbl in [(deploy, f"{deploy} (default)"), (best, f"{best} (best F1)")]:
        ax.axvline(t, color=INK_2, ls=":", lw=1.2)
        ax.text(t, 0.08, " " + lbl, color=INK_2, fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("confidence threshold"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
    ax.set_title("Team photos: threshold sweep (IoU ≥ 0.5)"); ax.legend(loc="center left")
    fig.tight_layout(); fig.savefig(BUILD / "fig_sweep.png"); plt.close(fig)


def fig_breakdown(size, person):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1, 1.4]})
    a.bar(range(len(size)), size["recall"], 0.6, color=SERIES[0])
    for i, (r, n) in enumerate(zip(size["recall"], size["hands"])):
        a.text(i, r + 0.02, f"{r:.2f}  (n={n})", ha="center", fontsize=8.5, color=INK)
    a.set_xticks(range(len(size)), [s.replace(" (", "\n(") for s in size.index])
    a.set_ylim(0, 1.05); a.set_title("Recall by hand size (share of image)"); a.grid(axis="x", visible=False)
    x, w = np.arange(len(person)), 0.26
    for i, (col, lbl) in enumerate([("precision", "precision"), ("recall", "recall"), ("mean_IoU_correct", "mean box IoU")]):
        b.bar(x + (i - 1) * w, person[col], w - 0.02, color=SERIES[i], label=lbl)
    b.set_xticks(x, [f"{p}\n({int(n)} hands)" for p, n in zip(person.index, person["hands"])])
    b.set_ylim(0, 1.18); b.set_yticks(np.arange(0, 1.01, 0.2)); b.set_title("By team member (conf 0.25)")
    b.legend(ncol=3, fontsize=8.5, loc="upper center")
    b.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(BUILD / "fig_breakdown.png"); plt.close(fig)


# --------------------------------------------------------------------------- report

def table(df, fmt="{:.3f}", index_name=""):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for idx, row in zip(df.index, df.itertuples(index=False)):   # itertuples keeps int columns int
        cells = "".join(f"<td>{fmt.format(v) if isinstance(v, (float, np.floating)) else v}</td>" for v in row)
        body += f"<tr><th>{idx}</th>{cells}</tr>"
    return f"<table><thead><tr><th>{index_name}</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(d):
    m, ea, pc, h = d["metrics"], d["ea"], d["per_class"], d["history"]
    hp = m["hyperparameters"]
    man = d["manifest"]
    best_epoch = int(h.loc[h["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    best_row = h.loc[h["epoch"] == best_epoch].iloc[0]
    ov = pd.DataFrame(m["overall"]).T
    val, team = ov.iloc[0], ov.iloc[2]
    gap = val["mAP50"] - team["mAP50"]
    cls, size, person = d["cls"], d["size"], d["person"]
    deploy, best = ea["conf"], ea["best_F1_conf"]
    ci = bootstrap(d["hands"], d["preds"])
    dups = duplicate_pairs(man)
    fp = d["preds"][d["preds"]["true_class"] == "background"]["pred_class"].value_counts()
    sp = ea["predict_speed_ms_per_image"]
    total_ms = sum(sp.values())
    n_team = int((man["source"] == "team").sum())
    worst_cls = cls["recall"].idxmin()
    small = size.iloc[0]
    area = d["hands"]["area_frac"]
    train_area = np.concatenate([np.loadtxt(f, ndmin=2)[:, 3] * np.loadtxt(f, ndmin=2)[:, 4]
                                 for f in (ROOT / "dataset" / "labels" / "train").glob("*.txt")])
    train_small = int((train_area < 0.01).sum())
    small_by_person = d["hands"][d["hands"]["area_frac"] < 0.01].groupby("person").size()
    small_owner = small_by_person.idxmax()
    iou_low = person["mean_IoU_correct"].idxmin()
    iou_low_note = (f" and precision ({person.loc[iou_low, 'precision']:.2f}), which points to a different box style rather than a harder set of photos"
                    if person["precision"].idxmin() == iou_low else
                    f" even though its precision is {person.loc[iou_low, 'precision']:.2f}: the model finds those hands, but the boxes "
                    "are drawn differently, a labelling-style difference rather than harder photos")

    counts = pd.crosstab(man["split"] + " · " + man["source"].map({"hagrid": "HaGRID", "team": "our photos"}),
                         man["class"])[CLASSES]
    counts["total"] = counts.sum(axis=1)
    counts = counts.reindex([i for i in ["train · HaGRID", "val · HaGRID", "test · HaGRID", "test · our photos"] if i in counts.index])

    ov_tbl = ov.copy(); ov_tbl.index = [i.replace("test, team only", "test, our photos only") for i in ov_tbl.index]
    team_cls = pc.loc["test (team only)"].loc[CLASSES, ["precision", "recall", "mAP50", "mAP50-95"]].copy()
    team_cls["hands"] = cls["hands"].astype(int)
    team_cls["missed @0.25"] = cls["missed"].astype(int)
    team_cls["wrong class @0.25"] = cls["wrong_class"].astype(int)
    pers_tbl = person[["hands", "predictions", "correct", "precision", "recall", "mean_IoU_correct"]].copy()
    for c in ["hands", "predictions", "correct"]:
        pers_tbl[c] = pers_tbl[c].astype(int)
    pers_tbl = pers_tbl.rename(columns={"mean_IoU_correct": "mean IoU"})
    size_tbl = size.copy(); size_tbl["hands"] = size_tbl["hands"].astype(int); size_tbl["missed"] = size_tbl["missed"].astype(int)

    dup_txt = "; ".join(f"<code>{a}</code> ({pa}) = <code>{b}</code> ({pb})" for a, pa, b, pb in dups)
    data_issues = []
    if dups:
        data_issues.append(("Duplicate photos.", (f"{len(dups)} pairs of our test photos are" if len(dups) > 1 else "One pair of our test photos is")
                            + f" the same picture contributed by two members ({dup_txt}). "
                            f"Those hands count twice, so the test set has {n_team - len(dups)} distinct photos, not {n_team}."))
    hands, preds = d["hands"], d["preds"]
    loose_missed = int((hands["file"].isin(LOOSE_BOXES) & ~hands["correct"]).sum())
    label_bg = int((preds["file"].isin(LOOSE_BOXES + list(UNLABELLED)) & (preds["true_class"] == "background")).sum())
    code = lambda f: f"<code>{Path(f).stem}</code>"
    data_issues.append(("Labelling errors (human error).",
        f"Checking every photo against the model's predictions found label mistakes that were left in the test set: boxes drawn "
        f"much larger than the fist in {', '.join(code(f) for f in LOOSE_BOXES)}; gesturing hands that were never labelled in "
        + "; ".join(f"{code(f)} ({v})" for f, v in UNLABELLED.items())
        + "; and an unclear gesture in " + "; ".join(f"{code(f)} ({v})" for f, v in AMBIGUOUS.items()) + ". "
        f"A tight, correct box on a loose label overlaps it by less than 0.5 IoU and counts as a miss plus a false positive, and a detection "
        f"of an unlabelled hand counts as a false positive. At conf {deploy} these cause {loose_missed} of the {ea['missed_hands']} missed hands "
        f"and {label_bg} of the {ea['background_false_positives']} background predictions, so the scores on our photos (fist most of all) are slightly pessimistic."))

    tuning = ROOT / "results" / "tuning_imgsz_summary.csv"
    if tuning.exists():
        t = pd.read_csv(tuning, index_col="run").sort_values("imgsz")
        n_small = int(small["hands"])
        small_found = (t["small-hand recall"] * n_small).round().astype(int)
        small_change = int((small_found - small_found["baseline"]).abs().max())
        imgsz_note = ("We tested other image sizes (<code>Gesture_YOLO_Tuning_imgsz.ipynb</code>): team mAP50 was "
                      + ", ".join(f"{r['team mAP50']:.3f} at {int(r['imgsz'])}" for _, r in t.iterrows())
                      + ", and small hands found were " + ", ".join(f"{n}" for n in small_found) + f" of {n_small}, "
                      f"so no size changed small-hand recall by more than {small_change} hand{'s' if small_change != 1 else ''}. "
                      "The training set is not short of small hands; it is short of our kind "
                      "of scene (crowds, outdoors, several people). More varied training photos and context-changing augmentation "
                      "(mosaic, mixup, scale) are the more promising levers.")
    else:
        imgsz_note = "Larger image sizes (<code>imgsz</code> 800-960) or scale augmentation are the obvious things to try."
    css = (Path(__file__).parent / "report.css").read_text()
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Baseline Model Evaluation</title>
<style>{css}</style></head><body>

<header>
  <p class="kicker">IAT 360 Computer Vision Project · Group 6 · Gesture-controlled photo booth</p>
  <h1>Baseline model evaluation</h1>
  <p class="byline">YOLOv8s fine-tuned to detect four hand gestures · Ben Nichiporik, Daffodil Ho, Joseph Lee, Thu Nguyen · {date.today():%B %Y}</p>
</header>

<section class="tiles">
  <div class="tile"><span class="num">{val['mAP50']:.3f}</span><span class="lbl">mAP50, validation<br>(HaGRID, {int(ov.index[0].split('(')[1][:-1])} images)</span></div>
  <div class="tile accent"><span class="num">{team['mAP50']:.3f}</span><span class="lbl">mAP50, our own photos<br>({n_team} images, headline)</span></div>
  <div class="tile"><span class="num">{cls.loc[worst_cls, 'recall']:.2f}</span><span class="lbl">{worst_cls} recall on our photos<br>(weakest class, conf 0.25)</span></div>
  <div class="tile"><span class="num">{total_ms:.1f} ms</span><span class="lbl">per image on an RTX 5060 Ti<br>(≈{1000 / total_ms:.0f} frames/s)</span></div>
</section>

<p class="lead">The baseline is close to perfect on held-out HaGRID images but loses {gap:.2f} mAP50 on the photos our group
collected. Its weakest spots are <b>small hands in wide shots</b> (only {small['recall']:.0%} found) and <b>fists</b>
({cls.loc['fist', 'recall']:.0%} found), which the model either misses or confuses with a thumbs-down. The model is fast enough for a live photo booth, and a higher confidence threshold
({best['conf']}) trades a little recall for noticeably fewer false triggers.</p>

<h2>1. What was evaluated</h2>
<div class="cols">
<div>
<p><b>Model.</b> YOLOv8s pretrained on COCO, fine-tuned on our data (transfer learning; all layers trained).
{m.get('epochs_run', len(h))} epochs were run; the checkpoint with the best validation fitness (epoch {best_epoch}) is the baseline,
<code>models/baseline_best.pt</code>.</p>
<p><b>Protocol.</b> Every model in the project is scored by the same code (<code>gesture_eval.py</code>) on three sets:
HaGRID validation (used to pick the checkpoint), the full test split, and <b>our own photos only</b>, the
honest estimate of photo-booth performance. Sections 4-6 then score our photos at a deployment threshold
(confidence ≥ {deploy}, a prediction counts as correct if its box overlaps a labelled hand with IoU ≥ 0.5 and the class matches).</p>
</div>
<div>
<table class="kv"><tbody>
<tr><th>image size</th><td>{hp['imgsz']}</td><th>optimizer</th><td>{hp['optimizer']}</td></tr>
<tr><th>epochs</th><td>{hp['epochs']} (patience {hp['patience']})</td><th>lr0</th><td>{hp['lr0']}</td></tr>
<tr><th>batch</th><td>{hp['batch']}</td><th>momentum</th><td>{hp['momentum']}</td></tr>
<tr><th>seed</th><td>{hp['seed']} (deterministic)</td><th>weight decay</th><td>{hp['weight_decay']}</td></tr>
<tr><th>flips</th><td>LR {hp['fliplr']}, UD {hp['flipud']}</td><th>other aug.</th><td>Ultralytics defaults (mosaic, HSV, scale)</td></tr>
</tbody></table>
</div>
</div>
{table(counts, fmt="{:.0f}", index_name="split · source")}
<p class="caption">Images per split and class. HaGRID images 001-368 per class; our photos are 369-400 (8 per member per class).</p>

<h2>2. Training behaviour</h2>
<img src="fig_training.png" class="full">
<p>Transfer learning works quickly: validation mAP50 is already {h['metrics/mAP50(B)'].iloc[0]:.2f} after one epoch, because
the COCO-pretrained backbone already provides general shape and edge features. Accuracy then climbs slowly. The best epoch ({best_epoch}) reaches mAP50
{best_row['metrics/mAP50(B)']:.3f} and mAP50-95 {best_row['metrics/mAP50-95(B)']:.3f}. Validation losses keep falling
alongside the training losses through epoch {len(h)}, so there is no sign of overfitting <i>to HaGRID</i>. Early stopping never triggered.
The overfitting that does matter is to HaGRID's style of photo, which only the test on our own photos reveals.</p>

<h2>3. Overall results</h2>
{table(ov_tbl, index_name="evaluation set")}
<p>The validation scores are inflated: HaGRID validation images come from the same collection as the training
images (one person standing and facing a phone camera, mostly indoors, one hand per photo). On our own photos, mAP50 drops by {gap:.2f} and mAP50-95 by
{val['mAP50-95'] - team['mAP50-95']:.2f}. The larger drop in mAP50-95 means the model usually <i>finds</i> the hand but its box doesn't
line up tightly with ours. Correct detections have a mean IoU of only {ea['mean_IoU_correct']:.2f} with our labels. Part of this is
annotation style rather than model error: each member drew boxes in MakeSense slightly differently (some include the wrist).
At conf {deploy} on our photos, the 95% bootstrap intervals (resampling photos) are precision
{ci['precision'][0]:.2f}-{ci['precision'][1]:.2f}, recall {ci['recall'][0]:.2f}-{ci['recall'][1]:.2f} and F1
{ci['F1'][0]:.2f}-{ci['F1'][1]:.2f}. A single model's F1 is uncertain by about ±{(ci['F1'][1] - ci['F1'][0]) / 2:.2f}, so small
differences between tuned models should not be over-read.</p>

<h2>4. Per-class performance</h2>
<img src="fig_per_class.png" class="full">
{table(team_cls, index_name="our photos")}
<p class="caption">Precision, recall and mAP from Ultralytics; missed and wrong-class counts at conf {deploy}, IoU 0.5.</p>
<p><b>Palm and like transfer well</b>: their shapes are distinctive (spread fingers, upward thumb).
<b>Fist is the weakest class</b>: {int(cls.loc['fist', 'missed'])} of {int(cls.loc['fist', 'hands'])} fists are missed entirely
({loose_missed} of them only because their labels were drawn too loosely; Section 8).
A fist is a compact, low-texture blob that looks like many other things at small scale, and several of our fists are
raised in crowd scenes with busy backgrounds, which HaGRID never shows (Section 6). <b>Dislike</b> is confused with fist
in both directions: a thumbs-down seen from the side is a fist with a short thumb, and these are the highest-confidence errors (Section 6).</p>

<h2>5. Errors at the deployment threshold</h2>
<div class="cols">
<div><img src="../../results/baseline/confusion_conf025.png" class="half"></div>
<div><img src="fig_sweep.png" class="half"></div>
</div>
<p>At conf {deploy} the model makes {ea['predictions']} predictions for {ea['hands']} labelled hands in our {ea['images']} photos:
<b>{ea['correct']} correct</b>, <b>{ea['missed_hands']} hands missed</b>, <b>{ea['wrong_class']} found with the wrong class</b>, and
<b>{ea['background_false_positives']} predictions on background</b> (no labelled hand there; by predicted class:
{", ".join(f"{k} {v}" for k, v in fp.items())}). Missing a hand is the most common error. Ultralytics' own confusion matrix uses a
near-zero confidence threshold, so its background row looks far worse than what a user would see. The matrix here uses the real threshold.</p>
<p><b>Choosing the threshold.</b> F1 is highest at conf {best['conf']} (precision {best['precision']:.2f}, recall {best['recall']:.2f},
F1 {best['F1']:.2f}), compared with precision {ea['precision']:.2f} and recall {ea['recall']:.2f} at {deploy}. For a photo booth a false
trigger (taking a picture nobody asked for) is worse than a missed gesture (the user just holds it a moment longer), so we
recommend conf ≈ {best['conf']} plus requiring the gesture over several consecutive frames.</p>

<h2>6. Where the model fails</h2>
<img src="fig_breakdown.png" class="full">
<div class="cols">
<div>{table(size_tbl, fmt="{:.2f}", index_name="hand size")}</div>
<div>{table(pers_tbl, fmt="{:.2f}", index_name="member")}</div>
</div>
<p><b>Hand size is the clearest driver.</b> The median labelled hand covers {area.median() * 100:.1f}% of the image, but
{int(small['hands'])} hands cover under 1%, and only {small['recall']:.0%} of those are found. At imgsz {hp['imgsz']},
a hand covering 0.3% of the image is only about {np.sqrt(0.003) * hp['imgsz']:.0f} px across after resizing, which leaves little detail.
Size alone is not the cause, though: {train_small} of the {len(train_area)} training hands ({train_small / len(train_area):.0%}) are also under 1%
of the image, and the median training hand ({np.median(train_area) * 100:.1f}%) is smaller than ours. The difference is context.
HaGRID's small hands belong to one person posing in a plain room; ours are in crowds, outdoors and at odd angles.
<b>Per member</b>, the differences follow photo choice and labelling style. {small_owner}'s photos hold
{small_by_person[small_owner]} of the {int(small['hands'])} small hands, which lowers recall there.
{iou_low}'s photos have the lowest mean box IoU ({person.loc[iou_low, 'mean_IoU_correct']:.2f}){iou_low_note}.</p>
<img src="../../results/baseline/failure_cases.jpg" class="full">
<p class="caption">Green = our label, red = prediction (conf ≥ {deploy}). Top row: smallest missed hands. Bottom row: wrong-class
predictions: fist/dislike and palm/like confusions.</p>

<h2>7. Speed and size</h2>
<p>Per image at imgsz {hp['imgsz']} (batch 1, RTX 5060 Ti): preprocess {sp['preprocess']:.1f} ms, inference {sp['inference']:.1f} ms,
postprocess {sp['postprocess']:.1f} ms, <b>{total_ms:.1f} ms in total (≈{1000 / total_ms:.0f} frames/s)</b>; repeated runs vary by a few ms. The model is
{(ROOT / 'models' / 'baseline_best.pt').stat().st_size / 1e6:.0f} MB (11.1 M parameters, 28.4 GFLOPs). A webcam photo booth needs about
15-30 frames/s, so a GPU has headroom to spare. CPU-only speed was not measured.</p>

<h2>8. Limits of this evaluation</h2>
<ul>
<li><b>Small test set.</b> {ea['hands']} hands in {n_team} photos. The bootstrap interval for F1 in Section 3 spans {ci['F1'][1] - ci['F1'][0]:.2f}.</li>
{"".join(f"<li><b>{title}</b> {text}</li>" for title, text in data_issues)}
<li><b>Validation is optimistic.</b> It is HaGRID-only and drawn from the same distribution as training, so checkpoint selection
does not reward robustness to our kind of photo.</li>
<li><b>One training seed.</b> Run-to-run variation was not measured; it should be kept in mind when comparing tuned models.</li>
<li><b>Labels differ by annotator.</b> Box styles vary between members, which depresses mAP50-95 and IoU-based matching.</li>
</ul>

<h2>9. Takeaways for hyperparameter tuning</h2>
<ol>
<li><b>Small hands</b> are the main failure mode. {imgsz_note}</li>
<li><b>Fist vs dislike</b> needs more varied fist examples or more epochs of fine-grained learning; watch fist recall in every comparison.</li>
<li>Judge tuned models on <b>our photos</b> (team-only mAP50 {team['mAP50']:.3f}, F1 {ea['F1']:.3f} at conf {deploy}), not on validation,
which is saturated at {val['mAP50']:.2f}.</li>
</ol>
</body></html>"""


def main():
    BUILD.mkdir(exist_ok=True)
    d = load()
    h = d["history"]
    fig_training(h, int(h.loc[h["metrics/mAP50-95(B)"].idxmax(), "epoch"]))
    fig_per_class(d["per_class"])
    fig_sweep(d["sweep"], d["ea"]["conf"], d["ea"]["best_F1_conf"]["conf"])
    fig_breakdown(d["size"], d["person"])
    html = BUILD / "Baseline_Evaluation_Report.html"
    html.write_text(build_html(d), encoding="utf-8")

    browser = next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        sys.exit(f"No Edge/Chrome found; open {html} in a browser and print it to PDF.")
    subprocess.run([str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", html.as_uri()], check=True, capture_output=True, timeout=120)
    print("wrote", PDF)


if __name__ == "__main__":
    main()
