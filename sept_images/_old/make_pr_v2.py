#!/usr/bin/env python3
"""Redraw the pooled participant-level precision-recall figure for clarity.

Reads the same August full-cohort runs and uses the same participant aggregation
(mean probability over a participant's images) as
2026-08_srep_revision_participant_level/make_figures.py, so AP values match the
published figure. Writes new files only; nothing existing is modified.
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "2026-08_srep_revision_participant_level" / "experiments"
OUT = HERE.parent / "supplementary"
K = 3
CLASSES = ["CDR 0 (Non-Demented)", "CDR 0.5 (Very Mild)", "CDR $\\geq$ 1 (Demented)"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
# fixed categorical order (validated palette); secondary encoding = line style
ARMS = [
    ("FAFL",  "exp01_main_sweep",     "FA_FL",         "FA-FL",                 "#2a78d6", "-"),
    ("WCE",   "exp01_main_sweep",     "WCE",           "Weighted CE",           "#eb6834", "-"),
    ("LDAM",  "exp01_main_sweep",     "LDAM",          "LDAM",                  "#1baf7a", "-"),
    ("CB",    "exp02_class_balanced", "ClassBalanced", "Class-balanced",        "#eda100", "-"),
    ("FL137", "exp07_focal_gamma137", "FocalLoss",     "Focal $\\gamma$=1.37",  "#e87ba4", (0, (5, 2))),
    ("FL200", "exp08_focal_gamma200", "FocalLoss",     "Focal $\\gamma$=2.0",   "#008300", (0, (2, 1.5))),
    ("FL300", "exp01_main_sweep",     "FocalLoss",     "Focal $\\gamma$=3.0",   "#4a3aa7", (0, (6, 2, 1.5, 2))),
]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def load_arm(exp, loss):
    parts = []
    for mf in sorted((EXP / exp / "outputs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE": continue
        if m.get("loss", {}).get("type") != loss: continue
        if m.get("split", {}).get("group_by") != "participant": continue
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        parts.append((int(m["split"]["fold"]), d["participant_id"].astype(str),
                      d["labels"].astype(int), d["probabilities"].astype(np.float64),
                      d["image_path"].astype(str)))
    folds = sorted(p[0] for p in parts)
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"{exp}/{loss}: folds {folds}")
    pid = np.concatenate([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
    P = np.concatenate([p[3] for p in parts]); img = np.concatenate([p[4] for p in parts])
    o = np.lexsort((img, pid))
    return pid[o], y[o], P[o]

DATA = {k: load_arm(e, l) for k, e, l, *_ in ARMS}
base_pid, base_y, _ = DATA["FAFL"]
for k in DATA:
    assert np.array_equal(DATA[k][0], base_pid) and np.array_equal(DATA[k][1], base_y), k
upid = np.array(sorted(set(base_pid))); idx = {p: i for i, p in enumerate(upid)}
gi = np.array([idx[p] for p in base_pid]); cnt = np.bincount(gi)
PY = np.zeros(len(upid), int); PY[gi] = base_y
def participant(P):
    A = np.zeros((len(upid), K))
    for c in range(K): A[:, c] = np.bincount(gi, weights=P[:, c], minlength=len(upid)) / cnt
    return A
PART = {k: participant(DATA[k][2]) for k in DATA}

def pr_pts(score, pos):
    o = np.argsort(-score); p = pos[o].astype(float)
    tp = np.cumsum(p); n = np.arange(1, len(p) + 1)
    return tp / max(tp[-1], 1), tp / n
def ap(score, pos):
    o = np.argsort(-score); p = pos[o].astype(float)
    prec = np.cumsum(p) / np.arange(1, len(p) + 1)
    return float(prec[p == 1].mean())

# ---- bootstrap: resample participants within true class, B = 2000
rng = np.random.default_rng(20260930); B = 2000
cls_idx = [np.flatnonzero(PY == c) for c in range(K)]
draws = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in cls_idx]) for _ in range(B)]
rows = []
AP = {}
for key, *_r in ARMS:
    for c in range(K):
        s = PART[key][:, c]; pos = PY == c
        v = ap(s, pos)
        bs = np.array([ap(s[d], pos[d]) for d in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        AP[(key, c)] = (v, lo, hi)
        rows.append([key, CLASSES[c].split(" (")[0].replace("$\\geq$", ">="), f"{v:.4f}", f"{lo:.4f}", f"{hi:.4f}", int(pos.sum())])
with open(OUT / "ap_pooled_participant_ci.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm", "class", "AP", "ci_low", "ci_high", "n_positive"]); w.writerows(rows)

# ---- Figure A: PR curves, shared legend, zoomed CDR 0 panel, prevalence labelled
fig, axes = plt.subplots(1, K, figsize=(9.6, 3.5))
for c, ax in enumerate(axes):
    pos = PY == c; prev = pos.mean()
    for key, _, _, lab, col, ls in ARMS[::-1]:          # FA-FL drawn last, on top
        x, yv = pr_pts(PART[key][:, c], pos)
        ax.plot(np.r_[0, x], np.r_[yv[0], yv], color=col, ls=ls,
                lw=2.1 if key == "FAFL" else 1.3, drawstyle="steps-post",
                zorder=4 if key == "FAFL" else 3, alpha=1.0 if key == "FAFL" else 0.9)
    ax.axhline(prev, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
    aps = [AP[(k, c)][0] for k, *_ in ARMS]
    lo_y = 0.6 if c == 0 else 0.0
    ax.text(0.02 if c == 0 else 0.98, prev + (0.012 if c == 0 else 0.02), f"prevalence {prev:.2f}", color=MUTED, fontsize=7.5, va="bottom", ha="left" if c == 0 else "right")
    ax.set_title(f"{CLASSES[c]}\nAP {min(aps):.3f} to {max(aps):.3f}", color=INK)
    ax.set_xlim(0, 1.0); ax.set_ylim(lo_y, 1.02); ax.grid(alpha=0.9, zorder=0)
    ax.set_xlabel("Recall")
axes[0].set_ylabel("Precision")
handles = [Line2D([], [], color=col, ls=ls, lw=2.1 if key == "FAFL" else 1.5, label=lab)
           for key, _, _, lab, col, ls in ARMS]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
           bbox_to_anchor=(0.5, -0.07), handlelength=2.6, columnspacing=1.2)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_pr_pooled_v2.{ext}")
plt.close(fig)

# ---- Figure B: AP with 95% participant-bootstrap intervals, one panel per class
fig, axes = plt.subplots(1, K, figsize=(9.6, 3.0), sharey=True)
ypos = np.arange(len(ARMS))[::-1]
for c, ax in enumerate(axes):
    prev = (PY == c).mean()
    ax.axvline(prev, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
    for yi, (key, _, _, lab, col, ls) in zip(ypos, ARMS):
        v, lo, hi = AP[(key, c)]
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot(v, yi, "o", ms=6.5, color=col, mec="white", mew=1.2, zorder=3)
        ax.text(hi + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.2, color=MUTED)
    ax.set_title(CLASSES[c], color=INK)
    ax.set_xlim((0.74, 1.005) if c == 0 else (0.0, 0.85))
    ax.text(prev + (0.004 if c == 0 else 0.015), len(ARMS) - 0.45, "prevalence", color=MUTED, fontsize=7, ha="left")
    ax.grid(axis="x", alpha=0.9, zorder=0); ax.set_xlabel("Average precision")
axes[0].set_yticks(ypos); axes[0].set_yticklabels([a[3] for a in ARMS])
axes[0].set_ylim(-0.6, len(ARMS) - 0.1)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_ap_ci.{ext}")
plt.close(fig)
print("participants", len(upid), "class counts", np.bincount(PY).tolist())
for r in rows: print(*r)
