#!/usr/bin/env python3
"""Regenerate every Results figure for all SEVEN training objectives.

Run from the project root:

    python make_figures.py

Writes PNGs to results_rewrite/figures/. Reads only the stored prediction arrays
and manifests. No GPU, no retraining, no modification of any existing file.

Why this script exists rather than a patch to analysis/_common.py: that module
selects runs with `runs[runs["loss"] == loss]`, and all three focal arms carry
loss == "FocalLoss". Adding exp07 and exp08 to its default experiment list would
return three runs per fold and silently average across gamma, and the existing
duplicate guard would not catch it because the duplicates share a fold. This
script keys arms by (experiment, loss) instead and asserts exactly one run per
fold per arm.

Palette: Okabe-Ito subset, validated colourblind-safe over all pairs. Seven arms
are encoded as five hues plus line style within the focal family, so the three
focal configurations read as one family at three settings.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
if not (ROOT / "experiments").exists():
    ROOT = Path.cwd()
EXP  = ROOT / "experiments"
OUT  = ROOT / "results_rewrite" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
K = 3
CLASSES = ["Non Demented", "Very Mild", "Demented"]

# arm key -> (experiment, loss type, label, colour, linestyle, marker)
ARMS = [
    ("WCE",       "exp01_main_sweep",     "WCE",           "Weighted CE",        "#009E73", "-",  "D"),
    ("LDAM",      "exp01_main_sweep",     "LDAM",          "LDAM",               "#CC79A7", "-",  "X"),
    ("CB",        "exp02_class_balanced", "ClassBalanced", "Class-Balanced",     "#E69F00", "-",  "P"),
    ("FL137",     "exp07_focal_gamma137", "FocalLoss",     "Focal $\\gamma$=1.37","#D55E00", "-",  "s"),
    ("FL200",     "exp08_focal_gamma200", "FocalLoss",     "Focal $\\gamma$=2.0", "#D55E00", "--", "^"),
    ("FL300",     "exp01_main_sweep",     "FocalLoss",     "Focal $\\gamma$=3.0", "#D55E00", ":",  "v"),
    ("FAFL",      "exp01_main_sweep",     "FA_FL",         "FA-FL",              "#0072B2", "-",  "o"),
]
NEUTRAL = "#555555"
INK, GRID = "#1a1a1a", "#d8d8d8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---------------------------------------------------------------- loading
def load_arm(exp, loss):
    parts = []
    for mf in sorted((EXP / exp / "outputs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE":                       continue
        if m.get("loss", {}).get("type") != loss:               continue
        if m.get("split", {}).get("group_by") != "participant": continue
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        parts.append((int(m["split"]["fold"]),
                      d["participant_id"].astype(str),
                      d["labels"].astype(int),
                      d["probabilities"].astype(np.float64),
                      d["image_path"].astype(str)))
    folds = [p[0] for p in parts]
    if sorted(folds) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"{exp}/{loss}: expected exactly one run per fold 0-4, got folds {sorted(folds)}")
    parts.sort(key=lambda t: t[0])
    pid = np.concatenate([p[1] for p in parts])
    y   = np.concatenate([p[2] for p in parts])
    P   = np.concatenate([p[3] for p in parts])
    img = np.concatenate([p[4] for p in parts])
    fold = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
    o = np.lexsort((img, pid))
    return pid[o], y[o], P[o], fold[o]

print("loading seven arms ...")
DATA = {}
for key, exp, loss, *_ in ARMS:
    DATA[key] = load_arm(exp, loss)
base_pid, base_y, _, base_fold = DATA["FAFL"]
for key in DATA:
    if not np.array_equal(DATA[key][0], base_pid) or not np.array_equal(DATA[key][1], base_y):
        raise SystemExit(f"{key}: image ordering does not match; cannot compare arms")
upid = np.array(sorted(set(base_pid)))
pos  = {p: i for i, p in enumerate(upid)}
gi   = np.array([pos[p] for p in base_pid])
cnt  = np.bincount(gi)
py   = np.zeros(len(upid), int);  py[gi]   = base_y
pfold= np.zeros(len(upid), int);  pfold[gi]= base_fold
print(f"  {len(base_pid):,} images, {len(upid)} participants, folds "
      f"{np.bincount(pfold).tolist()}")

def participant(P):
    A = np.zeros((len(upid), K))
    for c in range(K):
        A[:, c] = np.bincount(gi, weights=P[:, c], minlength=len(upid)) / cnt
    return A

IMG  = {k: DATA[k][2] for k in DATA}
PART = {k: participant(DATA[k][2]) for k in DATA}
Y, PY = base_y, py

# ---------------------------------------------------------------- metrics
def cmat(a, b):
    M = np.zeros((K, K)); np.add.at(M, (a, b), 1); return M
def macro_f1(M):
    o = []
    for k in range(K):
        tp = M[k, k]; fp = M[:, k].sum() - tp; fn = M[k, :].sum() - tp
        o.append(0.0 if 2*tp + fp + fn == 0 else 2*tp / (2*tp + fp + fn))
    return float(np.mean(o))
def ece(P, yy, B=15):
    c = P.max(1); ok = (P.argmax(1) == yy).astype(float)
    idx = np.clip((c * B).astype(int), 0, B - 1); e = 0.0
    for k in range(B):
        m = idx == k
        if m.sum(): e += m.sum() / len(yy) * abs(ok[m].mean() - c[m].mean())
    return e
def reliability(P, yy, B=15):
    c = P.max(1); ok = (P.argmax(1) == yy).astype(float)
    idx = np.clip((c * B).astype(int), 0, B - 1)
    xs, ys, ws = [], [], []
    for k in range(B):
        m = idx == k
        if m.sum() >= 20:
            xs.append(c[m].mean()); ys.append(ok[m].mean()); ws.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ws)
def roc_pts(score, pos):
    o = np.argsort(-score); p = pos[o].astype(float)
    tp = np.cumsum(p); fp = np.cumsum(1 - p)
    return fp / max(fp[-1], 1), tp / max(tp[-1], 1)
def pr_pts(score, pos):
    o = np.argsort(-score); p = pos[o].astype(float)
    tp = np.cumsum(p); n = np.arange(1, len(p) + 1)
    return tp / max(tp[-1], 1), tp / n
def auc_of(x, y): return float(np.trapz(y, x))

def legend_handles(loc_ax, ncol=2):
    h = [Line2D([], [], color=c, ls=ls, marker=mk, ms=4, lw=1.8, label=lab)
         for _, _, _, lab, c, ls, mk in ARMS]
    return loc_ax.legend(handles=h, ncol=ncol, frameon=False, loc="lower right")

saved = []
def save(fig, name):
    p = OUT / name; fig.savefig(p); plt.close(fig); saved.append(name)
    print(f"  wrote {name}")

# ------------------------------------------------- 1. reliability diagram
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([0, 1], [0, 1], color=NEUTRAL, ls=(0, (4, 3)), lw=1.2, zorder=1,
        label="Perfect calibration")
for key, _, _, lab, c, ls, mk in ARMS:
    x, yv, _ = reliability(IMG[key], Y)
    ax.plot(x, yv, color=c, ls=ls, marker=mk, ms=4.5, lw=1.8,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            label=f"{lab}  ({ece(IMG[key], Y):.4f})")
ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
ax.set_xlim(0.3, 1.02); ax.set_ylim(0.0, 1.02); ax.grid(alpha=0.5, zorder=0)
ax.legend(frameon=False, fontsize=7.4, loc="upper left", title="Objective  (ECE)",
          title_fontsize=7.6)
save(fig, "fig_calibration_reliability.png")

# ------------------------------------------------------- 2. ECE by fold
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for key, _, _, lab, c, ls, mk in ARMS:
    v = [ece(IMG[key][base_fold == f], Y[base_fold == f]) for f in range(5)]
    ax.plot(range(1, 6), v, color=c, ls=ls, marker=mk, ms=5, lw=1.8,
            markeredgecolor="white", markeredgewidth=0.6, label=lab)
ax.set_xlabel("Test fold"); ax.set_ylabel("Expected calibration error")
ax.set_xticks(range(1, 6)); ax.grid(axis="y", alpha=0.5)
ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left")
save(fig, "fig_ece_by_fold.png")

# --------------------------------------------- 3. variance decomposition
accf = {k: [100 * (IMG[k][base_fold == f].argmax(1) == Y[base_fold == f]).mean()
            for f in range(5)] for k in DATA}
init = {}
for mf in sorted((EXP / "exp03_init_variance" / "outputs").glob("*/manifest.json")):
    m = json.loads(mf.read_text(encoding="utf-8"))
    if m.get("status") != "COMPLETE": continue
    init.setdefault(m["loss"]["type"], []).append(m["headline"]["test_accuracy"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.9, 1]})
for i, (key, _, _, lab, c, ls, mk) in enumerate(ARMS):
    a1.scatter([i]*5, accf[key], color=c, marker=mk, s=34, zorder=3,
               edgecolor="white", linewidth=0.6)
    a1.plot([i-0.24, i+0.24], [np.mean(accf[key])]*2, color=c, lw=2.2, zorder=4)
a1.set_xticks(range(len(ARMS)))
a1.set_xticklabels([a[3] for a in ARMS], rotation=38, ha="right")
a1.set_ylabel("Pooled accuracy (%)")
a1.set_title(f"Partition varies, seed fixed\nmean SD {np.mean([np.std(v, ddof=1) for v in accf.values()]):.2f} points",
             fontsize=9)
a1.grid(axis="y", alpha=0.5)
lbl = {"FA_FL": ("FA-FL", "#0072B2", "o"), "WCE": ("Weighted CE", "#009E73", "D")}
for i, (lo, v) in enumerate(sorted(init.items())):
    nm, c, mk = lbl.get(lo, (lo, NEUTRAL, "o"))
    a2.scatter([i]*len(v), v, color=c, marker=mk, s=34, zorder=3,
               edgecolor="white", linewidth=0.6)
    a2.plot([i-0.18, i+0.18], [np.mean(v)]*2, color=c, lw=2.2, zorder=4)
    a2.annotate(f"SD {np.std(v, ddof=1):.2f}", (i, max(v)), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=c, fontweight="bold")
a2.set_xticks(range(len(init)))
a2.set_xticklabels([lbl.get(k, (k,))[0] for k in sorted(init)], rotation=38, ha="right")
a2.set_title("Seed varies, partition fixed\nmean SD 0.59 points", fontsize=9)
a2.grid(axis="y", alpha=0.5)
save(fig, "fig_variance_decomposition.png")

# ------------------------------------------------- 4. confusion matrices
def confusion_panel(ax, M, title):
    Mn = M / M.sum(1, keepdims=True)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{int(M[i, j])}\n{Mn[i, j]*100:.0f}%", ha="center",
                    va="center", fontsize=8.5,
                    color="white" if Mn[i, j] > 0.55 else INK)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9.5)
    for s in ax.spines.values(): s.set_visible(False)
for key, fname in (("FAFL", "fig_confusion_fafl.png"), ("LDAM", "fig_confusion_ldam.png")):
    lab = dict((a[0], a[3]) for a in ARMS)[key]
    M = cmat(PY, PART[key].argmax(1))
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    confusion_panel(ax, M, f"{lab}   (macro-F1 {macro_f1(M):.3f})")
    save(fig, fname)

# --------------------------------------------------------- 5. ROC and PR
for kind, fname in (("roc", "fig_roc_pooled.png"), ("pr", "fig_pr_pooled.png")):
    fig, axes = plt.subplots(1, K, figsize=(9.6, 3.3), sharey=True)
    for c_i, ax in enumerate(axes):
        pos = (PY == c_i)
        for key, _, _, lab, col, ls, mk in ARMS:
            s = PART[key][:, c_i]
            if kind == "roc":
                x, yv = roc_pts(s, pos); v = auc_of(x, yv); tag = "AUC"
                ax.plot([0, 1], [0, 1], color=NEUTRAL, ls=(0, (4, 3)), lw=1.0, zorder=1)
            else:
                x, yv = pr_pts(s, pos); v = float(np.mean(yv[pos[np.argsort(-s)]])); tag = "AP"
                ax.axhline(pos.mean(), color=NEUTRAL, ls=(0, (4, 3)), lw=1.0, zorder=1)
            ax.plot(x, yv, color=col, ls=ls, lw=1.7, zorder=3, label=f"{lab} ({v:.3f})")
        ax.set_title(CLASSES[c_i], fontsize=9.5); ax.grid(alpha=0.5, zorder=0)
        ax.set_xlabel("False positive rate" if kind == "roc" else "Recall")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=6.6, loc="lower right" if kind == "roc" else "upper right")
    axes[0].set_ylabel("True positive rate" if kind == "roc" else "Precision")
    save(fig, fname)

# -------------------------------------------- 6. baseline-anchored frontier
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cn = np.bincount(PY, minlength=K); maj = int(cn.argmax())
Pc = np.zeros((len(PY), K)); Pc[:, maj] = 1.0
Mb = cmat(PY, Pc.argmax(1))
b_acc, b_f1 = np.trace(Mb) / Mb.sum(), macro_f1(Mb)
pri = cn / cn.sum(); sr_acc = float((pri**2).sum())
ax.axvline(b_acc, color=NEUTRAL, ls=(0, (5, 3)), lw=1.3, zorder=1)
ax.scatter([b_acc, sr_acc], [b_f1, 1/K], marker="*", s=170, color=NEUTRAL,
           zorder=4, label="Trivial baseline")
ax.annotate("Constant predictor", (b_acc, b_f1), textcoords="offset points",
            xytext=(-9, 8), ha="right", fontsize=7.8, color=NEUTRAL)
ax.annotate("Stratified random", (sr_acc, 1/K), textcoords="offset points",
            xytext=(10, 0), va="center", fontsize=7.8, color=NEUTRAL)
for key, _, _, lab, c, ls, mk in ARMS:
    M = cmat(PY, PART[key].argmax(1))
    ax.scatter([np.trace(M) / M.sum()], [macro_f1(M)], color=c, marker=mk, s=70,
               zorder=5, edgecolor="white", linewidth=0.8, label=lab)
ax.set_xlabel("Participant-level accuracy"); ax.set_ylabel("Participant-level macro-F1")
ax.grid(alpha=0.5, zorder=0); ax.margins(x=0.10, y=0.10)
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "fig_baseline_frontier.png")

# ------------------------------------------------------- 7. age-cutoff curve
try:
    import pandas as pd
    meta = pd.read_excel(ROOT / "oasis_cross-sectional.xlsx")
    meta["pid"] = meta["ID"].str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid").reindex(upid)
    AGE = meta["Age"].to_numpy(float); CDR = meta["CDR"].to_numpy(float)
    cuts = [("none", np.ones(len(upid), bool)), ("CDR", ~np.isnan(CDR))] + \
           [(f"$\\geq${c}", (~np.isnan(CDR)) & (AGE >= c)) for c in (50, 55, 60, 65, 70, 75)]
    xs = list(range(len(cuts))); base, best, bestf = [], [], []
    for _, m in cuts:
        im = np.isin(gi, np.where(m)[0]); yy = Y[im]
        c2 = np.bincount(yy, minlength=K)
        base.append(100 * c2.max() / c2.sum())
        best.append(max(100 * (IMG[k][im].argmax(1) == yy).mean() for k in DATA))
        bestf.append(max(macro_f1(cmat(yy, IMG[k][im].argmax(1))) for k in DATA))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.35, 1]})
    a1.plot(xs, base, color=NEUTRAL, ls=(0, (5, 3)), marker="s", ms=5, lw=1.8,
            label="Constant predictor")
    a1.plot(xs, best, color="#0072B2", marker="o", ms=5, lw=1.9,
            markeredgecolor="white", markeredgewidth=0.6, label="Best objective")
    a1.set_ylabel("Accuracy (%)"); a1.legend(frameon=False, fontsize=8)
    a1.grid(axis="y", alpha=0.5)
    a2.plot(xs, bestf, color="#0072B2", marker="o", ms=5, lw=1.9,
            markeredgecolor="white", markeredgewidth=0.6)
    a2.set_ylabel("Best macro-F1"); a2.grid(axis="y", alpha=0.5)
    a2.set_xticks(xs); a2.set_xticklabels([c[0] for c in cuts])
    a2.set_xlabel("Cohort restriction (CDR recorded and age threshold)")
    save(fig, "fig_age_curve.png")
except Exception as exc:                                    # noqa: BLE001
    print(f"  [skip] fig_age_curve.png: {exc}")

print(f"\nDONE. {len(saved)} figures in {OUT}")
for s in saved: print("   ", s)
