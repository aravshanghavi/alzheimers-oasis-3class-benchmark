#!/usr/bin/env python3
"""Regenerate every figure for the round-2 draft as 300-dpi PNG in sept_images/figures/.

Run from anywhere:  python make_all_figures.py

Data sources (read only; nothing outside sept_images/figures is written):
  Restricted cohort (166 participants): September runs, 2026-09_instrumented_arms/experiments,
      exp10, exp11, exp12, exp13, exp15, code fingerprint cb7ee123, one run per fold.
  Full cohort (347 participants): 2026-08_srep_revision_participant_level/experiments,
      code fingerprint 8378f120. These are the only complete five-fold full-cohort runs; the
      September tree holds fold-0 reproducibility reruns only.
  Audit tables: 2026-09_instrumented_arms/analysis/outputs (t23, t30), generated 29 Sep.
  Leakage and 2.5D comparisons: 2026-09_tier1_verified_stack/analysis/outputs.
Unit conventions: participant = mean predicted probability over the participant's test images.
Calibration figures are image level, raw probabilities, 15 equal-width bins, pooled over folds.
"""
import csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "figures"; OUT.mkdir(parents=True, exist_ok=True)
AUG = ROOT / "2026-08_srep_revision_participant_level" / "experiments"
SEP = ROOT / "2026-09_instrumented_arms" / "experiments"
TAB = ROOT / "2026-09_instrumented_arms" / "analysis" / "outputs"
T1 = ROOT / "2026-09_tier1_verified_stack" / "analysis" / "outputs"
K = 3
CLASS = ["CDR 0", "CDR 0.5", "CDR $\\geq$ 1"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
# key: (label, colour, line style). Fixed across every figure. Reference arm in neutral grey.
STYLE = {
    "FAFL":  ("FA-FL",                 "#2a78d6", "-"),
    "WCE":   ("WCE",                   "#eb6834", "-"),
    "LDAM":  ("LDAM",                  "#1baf7a", "-"),
    "CB":    ("CB",                    "#eda100", "-"),
    "FL137": ("FL ($\\gamma$=1.37)",   "#e87ba4", (0, (5, 2))),
    "FL200": ("FL ($\\gamma$=2.0)",    "#008300", (0, (2, 1.5))),
    "FL269": ("FL ($\\gamma$=2.69)",   "#e34948", (0, (6, 2, 1.5, 2))),
    "FL300": ("FL ($\\gamma$=3.0)",    "#4a3aa7", (0, (1, 1.2))),
    "UNW":   ("Unweighted ref.",       "#8a8984", (0, (4, 2))),
}
FULL_ARMS = [("FAFL", AUG, "exp01_main_sweep", "FA_FL", None), ("WCE", AUG, "exp01_main_sweep", "WCE", None),
             ("LDAM", AUG, "exp01_main_sweep", "LDAM", None), ("CB", AUG, "exp02_class_balanced", "ClassBalanced", 0.9999),
             ("FL137", AUG, "exp07_focal_gamma137", "FocalLoss", None), ("FL200", AUG, "exp08_focal_gamma200", "FocalLoss", None),
             ("FL300", AUG, "exp01_main_sweep", "FocalLoss", None)]
REST_ARMS = [("FAFL", SEP, "exp10_age60_main", "FA_FL", None), ("WCE", SEP, "exp10_age60_main", "WCE", None),
             ("LDAM", SEP, "exp10_age60_main", "LDAM", None), ("CB", SEP, "exp10_age60_main", "ClassBalanced", 0.9999),
             ("FL137", SEP, "exp11_age60_gamma137", "FocalLoss", None), ("FL200", SEP, "exp12_age60_gamma200", "FocalLoss", None),
             ("FL269", SEP, "exp13_age60_gamma269", "FocalLoss", None), ("FL300", SEP, "exp10_age60_main", "FocalLoss", None),
             ("UNW", SEP, "exp15_age60_unweighted", "ClassBalanced", 0.999)]
FP = {AUG: "8378f120", SEP: "cb7ee123"}
def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("  wrote", name)
def trapz(y, x):
    return float(getattr(np, "trapezoid", getattr(np, "trapz", None))(y, x))

# ------------------------------------------------------------------ loading
def load_arm(tree, exp, loss, beta):
    parts = {}
    for mf in sorted((tree / exp / "outputs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE" or m.get("loss", {}).get("type") != loss: continue
        if m.get("split", {}).get("group_by") != "participant": continue
        if not str(m.get("code_fingerprint", mf.parent.name)).startswith(FP[tree]) and FP[tree] not in mf.parent.name: continue
        if beta is not None and abs(float(m["loss"].get("params", {}).get("beta", -1)) - beta) > 1e-9: continue
        if not (mf.parent / "_COMPLETE").exists(): continue
        f = int(m["split"]["fold"])
        if f in parts: raise SystemExit(f"{exp}/{loss}: two runs for fold {f}")
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        parts[f] = (d["participant_id"].astype(str), d["labels"].astype(int), d["probabilities"].astype(float), d["image_path"].astype(str))
    if sorted(parts) != [0, 1, 2, 3, 4]: raise SystemExit(f"{exp}/{loss}: folds {sorted(parts)}")
    pid = np.concatenate([parts[f][0] for f in range(5)]); y = np.concatenate([parts[f][1] for f in range(5)])
    P = np.concatenate([parts[f][2] for f in range(5)]); img = np.concatenate([parts[f][3] for f in range(5)])
    fold = np.concatenate([np.full(len(parts[f][0]), f) for f in range(5)])
    o = np.lexsort((img, pid))
    return pid[o], y[o], P[o], fold[o]

class Cohort:
    def __init__(self, name, arms):
        self.name = name; self.keys = [a[0] for a in arms]
        self.data = {k: load_arm(t, e, l, b) for k, t, e, l, b in arms}
        pid, y, _, fold = self.data[self.keys[0]]
        for k in self.keys:
            assert np.array_equal(self.data[k][0], pid) and np.array_equal(self.data[k][1], y), k
        self.y, self.fold = y, fold
        up = np.array(sorted(set(pid))); ix = {p: i for i, p in enumerate(up)}
        self.gi = np.array([ix[p] for p in pid]); cnt = np.bincount(self.gi)
        self.py = np.zeros(len(up), int); self.py[self.gi] = y
        self.img = {k: self.data[k][2] for k in self.keys}
        self.part = {}
        for k in self.keys:
            A = np.zeros((len(up), K))
            for c in range(K): A[:, c] = np.bincount(self.gi, weights=self.img[k][:, c], minlength=len(up)) / cnt
            self.part[k] = A
        print(f"{name}: {len(y):,} images, {len(up)} participants, classes {np.bincount(self.py).tolist()}")

# ------------------------------------------------------------------ metrics
def ece_top(P, y, B=15):
    conf = P.max(1); ok = (P.argmax(1) == y).astype(float)
    b = np.minimum((conf * B).astype(int), B - 1); e = 0.0
    for i in range(B):
        m = b == i
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e
def reliability(P, y, B=15):
    conf = P.max(1); ok = (P.argmax(1) == y).astype(float)
    b = np.minimum((conf * B).astype(int), B - 1); xs, ys, ns = [], [], []
    for i in range(B):
        m = b == i
        if m.sum() >= 20: xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)
def ranked(score, pos):
    o = np.argsort(-score, kind="stable"); return pos[o].astype(float)
def roc_pts(score, pos):
    p = ranked(score, pos); tp = np.cumsum(p); fp = np.cumsum(1 - p)
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
def pr_pts(score, pos):
    p = ranked(score, pos); tp = np.cumsum(p); return tp / tp[-1], tp / np.arange(1, len(p) + 1)
def ap(score, pos):
    p = ranked(score, pos); prec = np.cumsum(p) / np.arange(1, len(p) + 1); return float(prec[p == 1].mean())

def legend_row(fig, keys, y=-0.06, ncol=None):
    h = [Line2D([], [], color=STYLE[k][1], ls=STYLE[k][2], lw=2.0 if k == "FAFL" else 1.5, label=STYLE[k][0]) for k in keys]
    fig.legend(handles=h, loc="lower center", ncol=ncol or len(keys), frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=2.6, columnspacing=1.1)

# ------------------------------------------------------------ curve figures
def fig_curves(C, tag):
    for kind in ("roc", "pr"):
        fig, axes = plt.subplots(1, K, figsize=(9.6, 3.4))
        for c, ax in enumerate(axes):
            pos = C.py == c; prev = pos.mean()
            vals = []
            for k in C.keys[::-1]:
                s = C.part[k][:, c]
                if kind == "roc":
                    x, yv = roc_pts(s, pos); vals.append(trapz(yv, x))
                    ax.plot(x, yv, color=STYLE[k][1], ls=STYLE[k][2], lw=2.0 if k == "FAFL" else 1.2, drawstyle="steps-post", zorder=4 if k == "FAFL" else 3)
                else:
                    x, yv = pr_pts(s, pos); vals.append(ap(s, pos))
                    ax.plot(np.r_[0, x], np.r_[yv[0], yv], color=STYLE[k][1], ls=STYLE[k][2], lw=2.0 if k == "FAFL" else 1.2, drawstyle="steps-post", zorder=4 if k == "FAFL" else 3)
            if kind == "roc":
                ax.plot([0, 1], [0, 1], color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
                ax.set_xlabel("False positive rate"); ax.set_ylim(0, 1.01)
                ax.set_title(f"{CLASS[c]}\nAUC {min(vals):.3f} to {max(vals):.3f}")
            else:
                ax.axhline(prev, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
                ax.text(0.98, prev - 0.012, f"prevalence {prev:.2f}", color=MUTED, fontsize=7, va="top", ha="right")
                ax.set_xlabel("Recall"); ax.set_ylim((prev - 0.1) if (c == 0 and prev > 0.5) else 0, 1.02)
                ax.set_title(f"{CLASS[c]}\nAP {min(vals):.3f} to {max(vals):.3f}")
            ax.set_xlim(0, 1); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        axes[0].set_ylabel("True positive rate" if kind == "roc" else "Precision")
        legend_row(fig, C.keys, ncol=min(len(C.keys), 9)); fig.tight_layout()
        save(fig, f"fig_{kind}_{tag}.png")

def fig_ap_ci(C, tag, B=2000):
    rng = np.random.default_rng(20260930)
    idx = [np.flatnonzero(C.py == c) for c in range(K)]
    draws = [np.concatenate([rng.choice(i, len(i), replace=True) for i in idx]) for _ in range(B)]
    fig, axes = plt.subplots(1, K, figsize=(9.6, 0.45 * len(C.keys) + 1.2), sharey=True)
    y = np.arange(len(C.keys))[::-1]
    rows = []
    for c, ax in enumerate(axes):
        pos = C.py == c; prev = pos.mean()
        ax.axvline(prev, color=MUTED, ls=(0, (4, 3)), lw=0.9)
        for yi, k in zip(y, C.keys):
            s = C.part[k][:, c]; v = ap(s, pos)
            bs = np.array([ap(s[d], pos[d]) for d in draws]); lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append([tag, STYLE[k][0], CLASS[c].replace("$\\geq$", ">="), f"{v:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            ax.plot([lo, hi], [yi, yi], color=STYLE[k][1], lw=1.8, solid_capstyle="round")
            ax.plot(v, yi, "o", ms=6, color=STYLE[k][1], mec="white", mew=1.1)
            ax.text(hi + 0.01, yi, f"{v:.3f}", va="center", fontsize=7, color=MUTED)
        ax.text(prev + 0.005, len(C.keys) - 0.45, "prevalence", color=MUTED, fontsize=7)
        ax.set_title(CLASS[c]); ax.set_xlabel("Average precision"); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_xlim((max(0, prev - 0.05), 1.005) if prev > 0.5 else (0, 0.9))
    axes[0].set_yticks(y); axes[0].set_yticklabels([STYLE[k][0] for k in C.keys]); axes[0].set_ylim(-0.6, len(C.keys) - 0.1)
    fig.tight_layout(); save(fig, f"fig_ap_ci_{tag}.png")
    with open(OUT / f"ap_ci_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cohort", "arm", "class", "AP", "ci_low", "ci_high"]); w.writerows(rows)

def fig_reliability(C, tag):
    n = len(C.keys); cols = 3 if n > 6 else n; rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(2.3 * cols, 2.3 * rws), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, C.keys):
        x, yv, nb = reliability(C.img[k], C.y)
        ax.plot([0, 1], [0, 1], color=MUTED, ls=(0, (4, 3)), lw=0.9)
        ax.plot(x, yv, "-o", color=STYLE[k][1], ms=3.5, lw=1.4)
        ax.set_title(f"{STYLE[k][0]}  (ECE {ece_top(C.img[k], C.y):.3f})", fontsize=7.8)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0.0, 1.0); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax in axes[n:]: ax.axis("off")
    for ax in axes[(rws - 1) * cols:]: ax.set_xlabel("Mean confidence")
    for i in range(rws): axes[i * cols].set_ylabel("Accuracy")
    fig.tight_layout(); save(fig, f"fig_reliability_{tag}.png")

def fig_confusion(C, tag, keys=("FAFL", "LDAM")):
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 2.8))
    for ax, k in zip(np.atleast_1d(axes), keys):
        pred = C.part[k].argmax(1); M = np.zeros((K, K), int)
        for t, p in zip(C.py, pred): M[t, p] += 1
        R = M / M.sum(1, keepdims=True)
        ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i, j]}\n{R[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color="white" if R[i, j] > 0.55 else INK)
        ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(CLASS); ax.set_yticklabels(CLASS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(STYLE[k][0])
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); save(fig, f"fig_confusion_{tag}.png")

def fig_variance(C):
    acc = {k: [100 * (C.img[k][C.fold == f].argmax(1) == C.y[C.fold == f]).mean() for f in range(5)] for k in C.keys}
    seeds = {}
    for mf in sorted((AUG / "exp03_init_variance" / "outputs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE" or not (mf.parent / "_COMPLETE").exists() or "8378f120" not in mf.parent.name: continue
        key = {"FA_FL": "FAFL", "WCE": "WCE"}.get(m["loss"]["type"])
        if key is None: continue
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        seeds.setdefault(key, []).append(100 * (d["probabilities"].argmax(1) == d["labels"]).mean())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [len(C.keys), 2.6]}, sharey=True)
    for i, k in enumerate(C.keys):
        a1.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), acc[k], s=16, color=STYLE[k][1], zorder=3)
        a1.plot([i - 0.25, i + 0.25], [np.mean(acc[k])] * 2, color=INK, lw=1.4)
    sd_fold = np.mean([np.std(v, ddof=1) for v in acc.values()])
    a1.set_xticks(range(len(C.keys))); a1.set_xticklabels([STYLE[k][0] for k in C.keys], rotation=35, ha="right")
    a1.set_ylabel("Image-level accuracy (%)"); a1.set_title(f"Five partitions, one seed (mean SD {sd_fold:.2f} points)", loc="left")
    for i, k in enumerate(["FAFL", "WCE"]):
        v = seeds.get(k, [])
        a2.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=16, color=STYLE[k][1], zorder=3)
        if v: a2.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=INK, lw=1.4)
    sds = {k: np.std(v, ddof=1) for k, v in seeds.items() if len(v) > 1}
    a2.set_xticks([0, 1]); a2.set_xticklabels(["FA-FL", "WCE"]); a2.set_xlim(-0.6, 1.6)
    a2.set_title("One partition (fold 0), five seeds\nSD " + ", ".join(f"{STYLE[k][0]} {s:.2f}" for k, s in sds.items()), loc="left")
    for ax in (a1, a2): ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig_variance_decomposition.png")
    print(f"    fold SD mean {sd_fold:.2f}; seed SDs " + ", ".join(f"{k} {s:.2f} (n={len(seeds[k])})" for k, s in sds.items()))

# --------------------------------------------------------- table-driven figures
t23 = list(csv.DictReader(open(TAB / "t23_protocol_audit_long.csv", encoding="utf-8")))
T23KEY = {"Weighted CE": "WCE", "LDAM": "LDAM", "Class-Balanced": "CB", "Focal g=1.37": "FL137", "Focal g=2.00": "FL200",
          "Focal g=2.69": "FL269", "Focal g=3.00": "FL300", "FA-FL": "FAFL", "Class-Balanced [exp15]": "UNW"}
ORDER = ["WCE", "LDAM", "CB", "FL137", "FL200", "FL269", "FL300", "FAFL", "UNW"]
def pick(req, cohort, metric, ref=None):
    return {T23KEY[r["arm"]]: r for r in t23 if r["requirement"] == req and r["cohort"] == cohort and r["unit"] == "participant"
            and r["metric"] == metric and (ref is None or r["reference"] == ref)}

def fig_floors():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw={"height_ratios": [7, 9]})
    L = iter("abcd")
    for r, (co, name) in enumerate((("full", "Full cohort, 347 participants"), ("restricted", "Restricted cohort, 166 participants"))):
        for c, (d, xl) in enumerate(((pick("R1", co, "accuracy"), "Accuracy minus constant predictor (percentage points)"),
                                     (pick("R4", co, "macro_f1", "META age+sex+eTIV+nWBV"), "Macro-F1 minus age + sex + eTIV + nWBV model"))):
            ax = axes[r, c]; arms = [a for a in ORDER if a in d][::-1]; y = np.arange(len(arms))
            ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 2)))
            for yi, a in zip(y, arms):
                v, lo, hi = (float(d[a][x]) for x in ("difference", "ci_low", "ci_high"))
                col = "#eb6834" if (lo > 0 or hi < 0) else "#2a78d6"
                ax.hlines(yi, lo, hi, color=col, lw=2, capstyle="round"); ax.plot(v, yi, "o", ms=5, color=col, mec="white", mew=1)
            ax.set_yticks(y); ax.set_yticklabels([STYLE[a][0] for a in arms] if c == 0 else [])
            ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.set_title(f"{next(L)}   {name}", loc="left")
            if r == 1: ax.set_xlabel(xl)
        axes[r, 0].set_xlim(-8, 19); axes[r, 1].set_xlim(-0.28, 0.30)
    fig.tight_layout(); save(fig, "fig_floors.png")

def fig_null():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, (m, title) in zip(axes, (("f1_NonDem", CLASS[0]), ("f1_VeryMild", CLASS[1]), ("f1_Dem", CLASS[2]))):
        d = pick("R2", "restricted", m); arms = [a for a in ORDER if a in d][::-1]; y = np.arange(len(arms))
        r0 = d[arms[0]]; sd = float(r0["null_sd"]); mean = float(r0["arm_value"]) - float(r0["z_against_null_sd"]) * sd
        ax.axvspan(float(r0["null_min"]), float(r0["null_max"]), color=GRID, zorder=0)
        ax.axvline(mean, color=MUTED, lw=1, ls=(0, (3, 2))); ax.axvline(mean + 2.5 * sd, color=MUTED, lw=0.8, ls=(0, (1, 2)))
        for yi, a in zip(y, arms):
            z = float(d[a]["z_against_null_sd"])
            ax.plot(float(d[a]["arm_value"]), yi, "o", ms=5.5, color="#eb6834" if z > 2.5 else "#2a78d6", mec="white", mew=1, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([STYLE[a][0] for a in arms]); ax.set_title(title)
        ax.set_xlabel("Participant-level F1"); ax.set_xlim(0, 0.8); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig_null_perclass.png")

def fig_gamma_ladder():
    e = {}
    for r in csv.DictReader(open(TAB / "t30_ece_estimator_robustness.csv", encoding="utf-8")):
        if (r["kind"] == "arm_ece" and r["cohort"] == "restricted" and r["unit"] == "image" and r["n_bins_requested"].startswith("15")
                and r["binning"] == "equal width" and r["bootstrap"].startswith("fold clustered")):
            e[(r["arm"], r["estimator"])] = tuple(float(r[x]) for x in ("ece", "ci_low", "ci_high"))
    lad = [(1.37, "Focal g=1.37"), (2.00, "Focal g=2.00"), (2.69, "Focal g=2.69"), (3.00, "Focal g=3.00")]; geff = 2.727
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for est, off, filled, lab in (("pooled", -0.018, True, "pooled over folds"), ("mean of folds", 0.018, False, "mean of per-fold values")):
        xs = [g + off for g, _ in lad]; v = [e[(a, est)] for _, a in lad]
        ax.vlines(xs, [t[1] for t in v], [t[2] for t in v], color="#4a3aa7", lw=1.4)
        ax.plot(xs, [t[0] for t in v], "o-", color="#4a3aa7", mfc="#4a3aa7" if filled else "white", ms=5.5, lw=1.2, label=f"Focal loss, {lab}")
        fv = e[("FA-FL", est)]
        ax.vlines(geff + off, fv[1], fv[2], color="#2a78d6", lw=1.4)
        ax.plot(geff + off, fv[0], "D", color="#2a78d6", mfc="#2a78d6" if filled else "white", ms=6, label=f"FA-FL at effective exponent, {lab}")
    ax.set_xticks([1.37, 2.0, 2.69, 3.0]); ax.set_xlabel("Focusing exponent $\\gamma$"); ax.set_ylabel("Image-level ECE (15 bins)")
    ax.set_ylim(0, 0.16); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "fig_gamma_ladder.png")

def fig_diff(path, dcol, lcol, hcol, name):
    rows = {r["Metric"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    right = [("Macro-F1", "Macro-F1"), ("F1 Very Mild", "F1, CDR 0.5"), ("F1 Demented", "F1, CDR $\\geq$ 1"), ("ECE", "ECE"), ("Brier", "Brier score")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, items, xl in ((a1, [("Accuracy (%)", "Accuracy")], "Difference (percentage points)"), (a2, right, "Difference")):
        y = np.arange(len(items))[::-1]
        for yi, (k, lab) in zip(y, items):
            v, lo, hi = (float(rows[k][c]) for c in (dcol, lcol, hcol))
            ax.hlines(yi, lo, hi, color="#2a78d6", lw=2, capstyle="round"); ax.plot(v, yi, "o", ms=5, color="#2a78d6", mec="white", mew=1)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 2))); ax.set_yticks(y); ax.set_yticklabels([l for _, l in items])
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.6, len(items) - 0.4)
    fig.tight_layout(); save(fig, name)

# ----------------------------------------------------------------------- run
if __name__ == "__main__":
    print("table-driven figures"); fig_floors(); fig_null(); fig_gamma_ladder()
    fig_diff(T1 / "leakage_ablation.csv", "Inflation (session - participant)", "CI low (diff)", "CI high (diff)", "fig_leakage_ablation.png")
    fig_diff(T1 / "multislice_comparison.csv", "Difference (2.5D - 2D)", "CI low (diff)", "CI high (diff)", "fig_multislice.png")
    for C, tag in ((Cohort("restricted", REST_ARMS), "restricted"), (Cohort("full", FULL_ARMS), "full")):
        print(f"{tag}: image-level ECE check (compare with t30 pooled 15-bin):",
              ", ".join(f"{STYLE[k][0]} {ece_top(C.img[k], C.y):.4f}" for k in C.keys))
        fig_curves(C, tag); fig_ap_ci(C, tag); fig_reliability(C, tag); fig_confusion(C, tag)
        if tag == "full": fig_variance(C)
    for src, dst in ((ROOT / "review_experiments" / "data_vvisualizations" / "augmentation_panel_Demented.png", "fig_augmentation_CDR_ge1.png"),
                     (ROOT / "review_experiments" / "data_vvisualizations" / "augmentation_panel_Non_Demented.png", "fig_augmentation_CDR0.png"),
                     (ROOT / "review_experiments" / "data_vvisualizations" / "augmentation_panel_Very_Mild_Demented.png", "fig_augmentation_CDR05.png"),
                     (TAB / "f19a_train_loss_share.png", "fig_train_loss_share_restricted.png")):
        if src.exists(): shutil.copy2(src, OUT / dst); print("  copied", dst)
        else: print("  MISSING", src)
    print("done:", OUT)
