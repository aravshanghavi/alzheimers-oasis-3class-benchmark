#!/usr/bin/env python3
"""Redraw three figures whose earlier versions were mislabelled or unreadable.

1. fig_gamma_ladder: restricted cohort, IMAGE-level ECE (15 equal-width bins) for the
   constant-exponent focal ladder and FA-FL, pooled and mean-of-folds estimators, with
   fold-clustered 95% intervals. Source: t30_ece_estimator_robustness.csv (kind=arm_ece).
   (The earlier f20d_gamma_ladder.png labelled these image-level values "participant-level".)
2. fig_leakage_ablation: FA-FL, full cohort, session-level minus participant-level grouping.
   Source: 2026-09_tier1_verified_stack/analysis/outputs/leakage_ablation.csv (image level).
3. fig_multislice: FA-FL, full cohort, 2.5D (three adjacent slices) minus 2D.
   Source: 2026-09_tier1_verified_stack/analysis/outputs/multislice_comparison.csv (image level).
Reads only; writes PNG and PDF into the parent folder.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[1]
T30 = ROOT / "2026-09_instrumented_arms" / "analysis" / "outputs" / "t30_ece_estimator_robustness.csv"
T1 = ROOT / "2026-09_tier1_verified_stack" / "analysis" / "outputs"
INK, MUTED, GRID, S1, S2 = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

# ------------------------------------------------------------------ 1. gamma ladder
ece = {}
for r in csv.DictReader(open(T30, encoding="utf-8")):
    if (r["kind"] == "arm_ece" and r["cohort"] == "restricted" and r["unit"] == "image"
            and r["n_bins_requested"].startswith("15") and r["binning"] == "equal width"
            and r["bootstrap"].startswith("fold clustered")):
        ece[(r["arm"], r["estimator"])] = (float(r["ece"]), float(r["ci_low"]), float(r["ci_high"]))
ladder = [(1.37, "Focal g=1.37"), (2.00, "Focal g=2.00"), (2.69, "Focal g=2.69"), (3.00, "Focal g=3.00")]
GEFF = 2.727   # FA-FL image-weighted effective exponent, five-fold mean (project record, from run manifests)
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for est, off, mfc, lab in (("pooled", -0.018, S1, "pooled over folds"), ("mean of folds", 0.018, "white", "mean of per-fold values")):
    xs = [g + off for g, _ in ladder]
    v = [ece[(a, est)][0] for _, a in ladder]
    lo = [ece[(a, est)][1] for _, a in ladder]; hi = [ece[(a, est)][2] for _, a in ladder]
    ax.vlines(xs, lo, hi, color=S1, lw=1.4)
    ax.plot(xs, v, "o-", color=S1, mfc=mfc, mec=S1, ms=5.5, lw=1.2, label=f"Focal loss, {lab}")
    fv, flo, fhi = ece[("FA-FL", est)]
    ax.vlines(GEFF + off, flo, fhi, color=S2, lw=1.4)
    ax.plot(GEFF + off, fv, "D", color=S2, mfc=S2 if est == "pooled" else "white", mec=S2, ms=6,
            label=f"FA-FL at its effective exponent, {lab}")
ax.set_xticks([1.37, 2.0, 2.69, 3.0]); ax.set_xlabel("Focusing exponent $\\gamma$")
ax.set_ylabel("Image-level ECE (15 bins)")
ax.set_ylim(0, 0.16); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.set_title("Restricted cohort, before temperature scaling", loc="left", fontsize=8, color=INK)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_gamma_ladder.{ext}")
plt.close(fig)

# ------------------------------------------------- 2 and 3. paired difference plots
def diff_plot(path, diff_col, lo_col, hi_col, title, xlab, fname):
    rows = {r["Metric"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    left = [("Accuracy (%)", "Accuracy")]
    right = [("Macro-F1", "Macro-F1"), ("F1 Very Mild", "F1, CDR 0.5"), ("F1 Demented", "F1, CDR $\\geq$ 1"),
             ("ECE", "ECE"), ("Brier", "Brier score")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, items, xl in ((a1, left, "Difference (percentage points)"), (a2, right, "Difference")):
        y = np.arange(len(items))[::-1]
        for yi, (k, lab) in zip(y, items):
            r = rows[k]; v, lo, hi = float(r[diff_col]), float(r[lo_col]), float(r[hi_col])
            ax.hlines(yi, lo, hi, color=S1, lw=2, capstyle="round")
            ax.plot(v, yi, "o", ms=5, color=S1, mec="white", mew=1)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in items])
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(items) - 0.4)
    a1.set_title(title, loc="left", fontsize=8, color=INK)
    fig.text(0.01, -0.04, xlab, fontsize=7, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)

diff_plot(T1 / "leakage_ablation.csv", "Inflation (session - participant)", "CI low (diff)", "CI high (diff)",
          "FA-FL, full cohort: session-level minus participant-level grouping",
          "Image level. Points: paired difference; bars: 95% participant-clustered bootstrap interval.",
          "fig_leakage_ablation")
diff_plot(T1 / "multislice_comparison.csv", "Difference (2.5D - 2D)", "CI low (diff)", "CI high (diff)",
          "FA-FL, full cohort: three-slice (2.5D) minus single-slice (2D) input",
          "Image level. Points: paired difference; bars: 95% participant-clustered bootstrap interval.",
          "fig_multislice")
print("ok", {k: v for k, v in ece.items() if k[0] == "FA-FL"})
