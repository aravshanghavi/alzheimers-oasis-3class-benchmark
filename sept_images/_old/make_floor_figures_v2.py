#!/usr/bin/env python3
"""Floors figure and per-class null figure, read directly from the current audit table.

Source: 2026-09_instrumented_arms/analysis/outputs/t23_protocol_audit_long.csv
(generated 29 Sep 2026 with exp15 included: 7 full-cohort arms, 9 restricted-cohort arms).
  R1, participant, accuracy  : arm minus constant majority predictor (percentage points)
  R4, participant, macro_f1  : arm minus META age+sex+eTIV+nWBV
  R2, participant, f1 per class: arm F1 against the permuted-label null (5 replicates)
Writes PNG and PDF next to this script's parent folder. Reads only; modifies nothing.
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
T23 = ROOT / "2026-09_instrumented_arms" / "analysis" / "outputs" / "t23_protocol_audit_long.csv"
INK, MUTED, GRID, S1, S2 = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
LABEL = {"Weighted CE": "WCE", "LDAM": "LDAM", "Class-Balanced": "CB",
         "Focal g=1.37": "FL (γ=1.37)", "Focal g=2.00": "FL (γ=2.0)", "Focal g=2.69": "FL (γ=2.69)",
         "Focal g=3.00": "FL (γ=3.0)", "FA-FL": "FA-FL", "Class-Balanced [exp15]": "Unweighted ref."}
ORDER = list(LABEL)

rows = list(csv.DictReader(open(T23, encoding="utf-8")))
def pick(req, cohort, metric, reference=None):
    out = {}
    for r in rows:
        if r["requirement"] != req or r["cohort"] != cohort or r["unit"] != "participant" or r["metric"] != metric:
            continue
        if reference and r["reference"] != reference:
            continue
        out[r["arm"]] = r
    return out

# ---------------------------------------------------------------- floors figure
panels = []
for cohort, name in (("full", "Full cohort, 347 participants"), ("restricted", "Restricted cohort, 166 participants")):
    acc = pick("R1", cohort, "accuracy")
    f1 = pick("R4", cohort, "macro_f1", "META age+sex+eTIV+nWBV")
    panels.append((name, acc, f1))

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw={"height_ratios": [7, 9]})
letters = iter("abcd")
for r, (name, acc, f1) in enumerate(panels):
    arms = [a for a in ORDER if a in acc][::-1]
    y = np.arange(len(arms))
    for c, (d, scale, xl) in enumerate(((acc, 1.0, "Accuracy minus constant predictor (percentage points)"),
                                        (f1, 1.0, "Macro-F1 minus age + sex + eTIV + nWBV model"))):
        ax = axes[r, c]
        v = np.array([float(d[a]["difference"]) for a in arms]) * scale
        lo = np.array([float(d[a]["ci_low"]) for a in arms]) * scale
        hi = np.array([float(d[a]["ci_high"]) for a in arms]) * scale
        excl = (lo > 0) | (hi < 0)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 2)))
        for yi, a, b, vv, e in zip(y, lo, hi, v, excl):
            col = S2 if e else S1
            ax.hlines(yi, a, b, color=col, lw=2, capstyle="round")
            ax.plot(vv, yi, "o", ms=5, color=col, mec="white", mew=1)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in arms] if c == 0 else [])
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if r == 1: ax.set_xlabel(xl)
        ax.set_title(f"{next(letters)}   {name}", loc="left", fontsize=8, color=INK)
    axes[r, 0].set_xlim(-8, 19); axes[r, 1].set_xlim(-0.28, 0.30)
fig.text(0.01, -0.01, "Points: difference; bars: 95% participant-clustered bootstrap interval. "
         "Orange: interval excludes zero.", fontsize=7, color=MUTED)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_floors.{ext}")
plt.close(fig)

# ------------------------------------------------ per-class null figure (restricted)
metrics = [("f1_NonDem", "CDR 0"), ("f1_VeryMild", "CDR 0.5"), ("f1_Dem", "CDR ≥ 1")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, (m, title) in zip(axes, metrics):
    d = pick("R2", "restricted", m)
    arms = [a for a in ORDER if a in d][::-1]
    y = np.arange(len(arms))
    r0 = d[arms[0]]
    nmin, nmax, nsd = float(r0["null_min"]), float(r0["null_max"]), float(r0["null_sd"])
    nmean = float(r0["arm_value"]) - float(r0["z_against_null_sd"]) * nsd
    ax.axvspan(nmin, nmax, color=GRID, zorder=0)
    ax.axvline(nmean, color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.axvline(nmean + 2.5 * nsd, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    for yi, a in zip(y, arms):
        v, z = float(d[a]["arm_value"]), float(d[a]["z_against_null_sd"])
        col = S2 if z > 2.5 else S1
        ax.plot(v, yi, "o", ms=5.5, color=col, mec="white", mew=1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in arms])
    ax.set_title(title, fontsize=8.5, color=INK); ax.set_xlabel("Participant-level F1")
    ax.set_xlim(0, 0.8); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.01, -0.03, "Shaded: range of 5 permuted-label replicates; dashed: null mean; dotted: null mean + 2.5 null SD. "
         "Orange: more than 2.5 null SD above the null mean.", fontsize=7, color=MUTED)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_null_perclass.{ext}")
plt.close(fig)
print("full arms", len(panels[0][1]), "restricted arms", len(panels[1][1]))
for name, acc, f1 in panels:
    for a in ORDER:
        if a in acc:
            print(name[:10], LABEL[a].ljust(16), "acc", acc[a]["difference"][:6], acc[a]["ci_low"][:6], acc[a]["ci_high"][:6], "| f1", f1[a]["difference"][:6], f1[a]["ci_low"][:6], f1[a]["ci_high"][:6])
