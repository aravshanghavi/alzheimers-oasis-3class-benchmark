#!/usr/bin/env python3
"""Calibration figures: reliability, per-class curves, fold spread, temperature effect.

Answers Reviewer 1 comment 5, which asked for reliability diagrams, per-class
calibration curves and seed-wise calibration variability -- none of which the
submitted manuscript provided.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from analysis._common import (CLASS_NAMES, ECE_BINS, LOSS_LABEL, LOSS_ORDER,  # noqa: E402
                              N_CLASSES, OUT, SuffStats, fold_metrics,
                              pooled_predictions, runs_frame, save_table)
from src.figures import GRID, INK, INK_MUTED, SERIES, SURFACE, _style, _save  # noqa: E402

# Five losses need five hues; SERIES supplies three, so extend with two more from
# the same validated ramp (yellow, violet) in fixed order -- never cycled.
HUES = SERIES + ["#eda100", "#4a3aa7"]


def reliability_curve(probs, labels, n_bins=ECE_BINS):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main() -> int:
    print("[a02] calibration figures")
    runs = runs_frame()
    fm = fold_metrics(runs)
    losses = [L for L in LOSS_ORDER if not fm[fm["loss"] == L].empty]

    pooled = {L: pooled_predictions(runs, L) for L in losses}
    P = {L: pooled[L][[f"p{c}" for c in range(N_CLASSES)]].to_numpy() for L in losses}
    Y = {L: pooled[L]["label"].to_numpy() for L in losses}
    ece = {L: SuffStats(pooled[L]).metrics()["ece"] for L in losses}

    # ---- pooled reliability, all losses on one axis --------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)), label="perfect calibration")
    for i, L in enumerate(losses):
        xs, ys, _ = reliability_curve(P[L], Y[L])
        ax.plot(xs, ys, color=HUES[i], lw=2, marker="o", ms=4.5,
                label=f"{LOSS_LABEL[L]} (ECE {ece[L]:.4f})")
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability, pooled over all participants\n"
                 f"({ECE_BINS} equal-width bins; bins with <20 images omitted)",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    _style(ax); _save(fig, OUT / "fig_reliability_pooled.png")
    print("    -> fig_reliability_pooled.png")

    # ---- per-class one-vs-rest calibration -----------------------------------
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4.4 * N_CLASSES, 4.2), sharey=True)
    edges = np.linspace(0, 1, ECE_BINS + 1)
    for c, ax in enumerate(np.atleast_1d(axes)):
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)))
        for i, L in enumerate(losses):
            y = (Y[L] == c).astype(float); p = P[L][:, c]
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (p > lo) & (p <= hi)
                if m.sum() >= 20:
                    xs.append(p[m].mean()); ys.append(y[m].mean())
            ax.plot(xs, ys, color=HUES[i], lw=2, marker="o", ms=4, label=LOSS_LABEL[L])
        ax.axhline((Y[losses[0]] == c).mean(), color=INK_MUTED, lw=1, ls=(0, (2, 3)), alpha=0.6)
        ax.set_title(f"{CLASS_NAMES[c]}\n(prevalence {(Y[losses[0]] == c).mean():.3f})",
                     fontsize=9, loc="left")
        ax.set_xlabel(f"predicted P({CLASS_NAMES[c]})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("observed frequency")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    fig.suptitle("Per-class calibration, one-vs-rest, pooled over all participants",
                 fontsize=10, x=0.02, ha="left", color=INK)
    _save(fig, OUT / "fig_calibration_perclass_pooled.png")
    print("    -> fig_calibration_perclass_pooled.png")

    # ---- fold-wise ECE spread ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, L in enumerate(losses):
        g = fm[fm["loss"] == L]
        ax.scatter(np.full(len(g), i) + np.linspace(-0.09, 0.09, len(g)), g["ece"],
                   color=HUES[i], s=42, zorder=3, label=None)
        ax.plot([i - 0.24, i + 0.24], [g["ece"].mean()] * 2, color=HUES[i], lw=2.5)
        ax.scatter([i], [fm[fm["loss"] == L]["ece_temp_scaled"].mean()], marker="_",
                   s=420, color=INK_MUTED, zorder=4)
    ax.set_xticks(range(len(losses)), [LOSS_LABEL[L] for L in losses], fontsize=8)
    ax.set_ylabel("ECE"); ax.set_ylim(bottom=0)
    ax.set_title("ECE across the five folds\n"
                 "coloured bar = fold mean, grey dash = mean after temperature scaling",
                 fontsize=10, loc="left")
    _style(ax); _save(fig, OUT / "fig_ece_by_fold.png")
    print("    -> fig_ece_by_fold.png")

    # ---- temperature scaling before/after ------------------------------------
    rows = []
    for L in losses:
        try:
            ts = pooled_predictions(runs, L, temperature_scaled=True)
        except SystemExit:
            continue
        m_ts = SuffStats(ts).metrics()
        rows.append({"Loss": LOSS_LABEL[L], "ECE": ece[L], "ECE post-T": m_ts["ece"],
                     "Brier": SuffStats(pooled[L]).metrics()["brier"],
                     "Brier post-T": m_ts["brier"],
                     "Mean T": fm[fm["loss"] == L]["temperature"].mean(),
                     "Accuracy (%)": SuffStats(pooled[L]).metrics()["accuracy"],
                     "Accuracy post-T (%)": m_ts["accuracy"]})
    if rows:
        t = pd.DataFrame(rows)
        save_table(t, "temperature_scaling_effect")
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        x = np.arange(len(t))
        ax.bar(x - 0.19, t["ECE"], width=0.36, color=SERIES[0], label="as trained")
        ax.bar(x + 0.19, t["ECE post-T"], width=0.36, color=SERIES[1],
               label="after temperature scaling")
        ax.set_xticks(x, list(t["Loss"]), fontsize=8)
        ax.set_ylabel("ECE (pooled)")
        ax.set_title("A single post-hoc temperature reorders the calibration ranking",
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
                  loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
        _style(ax); _save(fig, OUT / "fig_temperature_effect.png")
        print("    -> fig_temperature_effect.png")
        (OUT / "temperature_note.txt").write_text(
            "Temperature is a single scalar fitted by minimising NLL on each fold's "
            "VALIDATION split and applied to that fold's test split; accuracy is unchanged "
            "by construction. Reporting calibration only as trained would leave the "
            "obvious objection unanswered -- that a one-parameter post-hoc fit is nearly "
            "free -- so both are given. Where the ordering changes after scaling, the "
            "defensible claim narrows to calibration WITHOUT a held-out fit, and the "
            "manuscript should say so plainly.\n", encoding="utf-8")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
