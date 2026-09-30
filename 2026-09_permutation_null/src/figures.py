"""Per-run diagnostic figures.

Scope, deliberately: these are for eyeballing a single run without writing any
code. The manuscript's figures are NOT these -- they come from the analysis layer,
pooled across folds, because a figure from one fold overstates what one fold can
support. Everything here is reconstructible from predictions_*.npz at any time.

Design follows the project viz rules: one categorical hue per class assigned in
fixed order (never cycled), a single-hue light-to-dark ramp for magnitude, no
dual-axis plots, a legend whenever two or more series are present, recessive
grid and axes. The categorical triple below passed the palette validator for the
light surface (worst adjacent CVD dE 9.2 deutan, normal-vision dE 27.6).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")                      # no display on a headless training box
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_curve)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua -- fixed order
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e2e1dd"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#2a78d6", "#123a68"])

DPI = 140


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def _save(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def confusion_matrix_figure(cm: np.ndarray, class_names: Sequence[str], title: str,
                            out: Path) -> None:
    """Magnitude -> one hue, light to dark. Counts and row-normalised percentage
    are both printed, so identity never depends on reading a colour."""
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, np.maximum(row_sums, 1), out=np.zeros_like(cm)) * 100.0

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.imshow(pct, cmap=SEQ, vmin=0, vmax=100)
    n = len(class_names)
    ax.set_xticks(range(n), [c.replace(" ", "\n") for c in class_names], fontsize=8)
    ax.set_yticks(range(n), [c.replace(" ", "\n") for c in class_names], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10, pad=10)
    for i in range(n):
        for j in range(n):
            ink = "#ffffff" if pct[i, j] > 55 else INK
            ax.text(j, i, f"{int(cm[i, j]):,}\n{pct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=8.5, color=ink, linespacing=1.35)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)
    _save(fig, out)


def training_curve_figure(history: Sequence[Dict], out: Path) -> None:
    """Loss and accuracy are different scales, so they get two stacked axes --
    never a dual y-axis on one plot."""
    if not history:
        return
    ep = [h["epoch"] for h in history]
    n_warm = sum(1 for h in history if h["phase"] == "warmup")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                                   gridspec_kw={"hspace": 0.18})
    ax1.plot(ep, [h["train_loss"] for h in history], color=SERIES[0], lw=2, label="train")
    ax1.plot(ep, [h["val_loss"] for h in history], color=SERIES[1], lw=2, label="validation")
    ax1.set_ylabel("loss"); ax1.set_title("Training curve", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper right")

    ax2.plot(ep, [h["train_acc"] for h in history], color=SERIES[0], lw=2, label="train")
    ax2.plot(ep, [h["val_acc"] for h in history], color=SERIES[1], lw=2, label="validation")
    ax2.set_ylabel("accuracy (%)"); ax2.set_xlabel("epoch")
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower right")

    for ax in (ax1, ax2):
        _style(ax)
        if 0 < n_warm < len(ep):
            ax.axvline(ep[n_warm - 1] + 0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), alpha=0.7)
    ax1.text(ep[0], ax1.get_ylim()[1], " warm-up | fine-tuning", va="top", ha="left",
             fontsize=7.5, color=INK_MUTED)
    _save(fig, out)


def roc_pr_figure(probs: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
                  out_roc: Path, out_pr: Path) -> None:
    """One-vs-rest per class. Class identity is carried by the fixed hue order and
    repeated in the legend, so it never rests on colour alone."""
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    for i in range(n):
        y = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        ax.plot(fpr, tpr, color=SERIES[i], lw=2,
                label=f"{class_names[i]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)), label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC, one-vs-rest", fontsize=10, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    _style(ax); _save(fig, out_roc)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    for i in range(n):
        y = (labels == i).astype(int)
        prec, rec, _ = precision_recall_curve(y, probs[:, i])
        ax.plot(rec, prec, color=SERIES[i], lw=2,
                label=f"{class_names[i]} (AP {average_precision_score(y, probs[:, i]):.3f})")
        ax.axhline(y.mean(), color=SERIES[i], lw=1, ls=(0, (2, 3)), alpha=0.55)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall, one-vs-rest\n(dotted = class prevalence, the no-skill floor)",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    _style(ax); _save(fig, out_pr)


def reliability_figure(probs: np.ndarray, labels: np.ndarray, n_bins: int,
                       ece: float, out: Path) -> None:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            centres.append(conf[m].mean()); acc.append(correct[m].mean()); counts.append(int(m.sum()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    ax1.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)), label="perfect calibration")
    ax1.plot(centres, acc, color=SERIES[0], lw=2, marker="o", ms=5, label="observed")
    ax1.set_ylabel("empirical accuracy")
    ax1.set_title(f"Reliability ({n_bins} equal-width bins) — ECE {ece:.4f}", fontsize=10, loc="left")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    ax2.bar(centres, counts, width=1.0 / n_bins * 0.85, color=SERIES[0], alpha=0.55)
    ax2.set_ylabel("images"); ax2.set_xlabel("mean predicted confidence")
    for ax in (ax1, ax2):
        _style(ax)
    _save(fig, out)


def perclass_calibration_figure(probs: np.ndarray, labels: np.ndarray,
                               class_names: Sequence[str], n_bins: int, out: Path) -> None:
    """One-vs-rest reliability, one panel per class.

    Reviewer 1 asked for per-class calibration curves specifically. The top-label
    reliability diagram hides the thing that matters clinically: a model can be
    well calibrated overall while being badly overconfident on the rare class.
    """
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i, ax in enumerate(axes):
        y = (labels == i).astype(float)
        p = probs[:, i]
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p > lo) & (p <= hi)
            if m.sum() >= 5:
                xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)))
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=5)
        ax.axhline(y.mean(), color=INK_MUTED, lw=1, ls=(0, (2, 3)), alpha=0.6)
        ax.set_title(f"{class_names[i]}\n(prevalence {y.mean():.3f})", fontsize=9, loc="left")
        ax.set_xlabel("predicted P(class)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("Per-class calibration, one-vs-rest (bins with <5 images dropped)",
                 fontsize=10, x=0.02, ha="left", color=INK)
    _save(fig, out)


def risk_coverage_figure(probs: np.ndarray, labels: np.ndarray, out: Path) -> None:
    """Selective prediction: accuracy against the fraction of cases retained.

    This is the figure that makes the calibration argument concrete. If the model
    defers its least-confident cases to a clinician, how much does accuracy on the
    rest improve? A well-calibrated model climbs steeply; a miscalibrated one is
    flat, because its confidence does not rank its own errors.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(coverage * 100, cum_acc * 100, color=SERIES[0], lw=2, label="selective accuracy")
    ax.axhline(correct.mean() * 100, color=INK_MUTED, lw=1, ls=(0, (4, 3)),
               label=f"full coverage ({correct.mean() * 100:.2f}%)")
    for cov in (0.5, 0.8):
        k = max(int(cov * len(order)) - 1, 0)
        ax.plot([cov * 100], [cum_acc[k] * 100], marker="o", ms=7, color=SERIES[1], zorder=5)
        ax.annotate(f"{cum_acc[k] * 100:.1f}% at {int(cov * 100)}% coverage",
                    (cov * 100, cum_acc[k] * 100), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color=INK_MUTED)
    ax.set_xlabel("coverage: % of cases retained, most confident first")
    ax.set_ylabel("accuracy on retained (%)")
    ax.set_title("Risk-coverage — does confidence rank the model's own errors?",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower left")
    _style(ax); _save(fig, out)


def temperature_reliability_figure(before: np.ndarray, after: np.ndarray, labels: np.ndarray,
                                   n_bins: int, ece_before: float, ece_after: float,
                                   T: float, out: Path) -> None:
    """Reliability before and after temperature scaling, on one axis.

    If the gap between the two curves is small, the calibration advantage is
    intrinsic to the loss. If temperature scaling closes it, the honest claim is
    narrower: better calibrated out of the box, without a held-out fit.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def curve(p):
        conf, corr = p.max(axis=1), (p.argmax(axis=1) == labels).astype(float)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        return xs, ys

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)), label="perfect calibration")
    x0, y0 = curve(before)
    x1, y1 = curve(after)
    ax.plot(x0, y0, color=SERIES[0], lw=2, marker="o", ms=5, label=f"as trained (ECE {ece_before:.4f})")
    ax.plot(x1, y1, color=SERIES[1], lw=2, marker="s", ms=5,
            label=f"temperature scaled, T={T:.3f} (ECE {ece_after:.4f})")
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Effect of post-hoc temperature scaling (T fitted on validation)",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    _style(ax); _save(fig, out)


def write_run_figures(run_dir: Path, split: str, probs: np.ndarray, labels: np.ndarray,
                      cm, class_names: Sequence[str], ece: float, n_bins: int,
                      history: Sequence[Dict] | None = None,
                      temperature: Dict | None = None) -> list:
    """All per-run figures for one split. Never raises: a plotting failure must
    not lose a 45-minute training run."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    written = []
    try:
        confusion_matrix_figure(np.array(cm), class_names,
                                f"Confusion matrix ({split}) — count and row %",
                                fig_dir / f"confusion_matrix_{split}.png")
        roc_pr_figure(probs, labels, class_names,
                      fig_dir / f"roc_{split}.png", fig_dir / f"pr_{split}.png")
        reliability_figure(probs, labels, n_bins, ece, fig_dir / f"reliability_{split}.png")
        perclass_calibration_figure(probs, labels, class_names, n_bins,
                                    fig_dir / f"calibration_perclass_{split}.png")
        risk_coverage_figure(probs, labels, fig_dir / f"risk_coverage_{split}.png")
        if temperature is not None:
            temperature_reliability_figure(
                probs, temperature["probs_scaled"], labels, n_bins,
                ece, temperature["ece_after"], temperature["temperature"],
                fig_dir / f"temperature_scaling_{split}.png")
        if history:
            training_curve_figure(history, fig_dir / "training_curve.png")
        written = sorted(p.name for p in fig_dir.iterdir())
    except Exception as exc:                                   # noqa: BLE001
        (fig_dir / "FIGURES_FAILED.txt").write_text(repr(exc), encoding="utf-8")
    return written
