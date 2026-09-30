#!/usr/bin/env python3
"""Training-time diagnostics: WHY two objectives end up where they do.

    python analysis/a19_training_dynamics.py

WHAT THIS ANSWERS THAT NOTHING ELSE DOES
----------------------------------------
Every other script in this layer reads the END of a run: pooled test
predictions, final metrics, calibration of the selected checkpoint. They can
say that FA-FL landed at a different ECE from focal loss. They cannot say when
the two diverged, or which class drove it, or whether the difference was
present from the first epoch or appeared only after the head unfroze.

The FA-FL claim is specifically about WHERE THE LOSS SITS ACROSS CLASSES. That
is a training-time quantity. src/diagnostics.py now records it every epoch, per
class, for train and val, mirrored from src/losses.py and checked against the
criterion's own scalar on the first batch of every epoch. This script turns
those columns into figures.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No significance verdicts. Every quantity below is descriptive, and the figure
companions say so in as many words. The registered comparisons for this cohort
live in the age60_* families and are computed by the bootstrap scripts from
test predictions, where the unit of analysis is the participant. Training
curves are computed on images inside one fold's training split; attaching a
p-value to them would be a category error.

A HARD PRECONDITION
-------------------
If any run's recorded mirror drift exceeds tolerance, the per-class loss
numbers in that run are not trustworthy, and this script refuses to plot them
rather than producing a figure that looks fine and is wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import CLASS_NAMES, SHORT, save_table          # noqa: E402
from analysis._figure_md import Finding, Provenance, emit_figure     # noqa: E402
from analysis._paths import OUT                                      # noqa: E402
from analysis.discover import load_runs                              # noqa: E402

EXPERIMENTS = ["exp10_age60_main", "exp11_age60_gamma137", "exp12_age60_gamma200",
               "exp13_age60_gamma269"]
MIRROR_TOL = 1e-3          # generous: the integration test showed max 1.9e-06
SOURCE = "analysis/a19_training_dynamics.py"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _arm_label(manifest: Dict) -> str:
    loss = manifest.get("loss", {})
    t = loss.get("type")
    if t == "FocalLoss":
        g = loss.get("params", {}).get("gamma")
        return f"Focal g={float(g):.2f}" if g is not None else "Focal"
    return {"WCE": "Weighted CE", "LDAM": "LDAM", "ClassBalanced": "Class-Balanced",
            "FA_FL": "FA-FL"}.get(t, str(t))


def load_dynamics() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Returns (per-epoch frame, per-run frame, meta).

    The per-epoch frame is one row per (arm, fold, epoch) with every diagnostic
    column. The per-run frame is one row per run with the stored effective
    gamma, the selected epoch and the mirror-drift maximum.
    """
    frames: List[pd.DataFrame] = []
    runrows: List[Dict] = []
    fingerprints, cohorts = set(), set()

    runs = pd.concat([load_runs(e) for e in EXPERIMENTS if not load_runs(e).empty],
                     ignore_index=True) if any(not load_runs(e).empty for e in EXPERIMENTS) \
        else pd.DataFrame()
    if runs.empty:
        raise SystemExit(
            "no COMPLETE runs found for " + ", ".join(EXPERIMENTS) + ".\n"
            "These are the train-matched age-60 arms. Launch them with:\n"
            "    run_all.bat\n"
            "and come back when they finish.")

    for _, r in runs.iterrows():
        run_dir = Path(r["run_dir"])
        m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        curve = run_dir / "training_curve.csv"
        if not curve.is_file():
            print(f"  [warn] {r['run_id']}: no training_curve.csv, skipped")
            continue
        df = pd.read_csv(curve)
        if "tr_lossshare_c0" not in df.columns:
            print(f"  [warn] {r['run_id']}: no per-class diagnostics in the curve "
                  f"(train.diagnostics was off), skipped")
            continue

        drift = float(df["mirror_abs_diff"].max()) if "mirror_abs_diff" in df.columns else np.nan
        arm = _arm_label(m)
        df = df.assign(arm=arm, fold=r["fold"], run_id=r["run_id"])
        frames.append(df)
        fingerprints.add(m.get("code_fingerprint"))
        coh = m.get("cohort", {}) or {}
        cohorts.add(json.dumps({"filter": coh.get("filter", "none"),
                                "participants": coh.get("participants_after"),
                                "images": coh.get("images_after")}))

        crit = m.get("criterion", {}) or {}
        # 'selected' is object dtype: True/False on fine-tuning rows, NaN on warm-up
        # rows, which have no checkpoint decision. .eq(True) treats NaN as False
        # without the dtype downcast that .fillna() warns about.
        sel_col = df["selected"] if "selected" in df.columns else pd.Series(False, index=df.index)
        sel = df.index[sel_col.eq(True)]
        sel_epoch = int(df.loc[sel[-1], "epoch"]) if len(sel) else int(m.get("training", {})
                                                                      .get("best_epoch") or -1)
        va_ece = df.set_index("epoch")["va_ece"] if "va_ece" in df.columns else None
        runrows.append({
            "arm": arm, "fold": int(r["fold"]), "run_id": r["run_id"],
            "effective_gamma": crit.get("effective_gamma"),
            "effective_gamma_participant": crit.get("effective_gamma_participant_weighted"),
            "gamma_per_class": crit.get("gamma_per_class"),
            "train_class_counts": m.get("train_class_counts"),
            "selected_epoch": sel_epoch,
            "epochs_run": int(df["epoch"].max()),
            "val_ece_at_selected": float(va_ece.get(sel_epoch, np.nan)) if va_ece is not None else np.nan,
            "val_ece_min": float(va_ece.min()) if va_ece is not None else np.nan,
            "val_ece_argmin_epoch": int(va_ece.idxmin()) if va_ece is not None else -1,
            "mirror_abs_diff_max": drift,
            "test_ece": r.get("test_ece"), "test_macro_f1": r.get("test_macro_f1"),
            "test_accuracy": r.get("test_accuracy"),
        })

    if not frames:
        raise SystemExit("runs exist but none carry per-class diagnostics. "
                         "Set train.diagnostics: true and re-run.")

    epochs = pd.concat(frames, ignore_index=True)
    per_run = pd.DataFrame(runrows).sort_values(["arm", "fold"]).reset_index(drop=True)

    bad = per_run[per_run["mirror_abs_diff_max"] > MIRROR_TOL]
    if len(bad):
        raise SystemExit(
            f"{len(bad)} run(s) recorded a per-sample mirror drift above {MIRROR_TOL:g}:\n"
            + bad[["run_id", "mirror_abs_diff_max"]].to_string(index=False)
            + "\nsrc/diagnostics.py has drifted from src/losses.py for those criteria. Every "
              "per-class loss number in those runs is suspect, so nothing is plotted. Fix the "
              "mirror first.")

    # Every run in this set must be on the same cohort; pooling training curves
    # across different cohorts would compare optimisation on different data.
    if len(cohorts) > 1:
        raise SystemExit("these runs span more than one cohort definition:\n  "
                         + "\n  ".join(sorted(cohorts))
                         + "\nTraining dynamics are only comparable within one cohort.")
    coh0 = json.loads(sorted(cohorts)[0]) if cohorts else {}

    meta = {"fingerprints": sorted(f for f in fingerprints if f),
            "cohort_filter": coh0.get("filter", "none"),
            "n_participants": coh0.get("participants"),
            "n_images": coh0.get("images"),
            "n_runs": len(per_run),
            "arms": sorted(per_run["arm"].unique()),
            "folds": sorted(per_run["fold"].unique())}
    return epochs, per_run, meta


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _band(ax, sub: pd.DataFrame, col: str, label: str, color) -> None:
    """Mean over folds with a min-max band. Five folds is too few for a
    standard error to mean anything, so the full range is drawn instead."""
    g = sub.groupby("epoch")[col]
    mean, lo, hi = g.mean(), g.min(), g.max()
    ax.plot(mean.index, mean.values, label=label, color=color, lw=1.8)
    ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.15, lw=0)


def _arm_colors(arms: List[str]) -> Dict[str, object]:
    cmap = plt.get_cmap("tab10")
    return {a: cmap(i % 10) for i, a in enumerate(arms)}


def _warmup_marker(ax, epochs: pd.DataFrame) -> Optional[int]:
    """Where warm-up ends and full fine-tuning begins. Divergences that appear
    only after this line are a different phenomenon from ones present before it."""
    if "phase" not in epochs.columns:
        return None
    w = epochs[epochs["phase"] == "warmup"]["epoch"]
    if w.empty:
        return None
    boundary = int(w.max()) + 0.5
    ax.axvline(boundary, color="0.4", ls=":", lw=1)
    return boundary


def _per_class_panel(epochs, arms, colors, col_tpl, title, ylabel, stem, shows,
                     prov, findings, not_shown, extra_caveats=()):
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(5.2 * len(CLASS_NAMES), 4.0),
                             sharex=True)
    for c, ax in enumerate(axes):
        for a in arms:
            sub = epochs[epochs["arm"] == a]
            col = col_tpl.format(c=c)
            if col not in sub.columns:
                continue
            _band(ax, sub, col, a, colors[a])
        _warmup_marker(ax, epochs)
        ax.set_title(f"{SHORT[c]}  ({CLASS_NAMES[c]})", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    png, md = emit_figure(stem, fig, title=title, shows=shows, provenance=prov,
                          findings=findings, not_shown=not_shown,
                          extra_caveats=extra_caveats)
    plt.close(fig)
    return png, md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRAINING_CURVE_CAVEATS = (
    "Every quantity on this figure is computed on IMAGES inside one fold's training or "
    "validation split. It is a description of the optimisation, not an estimate of "
    "generalisation, and it carries no interval and no significance verdict.",
    "Validation here is the inner validation split used for checkpoint selection, not the "
    "held-out test fold. Reading it as a performance estimate would be selection on the "
    "same data.",
    "Bands show the full range across the five folds rather than a standard error. With "
    "five folds a standard error is an unstable estimate and a band drawn from one invites "
    "the reader to treat non-overlap as a test.",
)


def main() -> int:
    epochs, per_run, meta = load_dynamics()
    arms = meta["arms"]
    colors = _arm_colors(arms)

    if len(meta["fingerprints"]) > 1:
        raise SystemExit(
            f"these runs span {len(meta['fingerprints'])} code fingerprints "
            f"{[f[:8] for f in meta['fingerprints']]}. Training dynamics compared across "
            f"different code are not comparable. Re-run the odd ones out.")

    prov = Provenance(
        run_ids=list(per_run["run_id"]),
        code_fingerprint=meta["fingerprints"][0] if meta["fingerprints"] else "",
        # The cohort label drives the automatic caveats, so it is read from the
        # manifests rather than assumed from the experiment names.
        cohort="age60" if meta["cohort_filter"] == "cdr_age" else "full",
        unit="image",
        n_participants=meta["n_participants"], n_images=meta["n_images"],
        source_script=SOURCE)

    print(f"a19  {meta['n_runs']} run(s), {len(arms)} arm(s): {', '.join(arms)}")
    print(f"     mirror drift max over all runs: {per_run['mirror_abs_diff_max'].max():.2e} "
          f"(tolerance {MIRROR_TOL:g})")

    # -- 1. the mechanism measurement ---------------------------------------
    share_findings = []
    last = epochs[epochs["epoch"] == epochs["epoch"].max()]
    for a in arms:
        s = last[last["arm"] == a]
        for c in range(len(CLASS_NAMES)):
            col = f"tr_lossshare_c{c}"
            if col in s.columns and len(s):
                share_findings.append(Finding(
                    label=f"{a}: final-epoch training loss share, {SHORT[c]}",
                    value=float(s[col].mean()), fmt="{:.4f}"))
    _per_class_panel(
        epochs, arms, colors, "tr_lossshare_c{c}",
        "Share of total training loss carried by each class",
        "share of total training loss", "f19a_train_loss_share",
        shows=("Per-class share of the total training loss, by epoch. Each line is the mean "
               "over five grouped folds and the band is the full fold range. The dotted "
               "vertical line marks the end of warm-up, where the backbone unfreezes. The "
               "three panels sum to 1 at every epoch by construction."),
        prov=prov, findings=share_findings,
        not_shown=("Whether a difference in loss share causes a difference in test "
                   "performance. This figure shows where the objective put its weight, not "
                   "what that weight bought.",
                   "Gradient magnitude. Loss share and gradient share coincide only when the "
                   "per-sample gradient is proportional to the per-sample loss, which is not "
                   "generally true for focal-type objectives."),
        extra_caveats=TRAINING_CURVE_CAVEATS + (
            "This is the measurement the FA-FL mechanism claim is about: the claim is that "
            "adapting gamma per class moves loss onto the rare classes in a way a single "
            "gamma cannot. Read the FA-FL curve against the three constant-gamma focal "
            "curves, not against the weighted-CE curve.",))

    # -- 2. calibration over time -------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for a in arms:
        _band(ax, epochs[epochs["arm"] == a], "va_ece", a, colors[a])
    _warmup_marker(ax, epochs)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation ECE (15 bins)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("Validation calibration error during training")
    fig.tight_layout()
    ece_findings = [Finding(label=f"{a}: epoch of minimum validation ECE",
                            value=float(per_run[per_run['arm'] == a]['val_ece_argmin_epoch'].mean()),
                            fmt="{:.1f}") for a in arms]
    ece_findings += [Finding(label=f"{a}: mean gap between selected epoch and best-ECE epoch",
                             value=float((per_run[per_run['arm'] == a]['selected_epoch'] -
                                          per_run[per_run['arm'] == a]['val_ece_argmin_epoch']).mean()),
                             fmt="{:+.1f}") for a in arms]
    emit_figure("f19b_val_ece_by_epoch", fig,
                title="Validation calibration error during training",
                shows=("Expected calibration error on the inner validation split, 15 equal-width "
                       "confidence bins, by epoch. Mean over five folds with the fold range as a "
                       "band."),
                provenance=prov, findings=ece_findings,
                not_shown=("Test-set calibration. The checkpoint is selected on this curve, so "
                           "its minimum is optimistic by construction.",),
                extra_caveats=TRAINING_CURVE_CAVEATS + (
                    "A non-zero gap between the selected epoch and the epoch of lowest "
                    "validation ECE means the checkpoint metric and calibration disagree. That "
                    "is a property of the selection rule, which is identical across arms, so it "
                    "does not favour any one of them.",))
    plt.close(fig)

    # -- 3. per-class behaviour ---------------------------------------------
    _per_class_panel(
        epochs, arms, colors, "va_recall_c{c}",
        "Per-class recall on the validation split", "recall", "f19c_val_recall_by_class",
        shows=("Per-class recall on the inner validation split by epoch, mean over five folds "
               "with the fold range as a band."),
        prov=prov, findings=[],
        not_shown=("Test recall. This is the split the checkpoint is chosen on.",),
        extra_caveats=TRAINING_CURVE_CAVEATS)

    _per_class_panel(
        epochs, arms, colors, "va_trueprob_c{c}",
        "Mean probability assigned to the true class", "mean p(true class)",
        "f19d_val_true_prob_by_class",
        shows=("Mean probability the model assigns to the correct class, per class, on the "
               "inner validation split, by epoch."),
        prov=prov, findings=[],
        not_shown=("Whether the probabilities are calibrated. A rising true-class probability "
                   "with rising ECE means the model is getting more confident faster than it "
                   "is getting more correct.",),
        extra_caveats=TRAINING_CURVE_CAVEATS)

    _per_class_panel(
        epochs, arms, colors, "va_predshare_c{c}",
        "Share of validation predictions falling in each class", "share of predictions",
        "f19e_val_prediction_share",
        shows=("Fraction of all validation predictions that land in each class, by epoch. The "
               "three panels sum to 1 at every epoch."),
        prov=prov, findings=[],
        not_shown=("Accuracy. A model can match the class prior exactly and still be wrong "
                   "about every individual.",),
        extra_caveats=TRAINING_CURVE_CAVEATS + (
            "This is the collapse detector. An arm whose prediction share converges on the "
            "majority class has stopped separating classes, and its accuracy from that epoch "
            "onward is the class prior rather than a learned signal.",))

    # -- 4. the realized focusing exponent ----------------------------------
    have_gamma = per_run.dropna(subset=["effective_gamma"])
    if len(have_gamma):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        order = sorted(have_gamma["arm"].unique())
        for i, a in enumerate(order):
            v = have_gamma[have_gamma["arm"] == a]["effective_gamma"].to_numpy()
            ax.scatter(np.full_like(v, i, dtype=float), v, s=42, color=colors[a], zorder=3)
            ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color="0.2", lw=2, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.set_ylabel("frequency-weighted effective gamma")
        ax.grid(alpha=0.25, axis="y")
        ax.set_title("Realized focusing exponent, computed from each run's own training fold")
        fig.tight_layout()
        g_findings = [Finding(label=f"{a}: effective gamma (image-weighted, mean over folds)",
                              value=float(have_gamma[have_gamma["arm"] == a]["effective_gamma"].mean()),
                              fmt="{:.4f}") for a in order]
        g_findings += [Finding(
            label=f"{a}: effective gamma (participant-weighted, mean over folds)",
            value=float(have_gamma[have_gamma["arm"] == a]["effective_gamma_participant"].mean()),
            fmt="{:.4f}")
            for a in order
            if have_gamma[have_gamma["arm"] == a]["effective_gamma_participant"].notna().any()]
        emit_figure("f19f_effective_gamma", fig,
                    title="Realized focusing exponent per arm",
                    shows=("One point per run: the frequency-weighted mean of the per-class "
                           "focusing exponent over that run's own training fold, read from the "
                           "run manifest. The bar is the mean over folds. For a constant-gamma "
                           "focal arm this is that gamma by definition, so those arms appear as "
                           "flat reference levels."),
                    provenance=prov, findings=g_findings,
                    not_shown=("That FA-FL and a constant-gamma focal arm at the same effective "
                               "value behave the same. The effective gamma is a weighted mean, "
                               "and two objectives with the same mean can differ everywhere. "
                               "Whether they behave the same is what the age60_gamma_control_* "
                               "families test, from test predictions, at the participant level.",),
                    extra_caveats=(
                        "The effective gamma is a summary statistic of the objective, not a "
                        "result. It is reported here so the reader can see that restricting the "
                        "cohort changes it, and by how much, without taking that as evidence "
                        "for or against the mechanism claim.",
                        "Image-weighted and participant-weighted values are both reported. They "
                        "differ only if participants contribute unequal numbers of slices.",))
        plt.close(fig)

    # -- 5. the run-level table ---------------------------------------------
    tbl = per_run.drop(columns=["gamma_per_class", "train_class_counts"])
    save_table(tbl, "t19_training_dynamics_per_run")
    print(f"     wrote {OUT / 't19_training_dynamics_per_run.csv'}")
    print(f"     figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
