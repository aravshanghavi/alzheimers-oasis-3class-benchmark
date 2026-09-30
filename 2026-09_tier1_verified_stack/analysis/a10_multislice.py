#!/usr/bin/env python3
"""exp04 (2.5D) vs exp01 (2D): does adjacent-slice context help?

Reviewers 1 (#3) and 3 (#1) independently criticised the use of single 2D slices.
exp04 packs slices i-1, i, i+1 into the three input channels instead of a
replicated grayscale slice, clamping at the edges of the 100-160 band. Everything
else -- architecture, loss, hyperparameters, split seed, folds, initialisation --
is held identical to exp01.

This is a PROPERLY PAIRED comparison, unlike a08: both arms use participant-level
grouping with the same split seed, so the folds contain the same people and each
participant is scored once under each protocol. The bootstrap therefore resamples
participants ONCE and recomputes both arms on that same resample, which is what
makes the difference interval meaningful.

Either outcome is reportable. A gain says adjacent-slice context carries
information the 2D model was discarding, and that the reviewers were right. No
gain is evidence that within-slice information dominates at this resolution --
which is a stronger answer to the criticism than acknowledging it as a limitation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from analysis._common import (OUT, SuffStats, clustered_bootstrap,   # noqa: E402
                              fold_metrics, paired_difference, pooled_predictions,
                              save_table, wilcoxon_signed_rank)
from analysis.discover import load_runs                              # noqa: E402
from src.figures import INK_MUTED, SERIES, _style, _save             # noqa: E402

N_BOOT = 2000
LOSS = "FA_FL"
A, B = "2.5D (exp04)", "2D (exp01)"

METRICS = [("accuracy", None, "Accuracy (%)"), ("macro_f1", None, "Macro-F1"),
           ("per_class_f1", 0, "F1 Non Demented"), ("per_class_f1", 1, "F1 Very Mild"),
           ("per_class_f1", 2, "F1 Demented"), ("ece", None, "ECE"), ("brier", None, "Brier")]


def one_run_per_fold(runs: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Keep exactly one run per fold, the newest, and say loudly what was dropped.

    A re-run after a failure leaves two COMPLETE folders for the same fold.
    pooled_predictions() does NOT catch that -- its guard asks whether a
    participant appears in more than one FOLD, and two runs of the same fold
    share a fold number -- so the duplicated fold would silently enter the pooled
    set twice and carry double weight. Nothing downstream would flag it.
    """
    dupes = runs.groupby("fold").size()
    dupes = dupes[dupes > 1]
    if dupes.empty:
        return runs
    keep = runs.sort_values("run_id").groupby("fold", as_index=False).tail(1)
    dropped = runs[~runs["run_id"].isin(keep["run_id"])]
    print(f"  [warn] {arm}: {len(dropped)} duplicate run(s) for fold(s) "
          f"{sorted(dupes.index.tolist())}; keeping the newest per fold and DROPPING:")
    for _, r in dropped.iterrows():
        print(f"           {Path(r['run_dir']).name}")
    return keep.sort_values("fold")


def main() -> int:
    print("[a10] 2.5D multi-slice vs 2D")
    d2 = load_runs("exp01_main_sweep")
    d25 = load_runs("exp04_multislice_25d")
    if d25.empty:
        print("  exp04_multislice_25d has no completed runs -- skipping")
        return 0
    d2 = d2[d2["loss"] == LOSS]
    if d2.empty:
        print(f"  no exp01 runs for {LOSS} -- skipping")
        return 0
    d25 = one_run_per_fold(d25, "2.5D (exp04)")
    d2 = one_run_per_fold(d2, "2D (exp01)")
    if len(d25) != len(d2):
        print(f"  [warn] {len(d25)} 2.5D runs against {len(d2)} 2D runs; "
              f"the comparison is only paired over the folds present in both")

    ctx = sorted(d25["context_slices"].dropna().unique().tolist())
    print(f"  2.5D runs report context_slices={ctx} (must be [3])")
    if ctx != [3]:
        raise SystemExit("exp04 runs do not all carry context_slices=3 -- refusing to compare")

    stats = {A: SuffStats(pooled_predictions(d25, LOSS)),
             B: SuffStats(pooled_predictions(d2, LOSS))}
    print(f"  {stats[A].n_p} participants, {int(stats[A].n_img.sum()):,} images per arm")

    rows = []
    for metric, cls, label in METRICS:
        boot = clustered_bootstrap(stats, metric, n_boot=N_BOOT, per_class=cls)
        diff = paired_difference(boot, A, B)
        rows.append({"Metric": label,
                     "2.5D": boot[A]["point"], "2.5D CI low": boot[A]["ci_low"],
                     "2.5D CI high": boot[A]["ci_high"],
                     "2D": boot[B]["point"], "2D CI low": boot[B]["ci_low"],
                     "2D CI high": boot[B]["ci_high"],
                     "Difference (2.5D - 2D)": diff["diff"],
                     "CI low (diff)": diff["ci_low"], "CI high (diff)": diff["ci_high"],
                     "p (bootstrap)": diff["p_bootstrap"],
                     "CI excludes 0": diff["excludes_zero"]})
    tab = pd.DataFrame(rows)
    save_table(tab, "multislice_comparison")

    # fold-wise paired view -- conservative companion, never the primary evidence
    f25 = fold_metrics(d25).drop_duplicates("fold").set_index("fold").sort_index()
    f2 = fold_metrics(d2).drop_duplicates("fold").set_index("fold").sort_index()
    shared = sorted(set(f25.index) & set(f2.index))
    frows = []
    for col, label in [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1"), ("ece", "ECE")]:
        if col not in f25.columns or col not in f2.columns:
            continue
        x = f25.loc[shared, col].to_numpy(dtype=float)
        y = f2.loc[shared, col].to_numpy(dtype=float)
        w = wilcoxon_signed_rank(x, y)
        frows.append({"Metric": label, "n folds": len(shared),
                      "2.5D mean": float(x.mean()), "2D mean": float(y.mean()),
                      "Mean paired difference": float((x - y).mean()),
                      "SD of difference": float((x - y).std(ddof=1)) if len(shared) > 1 else np.nan,
                      "Wilcoxon p": w["p"], "min attainable p": w["min_attainable_p"],
                      "rank-biserial": w["rank_biserial"]})
    fw = pd.DataFrame(frows)
    save_table(fw, "multislice_foldwise")

    show = tab[tab["Metric"].isin(["Accuracy (%)", "Macro-F1", "F1 Very Mild",
                                   "F1 Demented", "ECE"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(show))
    pt = show["Difference (2.5D - 2D)"].to_numpy()
    lo = show["CI low (diff)"].to_numpy(); hi = show["CI high (diff)"].to_numpy()
    scale = np.where(show["Metric"] == "Accuracy (%)", 1.0, 100.0)   # put F1/ECE on a pp-like axis
    ax.errorbar(pt * scale, y, xerr=[(pt - lo) * scale, (hi - pt) * scale], fmt="o",
                color=SERIES[2], capsize=4, lw=1.4, markersize=5)
    ax.axvline(0.0, color=INK_MUTED, lw=1.0, ls="--")
    ax.set_yticks(y, [f"{m}\n(x100)" if m != "Accuracy (%)" else m for m in show["Metric"]],
                  fontsize=8)
    ax.set_xlabel("2.5D minus 2D, 95% participant-clustered CI")
    ax.set_title("Does adjacent-slice context help?", fontsize=10, loc="left")
    ax.invert_yaxis()
    _style(ax); _save(fig, OUT / "fig_multislice.png")
    print("    -> fig_multislice.png")

    sig = tab[tab["CI excludes 0"]]
    verdict = ("NO METRIC IMPROVED SIGNIFICANTLY. Adjacent-slice context did not help at this "
               "resolution. Report this as a result: the 2D formulation is not discarding "
               "information that a cheap 2.5D extension recovers, which answers Reviewer 1 #3 "
               "and Reviewer 3 #1 with evidence rather than an acknowledgement. It does NOT "
               "license any claim about true 3D volumetric models, which were not tested and "
               "cannot be tested on 8-bit JPEG slices of a partial brain."
               if sig.empty else
               "The following metrics changed significantly:\n" + "\n".join(
                   f"  {r['Metric']}: {r['Difference (2.5D - 2D)']:+.4f} "
                   f"[{r['CI low (diff)']:.4f}, {r['CI high (diff)']:.4f}], "
                   f"p = {r['p (bootstrap)']:.4f}" for _, r in sig.iterrows())
               + "\nReport the magnitude, and state that the 2.5D arm reuses the hyperparameters "
                 "tuned for the 2D arm without re-tuning, so this is a lower bound on what the "
                 "formulation could achieve.")

    note = (
        "2.5D MULTI-SLICE vs 2D\n"
        "======================\n\n"
        "Both arms use participant-level grouping with the same split seed, so the folds hold "
        "the same people and every participant is scored once under each protocol. The "
        "bootstrap draws one resample of participants and recomputes BOTH arms on it, so the "
        "difference is paired and the interval is not inflated by between-fold variance. This "
        "is the design a08 could not have.\n\n"
        + "\n".join(
            f"{r['Metric']}: 2.5D {r['2.5D']:.4f}, 2D {r['2D']:.4f}, difference "
            f"{r['Difference (2.5D - 2D)']:+.4f} [{r['CI low (diff)']:.4f}, "
            f"{r['CI high (diff)']:.4f}], p = {r['p (bootstrap)']:.4f}"
            for _, r in tab.iterrows())
        + "\n\nVERDICT\n" + verdict
        + "\n\nCAVEATS TO STATE IN THE MANUSCRIPT.\n"
          "1. Inter-slice spacing is not recoverable from the JPEG derivatives, so the three "
          "channels are adjacent in index but of unknown physical separation.\n"
          "2. Slices at the band edges (100 and 160) have no neighbour on one side and are "
          "clamped to the centre slice, so those samples are partially 2D.\n"
          "3. Hyperparameters were carried over from the 2D protocol without re-tuning, which "
          "is deliberate and consistent with the rest of the revision.\n"
          "4. Multiplicity: with seven metrics tested, a Bonferroni-corrected threshold is "
          "0.0071. State which results survive it.\n"
          "5. The fold-wise Wilcoxon companion cannot reach p < 0.05 with five folds -- its "
          "minimum attainable two-sided p is 0.0625. It is reported for completeness only.\n")
    (OUT / "multislice_note.txt").write_text(note, encoding="utf-8")
    print("    -> multislice_note.txt")
    print(tab[["Metric", "2.5D", "2D", "Difference (2.5D - 2D)", "CI low (diff)",
               "CI high (diff)", "p (bootstrap)", "CI excludes 0"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
