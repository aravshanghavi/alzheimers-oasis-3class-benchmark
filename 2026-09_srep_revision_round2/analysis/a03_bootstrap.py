#!/usr/bin/env python3
"""Participant-clustered bootstrap CIs, paired differences, and effect sizes.

This is the script that most needs to be right. The test set holds ~86,000
images but only 347 independent people, with 3-4 near-duplicate acquisitions of
each anatomical slice (r = 0.988). Resampling images would understate every
interval by more than an order of magnitude. Participants are resampled; every
loss is evaluated on the SAME resample so between-loss differences stay paired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis._common import (LOSS_LABEL, LOSS_ORDER, OUT, SHORT, SuffStats,  # noqa: E402
                              clustered_bootstrap, fold_metrics, paired_difference,
                              pooled_predictions, runs_frame, save_table,
                              wilcoxon_signed_rank)

N_BOOT = 2000
REFERENCE = "FA_FL"


def main() -> int:
    print(f"[a03] participant-clustered bootstrap ({N_BOOT} resamples)")
    runs = runs_frame()
    fm = fold_metrics(runs)
    losses = [L for L in LOSS_ORDER if not fm[fm["loss"] == L].empty]

    stats = {L: SuffStats(pooled_predictions(runs, L)) for L in losses}
    stats_ts = {}
    for L in losses:
        try:
            stats_ts[L] = SuffStats(pooled_predictions(runs, L, temperature_scaled=True))
        except SystemExit:
            pass
    print(f"  {stats[losses[0]].n_p} participants, "
          f"{int(stats[losses[0]].n_img.sum()):,} images per loss")

    METRICS = [("accuracy", None), ("macro_f1", None), ("ece", None), ("brier", None),
               ("per_class_f1", 1), ("per_class_f1", 2)]
    LABEL = {("accuracy", None): "Accuracy (%)", ("macro_f1", None): "Macro-F1",
             ("ece", None): "ECE", ("brier", None): "Brier",
             ("per_class_f1", 1): "F1 Very Mild", ("per_class_f1", 2): "F1 Demented"}

    ci_rows, diff_rows = [], []
    for metric, cls in METRICS:
        name = LABEL[(metric, cls)]
        print(f"  {name} ...")
        boot = clustered_bootstrap(stats, metric, N_BOOT, per_class=cls)
        for L in losses:
            b = boot[L]
            ci_rows.append({"Metric": name, "Loss": LOSS_LABEL[L], "Estimate": b["point"],
                            "CI low": b["ci_low"], "CI high": b["ci_high"],
                            "Bootstrap SD": b["boot_sd"]})
        for L in losses:
            if L == REFERENCE:
                continue
            d = paired_difference(boot, REFERENCE, L)
            g_ref = fm[fm["loss"] == REFERENCE].sort_values("fold")
            g_oth = fm[fm["loss"] == L].sort_values("fold")
            col = f"f1_{SHORT[cls]}" if cls is not None else metric
            w = wilcoxon_signed_rank(g_ref[col].to_numpy(), g_oth[col].to_numpy())
            diff_rows.append({"Metric": name, "Comparison": f"FA-FL vs {LOSS_LABEL[L]}",
                              "Difference": d["diff"], "CI low": d["ci_low"],
                              "CI high": d["ci_high"], "p (bootstrap)": d["p_bootstrap"],
                              "CI excludes 0": d["excludes_zero"],
                              "p (Wilcoxon, 5 folds)": w["p"],
                              "rank-biserial": w["rank_biserial"]})

    if stats_ts:
        print("  ECE after temperature scaling ...")
        boot_ts = clustered_bootstrap(stats_ts, "ece", N_BOOT)
        for L in stats_ts:
            b = boot_ts[L]
            ci_rows.append({"Metric": "ECE post-temperature", "Loss": LOSS_LABEL[L],
                            "Estimate": b["point"], "CI low": b["ci_low"],
                            "CI high": b["ci_high"], "Bootstrap SD": b["boot_sd"]})
        for L in stats_ts:
            if L == REFERENCE:
                continue
            d = paired_difference(boot_ts, REFERENCE, L)
            diff_rows.append({"Metric": "ECE post-temperature",
                              "Comparison": f"FA-FL vs {LOSS_LABEL[L]}", "Difference": d["diff"],
                              "CI low": d["ci_low"], "CI high": d["ci_high"],
                              "p (bootstrap)": d["p_bootstrap"],
                              "CI excludes 0": d["excludes_zero"],
                              "p (Wilcoxon, 5 folds)": float("nan"),
                              "rank-biserial": float("nan")})

    save_table(pd.DataFrame(ci_rows), "bootstrap_ci")
    save_table(pd.DataFrame(diff_rows), "bootstrap_paired_differences")

    note = (
        f"Confidence intervals are 2.5/97.5 percentiles of {N_BOOT} bootstrap resamples "
        "drawn over PARTICIPANTS with replacement, not over images. The test set contains "
        f"{int(stats[losses[0]].n_img.sum()):,} images but only {stats[losses[0]].n_p} "
        "independent participants, and each anatomical slice appears 3-4 times through "
        "repeated MPRAGE acquisitions (r = 0.988), so image-level resampling would "
        "understate interval width by more than an order of magnitude.\n\n"
        "All losses are evaluated on the SAME resample, so paired differences and their "
        "intervals are valid.\n\n"
        "Two tests are reported. The bootstrap interval on the paired difference is the "
        "primary evidence. The Wilcoxon signed-rank test across the five folds is a "
        "conservative companion and is reported for completeness -- with n = 5 its "
        "smallest attainable two-sided p is 0.0625, so it can never reach 0.05 regardless "
        "of effect size. Any claim resting on Wilcoxon alone would be underpowered by "
        "construction; this is stated rather than worked around.\n\n"
        "p-values are uncorrected. With four comparisons against the reference, a "
        "Bonferroni threshold would be 0.0125; both raw p and that threshold are given so "
        "the reader can apply whichever standard they prefer.\n")
    (OUT / "bootstrap_note.txt").write_text(note, encoding="utf-8")
    print("    -> bootstrap_note.txt")

    print("\n  Pooled estimates with 95% participant-clustered CI:")
    ci = pd.DataFrame(ci_rows)
    for m in ["Accuracy (%)", "Macro-F1", "ECE", "ECE post-temperature"]:
        s = ci[ci["Metric"] == m]
        if s.empty:
            continue
        print(f"   {m}")
        for _, r in s.iterrows():
            print(f"      {r['Loss']:<20}{r['Estimate']:>9.4f}  "
                  f"[{r['CI low']:.4f}, {r['CI high']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
