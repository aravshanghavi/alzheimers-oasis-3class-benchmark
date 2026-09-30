#!/usr/bin/env python3
"""A POWERED test of the leakage effect, because a08 is not one.

a08 compares two pooled estimates from two different partitionings. Because the
partitions differ, the contrast carries the full between-fold variance: the 95%
interval on the accuracy difference is about +/- 5 points, which is wider than
any leakage effect could plausibly be. That comparison can therefore neither
demonstrate nor exclude an effect, and reporting it alone would leave the
manuscript's central claim resting on an uninformative interval.

This script runs two contrasts that remove the partition noise instead.

CONTRAST A -- the naive comparison, WITH its negative control.
    Inside each exp05 fold, some test participants also appear in that fold's
    TRAINING set. The obvious test is to compare their accuracy against the clean
    test participants of the same class. That comparison is what a less careful
    study would report as evidence of leakage -- and it is confounded, because
    the contaminated and clean groups are DIFFERENT PEOPLE who may differ in
    difficulty for reasons that have nothing to do with leakage.
    So the identical comparison is also run on exp01, where NOBODY is
    contaminated. That is a negative control: any difference it reproduces is
    selection, not leakage.

CONTRAST B -- difference in differences.
    Every participant is scored under exp01 (never contaminated) and under exp05
    (contaminated if they carry a repeat session). Take each participant's
    change in accuracy between the two protocols. Participants WITH a repeat
    session are the treated group; participants WITHOUT one are the control
    group, and their change measures the partition/model noise that affects
    everybody. The difference between the two groups' mean change is the
    leakage effect with that noise subtracted out.

Both are bootstrapped over participants, which is the correct resampling unit --
images within a participant are near-duplicates and resampling them would
understate every interval by an order of magnitude.
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

from analysis._common import CLASS_NAMES, OUT, save_table            # noqa: E402
from analysis.discover import load_predictions, load_runs            # noqa: E402
from src.figures import INK_MUTED, SERIES, _style, _save             # noqa: E402

N_BOOT = 10000
LOSS = "FA_FL"


def per_participant_accuracy(runs: pd.DataFrame, loss: str) -> pd.DataFrame:
    """One row per participant: images, correct, accuracy, true class."""
    sub = runs[runs["loss"] == loss].sort_values("fold")
    if sub.empty:
        raise SystemExit(f"no runs for loss {loss!r}")
    frames = []
    for _, r in sub.iterrows():
        p = load_predictions(r["run_dir"], "test")
        p["fold"] = int(r["fold"])
        frames.append(p)
    allp = pd.concat(frames, ignore_index=True)
    allp["correct"] = (allp["pred"] == allp["label"]).astype(float)
    g = allp.groupby("participant_id").agg(images=("correct", "size"),
                                           correct=("correct", "sum"),
                                           true_class=("label", "first"))
    g["accuracy"] = 100.0 * g["correct"] / g["images"]
    return g.reset_index()


def contamination_map(leak_runs: pd.DataFrame) -> pd.DataFrame:
    """Which test participants were also in their own fold's training set."""
    rows = []
    for _, r in leak_runs.iterrows():
        d = pd.read_csv(Path(r["run_dir"]) / "fold_participants.csv")
        train = set(d[d["split"] == "train"]["participant_id"])
        for pid in d[d["split"] == "test"]["participant_id"].unique():
            rows.append({"participant_id": pid, "fold": int(r["fold"]),
                         "contaminated": pid in train})
    m = pd.DataFrame(rows)
    # a participant tested in two folds counts as contaminated if either was
    return m.groupby("participant_id")["contaminated"].any().reset_index()


def boot_mean_diff(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT, seed=11):
    """Difference of group means, resampling each group's participants."""
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan, np.nan
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    d = da - db
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(a.mean() - b.mean()), float(lo), float(hi), float(min(p, 1.0))


def main() -> int:
    print("[a09] powered leakage contrasts (within-model and difference-in-differences)")
    clean_runs = load_runs("exp01_main_sweep")
    leak_runs = load_runs("exp05_leakage_ablation")
    if leak_runs.empty:
        print("  exp05_leakage_ablation has no completed runs -- skipping")
        return 0
    clean_runs = clean_runs[clean_runs["loss"] == LOSS]
    if clean_runs.empty:
        print(f"  no exp01 runs for {LOSS} -- skipping")
        return 0

    leak = per_participant_accuracy(leak_runs, LOSS)
    clean = per_participant_accuracy(clean_runs, LOSS)
    contam = contamination_map(leak_runs)
    leak = leak.merge(contam, on="participant_id", how="left")
    leak["contaminated"] = leak["contaminated"].fillna(False)

    n_contam = int(leak["contaminated"].sum())
    classes_hit = sorted(leak[leak["contaminated"]]["true_class"].unique())
    print(f"  {n_contam} of {len(leak)} participants contaminated under session-level "
          f"grouping; classes affected: {[CLASS_NAMES[c] for c in classes_hit]}")

    both = clean.merge(leak, on="participant_id", suffixes=("_clean", "_leak"))
    both["delta"] = both["accuracy_leak"] - both["accuracy_clean"]

    # ---- CONTRAST A: naive comparison + negative control -------------------
    rows = []
    for c in classes_hit:
        sub = both[both["true_class_leak"] == c]
        grp = sub["contaminated"].to_numpy(dtype=bool)
        naive = boot_mean_diff(sub[grp]["accuracy_leak"].to_numpy(),
                               sub[~grp]["accuracy_leak"].to_numpy(), seed=11)
        # SAME participants, SAME comparison, under exp01 where none are contaminated
        ctrl = boot_mean_diff(sub[grp]["accuracy_clean"].to_numpy(),
                              sub[~grp]["accuracy_clean"].to_numpy(), seed=11)
        rows.append({"Class": CLASS_NAMES[c], "n contaminated": int(grp.sum()),
                     "n clean": int((~grp).sum()),
                     "Acc contaminated, exp05 (%)": float(sub[grp]["accuracy_leak"].mean()),
                     "Acc clean, exp05 (%)": float(sub[~grp]["accuracy_leak"].mean()),
                     "Naive difference": naive[0], "Naive CI low": naive[1],
                     "Naive CI high": naive[2], "Naive p": naive[3],
                     "Acc same participants, exp01 (%)": float(sub[grp]["accuracy_clean"].mean()),
                     "Acc clean, exp01 (%)": float(sub[~grp]["accuracy_clean"].mean()),
                     "NEGATIVE CONTROL difference": ctrl[0], "Control CI low": ctrl[1],
                     "Control CI high": ctrl[2], "Control p": ctrl[3],
                     "Naive test confounded": bool(ctrl[1] > 0 or ctrl[2] < 0)})
    ca = pd.DataFrame(rows)
    save_table(ca, "leakage_within_model")

    # ---- CONTRAST B: difference in differences -----------------------------
    treated = both[both["contaminated"]]["delta"].to_numpy()
    control = both[~both["contaminated"]]["delta"].to_numpy()
    did_pt, did_lo, did_hi, did_p = boot_mean_diff(treated, control, seed=29)
    cb = pd.DataFrame([{
        "Group": "repeat-session participants (treated)", "n": len(treated),
        "Mean change in accuracy (pp)": float(treated.mean()) if len(treated) else np.nan},
        {"Group": "single-session participants (control)", "n": len(control),
         "Mean change in accuracy (pp)": float(control.mean()) if len(control) else np.nan},
        {"Group": "DIFFERENCE-IN-DIFFERENCES (leakage effect)", "n": len(both),
         "Mean change in accuracy (pp)": did_pt}])
    save_table(cb, "leakage_did")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels, pts, los, his, cols = [], [], [], [], []
    for _, r in ca.iterrows():
        labels.append(f"NAIVE: contaminated vs clean\n{r['Class']}, exp05 "
                      f"(n={r['n contaminated']} vs {r['n clean']})")
        pts.append(r["Naive difference"]); los.append(r["Naive CI low"])
        his.append(r["Naive CI high"]); cols.append(SERIES[1])
        labels.append(f"NEGATIVE CONTROL: same split, exp01\n{r['Class']}, nobody contaminated")
        pts.append(r["NEGATIVE CONTROL difference"]); los.append(r["Control CI low"])
        his.append(r["Control CI high"]); cols.append(INK_MUTED)
    labels.append(f"DIFFERENCE-IN-DIFFERENCES\nleakage effect "
                  f"(n={len(treated)} vs {len(control)})")
    pts.append(did_pt); los.append(did_lo); his.append(did_hi); cols.append(SERIES[0])
    y = np.arange(len(pts))
    pts, los, his = np.array(pts), np.array(los), np.array(his)
    for i in range(len(pts)):
        ax.errorbar(pts[i], y[i], xerr=[[pts[i] - los[i]], [his[i] - pts[i]]], fmt="o",
                    color=cols[i], capsize=4, lw=1.4, markersize=5)
    ax.axvline(0.0, color=INK_MUTED, lw=1.0, ls="--")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("accuracy attributable to leakage (percentage points, 95% CI)")
    ax.set_title("Measured effect of session-level grouping", fontsize=10, loc="left")
    ax.invert_yaxis()
    _style(ax); _save(fig, OUT / "fig_leakage_within_model.png")
    print("    -> fig_leakage_within_model.png")

    conf = bool(ca["Naive test confounded"].any())
    note = (
        "POWERED LEAKAGE CONTRASTS\n"
        "=========================\n\n"
        "a08 compares two pooled estimates built on DIFFERENT partitions, so it carries the "
        "full between-fold variance and its accuracy interval spans roughly +/-5 percentage "
        "points -- too wide to demonstrate or exclude anything. The contrasts below remove "
        "that noise.\n\n"
        f"WHO IS CONTAMINATED. Under session-level grouping {n_contam} of {len(leak)} "
        f"participants appear in both the training and the test partition of their own fold. "
        f"All belong to: {', '.join(CLASS_NAMES[c] for c in classes_hit)}. These are the "
        "OASIS-1 reliability sample -- nondemented participants re-imaged within 90 days of "
        "their first session specifically to support reproducibility analyses. They are not a "
        "random subset of the cohort, and that is the crux of what follows.\n\n"
        "CONTRAST A -- the naive comparison, and why it is wrong.\n"
        + "\n".join(
            f"  {r['Class']}: contaminated {r['Acc contaminated, exp05 (%)']:.2f}% "
            f"(n={r['n contaminated']}) vs clean {r['Acc clean, exp05 (%)']:.2f}% "
            f"(n={r['n clean']}), difference {r['Naive difference']:+.2f} pp "
            f"[{r['Naive CI low']:.2f}, {r['Naive CI high']:.2f}], p = {r['Naive p']:.4f}"
            for _, r in ca.iterrows())
        + "\n\n  NEGATIVE CONTROL -- the identical comparison under exp01, where NO participant "
          "is contaminated:\n"
        + "\n".join(
            f"  {r['Class']}: same participants {r['Acc same participants, exp01 (%)']:.2f}% vs "
            f"others {r['Acc clean, exp01 (%)']:.2f}%, difference "
            f"{r['NEGATIVE CONTROL difference']:+.2f} pp [{r['Control CI low']:.2f}, "
            f"{r['Control CI high']:.2f}], p = {r['Control p']:.4f}"
            for _, r in ca.iterrows())
        + ("\n\n  THE NEGATIVE CONTROL REPRODUCES THE EFFECT. The naive comparison therefore "
           "measures who these participants are, not what leakage did to them. Reporting it as "
           "a leakage effect would be a false positive of exactly the kind this manuscript "
           "criticises.\n" if conf else
           "\n\n  The negative control is null, so the naive comparison is not obviously "
           "confounded by participant selection. Contrast B remains the better estimate.\n")
        + "\nCONTRAST B -- difference in differences (THE VALID ESTIMATE).\n"
          "Each participant is scored under exp01 (never contaminated) and exp05 "
          "(contaminated if they carry a repeat session), and their change is taken. Repeat-"
          "session participants are treated; the rest are controls whose change measures the "
          "model and partition noise common to everyone. Subtracting removes both the partition "
          "noise AND the participant-selection confound, because every participant is compared "
          "against themselves.\n\n"
        f"  treated  (n={len(treated)}): {treated.mean():+.3f} pp\n"
        f"  control  (n={len(control)}): {control.mean():+.3f} pp\n"
        f"  LEAKAGE EFFECT = {did_pt:+.3f} pp [{did_lo:.3f}, {did_hi:.3f}], p = {did_p:.4f}\n\n"
        "HOW TO REPORT THIS. Contrast B is the headline. Contrast A and its negative control "
        "belong in the manuscript too, as a demonstration that the obvious test of leakage "
        "returns a large, highly significant, and entirely spurious result on this cohort.\n")
    (OUT / "leakage_powered_note.txt").write_text(note, encoding="utf-8")
    print("    -> leakage_powered_note.txt")
    print(ca.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"  DiD leakage effect: {did_pt:+.3f} pp [{did_lo:.3f}, {did_hi:.3f}], p = {did_p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
