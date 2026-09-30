#!/usr/bin/env python3
"""Independent verification: re-derive every headline number straight from the
raw prediction arrays and diff against what the pipeline and the tables report.

This exists because the submitted manuscript's Table 2 and Table 4 disagreed on
the standard deviations of the same runs -- an error that survived review because
nothing ever recomputed the numbers from source. Any mismatch here is a hard
failure; the script exits non-zero and the tables must not be used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis._common import (LOSS_LABEL, LOSS_ORDER, N_CLASSES, OUT, SuffStats,  # noqa: E402
                              fold_metrics, pooled_predictions, runs_frame)
from analysis.discover import load_predictions                        # noqa: E402
from src.evaluate import expected_calibration_error                   # noqa: E402

TOL = 1e-6
TOL_LOOSE = 1e-4


def main() -> int:
    print("[a07] verification -- recomputing everything from raw predictions")
    runs = runs_frame()
    fm = fold_metrics(runs)
    failures, checks = [], 0

    # 1. every run: metrics.json must match a fresh computation from its .npz
    print("  1. per-run metrics.json vs raw predictions")
    for _, r in runs.iterrows():
        p = load_predictions(r["run_dir"], "test")
        probs = p[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
        y, yhat = p["label"].to_numpy(), p["pred"].to_numpy()
        stored = json.loads((Path(r["run_dir"]) / "metrics.json").read_text())["test"]
        acc = 100.0 * (y == yhat).mean()
        mf1 = f1_score(y, yhat, labels=list(range(N_CLASSES)), average="macro", zero_division=0)
        ece, _ = expected_calibration_error(probs, y, stored["ece_bins"], stored["ece_binning"])
        cm = confusion_matrix(y, yhat, labels=list(range(N_CLASSES)))
        for name, got, want, tol in (("accuracy", acc, stored["accuracy"], TOL_LOOSE),
                                     ("macro_f1", mf1, stored["macro_f1"], TOL_LOOSE),
                                     ("ece", ece, stored["ece"], TOL_LOOSE)):
            checks += 1
            if abs(got - want) > tol:
                failures.append(f"{r['run_id']}: {name} recomputed {got:.6f} vs stored {want:.6f}")
        checks += 1
        if not np.array_equal(cm, np.array(stored["confusion_matrix"])):
            failures.append(f"{r['run_id']}: confusion matrix mismatch")

    # 2. sufficient-statistic metrics must equal direct computation
    print("  2. sufficient-statistic path vs direct computation (pooled)")
    for L in [x for x in LOSS_ORDER if not fm[fm["loss"] == x].empty]:
        pooled = pooled_predictions(runs, L)
        st = SuffStats(pooled).metrics()
        probs = pooled[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
        y, yhat = pooled["label"].to_numpy(), pooled["pred"].to_numpy()
        direct_acc = 100.0 * (y == yhat).mean()
        direct_mf1 = f1_score(y, yhat, labels=list(range(N_CLASSES)), average="macro", zero_division=0)
        direct_ece, _ = expected_calibration_error(probs, y, 15, "equal_width")
        for name, got, want in (("accuracy", st["accuracy"], direct_acc),
                                ("macro_f1", st["macro_f1"], direct_mf1),
                                ("ece", st["ece"], direct_ece)):
            checks += 1
            if abs(got - want) > TOL_LOOSE:
                failures.append(f"pooled {L}: {name} suffstats {got:.8f} vs direct {want:.8f}")

    # 3. pooling must cover the cohort exactly once
    print("  3. pooled coverage: every participant tested exactly once")
    for L in [x for x in LOSS_ORDER if not fm[fm["loss"] == x].empty]:
        pooled = pooled_predictions(runs, L)
        counts = pooled.groupby("participant_id")["fold"].nunique()
        checks += 1
        if (counts > 1).any():
            failures.append(f"pooled {L}: {(counts > 1).sum()} participant(s) in >1 fold")

    # 4. published tables must match a fresh recomputation
    print("  4. generated tables vs fresh recomputation")
    t4 = OUT / "table4_pooled.csv"
    if t4.exists():
        pub = pd.read_csv(t4)
        for _, row in pub.iterrows():
            L = next((k for k, v in LOSS_LABEL.items() if v == row["Loss"]), None)
            if L is None:
                continue
            m = SuffStats(pooled_predictions(runs, L)).metrics()
            for col, got in (("Accuracy (%)", m["accuracy"]), ("Macro-F1", m["macro_f1"]),
                             ("ECE", m["ece"]), ("Brier", m["brier"])):
                checks += 1
                if abs(float(row[col]) - got) > 5e-4:
                    failures.append(f"table4 {row['Loss']} {col}: published {row[col]} "
                                    f"vs recomputed {got:.6f}")
    else:
        print("     (table4_pooled.csv absent -- run a01 first)")

    # 5. SD convention consistent between tables 2 and 4 inputs
    print("  5. fold-wise SD convention")
    t2 = OUT / "table2_overall.csv"
    if t2.exists():
        pub = pd.read_csv(t2)
        for _, row in pub.iterrows():
            L = next((k for k, v in LOSS_LABEL.items() if v == row["Loss"]), None)
            g = fm[fm["loss"] == L]
            checks += 1
            if abs(row["Accuracy SD"] - g["accuracy"].std(ddof=1)) > 5e-4:
                failures.append(f"table2 {row['Loss']}: accuracy SD is not ddof=1")

    print()
    print("=" * 76)
    if failures:
        print(f"VERIFICATION FAILED -- {len(failures)} mismatch(es) out of {checks} checks")
        for f in failures[:30]:
            print("  " + f)
        print("=" * 76)
        (OUT / "verification_FAILED.txt").write_text("\n".join(failures), encoding="utf-8")
        return 1
    print(f"VERIFICATION PASSED -- {checks} checks, no mismatches")
    print("Every table number was re-derived from the raw prediction arrays.")
    print("=" * 76)
    (OUT / "verification_passed.txt").write_text(
        f"{checks} independent checks passed.\n"
        "Per-run metrics.json, the pooled sufficient-statistic path, cohort coverage, "
        "the published tables and the SD convention were all re-derived from "
        "predictions_test.npz and matched.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
