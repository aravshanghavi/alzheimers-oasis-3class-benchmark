#!/usr/bin/env python3
"""a22: the permuted-label null distribution, against the real arms.

Reads the exp14 permuted runs from this tree and the real age-60 runs from
2026-09_instrumented_arms, pools each replicate's five test folds to one
prediction per participant, and reports where the real arms fall in the null.

Unit of analysis is the PARTICIPANT. Each participant's predicted distribution
is the mean of the predicted distributions over that participant's images, then
argmax. That is the manuscript's Table 4 rule, and it is the unit the paper
should report everywhere. Image-level values are printed alongside only so the
difference is visible, never as the headline.

    python analysis/a22_permutation_null.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.runfilter import permuted_spec, production_reason  # noqa: E402
REAL_TREE = REPO_ROOT.parent / "2026-09_instrumented_arms"
OUT_DIR = REPO_ROOT / "analysis" / "outputs"
K = 3
REAL_ARMS = {("exp10", "WCE"): "Weighted CE", ("exp10", "LDAM"): "LDAM",
             ("exp10", "ClassBalanced"): "Class-Balanced", ("exp10", "FocalLoss"): "Focal g=3.00",
             ("exp10", "FA_FL"): "FA-FL", ("exp11", "FocalLoss"): "Focal g=1.37",
             ("exp12", "FocalLoss"): "Focal g=2.00", ("exp13", "FocalLoss"): "Focal g=2.69"}


def macro_f1(pred: np.ndarray, lab: np.ndarray) -> float:
    fs = []
    for k in range(K):
        tp = int(((pred == k) & (lab == k)).sum())
        fp = int(((pred == k) & (lab != k)).sum())
        fn = int(((pred != k) & (lab == k)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def participant_level(runs):
    """runs: list of (probabilities, labels, participant_id). One row per participant."""
    P, L = [], []
    for pr, lab, pid in runs:
        for u in np.unique(pid):
            m = pid == u
            P.append(pr[m].mean(axis=0))
            L.append(lab[m][0])
    return np.asarray(P), np.asarray(L)


def load_run(d: Path):
    f = d / "predictions_test.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return (z["probabilities"].astype(np.float64), z["labels"],
            z["participant_id"].astype(str))


def collect_null():
    """PRODUCTION permuted runs only, keyed by seed, one entry per fold.

    Smoke runs (two epochs, 64 pixels, no pretrained weights) are written into
    this same directory by `run_permnull.py --smoke` and carry a real seed and
    fold in their manifest. Counting one would push a replicate to six runs and
    silently drop it from the null. They are excluded by resolved config, and
    any duplicate (seed, fold) keeps the most recent run and says so.
    """
    by_spec = {}
    excluded = []
    base = REPO_ROOT / "experiments" / "exp14_permnull_wce" / "outputs"
    if not base.is_dir():
        return {}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "_COMPLETE").exists():
            continue
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (m.get("permutation") or {}).get("enabled"):
            continue
        why = production_reason(m)
        if why is not None:
            excluded.append((d.name, why))
            continue
        spec = permuted_spec(m)
        if spec is None:
            excluded.append((d.name, "seed or fold missing from manifest"))
            continue
        prev = by_spec.get(spec)
        started = m.get("started_utc", "")
        if prev is None or started > prev[0]:
            if prev is not None:
                print(f"  NOTE duplicate {spec}: keeping the newer run {d.name}")
            by_spec[spec] = (started, d)
        else:
            print(f"  NOTE duplicate {spec}: keeping the newer run {prev[1].name}")

    if excluded:
        print(f"  Excluded {len(excluded)} non-production run(s) from the null:")
        for name, why in excluded:
            print(f"    {name}  ->  {why}")

    out = {}
    for (seed, _fold), (_started, d) in sorted(by_spec.items()):
        r = load_run(d)
        if r is not None:
            out.setdefault(seed, []).append(r)
    return out


def collect_real():
    out = {}
    if not (REAL_TREE / "experiments").is_dir():
        return out
    for d in sorted((REAL_TREE / "experiments").glob("exp1*/outputs/*__exp1*")):
        if not d.is_dir() or not (d / "_COMPLETE").exists():
            continue
        parts = d.name.split("__")
        if len(parts) < 3:
            continue
        arm = REAL_ARMS.get((parts[1], parts[2]))
        if arm is None:
            continue
        r = load_run(d)
        if r is not None:
            out.setdefault(arm, []).append(r)
    return out


def main() -> int:
    null_runs = collect_null()
    real_runs = collect_real()

    if not null_runs:
        print("No COMPLETE permuted runs found. Run run_permnull.py first.")
        return 1

    print("=" * 74)
    print("PERMUTED-LABEL NULL, participant level, pooled over five test folds")
    print("=" * 74)
    print(f"{'perm seed':>10}{'folds':>7}{'participants':>14}{'accuracy %':>12}{'macro-F1':>10}")

    null_acc, null_f1, usable = [], [], []
    for seed in sorted(null_runs):
        runs = null_runs[seed]
        if len(runs) != 5:
            print(f"{seed:>10}{len(runs):>7}   INCOMPLETE, skipped (need 5 folds)")
            continue
        P, L = participant_level(runs)
        pred = P.argmax(axis=1)
        acc = 100.0 * float((pred == L).mean())
        f1 = macro_f1(pred, L)
        null_acc.append(acc)
        null_f1.append(f1)
        usable.append(seed)
        print(f"{seed:>10}{len(runs):>7}{len(L):>14}{acc:>12.4f}{f1:>10.4f}")

    if not usable:
        print("\nNo replicate has all five folds COMPLETE yet.")
        return 1

    null_acc_a, null_f1_a = np.asarray(null_acc), np.asarray(null_f1)
    n = len(usable)
    print(f"\n  null replicates: {n}   "
          f"accuracy mean {null_acc_a.mean():.4f} [min {null_acc_a.min():.4f}, max {null_acc_a.max():.4f}]")
    print(f"  {'':17}macro-F1 mean {null_f1_a.mean():.4f} "
          f"[min {null_f1_a.min():.4f}, max {null_f1_a.max():.4f}]")
    print(f"\n  p-value floor with {n} replicates: {1.0/(n+1):.4f}. "
          "Report replicates individually; this does not support a fitted tail.")

    if real_runs:
        print("\n" + "=" * 74)
        print("REAL ARMS AGAINST THE NULL (participant level)")
        print("=" * 74)
        print(f"{'arm':<17}{'accuracy %':>12}{'macro-F1':>10}{'null >= F1':>12}{'p':>9}   verdict")
        rows = []
        for arm in ["Weighted CE", "LDAM", "Class-Balanced", "Focal g=1.37", "Focal g=2.00",
                    "Focal g=2.69", "Focal g=3.00", "FA-FL"]:
            runs = real_runs.get(arm)
            if not runs or len(runs) != 5:
                continue
            P, L = participant_level(runs)
            pred = P.argmax(axis=1)
            acc = 100.0 * float((pred == L).mean())
            f1 = macro_f1(pred, L)
            ge = int((null_f1_a >= f1).sum())
            p = (1 + ge) / (1 + n)
            verdict = "inside the null" if ge > 0 else f"outside all {n} replicates"
            print(f"{arm:<17}{acc:>12.4f}{f1:>10.4f}{ge:>12}{p:>9.4f}   {verdict}")
            rows.append((arm, acc, f1, ge, p, verdict))

        L0 = participant_level(next(iter(real_runs.values())))[1]
        maj = int(np.bincount(L0, minlength=K).argmax())
        cpred = np.full(len(L0), maj)
        print(f"{'Constant majority':<17}{100.0*float((cpred==L0).mean()):>12.4f}"
              f"{macro_f1(cpred, L0):>10.4f}")

        print("\nHow to read this. A real arm whose macro-F1 is exceeded by no null replicate")
        print("has discriminative signal beyond what this pipeline manufactures from permuted")
        print("labels. An arm inside the null does not, and any claim of real signal for that")
        print(f"arm must come out of the paper. With {n} replicates the smallest reportable")
        print(f"p-value is {1.0/(n+1):.4f}, so 'outside all replicates' is the strongest available")
        print("statement, not 'p < 0.05'.")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "null": {"seeds": usable, "accuracy": null_acc, "macro_f1": null_f1,
                     "n_replicates": n, "p_value_floor": 1.0 / (n + 1)},
            "real": [{"arm": a, "accuracy": ac, "macro_f1": f, "null_ge": g,
                      "p": pv, "verdict": v} for a, ac, f, g, pv, v in rows],
            "unit": "participant",
            "note": ("Null permutes participant-level labels within train and val only; "
                     "test folds retain true labels. Real arms are from "
                     "2026-09_instrumented_arms exp10-exp13."),
        }
        (OUT_DIR / "t22_permutation_null.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWritten: {OUT_DIR / 't22_permutation_null.json'}")
    else:
        print(f"\nReal arms not found under {REAL_TREE}. Null reported alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
