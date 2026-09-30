#!/usr/bin/env python3
"""a40: what survives if CDR 0.5 is not treated as its own category?

    python analysis/a40_cdr05_collapse.py
    python analysis/a40_cdr05_collapse.py --quick      # 200 resamples
    python analysis/a40_cdr05_collapse.py --boot 10000

WHY A REVIEWER WILL ASK
-----------------------
CDR 0.5 is not a diagnosis of dementia. In the Clinical Dementia Rating it is
"very mild impairment", historically "questionable dementia", and a person scored
0.5 may be cognitively normal for their age, may have mild cognitive impairment,
or may be in the earliest stage of a dementia. Its content depends on the rater
and on the era of the rating. On the restricted cohort it is also the second
largest class, 58 of 166 participants, so it carries a third of the weight of
every macro-averaged number in the paper.

A three-class task whose middle class is contested is a task whose middle class a
reviewer will want removed. There are exactly two defensible ways to remove it
and they point in opposite directions:

    CDR 0.5 merged into CDR 0        treat questionable dementia as not dementia
    CDR 0.5 merged into CDR >= 1     treat any impairment as the thing to detect

Neither is more correct than the other. Which one is charitable to the paper
depends on the result, which is why both are computed and both are reported.

THIS IS A POST-HOC REDUCTION OF A PRE-SPECIFIED THREE-CLASS TASK
----------------------------------------------------------------
Said plainly because it will otherwise be read as a second primary analysis. The
registered task is three-class. The networks were trained on three classes with
three-class losses, and their decision rule is a three-way argmax. Nothing here
retrains anything. A reduction applies a map to labels and to predictions that
were both produced under the three-class rule. The reduced numbers are therefore
not what a model built for the binary task would achieve; they are what these
models deliver when their output is read as a binary decision. The distinction
matters in both directions: a binary model would likely do better, and a binary
task is also easier, so a favourable reduced number is not a new result.

THE CONSTANT PREDICTOR MUST BE RECOMPUTED INSIDE EACH REDUCTION
---------------------------------------------------------------
This is where the reduction can mislead most cheaply. Merging CDR 0.5 into CDR 0
makes the negative class 143 of 166 participants, so a predictor that says
"normal" to everyone scores 86 percent accuracy. An arm reporting 85 percent
under that reduction looks strong beside the paper's 57 percent and is in fact
WORSE than saying nothing. So every reduction carries its own constant
majority-class predictor, built from that reduction's own labels, and every arm
is differenced against it on the same bootstrap draw.

WHAT IS REPORTED, AND WHICH DRAW EACH QUANTITY USES
---------------------------------------------------
Accuracy against the within-reduction constant, F1 of the positive class,
macro-F1 over the two reduced classes, and the midrank AUC of the reduced
positive-class score. The three-class task is carried in the same table as the
reference row so that "did the conclusion change" is a comparison inside one file
rather than across two.

Accuracy and the F1 scores use the unstratified participant-clustered draw, which
is a23's and a35's arrangement for count metrics and lets prevalence vary as it
would in a new sample. AUC uses the draw stratified within the ORIGINAL three
classes, because an unstratified draw can empty the 23-participant CDR >= 1 class
and an AUC on an empty class is undefined rather than noisy. Draws in which a
quantity does not exist are counted in the table rather than dropped silently.

MULTIPLICITY
------------
No family in analysis/_registry.py covers a binary reduction. The declared
arm-vs-constant families are three-class and were realised at the image level on
accuracy; absorbing two reductions times two cohorts into them would break their
declared sizes. So every row carries the literal string "undeclared", every
interval is a nominal 95 percent one, and nothing here is a pre-registered test.
Nothing is written to the registry: the changelog entry is PRINTED for the first
author to paste.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import save_table                                       # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                              # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, FULL_EXPERIMENTS,                 # noqa: E402
                                    RESTRICTED_EXPERIMENTS, bootstrap_p,
                                    changelog_entry, find_full_tree,
                                    percentile_interval,
                                    stratified_participant_bootstrap,
                                    unstratified_participant_bootstrap)
from analysis.a23_protocol_audit import ARM_ORDER, CONSTANT, warn             # noqa: E402
from analysis.a25_increment_over_metadata import auc_draws, multiplicity      # noqa: E402
from analysis.a36_operating_point import counts                              # noqa: E402

SOURCE = "analysis/a40_cdr05_collapse.py"
EXPERIMENT_ID = "a40"
TABLE = "t40_cdr05_collapse"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

POST_HOC = ("merging CDR 0.5 is a POST-HOC REDUCTION of a pre-specified three-class task. "
            "No model was retrained; the three-class argmax is mapped through the same "
            "reduction as the labels")

THREE_CLASS = "three class, no reduction"
MERGE_DOWN = "CDR 0.5 merged into CDR 0"
MERGE_UP = "CDR 0.5 merged into CDR >= 1"
REDUCTIONS = [THREE_CLASS, MERGE_DOWN, MERGE_UP]

POSITIVE_NAME = {MERGE_DOWN: "CDR >= 1", MERGE_UP: "any impairment"}

UNSTRATIFIED = "unstratified participant"
STRATIFIED = "stratified participant, resampled within the ORIGINAL three classes"

METRICS = ["accuracy percent", "F1 positive class", "macro F1", "AUC"]
BOOTSTRAP_FOR = {"accuracy percent": UNSTRATIFIED, "F1 positive class": UNSTRATIFIED,
                 "macro F1": UNSTRATIFIED, "AUC": STRATIFIED}


def reduce_labels(y: np.ndarray, reduction: str) -> np.ndarray:
    """The binary truth under one reduction. Positive is always the impaired side."""
    if reduction == MERGE_DOWN:
        return (y == 2).astype(int)
    if reduction == MERGE_UP:
        return (y > 0).astype(int)
    raise ValueError(f"{reduction!r} is not a binary reduction")


def reduce_pred(pred: np.ndarray, reduction: str) -> np.ndarray:
    return reduce_labels(pred, reduction)


def reduce_score(probs: np.ndarray, reduction: str) -> np.ndarray:
    """P(positive) under the reduction, by summing the merged classes' probabilities."""
    if reduction == MERGE_DOWN:
        return probs[:, 2]
    if reduction == MERGE_UP:
        return probs[:, 1] + probs[:, 2]
    raise ValueError(f"{reduction!r} is not a binary reduction")


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(num), np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def binary_metrics(truth: np.ndarray, pred: np.ndarray, W: np.ndarray) -> Dict[str, np.ndarray]:
    """Accuracy, positive-class F1 and macro-F1 on every resample at once.

    counts() is a36's, reused rather than rewritten, so a fix to the 2x2
    bookkeeping reaches both files.
    """
    c = counts(truth, pred, W)
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    f1p = safe_ratio(2.0 * c["tp"], 2.0 * c["tp"] + c["fp"] + c["fn"])
    f1n = safe_ratio(2.0 * c["tn"], 2.0 * c["tn"] + c["fn"] + c["fp"])
    return {"accuracy percent": 100.0 * safe_ratio(c["tp"] + c["tn"], n),
            "F1 positive class": f1p,
            "macro F1": np.nanmean(np.stack([f1p, f1n]), axis=0)}


def three_class_metrics(y: np.ndarray, pred: np.ndarray, W: np.ndarray) -> Dict[str, np.ndarray]:
    """The unreduced reference: accuracy and three-class macro-F1 on every resample."""
    ind = np.zeros((len(y), N_CLASSES * N_CLASSES))
    ind[np.arange(len(y)), y * N_CLASSES + pred] = 1.0
    cm = (W @ ind).reshape(-1, N_CLASSES, N_CLASSES)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    f1 = safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    total = cm.sum(axis=(-2, -1))
    return {"accuracy percent": 100.0 * safe_ratio(tp.sum(axis=-1), total),
            "F1 positive class": np.full(len(total), np.nan),
            "macro F1": np.nanmean(f1, axis=-1)}


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Headline metrics under two binary reductions of CDR 0.5.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"paired bootstrap resamples (default {N_BOOT})")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--full-from", default=None, metavar="EXPERIMENTS_DIR")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"a40  CDR 0.5 collapsed, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     {POST_HOC}")

    cohorts: List[Cohort] = [Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)]
    full_dir = find_full_tree(args.full_from)
    if full_dir is None:
        warn("no full-cohort experiments folder found, so only the restricted cohort is "
             "reported. Pass --full-from to include the full cohort.")
    else:
        try:
            cohorts.insert(0, Cohort("full", FULL_EXPERIMENTS, full_dir))
        except SystemExit as exc:                                   # noqa: PERF203
            warn(f"full cohort unusable ({exc}); reporting the restricted cohort alone")
    for coh in cohorts:
        c = np.bincount(coh.labels, minlength=N_CLASSES)
        print(f"     {coh.name:<11} {len(coh.arms)} arms, {coh.n_participants} participants, "
              f"{coh.n_images:,} images, CDR 0 / 0.5 / >=1 = {c[0]} / {c[1]} / {c[2]}")

    rows: List[Dict] = []
    beats: Dict[Tuple[str, str, str], Tuple[int, int]] = {}

    for coh in cohorts:
        y = coh.labels
        n = coh.n_participants
        arms = [a for a in ARM_ORDER if a in coh.arms] + \
               [a for a in sorted(coh.arms) if a not in ARM_ORDER]
        W_un = multiplicity(unstratified_participant_bootstrap(
            n, n_boot, "a40:count", BASE_SEED, cohort=coh.name), n)
        strat_idx = stratified_participant_bootstrap(y, n_boot, "a40:auc", BASE_SEED,
                                                    cohort=coh.name)
        W_st = multiplicity(strat_idx, n)
        one = np.ones((1, n))

        for reduction in REDUCTIONS:
            binary = reduction != THREE_CLASS
            if binary:
                truth = reduce_labels(y, reduction)
                n_pos = int(truth.sum())
                maj = int(np.bincount(truth, minlength=2).argmax())
                const_pred = np.full(n, maj, dtype=int)
                const_score = np.full(n, 0.5)
            else:
                truth = y
                n_pos = int((y > 0).sum())
                maj = int(np.bincount(y, minlength=N_CLASSES).argmax())
                const_pred = np.full(n, maj, dtype=int)
                const_score = None
            const_acc = 100.0 * float((const_pred == truth).mean())
            print(f"\n     {coh.name} / {reduction}: positive {n_pos} of {n}, majority class "
                  f"{maj}, constant accuracy {const_acc:.2f} percent")

            # -- every arm and the constant predictor on the SAME draws ------------
            draw: Dict[Tuple[str, str], np.ndarray] = {}
            point: Dict[Tuple[str, str], float] = {}
            entries = arms + [CONSTANT]
            for a in entries:
                if a == CONSTANT:
                    pr = const_pred
                    sc = const_score
                else:
                    frame = coh.part[a]
                    pr = frame["pred"].to_numpy().astype(int)
                    sc = frame[[f"p{c}" for c in range(N_CLASSES)]].to_numpy(dtype=float)
                if binary:
                    pr_b = pr if a == CONSTANT else reduce_pred(pr, reduction)
                    m_draw = binary_metrics(truth, pr_b, W_un)
                    m_point = binary_metrics(truth, pr_b, one)
                else:
                    m_draw = three_class_metrics(truth, pr, W_un)
                    m_point = three_class_metrics(truth, pr, one)
                for k, v in m_draw.items():
                    draw[(a, k)] = v
                    point[(a, k)] = float(m_point[k][0])
                if binary:
                    s = np.full(n, 0.5) if a == CONSTANT else reduce_score(sc, reduction)
                    draw[(a, "AUC")] = auc_draws(s, truth.astype(bool), W_st)
                    point[(a, "AUC")] = float(auc_draws(s, truth.astype(bool), one)[0])
                else:
                    draw[(a, "AUC")] = np.full(n_boot, np.nan)
                    point[(a, "AUC")] = float("nan")

            for a in entries:
                for metric in METRICS:
                    d = draw[(a, metric)]
                    lo, hi = percentile_interval(d)
                    rows.append({
                        "kind": "value", "cohort": coh.name, "reduction": reduction,
                        "unit": "participant", "arm": a, "metric": metric,
                        "bootstrap": BOOTSTRAP_FOR[metric],
                        "value": point[(a, metric)], "ci_low": lo, "ci_high": hi,
                        "ci_level": 0.95,
                        "positive_class": POSITIVE_NAME.get(reduction, "n/a, three class"),
                        "n_positive": n_pos, "n_participants": n,
                        "majority_class": maj, "constant_accuracy_percent": const_acc,
                        "n_undefined_draws": int(np.isnan(d).sum()),
                        "n_bootstrap": n_boot, "family": UNDECLARED,
                        "code_fingerprint": coh.fingerprint.get(a, ""),
                        "note": POST_HOC})
            for a in arms:
                for metric in METRICS:
                    if metric == "AUC" and not binary:
                        continue
                    d = draw[(a, metric)] - draw[(CONSTANT, metric)]
                    lo, hi = percentile_interval(d)
                    st = bootstrap_p(d)
                    rows.append({
                        "kind": "vs_constant", "cohort": coh.name, "reduction": reduction,
                        "unit": "participant", "arm": a, "metric": metric,
                        "bootstrap": BOOTSTRAP_FOR[metric], "reference": CONSTANT,
                        "value": point[(a, metric)],
                        "reference_value": point[(CONSTANT, metric)],
                        "difference": point[(a, metric)] - point[(CONSTANT, metric)],
                        "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                        "p_bootstrap": st["p_bootstrap"],
                        "p_at_resolution_floor": st["p_at_resolution_floor"],
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "beats_constant": bool(lo > 0),
                        "positive_class": POSITIVE_NAME.get(reduction, "n/a, three class"),
                        "n_positive": n_pos, "n_participants": n,
                        "constant_accuracy_percent": const_acc,
                        "n_undefined_draws": int(np.isnan(d).sum()),
                        "n_bootstrap": n_boot, "family": UNDECLARED,
                        "code_fingerprint": coh.fingerprint.get(a, ""),
                        "note": "the constant predictor is rebuilt from THIS reduction's "
                                "own labels, which is the only way the comparison stays "
                                "honest when the majority share moves"})
            for metric in METRICS:
                sub = [r for r in rows if r["kind"] == "vs_constant"
                       and r["cohort"] == coh.name and r["reduction"] == reduction
                       and r["metric"] == metric]
                if sub:
                    beats[(coh.name, reduction, metric)] = (
                        int(sum(r["beats_constant"] for r in sub)), len(sub))

    table = pd.DataFrame(rows)
    lead = ["kind", "cohort", "reduction", "arm", "metric", "unit", "bootstrap", "value",
            "reference_value", "difference", "ci_low", "ci_high", "ci_level",
            "p_bootstrap", "beats_constant", "family"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    val = table[table["kind"] == "value"]
    print("\n" + "=" * 104)
    print("a40  WHAT SURVIVES IF CDR 0.5 IS NOT ITS OWN CATEGORY?")
    print("=" * 104)
    print(f"  {POST_HOC}.")
    for coh in cohorts:
        print(f"\n  == cohort: {coh.name}, {coh.n_participants} participants, participant "
              f"level")
        for reduction in REDUCTIONS:
            v = val[(val["cohort"] == coh.name) & (val["reduction"] == reduction)]
            if v.empty:
                continue
            cst = v[v["arm"] == CONSTANT].iloc[0]
            print(f"     -- {reduction}   (positive {int(cst['n_positive'])} of "
                  f"{int(cst['n_participants'])}, constant predictor accuracy "
                  f"{cst['constant_accuracy_percent']:.2f} percent)")
            shown = [m for m in METRICS
                     if not v[(v["metric"] == m)]["value"].isna().all()]
            print(f"        {'arm':<16}" + "".join(f"{m:>24}" for m in shown))
            arms = [a for a in ARM_ORDER if a in coh.arms] + \
                   [a for a in sorted(coh.arms) if a not in ARM_ORDER]
            for a in arms + [CONSTANT]:
                cells = []
                for m in shown:
                    r = v[(v["arm"] == a) & (v["metric"] == m)].iloc[0]
                    cells.append(f"{r['value']:>8.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
                print(f"        {a:<16}" + "".join(f"{c:>24}" for c in cells))
            line = []
            for m in shown:
                got = beats.get((coh.name, reduction, m))
                line.append(f"{got[0]} of {got[1]}" if got else "n/a")
            print(f"        {'beats constant':<16}" + "".join(f"{x:>24}" for x in line))

    print(f"\n  DOES THE CONCLUSION CHANGE?")
    print(f"  The three-class row is the paper's own task. Compare the beats-constant "
          f"counts down each column.")
    for coh in cohorts:
        for metric in ("accuracy percent", "macro F1"):
            got = [beats.get((coh.name, r, metric)) for r in REDUCTIONS]
            cells = "   ".join(f"{r}: {g[0]} of {g[1]}" if g else f"{r}: n/a"
                               for r, g in zip(REDUCTIONS, got))
            base = got[0]
            changed = any(g and base and g[0] != base[0] for g in got[1:])
            print(f"     {coh.name:<11}{metric:<18}{cells}"
                  f"   -> {'CHANGES' if changed else 'unchanged'}")
    print(f"  A count that moves is a reduction the paper must report. A count that does "
          f"not move is a")
    print(f"  robustness statement the paper is entitled to make, in one sentence, naming "
          f"both reductions.")
    print(f"\n  THE TRAP IN THE '{MERGE_DOWN}' ROWS. The constant predictor there scores "
          f"high accuracy because")
    print(f"  the negative class is most of the cohort. An arm's raw accuracy under that "
          f"reduction looks far")
    print(f"  better than the paper's three-class number and means less. Only the "
          f"difference against the")
    print(f"  within-reduction constant is comparable.")
    print(f"\n  EVERY comparison here is UNDECLARED. No registry family covers a binary "
          f"reduction, no p-value")
    print(f"  here is corrected, and every interval is a nominal 95 percent one.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "headline metrics recomputed under two post-hoc binary reductions of the "
        "three-class task, CDR 0.5 merged into CDR 0 and CDR 0.5 merged into CDR >= 1, on "
        "both cohorts and all arms, each against that reduction's own constant "
        "majority-class predictor",
        "accuracy, positive-class F1, macro-F1 and midrank AUC",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
