#!/usr/bin/env python3
"""a36: what does the argmax operating point actually deliver to a clinician?

    python analysis/a36_operating_point.py
    python analysis/a36_operating_point.py --quick      # 200 resamples
    python analysis/a36_operating_point.py --boot 10000

THE QUESTION A CLINICAL REVIEWER WILL ASK FIRST
-----------------------------------------------
Accuracy, macro-F1, ECE and Brier all describe a model. None of them tells a
clinician what happens to a patient. The four numbers that do are sensitivity,
specificity, positive predictive value and negative predictive value at the
operating point the model is actually used at, which here is the argmax of the
participant-level mean predicted distribution. That is the same rule the
manuscript's Table 4 uses, so this file adds no new decision rule; it reports the
consequences of the one already in the paper.

Two binary readings of the three-class problem are reported, because they are the
two a clinician would recognise:

    CDR >= 1 versus rest        does the model find the people with dementia
    any impairment versus CDR 0 does the model find the people who are not normal

THE NUMBER THIS FILE EXISTS FOR
-------------------------------
On the restricted cohort the best case is 9 of 23 CDR >= 1 participants
identified, a sensitivity of 0.391. That number is currently quoted without an
interval. Nine successes out of twenty three is a small-sample proportion, and a
reviewer who does not know its interval cannot tell whether 0.391 is
distinguishable from chance or from 0.6. So the interval on it is computed here
and printed at the top of the summary panel rather than buried in a table.

WHY THE STRATIFIED BOOTSTRAP, AND WHAT IT COSTS
-----------------------------------------------
Participants are resampled WITHIN their true CDR class, so every draw holds 85
CDR 0, 58 CDR 0.5 and 23 CDR >= 1. An unstratified draw over 166 people returns
zero CDR >= 1 participants often enough to matter, and a sensitivity computed on
zero positives is not a noisy estimate of sensitivity, it is undefined. The
stratified draw removes that failure mode and is what every reported interval in
this file uses.

The cost is real and is stated rather than hidden. Stratifying fixes the class
prevalence at its observed value, so a stratified interval on PPV or NPV is
CONDITIONAL ON THAT PREVALENCE. PPV and NPV are functions of prevalence, so the
conditional interval is narrower than one that also let the 23 vary. Both are
therefore computed: the stratified interval is the one reported, and the
unstratified companion is carried beside it in the table with the count of draws
in which it was undefined. If the two disagree, the disagreement is the finding
and not a bug.

Sensitivity and specificity do not depend on prevalence, so for those two the
stratified interval is the natural object and the companion is a check.

MULTIPLICITY
------------
No family in analysis/_registry.py covers an operating-point analysis. The
declared families are about differences between arms and against a constant
predictor on accuracy, macro-F1, ECE, QWK and AUROC. Absorbing 8 arms times 2
contrasts times 4 quantities into any of them would break a declared size. So
every row carries the literal string "undeclared", every interval is a nominal 95
percent interval, and nothing here is a pre-registered test. Nothing is written
to the registry: the changelog entry is PRINTED for the first author to paste.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import save_table                                       # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                              # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, RESTRICTED_EXPERIMENTS,           # noqa: E402
                                    changelog_entry, percentile_interval,
                                    stratified_participant_bootstrap,
                                    unstratified_participant_bootstrap)
from analysis.a23_protocol_audit import ARM_ORDER                             # noqa: E402
from analysis.a25_increment_over_metadata import multiplicity                 # noqa: E402

SOURCE = "analysis/a36_operating_point.py"
EXPERIMENT_ID = "a36"
TABLE = "t36_operating_point"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

OPERATING_POINT = ("argmax of the participant-level mean predicted distribution, "
                   "the manuscript's Table 4 rule, no threshold is tuned here")

SEVERE = "CDR >= 1 versus rest"
ANY_IMPAIRMENT = "any impairment versus CDR 0"
CONTRASTS = [SEVERE, ANY_IMPAIRMENT]

STRATIFIED = "stratified participant, resampled within true CDR class"
UNSTRATIFIED = "unstratified participant"

QUANTITIES = ["sensitivity", "specificity", "ppv", "npv"]
PREVALENCE_DEPENDENT = {"sensitivity": False, "specificity": False,
                        "ppv": True, "npv": True}

# The headline the panel leads with. Recomputed, never quoted: if the data no
# longer gives 9 of 23 the script says so instead of printing a stale sentence.
HEADLINE_NUMERATOR = 9
HEADLINE_DENOMINATOR = 23
HEADLINE_SENSITIVITY = 0.391


def binary_view(labels: np.ndarray, pred: np.ndarray,
                contrast: str) -> Tuple[np.ndarray, np.ndarray]:
    """True positive flag and predicted positive flag for one binary reading."""
    if contrast == SEVERE:
        return labels == 2, pred == 2
    if contrast == ANY_IMPAIRMENT:
        return labels > 0, pred > 0
    raise ValueError(f"unknown contrast {contrast!r}")


def counts(truth: np.ndarray, flag: np.ndarray, W: np.ndarray) -> Dict[str, np.ndarray]:
    """Weighted 2x2 counts for every resample at once.

    W is (n_boot, n) multiplicities, so each cell is one matrix-vector product
    against a fixed indicator. Row 0 of W is used by the caller for the point
    estimate by passing a row of ones, which keeps the point estimate and the
    draws on exactly one code path.
    """
    t = np.asarray(truth, dtype=bool)
    f = np.asarray(flag, dtype=bool)
    return {"tp": W @ (t & f).astype(float), "fp": W @ (~t & f).astype(float),
            "fn": W @ (t & ~f).astype(float), "tn": W @ (~t & ~f).astype(float)}


def rates(c: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The four clinical rates, NaN where the denominator is empty.

    A NaN here is not a missing value to be filled. It is a draw in which the
    quantity does not exist: no positives means no sensitivity. Counting those
    draws is how this file reports the cost of the unstratified companion.
    """
    def safe(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        out = np.full(np.shape(num), np.nan, dtype=float)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out
    return {"sensitivity": safe(c["tp"], c["tp"] + c["fn"]),
            "specificity": safe(c["tn"], c["tn"] + c["fp"]),
            "ppv": safe(c["tp"], c["tp"] + c["fp"]),
            "npv": safe(c["tn"], c["tn"] + c["fn"])}


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Sensitivity, specificity, PPV and NPV at the argmax operating point.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"participant-clustered resamples (default {N_BOOT})")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"a36  operating point, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")
    print(f"     operating point: {OPERATING_POINT}")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    y = coh.labels
    cls = np.bincount(y, minlength=N_CLASSES)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images, CDR 0 / 0.5 / >=1 = {cls[0]} / {cls[1]} / {cls[2]}")

    arms = [a for a in ARM_ORDER if a in coh.arms] + \
           [a for a in sorted(coh.arms) if a not in ARM_ORDER]

    # -- the two shared draws. Every arm is scored on the same rows. --------------
    strat_idx = stratified_participant_bootstrap(y, n_boot, "a36:strat", BASE_SEED)
    unstrat_idx = unstratified_participant_bootstrap(coh.n_participants, n_boot,
                                                     "a36:unstrat", BASE_SEED)
    W = {STRATIFIED: multiplicity(strat_idx, coh.n_participants),
         UNSTRATIFIED: multiplicity(unstrat_idx, coh.n_participants)}
    one = np.ones((1, coh.n_participants))
    print(f"     two draws built: {STRATIFIED}, and {UNSTRATIFIED} as a companion")

    rows: List[Dict] = []
    point: Dict[Tuple[str, str, str], float] = {}
    interval: Dict[Tuple[str, str, str, str], Tuple[float, float]] = {}

    for arm in arms:
        frame = coh.part[arm]
        if not np.array_equal(frame["label"].to_numpy().astype(int), y):
            raise SystemExit(f"a36 STOPPED. {arm} label order does not match the cohort; "
                             f"every count below would be scored against the wrong truth.")
        pred = frame["pred"].to_numpy().astype(int)
        for contrast in CONTRASTS:
            truth, flag = binary_view(y, pred, contrast)
            c_point = counts(truth, flag, one)
            r_point = rates(c_point)
            n_pos, n_neg = int(truth.sum()), int((~truth).sum())
            n_flag = int(flag.sum())
            for q in QUANTITIES:
                point[(arm, contrast, q)] = float(r_point[q][0])
            for boot_name, Wm in W.items():
                r_draw = rates(counts(truth, flag, Wm))
                for q in QUANTITIES:
                    d = r_draw[q]
                    lo, hi = percentile_interval(d)
                    n_undef = int(np.isnan(d).sum())
                    interval[(arm, contrast, q, boot_name)] = (lo, hi)
                    rows.append({
                        "arm": arm, "contrast": contrast, "quantity": q,
                        "bootstrap": boot_name,
                        "reported_interval": boot_name == STRATIFIED,
                        "value": float(r_point[q][0]),
                        "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                        "prevalence_dependent": PREVALENCE_DEPENDENT[q],
                        "n_undefined_draws": n_undef,
                        "tp": int(c_point["tp"][0]), "fp": int(c_point["fp"][0]),
                        "fn": int(c_point["fn"][0]), "tn": int(c_point["tn"][0]),
                        "n_true_positive": n_pos, "n_true_negative": n_neg,
                        "n_flagged_positive": n_flag,
                        "n_participants": coh.n_participants,
                        "operating_point": OPERATING_POINT,
                        "n_bootstrap": n_boot, "family": UNDECLARED,
                        "code_fingerprint": coh.fingerprint.get(arm, ""),
                        "note": ("reported interval, conditional on the observed class "
                                 "counts 85 / 58 / 23"
                                 if boot_name == STRATIFIED else
                                 "companion only. An unstratified draw can empty the "
                                 "23-participant class, which is why it is not the "
                                 "reported interval")})
        print(f"       {arm:<16} scored on both contrasts")

    table = pd.DataFrame(rows)
    lead = ["arm", "contrast", "quantity", "bootstrap", "reported_interval", "value",
            "ci_low", "ci_high", "ci_level", "tp", "fp", "fn", "tn"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    sens = {a: point[(a, SEVERE, "sensitivity")] for a in arms}
    best_arm = max(sens, key=lambda k: sens[k])
    best = table[(table["arm"] == best_arm) & (table["contrast"] == SEVERE) &
                 (table["quantity"] == "sensitivity") &
                 (table["bootstrap"] == STRATIFIED)].iloc[0]

    print("\n" + "=" * 100)
    print("a36  WHAT THE ARGMAX OPERATING POINT DELIVERS, WITH INTERVALS")
    print("=" * 100)
    print(f"  THE NUMBER THE REVIEWER WILL ASK ABOUT.")
    print(f"  Best sensitivity for {SEVERE} is {best_arm}: "
          f"{int(best['tp'])} of {int(best['tp'] + best['fn'])} participants found,")
    print(f"  sensitivity {best['value']:.3f}, 95 percent interval "
          f"[{best['ci_low']:.3f}, {best['ci_high']:.3f}].")
    print(f"  That interval is {best['ci_high'] - best['ci_low']:.3f} wide on 23 "
          f"positive participants. Any claim that rests on 0.39")
    print(f"  rather than on the bottom of that interval is a claim this cohort cannot "
          f"support.")
    if not (int(best["tp"]) == HEADLINE_NUMERATOR
            and int(best["tp"] + best["fn"]) == HEADLINE_DENOMINATOR):
        print(f"  !! The manuscript quotes {HEADLINE_NUMERATOR} of {HEADLINE_DENOMINATOR}, "
              f"sensitivity {HEADLINE_SENSITIVITY:.3f}. This run gives "
              f"{int(best['tp'])} of {int(best['tp'] + best['fn'])}. Reconcile before "
              f"quoting either.")

    for contrast in CONTRASTS:
        truth, _ = binary_view(y, np.zeros_like(y), contrast)
        print(f"\n  -- {contrast}   ({int(truth.sum())} positive, "
              f"{int((~truth).sum())} negative of {coh.n_participants})")
        print(f"     {'arm':<16}" + "".join(f"{q:>22}" for q in QUANTITIES))
        for arm in arms:
            cells = []
            for q in QUANTITIES:
                lo, hi = interval[(arm, contrast, q, STRATIFIED)]
                cells.append(f"{point[(arm, contrast, q)]:.3f} [{lo:.2f}, {hi:.2f}]")
            print(f"     {arm:<16}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n  EVERY interval above is the {STRATIFIED} draw at {n_boot:,} resamples.")
    print(f"  Stratifying is what keeps the 23-participant CDR >= 1 class from emptying "
          f"in a draw and")
    print(f"  making sensitivity undefined. It also fixes prevalence, so the PPV and NPV "
          f"intervals are")
    print(f"  CONDITIONAL on 85 / 58 / 23. The unstratified companion is in the CSV under "
          f"bootstrap =")
    un = table[table["bootstrap"] == UNSTRATIFIED]
    print(f"  '{UNSTRATIFIED}', where {int((un['n_undefined_draws'] > 0).sum())} of "
          f"{len(un)} rows had at least one undefined draw")
    print(f"  (worst {int(un['n_undefined_draws'].max())} of {n_boot:,}).")
    print(f"\n  EVERY comparison here is UNDECLARED. No registry family covers an "
          f"operating-point analysis,")
    print(f"  no p-value is reported, and every interval is a nominal 95 percent one.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "sensitivity, specificity, PPV and NPV of all eight restricted-cohort arms at the "
        "argmax operating point, for CDR >= 1 versus rest and for any impairment versus "
        "CDR 0, with participant-clustered intervals",
        "sensitivity, specificity, positive predictive value, negative predictive value",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
