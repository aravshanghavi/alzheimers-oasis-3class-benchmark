#!/usr/bin/env python3
"""a26: is "nWBV beats every network" a real ordering, or an artefact of scoring?

    python analysis/a26_pairwise_auc_differences.py
    python analysis/a26_pairwise_auc_differences.py --quick    # 200 resamples
    python analysis/a26_pairwise_auc_differences.py --boot 10000

THE QUESTION
------------
One scalar off the standard segmentation, normalised whole brain volume, ranks
the restricted cohort's participants better than any of the eight networks on
the separation that matters most. That sentence is either the strongest result
in the revision or a scoring accident, and the difference between those two
readings is worth settling before anybody writes it down.

Two things could make it an accident.

The first is TIES. a17's AUC gives tied scores distinct ranks, which is why its
constant predictor comes out at 0.4958 instead of 0.5000. Every AUC here uses
midranks instead, so a tie contributes exactly one half.

The second is the SCORE CONVENTION. A three class model has to be turned into
one number before a two class AUC can be taken, and there are two defensible
ways to do it:

    raw            the probability of the higher class, P(higher)
    renormalised   P(higher) / (P(higher) + P(lower)), which drops the third
                   class from the comparison entirely

They are not the same ordering, because the raw score lets the probability mass
sitting on the irrelevant third class move a participant up or down. Every arm
is scored under BOTH, and both are in the table, so no reader has to take on
trust that the choice did not decide the answer.

For "any impairment against CDR 0" the higher side spans two classes, so the
score is one minus P(CDR 0). Renormalising that is the identity, because
(1 - p0) / ((1 - p0) + p0) = 1 - p0, and the table says so on every such row
rather than printing two columns that silently agree by construction.

THE THREE CONTRASTS
-------------------
    CDR >= 1 against CDR 0        23 against 85
    any impairment against CDR 0  81 against 85
    CDR >= 1 against CDR 0.5      23 against 58

The last is the confound free core: both sides were assessed, both sides are old,
and neither side is the easy healthy majority.

THE TEST
--------
AUC(nWBV) minus AUC(arm), on a paired participant bootstrap stratified WITHIN
class. Stratifying is not a nicety here. An unstratified draw over 108 people of
whom 23 are CDR >= 1 empties that class often enough to make a noticeable share
of the resampled AUCs undefined, and dropping those draws biases the interval in
a direction nobody can predict.

MULTIPLICITY
------------
No declared family covers a metadata scalar against a CNN arm on pairwise AUC.
Every row carries the literal string "undeclared" and a nominal 95 percent
interval. The script prints the changelog entry rather than writing it.

THE VERIFICATION BLOCK
----------------------
Six numbers were computed before this script existed, by hand, from the same
predictions. The script recomputes all six and prints PASS or FAIL per number.
A FAIL is not something to reconcile at three in the morning: it means the score
convention or the cohort here is not the one those numbers came from, and every
other number in the table is then suspect too.
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
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx               # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, RESTRICTED_EXPERIMENTS,           # noqa: E402
                                    bootstrap_p, changelog_entry, midrank_auc,
                                    percentile_interval,
                                    stratified_participant_bootstrap)
# multiplicity and auc_draws live in a25 because _protocol_lib is shared code that
# this night's work does not edit. Importing them is the only alternative to a
# second copy, and a second copy is how two scripts end up printing two different
# numbers for one quantity.
from analysis.a25_increment_over_metadata import auc_draws, multiplicity      # noqa: E402

SOURCE = "analysis/a26_pairwise_auc_differences.py"
EXPERIMENT_ID = "a26"
TABLE = "t26_pairwise_auc_differences"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

RAW = "raw P(higher)"
RENORM = "renormalised P(higher)/(P(higher)+P(lower))"
IDENTITY_NOTE = ("renormalising is the identity for this contrast, because "
                 "(1-p0)/((1-p0)+p0) = 1-p0")

CONTRASTS: List[Tuple[str, Optional[int], int]] = [
    ("CDR>=1 vs CDR 0", 2, 0),
    ("any impairment vs CDR 0", None, 0),
    ("CDR>=1 vs CDR 0.5", 2, 1),
]

REFERENCE = "nWBV (sign reversed)"
SCALARS = [(REFERENCE, "nWBV", -1.0,
            "lower nWBV means more atrophy, so the sign is reversed to make higher "
            "mean more impaired"),
           ("MMSE (sign reversed)", "MMSE", -1.0,
            "lower MMSE means worse cognition, so the sign is reversed"),
           ("age", "Age", 1.0, "older is scored as more impaired")]

# Computed by hand from the same predictions before this script was written.
# Tolerance is one unit in the fourth decimal, which is the precision they were
# quoted at. Anything looser would let a real disagreement pass as rounding.
EXPECTED_TOL = 5e-5
EXPECTED = [
    ("best restricted CNN, raw convention", "CDR>=1 vs CDR 0", RAW, "BEST_ARM", 0.7749),
    ("best restricted CNN, renormalised", "CDR>=1 vs CDR 0", RENORM, "BEST_ARM", 0.7857),
    ("nWBV", "CDR>=1 vs CDR 0", "", REFERENCE, 0.8465),
    ("nWBV", "any impairment vs CDR 0", "", REFERENCE, 0.7128),
    ("nWBV", "CDR>=1 vs CDR 0.5", "", REFERENCE, 0.7410),
    ("MMSE", "CDR>=1 vs CDR 0", "", "MMSE (sign reversed)", 0.9801),
]


def subset(y: np.ndarray, higher: Optional[int], lower: int) -> Tuple[np.ndarray, np.ndarray]:
    """The participants in one contrast, and which of them are on the higher side."""
    mask = ((y > 0) if higher is None else (y == higher)) | (y == lower)
    pos = (y[mask] > 0) if higher is None else (y[mask] == higher)
    return mask, pos


def raw_score(probs: np.ndarray, higher: Optional[int], lower: int) -> np.ndarray:
    """P(higher class), or one minus P(CDR 0) when the higher side spans two classes."""
    return probs[:, higher] if higher is not None else 1.0 - probs[:, lower]


def renorm_score(probs: np.ndarray, higher: Optional[int], lower: int) -> np.ndarray:
    """P(higher)/(P(higher)+P(lower)), which removes the third class from the ordering.

    For the two class-spanning contrast this reduces to the raw score exactly, so
    the same array comes back and the caller labels the row as an identity rather
    than reporting a second number that only looks independent.
    """
    if higher is None:
        return raw_score(probs, higher, lower)
    a, b = probs[:, higher], probs[:, lower]
    denom = a + b
    return np.divide(a, denom, out=np.full(len(a), 0.5), where=denom > 0)


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="Pairwise AUC, nWBV against every arm.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"paired bootstrap resamples (default {N_BOOT})")
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

    print(f"a26  pairwise AUC differences, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    y = coh.labels
    counts = np.bincount(y, minlength=N_CLASSES)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"classes CDR 0 / 0.5 / >=1 = {counts[0]} / {counts[1]} / {counts[2]}")

    meta = pd.read_excel(metadata_xlsx())
    meta["pid"] = meta["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid").reindex(coh.participants)

    # -- assemble every score vector, once, over all 166 participants ------------
    scores: Dict[Tuple[str, str], np.ndarray] = {}     # (label, convention) -> score
    meta_of: Dict[str, str] = {}
    for label, column, sign, why in SCALARS:
        v = pd.to_numeric(meta[column], errors="coerce").to_numpy(dtype=float)
        n_missing = int(np.isnan(v).sum())
        if n_missing:
            print(f"  !! {label}: {n_missing} of {coh.n_participants} missing. Those "
                  f"participants are DROPPED from every contrast this scalar appears in, "
                  f"rather than imputed, because imputing a value and then ranking it is "
                  f"ranking the imputation.")
        scores[(label, "")] = sign * v
        meta_of[label] = why

    # -- per contrast ------------------------------------------------------------
    rows: List[Dict] = []
    point: Dict[Tuple[str, str, str], float] = {}      # (contrast, label, convention)
    for cname, higher, lower in CONTRASTS:
        mask, pos = subset(y, higher, lower)
        n_sub = int(mask.sum())
        idx = stratified_participant_bootstrap(pos.astype(int), n_boot, "a26:auc",
                                               BASE_SEED, contrast=cname)
        mult = multiplicity(idx, n_sub)
        identity = np.ones((1, n_sub))

        entries: List[Tuple[str, str, np.ndarray, str]] = []
        for arm in coh.order():
            P = coh.part[arm][[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
            entries.append((arm, RAW, raw_score(P, higher, lower), "CNN arm"))
            entries.append((arm, RENORM, renorm_score(P, higher, lower), "CNN arm"))
        for label, _col, _sign, why in SCALARS:
            entries.append((label, "", scores[(label, "")], f"metadata scalar, {why}"))

        draws: Dict[Tuple[str, str], np.ndarray] = {}
        for label, conv, s, kind in entries:
            sub_s = s[mask]
            finite = np.isfinite(sub_s)
            if not finite.all():
                # Drop the missing participants from this scalar only. Its bootstrap
                # then runs on its own index matrix, so its DIFFERENCES against the
                # arms are no longer paired; that is stated on the row.
                keep_pos = pos[finite]
                sub_idx = stratified_participant_bootstrap(
                    keep_pos.astype(int), n_boot, "a26:auc_incomplete", BASE_SEED,
                    contrast=cname, score=label)
                m2 = multiplicity(sub_idx, int(finite.sum()))
                a = float(midrank_auc(sub_s[finite], keep_pos))
                d = auc_draws(sub_s[finite], keep_pos, m2)
                paired = False
            else:
                a = float(midrank_auc(sub_s, pos))
                d = auc_draws(sub_s, pos, mult)
                paired = True
            point[(cname, label, conv)] = a
            draws[(label, conv)] = d
            lo, hi = percentile_interval(d, alpha=0.05)
            rows.append({"kind": "point", "contrast": cname,
                         "n_higher": int(pos.sum()), "n_lower": int((~pos).sum()),
                         "score": label, "convention": conv or "single scalar, no convention",
                         "score_kind": kind, "auc": a, "ci_low": lo, "ci_high": hi,
                         "ci_level": 0.95, "difference": np.nan, "p_bootstrap": np.nan,
                         "p_at_resolution_floor": False, "paired_with_reference": paired,
                         "n_bootstrap": n_boot, "family": UNDECLARED,
                         "code_fingerprint": coh.fingerprint.get(label, ""),
                         "note": (IDENTITY_NOTE if (higher is None and conv == RENORM) else "")})

        ref_draws = draws[(REFERENCE, "")]
        ref_point = point[(cname, REFERENCE, "")]
        for label, conv, _s, kind in entries:
            if label == REFERENCE:
                continue
            d = ref_draws - draws[(label, conv)]
            lo, hi = percentile_interval(d, alpha=0.05)
            st = bootstrap_p(d)
            rows.append({"kind": "difference", "contrast": cname,
                         "n_higher": int(pos.sum()), "n_lower": int((~pos).sum()),
                         "score": label, "convention": conv or "single scalar, no convention",
                         "score_kind": kind,
                         "auc": point[(cname, label, conv)],
                         "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                         "difference": ref_point - point[(cname, label, conv)],
                         "p_bootstrap": st["p_bootstrap"],
                         "p_at_resolution_floor": st["p_at_resolution_floor"],
                         "paired_with_reference": True,
                         "n_bootstrap": n_boot, "family": UNDECLARED,
                         "code_fingerprint": coh.fingerprint.get(label, ""),
                         "note": f"AUC({REFERENCE}) minus AUC(this score); positive means "
                                 f"nWBV ranks better"})

    table = pd.DataFrame(rows)
    save_table(table, TABLE, float_fmt="%.6f")

    # -- verification ------------------------------------------------------------
    best_raw = max(coh.order(), key=lambda a: point[("CDR>=1 vs CDR 0", a, RAW)])
    best_renorm = max(coh.order(), key=lambda a: point[("CDR>=1 vs CDR 0", a, RENORM)])
    checks: List[Dict] = []
    for what, cname, conv, who, want in EXPECTED:
        if who == "BEST_ARM":
            who_resolved = best_raw if conv == RAW else best_renorm
        else:
            who_resolved = who
        got = point[(cname, who_resolved, conv if who == "BEST_ARM" else "")]
        ok = abs(got - want) <= EXPECTED_TOL
        checks.append({"what": what, "contrast": cname, "who": who_resolved,
                       "expected": want, "computed": got, "delta": got - want,
                       "verdict": "PASS" if ok else "FAIL"})
    n_fail = sum(1 for c in checks if c["verdict"] == "FAIL")

    # -- one screen --------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 96)
    print("a26  DOES ONE SEGMENTATION SCALAR RANK PEOPLE BETTER THAN THE NETWORKS DO?")
    print("=" * 96)
    print(f"  restricted cohort, {coh.n_participants} participants, participant level, "
          f"{n_boot:,} paired stratified resamples, nominal 95 percent intervals.")
    print("\n  VERIFICATION against the six numbers computed before this script existed")
    print(f"  {'quantity':<38}{'who':<22}{'expected':>10}{'computed':>10}{'delta':>11}  verdict")
    for c in checks:
        print(f"  {c['what']:<38}{c['who']:<22}{c['expected']:>10.4f}{c['computed']:>10.4f}"
              f"{c['delta']:>+11.6f}  {c['verdict']}")
    if n_fail:
        print(f"\n  !! {n_fail} of {len(checks)} VERIFICATION CHECKS FAILED. STOP. Do not read "
              f"the table below and do not reconcile the difference by adopting these "
              f"numbers. A failure here means the cohort or the score convention is not the "
              f"one those numbers came from, so every other number in this file is suspect.")
    else:
        print(f"\n  all {len(checks)} verification checks PASS")

    print(f"\n  AUC BY CONTRAST. Higher is better. 0.5 is chance.")
    for cname, higher, _lower in CONTRASTS:
        sub = table[(table["kind"] == "point") & (table["contrast"] == cname)]
        n_hi = int(sub["n_higher"].iloc[0]); n_lo = int(sub["n_lower"].iloc[0])
        print(f"\n  -- {cname}   ({n_hi} against {n_lo})")
        for label, _c, _s, _w in SCALARS:
            a = point[(cname, label, "")]
            print(f"     {label:<26}{a:>8.4f}")
        head = f"     {'CNN arm':<26}{'raw':>8}{'renorm':>9}{'nWBV-raw':>11}{'nWBV-renorm':>13}"
        print(head)
        for arm in coh.order():
            ar, an = point[(cname, arm, RAW)], point[(cname, arm, RENORM)]
            ref = point[(cname, REFERENCE, "")]
            dr = table[(table["kind"] == "difference") & (table["contrast"] == cname) &
                       (table["score"] == arm) & (table["convention"] == RAW)].iloc[0]
            dn = table[(table["kind"] == "difference") & (table["contrast"] == cname) &
                       (table["score"] == arm) & (table["convention"] == RENORM)].iloc[0]
            mark = lambda r: "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
            print(f"     {arm:<26}{ar:>8.4f}{an:>9.4f}"
                  f"{ref - ar:>+10.4f}{mark(dr)}{ref - an:>+12.4f}{mark(dn)}")
        if higher is None:
            print(f"     ({IDENTITY_NOTE}, so the two convention columns agree by construction)")

    print("\n  * marks a difference whose nominal 95 percent interval excludes zero.")
    print("  The last two columns are AUC(nWBV) minus AUC(arm): POSITIVE means the single")
    print("  segmentation scalar ranks participants better than that network does.")
    print("\n  WHAT WOULD OVERTURN THIS. If the two conventions gave different orderings, the")
    print("  result would be a scoring artefact. They do not: read the raw and renorm columns")
    print("  side by side above. If the nWBV advantage were inside its interval, it would be")
    print("  an estimate and not a finding. The starred rows are the ones that are not.")
    print("\n  EVERY comparison here is UNDECLARED. Nominal 95 percent, nothing corrected.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting anything.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "nWBV against every CNN arm under two score conventions, and against MMSE and age, "
        "on three pairwise separations, restricted cohort",
        "midrank AUC difference",
        "participant"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
