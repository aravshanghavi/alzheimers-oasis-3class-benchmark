#!/usr/bin/env python3
"""a37: is the age 60 cutoff load-bearing, or would 65 or 70 say the same thing?

    python analysis/a37_age_threshold.py
    python analysis/a37_age_threshold.py --quick      # 200 resamples
    python analysis/a37_age_threshold.py --boot 10000

EVALUATION SIDE ONLY. NOTHING IS RETRAINED.
-------------------------------------------
This is the first thing to understand about every number in this file, and it is
repeated in the output header so it cannot be lost between the script and the
manuscript. The networks are the SAME eight restricted-cohort arms, trained once
on the participants aged 60 and over. All this file does is score them again on
two nested subsets of the very participants they were already scored on. No
model is refitted, no fold is redrawn, no hyperparameter is touched.

That has a consequence worth stating plainly. An arm evaluated on the 70 and over
subset was trained on a population that includes people in their sixties, so
these rows answer "does the conclusion survive a stricter evaluation cohort", not
"what would a model trained only on the old look like". The second question needs
GPU time this revision does not have.

WHY THE QUESTION MATTERS
------------------------
The paper's primary cohort is defined by a CDR assessment plus an age floor of 60.
Sixty is a defensible floor and also an arbitrary one. A reviewer is entitled to
ask whether the conclusions are a property of dementia detection or a property of
that particular number. If the picture at 65 and at 70 looks like the picture at
60, the floor is not load-bearing and the paper can say so. If it changes, the
paper has to say that instead.

THE CONSTANT PREDICTOR MOVES WITH THE SUBSET, WHICH IS THE POINT
----------------------------------------------------------------
Raising the age floor removes mostly younger CDR 0 participants, so the majority
share shifts under the model. An arm can gain or lose several points of accuracy
between these subsets purely because the floor moved. So every subset carries its
OWN constant majority-class predictor, built from the labels of that subset, and
every arm is differenced against that one. A raw accuracy comparison across the
three subsets without this is uninterpretable, and it is exactly the comparison a
hurried reader makes.

TWO BOOTSTRAP ARRANGEMENTS, LABELLED, NEVER MIXED
-------------------------------------------------
WITHIN SUBSET, which is a23's and a35's arrangement: an unstratified
participant-clustered resample of that subset's own participants, with every arm
and its constant predictor scored on the same draw, so arm minus constant is
paired.

ACROSS SUBSETS: the 65-and-over and 70-and-over sets are NESTED inside the
60-and-over set, so one resample of the 166 induces a resample of each, and the
change in accuracy is paired inside every draw. That is the only arrangement that
can put an interval on "did raising the floor change anything". Those rows say so
on their face.

MULTIPLICITY
------------
No family in analysis/_registry.py covers an age floor other than 60.
age_cohort_shift is declared over the full cohort against the age-60 cohort and
cannot absorb two further floors without breaking its declared size. So every row
carries the literal string "undeclared", every interval is a nominal 95 percent
one, and nothing here is a pre-registered test. Nothing is written to the
registry: the changelog entry is PRINTED for the first author to paste.
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

from analysis._common import (SuffStats, clustered_bootstrap,                  # noqa: E402
                              paired_difference, save_table)
from analysis._determinism import seed_for                                     # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx                # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, RESTRICTED_EXPERIMENTS,            # noqa: E402
                                    bootstrap_p, changelog_entry,
                                    percentile_interval,
                                    unstratified_participant_bootstrap)
from analysis.a23_protocol_audit import ARM_ORDER, CONSTANT, constant_frame    # noqa: E402
from analysis.a25_increment_over_metadata import (accuracy_percent, cm_draws,  # noqa: E402
                                                  macro_f1, multiplicity)

SOURCE = "analysis/a37_age_threshold.py"
EXPERIMENT_ID = "a37"
TABLE = "t37_age_threshold"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

EVALUATION_ONLY = ("EVALUATION SIDE ONLY. The eight arms are the age-60-trained "
                   "restricted-cohort models, rescored on nested subsets. Nothing was "
                   "retrained, no fold was redrawn.")

FLOORS = [60, 65, 70]
METRICS = ["accuracy", "macro_f1"]
HIGHER_BETTER = {"accuracy": True, "macro_f1": True}

WITHIN = "within subset, unstratified participant clustered, a23's arrangement"
ACROSS = ("across subsets, one resample of the age-60 cohort induces a resample of each "
          "nested subset")

# A subset this small stops being a cohort and starts being a case series. The
# script refuses to report below it rather than printing an interval nobody should
# read. 10 CDR >= 1 participants is already thin; below that the confusion matrix
# has empty rows and macro-F1 becomes a function of which class vanished.
MIN_PER_CLASS = 5


def label(floor: int) -> str:
    return f"age {floor} and over"


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Restricted-cohort arms rescored at age floors of 65 and 70.")
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

    print(f"a37  age floor sensitivity, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     {EVALUATION_ONLY}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    y = coh.labels
    cls = np.bincount(y, minlength=N_CLASSES)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images, CDR 0 / 0.5 / >=1 = {cls[0]} / {cls[1]} / {cls[2]}")

    meta = pd.read_excel(metadata_xlsx())
    meta["pid"] = meta["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid").reindex(coh.participants)
    age = pd.to_numeric(meta["Age"], errors="coerce").to_numpy()
    if np.isnan(age).any():
        raise SystemExit(f"a37 STOPPED. Age is missing for {int(np.isnan(age).sum())} of the "
                         f"{coh.n_participants} restricted-cohort participants, so the age "
                         f"floors cannot be applied without dropping people silently.")
    if age.min() < 60:
        raise SystemExit(f"a37 STOPPED. The restricted cohort should be aged 60 and over and "
                         f"its youngest participant is {age.min():.0f}. The subsets below "
                         f"would not be the ones the paper names.")
    print(f"     age range {age.min():.0f} to {age.max():.0f}, median {np.median(age):.0f}")

    masks: Dict[int, np.ndarray] = {f: age >= f for f in FLOORS}
    usable: List[int] = []
    for f in FLOORS:
        m = masks[f]
        c = np.bincount(y[m], minlength=N_CLASSES)
        thin = [i for i in range(N_CLASSES) if c[i] < MIN_PER_CLASS]
        print(f"     {label(f):<20} n={int(m.sum()):>4}  CDR 0 {c[0]:>3}  CDR 0.5 {c[1]:>3}  "
              f"CDR >= 1 {c[2]:>3}   constant "
              f"{100.0 * c.max() / max(int(m.sum()), 1):>6.2f}%"
              + (f"   !! class {thin} below {MIN_PER_CLASS}" if thin else ""))
        if thin:
            print(f"  !! {label(f)} is NOT REPORTED. A class with fewer than "
                  f"{MIN_PER_CLASS} participants makes macro-F1 a function of which class "
                  f"vanished rather than of the model.")
        else:
            usable.append(f)
    if 60 not in usable:
        raise SystemExit("a37 STOPPED. The age-60 reference subset itself is not usable.")

    arms = [a for a in ARM_ORDER if a in coh.arms] + \
           [a for a in sorted(coh.arms) if a not in ARM_ORDER]
    rows: List[Dict] = []

    # -- within subset -------------------------------------------------------------
    for f in usable:
        m = masks[f]
        keep = set(coh.participants[m])
        frames = {a: coh.part[a][coh.part[a]["participant_id"].isin(keep)]
                  .reset_index(drop=True) for a in arms}
        stats: Dict[str, SuffStats] = {a: SuffStats(fr) for a, fr in frames.items()}
        const, maj, prior = constant_frame(frames[arms[0]])
        stats[CONSTANT] = SuffStats(const)
        c = np.bincount(y[m], minlength=N_CLASSES)
        print(f"\n     {label(f)}: within-subset bootstrap over {int(m.sum())} "
              f"participants, {len(stats)} entries ...")
        boots = {metric: clustered_bootstrap(
            stats, metric, n_boot=n_boot,
            seed=seed_for(BASE_SEED, "a37:within", floor=f, metric=metric))
            for metric in METRICS}
        for metric in METRICS:
            b = boots[metric]
            for a in arms + [CONSTANT]:
                rows.append({
                    "kind": "within_subset", "subset": label(f), "age_floor": f,
                    "unit": "participant", "arm": a, "metric": metric,
                    "bootstrap": WITHIN, "value": b[a]["point"],
                    "ci_low": b[a]["ci_low"], "ci_high": b[a]["ci_high"], "ci_level": 0.95,
                    "n_participants": int(m.sum()), "n_cdr0": int(c[0]),
                    "n_cdr05": int(c[1]), "n_cdr_ge1": int(c[2]),
                    "majority_class": maj,
                    "constant_accuracy_percent": 100.0 * prior[maj],
                    "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": coh.fingerprint.get(a, ""),
                    "retrained": False, "note": EVALUATION_ONLY})
            for a in arms:
                d = paired_difference(b, a, CONSTANT)
                rows.append({
                    "kind": "vs_constant", "subset": label(f), "age_floor": f,
                    "unit": "participant", "arm": a, "metric": metric,
                    "bootstrap": WITHIN, "reference": CONSTANT,
                    "value": b[a]["point"], "reference_value": b[CONSTANT]["point"],
                    "difference": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                    "ci_level": d["ci_level"], "p_bootstrap": d["p_bootstrap"],
                    "p_at_resolution_floor": d["p_at_resolution_floor"],
                    "excludes_zero": d["excludes_zero"],
                    "beats_constant": bool(d["excludes_zero"]
                                           and d["ci_low"] > 0),
                    "n_participants": int(m.sum()),
                    "constant_accuracy_percent": 100.0 * prior[maj],
                    "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": coh.fingerprint.get(a, ""),
                    "retrained": False,
                    "note": "the constant predictor is built from THIS subset's own labels, "
                            "so the comparison is not contaminated by the floor moving"})

    # -- across subsets, nested ----------------------------------------------------
    idx = unstratified_participant_bootstrap(coh.n_participants, n_boot, "a37:across",
                                             BASE_SEED)
    W = multiplicity(idx, coh.n_participants)
    print(f"\n     across subsets: one resample of the {coh.n_participants} age-60 "
          f"participants, restricted to each nested subset ...")
    draws: Dict[Tuple[str, int, str], np.ndarray] = {}
    pts: Dict[Tuple[str, int, str], float] = {}
    for a in arms:
        frame = coh.part[a]
        pred = frame["pred"].to_numpy().astype(int)
        lab = frame["label"].to_numpy().astype(int)
        for f in usable:
            m = masks[f]
            cms = cm_draws(pred[m], lab[m], W[:, m])
            one = cm_draws(pred[m], lab[m], np.ones((1, int(m.sum()))))
            draws[(a, f, "accuracy")] = accuracy_percent(cms)
            draws[(a, f, "macro_f1")] = macro_f1(cms)
            pts[(a, f, "accuracy")] = float(accuracy_percent(one)[0])
            pts[(a, f, "macro_f1")] = float(macro_f1(one)[0])

    for a in arms:
        for f in [x for x in usable if x != 60]:
            for metric in METRICS:
                d = draws[(a, f, metric)] - draws[(a, 60, metric)]
                lo, hi = percentile_interval(d)
                st = bootstrap_p(d)
                rows.append({
                    "kind": "across_subsets", "subset": f"{label(f)} minus {label(60)}",
                    "age_floor": f, "unit": "participant", "arm": a, "metric": metric,
                    "bootstrap": ACROSS,
                    "value": pts[(a, f, metric)], "reference_value": pts[(a, 60, metric)],
                    "difference": pts[(a, f, metric)] - pts[(a, 60, metric)],
                    "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                    "p_bootstrap": st["p_bootstrap"],
                    "p_at_resolution_floor": st["p_at_resolution_floor"],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_participants": int(masks[f].sum()),
                    "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": coh.fingerprint.get(a, ""), "retrained": False,
                    "note": "nested subsets, so one draw of the 166 pairs the two sides. "
                            "This is not the within-subset arrangement and must not be "
                            "read as one"})

    table = pd.DataFrame(rows)
    lead = ["kind", "subset", "age_floor", "arm", "metric", "unit", "bootstrap", "value",
            "reference_value", "difference", "ci_low", "ci_high", "ci_level",
            "p_bootstrap", "excludes_zero", "family"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    w = table[table["kind"] == "within_subset"]
    vc = table[table["kind"] == "vs_constant"]
    ac = table[table["kind"] == "across_subsets"]

    print("\n" + "=" * 100)
    print("a37  IS THE AGE 60 CUTOFF LOAD-BEARING?")
    print("=" * 100)
    print(f"  {EVALUATION_ONLY}")
    print(f"\n  {'subset':<20}{'n':>5}{'CDR 0':>7}{'0.5':>6}{'>=1':>6}"
          f"{'constant acc':>15}")
    for f in usable:
        r = w[(w["age_floor"] == f) & (w["arm"] == CONSTANT) &
              (w["metric"] == "accuracy")].iloc[0]
        print(f"  {label(f):<20}{int(r['n_participants']):>5}{int(r['n_cdr0']):>7}"
              f"{int(r['n_cdr05']):>6}{int(r['n_cdr_ge1']):>6}{r['value']:>14.2f}%")

    for metric in METRICS:
        print(f"\n  -- {metric}, within-subset value and 95 percent interval")
        print(f"     {'arm':<16}" + "".join(f"{label(f):>26}" for f in usable))
        for a in arms + [CONSTANT]:
            cells = []
            for f in usable:
                r = w[(w["age_floor"] == f) & (w["arm"] == a) &
                      (w["metric"] == metric)].iloc[0]
                cells.append(f"{r['value']:>8.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
            print(f"     {a:<16}" + "".join(f"{c:>26}" for c in cells))
        line = []
        for f in usable:
            g = vc[(vc["age_floor"] == f) & (vc["metric"] == metric)]
            line.append(f"{int(g['beats_constant'].sum())} of {len(g)}")
        print(f"     {'beats constant':<16}" + "".join(f"{x:>26}" for x in line))

    print(f"\n  CHANGE FROM THE AGE 60 COHORT, paired across the nested subsets:")
    for metric in METRICS:
        print(f"    -- {metric}")
        for f in [x for x in usable if x != 60]:
            g = ac[(ac["age_floor"] == f) & (ac["metric"] == metric)]
            n_sig = int(g["excludes_zero"].sum())
            print(f"       {label(f):<20} range {g['difference'].min():+.3f} to "
                  f"{g['difference'].max():+.3f}, interval excludes zero for "
                  f"{n_sig} of {len(g)} arms")

    acc60 = vc[(vc["age_floor"] == 60) & (vc["metric"] == "accuracy")]
    verdict_lines = []
    for f in [x for x in usable if x != 60]:
        g = vc[(vc["age_floor"] == f) & (vc["metric"] == "accuracy")]
        verdict_lines.append(f"{label(f)}: {int(g['beats_constant'].sum())} of {len(g)} arms "
                             f"beat their own constant predictor on accuracy")
    print(f"\n  WHAT THIS SAYS ABOUT THE CUTOFF")
    print(f"     age 60: {int(acc60['beats_constant'].sum())} of {len(acc60)} arms beat "
          f"their own constant predictor on accuracy")
    for line in verdict_lines:
        print(f"     {line}")
    print(f"     If those counts match, the age 60 floor is not carrying the conclusion. "
          f"If they diverge,")
    print(f"     the paper has to report the divergence rather than the floor it prefers.")

    print(f"\n  EVERY comparison here is UNDECLARED. No registry family covers an age floor "
          f"other than 60,")
    print(f"  no p-value here is corrected, and every interval is a nominal 95 percent one.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "the eight age-60-trained restricted-cohort arms rescored, without retraining, on "
        "the nested subsets aged 65 and over and 70 and over, each against that subset's "
        "own constant majority-class predictor",
        "accuracy and macro-F1",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
