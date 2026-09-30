#!/usr/bin/env python3
"""a35: how much of the accuracy drop is the missing labels rather than the age?

    python analysis/a35_assessed_any_age.py
    python analysis/a35_assessed_any_age.py --quick      # 200 resamples
    python analysis/a35_assessed_any_age.py --boot 10000

WHAT THE RESTRICTION ACTUALLY REMOVED
-------------------------------------
Going from the full cohort to the restricted one takes away two different things
at the same time, and the paper currently reports their combined effect as
though it were one.

    147 controls who were NEVER CDR-ASSESSED. Their Non-Demented label is an
        absence of an assessment, not a negative assessment. Every one of them is
        under 60.
    34 further controls who WERE assessed and are under 60. Removing them is the
        age restriction proper.

Scoring the full-cohort networks on the 200 participants who have a CDR of any
age, 119 CDR 0 plus 58 CDR 0.5 plus 23 CDR >= 1, sits exactly between the two
cohorts and isolates the first effect from the second. Whatever accuracy is lost
going from 347 to 200 is the label-validity effect alone. Whatever is lost going
from 200 to 166 is the age restriction alone.

That is one row in the composition table and one sentence, and neither can be
written without this number.

THE CONSTANT PREDICTOR MOVES, WHICH IS HALF THE POINT
------------------------------------------------------
The majority share is 266/347 = 76.7 percent on the full cohort, 119/200 = 59.5
percent here, and 85/166 = 51.2 percent on the restricted cohort. An arm can lose
seventeen points of accuracy between the first two and have learned nothing about
dementia, because the floor moved under it by the same amount. So every subset
here carries its OWN constant predictor, built from the very labels it is scored
against, and every arm is differenced against that one rather than against a
number borrowed from another cohort.

THE BOOTSTRAP, IN TWO PARTS
---------------------------
WITHIN SUBSET, which is a23's arrangement and the one the brief asks for: an
unstratified participant-clustered resample of that subset's own participants,
with every arm and its constant predictor scored on the same draw, so arm minus
constant is paired. a23 refuses to difference two subsets inside one of these,
because the two sides do not share a participant set.

ACROSS SUBSETS, reported separately and labelled: the 200 and the 166 are nested
inside the 347, so one resample of the 347 induces a resample of each. Taking the
drop inside each draw does pair them, and that is the only way to put an interval
on "how much did restricting cost". Those rows say so on their face and are never
mixed with the within-subset ones.

MULTIPLICITY
------------
No family in analysis/_registry.py covers the CDR-assessed any-age subset.
age_cohort_shift is declared over the full cohort against the age-60 cohort, and
arm_vs_constant_accuracy was realised by a20 at the image level on a different
cohort. Absorbing these comparisons into either would break a declared size. So
every row carries the literal string "undeclared", every interval is a nominal 95
percent one, and nothing here is a pre-registered test. Nothing is written to the
registry: the changelog entry is PRINTED for the first author to paste.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import (SuffStats, clustered_bootstrap,                  # noqa: E402
                              paired_difference, save_table)
from analysis._determinism import seed_for                                     # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx                # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, FULL_EXPERIMENTS,                  # noqa: E402
                                    RESTRICTED_EXPERIMENTS, bootstrap_p,
                                    changelog_entry, find_full_tree,
                                    percentile_interval,
                                    unstratified_participant_bootstrap)
from analysis.a23_protocol_audit import (ARM_ORDER, CONSTANT,                  # noqa: E402
                                         constant_frame, warn)
from analysis.a25_increment_over_metadata import (accuracy_percent, cm_draws,  # noqa: E402
                                                  macro_f1, multiplicity)

SOURCE = "analysis/a35_assessed_any_age.py"
EXPERIMENT_ID = "a35"
TABLE = "t35_assessed_any_age"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

METRICS = ["accuracy", "macro_f1", "ece", "brier"]
HIGHER_BETTER = {"accuracy": True, "macro_f1": True, "ece": False, "brier": False}

FULL = "full cohort, every participant"
ASSESSED = "CDR assessed, any age"
RESTRICTED = "CDR assessed, age 60 and over"
SUBSET_ORDER = [FULL, ASSESSED, RESTRICTED]

# The counts the brief states. They are VERIFIED against the metadata before
# anything is computed, and a mismatch stops the script, because a subset that is
# not the one the paper describes would produce a number nobody could use.
EXPECTED = {ASSESSED: (200, 119, 58, 23), RESTRICTED: (166, 85, 58, 23),
            FULL: (347, 266, 58, 23)}

WITHIN = "within subset, unstratified participant clustered, a23's arrangement"
ACROSS = "across subsets, one resample of the 347 induces a resample of each nested subset"


def subset_counts(labels: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
    c = np.bincount(labels[mask], minlength=N_CLASSES)
    return int(mask.sum()), int(c[0]), int(c[1]), int(c[2])


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Full-cohort models on the CDR-assessed participants of any age.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"participant-clustered resamples (default {N_BOOT})")
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

    print(f"a35  CDR-assessed participants of any age, {n_boot:,} resamples, "
          f"base seed {BASE_SEED}")

    full_dir = find_full_tree(args.full_from)
    if full_dir is None:
        raise SystemExit(
            "a35 needs the full-cohort runs and no experiments folder holding them was "
            "found. Pass --full-from pointing at the tree that holds exp01, exp02, exp07 "
            "and exp08.")
    full = Cohort("full", FULL_EXPERIMENTS, full_dir)
    restricted = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     full:       {len(full.arms)} arms, {full.n_participants} participants, "
          f"{full.n_images:,} images, from {full_dir}")
    print(f"     restricted cohort defined by the exp10 to exp13 runs: "
          f"{restricted.n_participants} participants")

    # -- the counts, verified before anything is computed ------------------------
    meta = pd.read_excel(metadata_xlsx())
    meta["pid"] = meta["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid").reindex(full.participants)
    cdr = pd.to_numeric(meta["CDR"], errors="coerce").to_numpy()
    age = pd.to_numeric(meta["Age"], errors="coerce").to_numpy()
    if np.isnan(age).any():
        raise SystemExit(f"a35 STOPPED. Age is missing for {int(np.isnan(age).sum())} of "
                         f"the {full.n_participants} full-cohort participants.")

    masks = {FULL: np.ones(full.n_participants, dtype=bool),
             ASSESSED: np.isfinite(cdr),
             RESTRICTED: np.isin(full.participants, restricted.participants)}

    problems = []
    for name in SUBSET_ORDER:
        got = subset_counts(full.labels, masks[name])
        want = EXPECTED[name]
        print(f"     {name:<32} n={got[0]:>3}  CDR 0 {got[1]:>3}  CDR 0.5 {got[2]:>3}  "
              f"CDR >= 1 {got[3]:>3}   expected {want}")
        if got != want:
            problems.append(f"{name}: got {got}, expected {want}")
    derived_restricted = np.isfinite(cdr) & (age >= 60)
    if not np.array_equal(derived_restricted, masks[RESTRICTED]):
        problems.append(
            f"the restricted cohort from the exp10 to exp13 runs differs from 'CDR assessed "
            f"and age at least 60' in the metadata for "
            f"{int((derived_restricted != masks[RESTRICTED]).sum())} participant(s)")
    unassessed = ~masks[ASSESSED]
    if unassessed.any():
        print(f"     the {int(unassessed.sum())} never-assessed participants are all CDR 0 "
              f"by label ({int((full.labels[unassessed] != 0).sum())} exceptions) and range "
              f"in age from {age[unassessed].min():.0f} to {age[unassessed].max():.0f}")
        if age[unassessed].max() >= 60:
            problems.append(f"{int((age[unassessed] >= 60).sum())} never-assessed "
                            f"participant(s) are 60 or over, so the two effects this script "
                            f"separates are not disjoint")
    if problems:
        raise SystemExit("a35 STOPPED. The cohort does not match the description this "
                         "experiment is built on:\n  " + "\n  ".join(problems) +
                         "\nEvery number below would describe a different subset from the "
                         "one the paper names.")
    print(f"     constant predictor accuracy within {ASSESSED}: "
          f"{100.0 * EXPECTED[ASSESSED][1] / EXPECTED[ASSESSED][0]:.2f} percent "
          f"({EXPECTED[ASSESSED][1]} of {EXPECTED[ASSESSED][0]})")

    arms = [a for a in ARM_ORDER if a in full.arms]
    rows: List[Dict] = []

    # -- within subset, a23's arrangement -----------------------------------------
    for name in SUBSET_ORDER:
        m = masks[name]
        keep = set(full.participants[m])
        frames = {a: full.part[a][full.part[a]["participant_id"].isin(keep)]
                  .reset_index(drop=True) for a in arms}
        stats: Dict[str, SuffStats] = {a: SuffStats(f) for a, f in frames.items()}
        const, maj, prior = constant_frame(frames[arms[0]])
        stats[CONSTANT] = SuffStats(const)
        print(f"\n     {name}: participant-clustered bootstrap over "
              f"{int(m.sum())} participants, {len(stats)} entries ...")
        boots = {metric: clustered_bootstrap(
            stats, metric, n_boot=n_boot,
            seed=seed_for(BASE_SEED, "a35:within", subset=name, metric=metric))
            for metric in METRICS}
        for metric in METRICS:
            b = boots[metric]
            for a in arms + [CONSTANT]:
                rows.append({
                    "kind": "within_subset", "subset": name, "unit": "participant",
                    "arm": a, "metric": metric, "bootstrap": WITHIN,
                    "value": b[a]["point"], "ci_low": b[a]["ci_low"],
                    "ci_high": b[a]["ci_high"], "ci_level": 0.95,
                    "n_participants": int(m.sum()),
                    "n_cdr0": EXPECTED[name][1], "n_cdr05": EXPECTED[name][2],
                    "n_cdr_ge1": EXPECTED[name][3],
                    "majority_class": maj,
                    "constant_accuracy_percent": 100.0 * prior[maj],
                    "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": full.fingerprint.get(a, ""),
                    "note": "full-cohort-trained model scored on this subset; the constant "
                            "predictor is built from this subset's own labels"})
            for a in arms:
                d = paired_difference(b, a, CONSTANT)
                rows.append({
                    "kind": "vs_constant", "subset": name, "unit": "participant",
                    "arm": a, "metric": metric, "bootstrap": WITHIN,
                    "reference": CONSTANT,
                    "value": b[a]["point"], "reference_value": b[CONSTANT]["point"],
                    "difference": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                    "ci_level": d["ci_level"], "p_bootstrap": d["p_bootstrap"],
                    "p_at_resolution_floor": d["p_at_resolution_floor"],
                    "excludes_zero": d["excludes_zero"],
                    "beats_constant": bool(
                        d["excludes_zero"] and
                        (d["ci_low"] > 0 if HIGHER_BETTER[metric] else d["ci_high"] < 0)),
                    "n_participants": int(m.sum()),
                    "constant_accuracy_percent": 100.0 * prior[maj],
                    "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": full.fingerprint.get(a, ""),
                    "note": "no declared family covers this subset, so the interval is a "
                            "nominal 95 percent one and this is not a pre-registered test"})

    # -- across subsets, nested, one shared draw ----------------------------------
    idx = unstratified_participant_bootstrap(full.n_participants, n_boot, "a35:across",
                                             BASE_SEED)
    W = multiplicity(idx, full.n_participants)
    print(f"\n     across subsets: one resample of the {full.n_participants} full-cohort "
          f"participants, restricted to each nested subset ...")
    across: Dict[Tuple[str, str, str], np.ndarray] = {}
    across_point: Dict[Tuple[str, str, str], float] = {}
    for a in arms:
        frame = full.part[a]
        pred = frame["pred"].to_numpy().astype(int)
        lab = frame["label"].to_numpy().astype(int)
        for name in SUBSET_ORDER:
            m = masks[name]
            cms = cm_draws(pred[m], lab[m], W[:, m])
            one = cm_draws(pred[m], lab[m], np.ones((1, int(m.sum()))))
            across[(a, name, "accuracy")] = accuracy_percent(cms)
            across[(a, name, "macro_f1")] = macro_f1(cms)
            across_point[(a, name, "accuracy")] = float(accuracy_percent(one)[0])
            across_point[(a, name, "macro_f1")] = float(macro_f1(one)[0])

    steps = [("label validity, full minus assessed any age", FULL, ASSESSED),
             ("age restriction, assessed any age minus age 60 and over", ASSESSED,
              RESTRICTED),
             ("total, full minus age 60 and over", FULL, RESTRICTED)]
    for a in arms:
        for label, hi, lo in steps:
            for metric in ("accuracy", "macro_f1"):
                d = across[(a, hi, metric)] - across[(a, lo, metric)]
                ci_lo, ci_hi = percentile_interval(d)
                stats_d = bootstrap_p(d)
                rows.append({
                    "kind": "across_subsets", "subset": f"{hi} minus {lo}",
                    "unit": "participant", "arm": a, "metric": metric,
                    "bootstrap": ACROSS, "step": label,
                    "value": across_point[(a, hi, metric)],
                    "reference_value": across_point[(a, lo, metric)],
                    "difference": across_point[(a, hi, metric)]
                    - across_point[(a, lo, metric)],
                    "ci_low": ci_lo, "ci_high": ci_hi, "ci_level": 0.95,
                    "p_bootstrap": stats_d["p_bootstrap"],
                    "p_at_resolution_floor": stats_d["p_at_resolution_floor"],
                    "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
                    "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": full.fingerprint.get(a, ""),
                    "note": "the two subsets are NESTED inside the full cohort, so one "
                            "resample of the 347 induces a resample of each and the drop "
                            "is paired inside every draw. This is not a23's within-subset "
                            "arrangement and is labelled so it cannot be read as one"})

    table = pd.DataFrame(rows)
    lead = ["kind", "subset", "step", "arm", "metric", "unit", "bootstrap", "value",
            "reference_value", "difference", "ci_low", "ci_high", "ci_level",
            "p_bootstrap", "excludes_zero", "family"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    w = table[table["kind"] == "within_subset"]
    print("\n" + "=" * 100)
    print("a35  WHERE DOES THE DROP COME FROM: THE MISSING LABELS, OR THE AGE?")
    print("=" * 100)
    print("  Full-cohort-trained networks, participant level, scored on three nested "
          "participant sets.")
    print("  Each set carries its OWN constant majority-class predictor, built from that "
          "set's labels.")
    print(f"\n  {'subset':<32}{'n':>5}{'CDR 0':>7}{'0.5':>5}{'>=1':>5}"
          f"{'constant acc':>14}")
    for name in SUBSET_ORDER:
        n, c0, c1, c2 = EXPECTED[name]
        const_acc = w[(w["subset"] == name) & (w["arm"] == CONSTANT) &
                      (w["metric"] == "accuracy")]["value"].iloc[0]
        print(f"  {name:<32}{n:>5}{c0:>7}{c1:>5}{c2:>5}{const_acc:>13.2f}%")

    print(f"\n  ACCURACY and MACRO-F1 per arm, with the within-subset 95 percent interval.")
    print(f"  {'arm':<16}" + "".join(f"{s.split(',')[0][:14]:>26}" for s in SUBSET_ORDER))
    for metric in ("accuracy", "macro_f1"):
        print(f"    -- {metric}")
        for a in arms:
            cells = []
            for name in SUBSET_ORDER:
                r = w[(w["subset"] == name) & (w["arm"] == a) &
                      (w["metric"] == metric)].iloc[0]
                cells.append(f"{r['value']:>8.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
            print(f"  {a:<16}" + "".join(f"{c:>26}" for c in cells))
        r = [w[(w["subset"] == name) & (w["arm"] == CONSTANT) &
               (w["metric"] == metric)].iloc[0] for name in SUBSET_ORDER]
        print(f"  {CONSTANT:<16}" + "".join(
            f"{x['value']:>8.3f} [{x['ci_low']:.2f}, {x['ci_high']:.2f}]".rjust(26)
            for x in r))

    print(f"\n  BEATS ITS OWN CONSTANT PREDICTOR (interval excluding zero in the arm's "
          f"favour):")
    vc = table[table["kind"] == "vs_constant"]
    for metric in ("accuracy", "macro_f1"):
        line = []
        for name in SUBSET_ORDER:
            g = vc[(vc["subset"] == name) & (vc["metric"] == metric)]
            line.append(f"{int(g['beats_constant'].sum())} of {len(g)}")
        print(f"    {metric:<12}" + "".join(f"{x:>26}" for x in line))

    print(f"\n  WHERE THE DROP COMES FROM, accuracy points, paired across the nested "
          f"subsets:")
    ac = table[(table["kind"] == "across_subsets") & (table["metric"] == "accuracy")]
    print(f"  {'arm':<16}{'labels (347->200)':>28}{'age (200->166)':>28}"
          f"{'total':>10}")
    for a in arms:
        g = ac[ac["arm"] == a]
        lab = g[g["step"].str.startswith("label")].iloc[0]
        agr = g[g["step"].str.startswith("age")].iloc[0]
        tot = g[g["step"].str.startswith("total")].iloc[0]
        print(f"  {a:<16}"
              f"{lab['difference']:>10.2f} [{lab['ci_low']:+.2f}, {lab['ci_high']:+.2f}]"
              f"{agr['difference']:>12.2f} [{agr['ci_low']:+.2f}, {agr['ci_high']:+.2f}]"
              f"{tot['difference']:>10.2f}")
    lab_all = ac[ac["step"].str.startswith("label")]
    age_all = ac[ac["step"].str.startswith("age")]
    tot_all = ac[ac["step"].str.startswith("total")]
    share = (lab_all["difference"].to_numpy() / tot_all["difference"].to_numpy())
    print(f"\n  The unassessed labels alone account for {lab_all['difference'].min():.2f} "
          f"to {lab_all['difference'].max():.2f} accuracy points,")
    print(f"  which is {100 * share.min():.0f} to {100 * share.max():.0f} percent of the "
          f"total drop. The age restriction proper accounts for")
    print(f"  {age_all['difference'].min():.2f} to {age_all['difference'].max():.2f} "
          f"points. Intervals exclude zero for "
          f"{int(lab_all['excludes_zero'].sum())} of {len(lab_all)} arms on the label step")
    print(f"  and {int(age_all['excludes_zero'].sum())} of {len(age_all)} on the age step.")
    print(f"\n  The constant predictor's own accuracy falls from "
          f"{100.0 * EXPECTED[FULL][1] / EXPECTED[FULL][0]:.2f} to "
          f"{100.0 * EXPECTED[ASSESSED][1] / EXPECTED[ASSESSED][0]:.2f} to "
          f"{100.0 * EXPECTED[RESTRICTED][1] / EXPECTED[RESTRICTED][0]:.2f} percent across "
          f"the same three sets,")
    print(f"  so a drop of that size is what a model that learned nothing would also show.")

    print("\n  EVERY comparison here is UNDECLARED. No registry family covers the "
          "CDR-assessed any-age")
    print("  subset, no p-value here is corrected, and the intervals are nominal 95 "
          "percent.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "full-cohort-trained arms scored on the 200 CDR-assessed participants of any age, "
        "against that subset's own constant majority-class predictor, and the accuracy "
        "drop split into a label-validity step and an age step across the three nested "
        "participant sets",
        "accuracy, macro-F1, ECE and Brier",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
