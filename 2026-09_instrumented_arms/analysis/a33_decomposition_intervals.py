#!/usr/bin/env python3
"""a33: intervals on the two halves of the cohort-restriction accuracy drop.

    python analysis/a33_decomposition_intervals.py
    python analysis/a33_decomposition_intervals.py --quick      # 200 resamples
    python analysis/a33_decomposition_intervals.py --boot 10000

THE CLAIM THIS PUTS AN INTERVAL ON
----------------------------------
Scoring the full-cohort networks on the restricted cohort costs 22 to 27 points
of accuracy. The manuscript splits that drop in two:

    CLASS REBALANCING     the restricted cohort has 85 controls against 266, so
                          the majority class falls from 77 percent to 51 percent
                          and any predictor that leans on the majority loses
                          accuracy for purely arithmetic reasons.
    AGE-SPECIFIC SELECTION what is left after the arithmetic: the 85 controls who
                          remain are older and were actually CDR-assessed, and
                          they are harder.

The split currently rests on point estimates, for four of seven arms. A split
with no interval cannot be argued with, and an age-specific component whose
interval includes zero would mean the whole drop is arithmetic. So the split is
recomputed here for ALL SEVEN full-cohort arms, with an interval on each half.

THE COUNTERFACTUAL
------------------
Build a cohort with the restricted cohort's class counts, 85 / 58 / 23, but draw
the 85 controls AT RANDOM from all 266 full-cohort controls rather than taking
the 85 older assessed ones. Call its accuracy (d). Then

    total drop        = (a) full cohort        minus (b) restricted cohort
    rebalancing       = (a) full cohort        minus (d) counterfactual
    age-specific      = (d) counterfactual     minus (b) restricted cohort

and the two components sum to the total drop by construction, for every draw and
every resample, which is checked in code rather than assumed.

THE NESTING, WHICH A REVIEWER WILL ASK ABOUT
--------------------------------------------
There are two sources of variation and they are not the same thing.

    DRAW variation      which 85 of the 266 controls the counterfactual happened
                        to pick. It is a property of the procedure.
    SAMPLING variation  which 347 people OASIS-1 happened to contain. It is the
                        usual thing a bootstrap covers.

The nesting is: OUTER, a participant-clustered bootstrap of the 347 full-cohort
participants; INNER, one independent counterfactual draw inside each outer
resample. B outer resamples therefore carry B counterfactual draws, and the
percentile interval over the B outer values covers both sources at once. That is
the PRIMARY interval, and it is the widest of the three reported, which is the
direction an interval should err in.

Two narrower intervals are reported beside it, labelled, because a reader who is
told only the widest number cannot see where the width came from:

    sampling only   the outer bootstrap with the counterfactual replaced by its
                    exact expectation over draws inside each resample. Because
                    accuracy is linear in per-participant correctness, that
                    expectation is closed form: 85 times the mean correctness of
                    the resample's 266 controls, plus the 58 and the 23.
    draw only       the 10,000 draws on the observed cohort, with no resampling.

THE OUTER BOOTSTRAP IS STRATIFIED, ON PURPOSE
---------------------------------------------
Participants are resampled within four cells: the 85 restricted controls, the
181 other controls, the 58 CDR 0.5 and the 23 CDR >= 1. Every cell keeps its
observed size. This differs from a21 and a23, which use an unstratified draw, and
the reason is that the counterfactual is DEFINED by fixed counts. An unstratified
draw that returns 249 controls instead of 266 has not estimated the uncertainty
of this quantity, it has computed a different quantity. Fixing the four cells
also keeps (a), (b) and (d) on one underlying draw, which is what makes their
differences paired.

WHAT IS REUSED FROM a21, AND WHAT IS NOT
-----------------------------------------
a21_age_confound.py computes a DIFFERENT decomposition: test composition against
training effect, at the IMAGE level, for five of the seven arms. It does not
compute the rebalancing versus age-specific split at all, so there are no point
estimates there to reuse. What a21 does provide is (a) and (b), the two ends of
the total drop, and those are reproduced here EXACTLY at the image level as a
check on the loading path. If the image-level reproduction disagrees with
t21_age_confound.csv the script STOPS, because two files printing different
values for one quantity is the failure this project has already paid for once.

The reported decomposition is at the PARTICIPANT level, which is the unit the
manuscript declares as primary. Participant-level (a) and (b) are NOT expected to
equal t21's image-level values, and the difference between the two units is
reported rather than reconciled away.

MULTIPLICITY
------------
The total drop per arm on accuracy is what the declared family age_cohort_shift
describes, word for word, at its declared size of seven. The registry's audit
currently lists that family as NOT RUN. This script computes it and reports it
against that family's corrected threshold. The declaration does not name a unit,
so the unit is printed on every row, and the image-level version of the same
comparison is reported as undeclared beside it.

No family covers a counterfactual decomposition, so the rebalancing and
age-specific components carry the literal string "undeclared" and nominal 95
percent intervals. Nothing is written to the registry: the changelog entry is
PRINTED for the first author to paste.
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

from analysis._common import paired_difference, save_table                     # noqa: E402
from analysis._determinism import seeded_rng                                   # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx                # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, FULL_EXPERIMENTS,                  # noqa: E402
                                    RESTRICTED_EXPERIMENTS, bootstrap_p,
                                    changelog_entry, find_full_tree,
                                    percentile_interval,
                                    stratified_participant_bootstrap)
from analysis._registry import REGISTRY                                        # noqa: E402
from analysis.a23_protocol_audit import ARM_ORDER, warn                        # noqa: E402

SOURCE = "analysis/a33_decomposition_intervals.py"
EXPERIMENT_ID = "a33"
TABLE = "t33_decomposition_intervals"
N_BOOT = 10000
QUICK_BOOT = 200
N_DRAWS = 10000
BASE_SEED = 12345
UNDECLARED = "undeclared"
TOTAL_DROP_FAMILY = "age_cohort_shift"

# a21 covers these five arms only. The other two full-cohort arms, the focal
# gamma controls, have no t21 row and are reported without a reproduction check.
A21_ARMS = ["Weighted CE", "LDAM", "Class-Balanced", "Focal g=3.00", "FA-FL"]
T21 = "t21_age_confound.csv"
T21_TOLERANCE = 1e-3          # t21 is stored at four decimals by save_table

NESTED = "nested: outer participant bootstrap, one counterfactual draw inside each"
SAMPLING_ONLY = "sampling only: outer bootstrap, counterfactual at its exact expectation"
DRAW_ONLY = "draw only: counterfactual draws on the observed cohort, no resampling"

COMPONENTS = ["total drop", "class rebalancing", "age-specific selection"]


# ---------------------------------------------------------------------------
# Cohort cells
# ---------------------------------------------------------------------------

def cell_vector(labels: np.ndarray, in_restricted: np.ndarray) -> np.ndarray:
    """Four strata: restricted controls, other controls, CDR 0.5, CDR >= 1.

    Named rather than inlined because the column layout of the bootstrap index
    matrix depends on it, and a silent change of cell order would move every
    interval without moving any code that looks wrong.
    """
    cells = np.where(labels == 0, np.where(in_restricted, 0, 1), labels + 1)
    return cells.astype(int)


def stratum_slices(cells: np.ndarray) -> Dict[int, slice]:
    """Where each stratum's columns sit in stratified_participant_bootstrap output.

    That helper lays the strata out contiguously in the order np.unique returns,
    so the slice boundaries are the cumulative cell sizes. This is read off the
    helper's contract rather than assumed, and asserted against it in main().
    """
    out, pos = {}, 0
    for c in np.unique(cells):
        m = int((cells == c).sum())
        out[int(c)] = slice(pos, pos + m)
        pos += m
    return out


# ---------------------------------------------------------------------------
# Accuracy pieces
# ---------------------------------------------------------------------------

def correctness(frame: pd.DataFrame) -> np.ndarray:
    return (frame["pred"].to_numpy().astype(int)
            == frame["label"].to_numpy().astype(int)).astype(np.float64)


def decompose(C: np.ndarray, sl: Dict[int, slice], n_controls_cf: int,
              sel: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """(a), (b), (d) and the two components, for one or many resamples.

    C is (n_draws, 347) correctness in bootstrap column order. `sel` picks the
    counterfactual's controls out of the 266 control columns; when it is None the
    counterfactual is replaced by its exact expectation over draws, which is what
    the sampling-only interval uses.
    """
    ctl = np.s_[sl[0].start:sl[1].stop]              # all 266 controls, cells 0 and 1
    imp = np.s_[sl[2].start:sl[3].stop]              # the 58 + 23 impaired
    n_all = C.shape[1]
    n_imp = sl[3].stop - sl[2].start
    n_restricted = (sl[0].stop - sl[0].start) + n_imp

    a = C.mean(axis=1) * 100.0
    b = (C[:, sl[0]].sum(axis=1) + C[:, imp].sum(axis=1)) / n_restricted * 100.0
    imp_sum = C[:, imp].sum(axis=1)
    n_cf = n_controls_cf + n_imp
    if sel is None:
        cf_controls = n_controls_cf * C[:, ctl].mean(axis=1)
    else:
        cf_controls = np.take_along_axis(C[:, ctl], sel, axis=1).sum(axis=1)
    d = (cf_controls + imp_sum) / n_cf * 100.0
    return {"a_full": a, "b_restricted": b, "d_counterfactual": d,
            "total drop": a - b, "class rebalancing": a - d,
            "age-specific selection": d - b,
            "n_full": np.full(len(a), n_all), "n_restricted": np.full(len(a), n_restricted),
            "n_counterfactual": np.full(len(a), n_cf)}


def draw_selection(n_controls: int, n_take: int, n_draws: int, tag: str,
                   **parts) -> np.ndarray:
    """(n_draws, n_take) column positions, sampled WITHOUT replacement each row.

    argpartition on a uniform matrix is used instead of a Python loop over
    rng.choice, which costs about a minute at 10,000 draws and buys nothing: a
    random partial order of independent uniforms is a uniform random subset.
    """
    rng = seeded_rng(tag, BASE_SEED, n_controls=n_controls, n_take=n_take, **parts)
    r = rng.random((n_draws, n_controls))
    return np.argpartition(r, n_take - 1, axis=1)[:, :n_take].astype(np.int64)


# ---------------------------------------------------------------------------
# a21 reproduction check
# ---------------------------------------------------------------------------

def image_level_ab(coh: Cohort, arm: str, keep: set) -> Tuple[float, float]:
    """(a) and (b) at the image level, which is the unit a21 reports."""
    f = coh.arms[arm]
    c = correctness(f)
    a = float(c.mean() * 100.0)
    m = f["participant_id"].isin(keep).to_numpy()
    b = float(c[m].mean() * 100.0)
    return a, b


def check_against_t21(coh: Cohort, keep: set) -> List[Dict]:
    """Reproduce t21's (a) and (b) at the image level, or stop.

    Two files printing two values for one quantity is the failure this project
    already spent a day reconciling. A mismatch here means the loading path
    differs from a21's, and every number below it would be suspect, so the script
    refuses to continue rather than printing a decomposition nobody can trust.
    """
    path = OUT / T21
    rows: List[Dict] = []
    if not path.is_file():
        warn(f"{T21} is not present, so the a21 reproduction check could not run. The "
             f"participant-level decomposition below is unchecked against a21.")
        return rows
    t21 = pd.read_csv(path)
    bad: List[str] = []
    for _, r in t21.iterrows():
        arm = str(r["arm"])
        if arm not in coh.arms:
            bad.append(f"{arm}: present in {T21} but not among the loaded full-cohort arms")
            continue
        a, b = image_level_ab(coh, arm, keep)
        da = a - float(r["a_train_full_test_full"])
        db = b - float(r["b_train_full_test_age60"])
        ok = bool(abs(da) <= T21_TOLERANCE and abs(db) <= T21_TOLERANCE)
        rows.append({"kind": "a21_reproduction", "arm": arm, "unit": "image",
                     "a_full_here": a, "a_full_t21": float(r["a_train_full_test_full"]),
                     "a_difference": da,
                     "b_restricted_here": b,
                     "b_restricted_t21": float(r["b_train_full_test_age60"]),
                     "b_difference": db,
                     "reproduces_t21_exactly": ok,
                     "tolerance": T21_TOLERANCE, "family": UNDECLARED,
                     "note": f"t21 is stored at four decimals, so the tolerance is "
                             f"{T21_TOLERANCE}. a21 reports the IMAGE unit; the "
                             f"decomposition below is reported at the PARTICIPANT unit "
                             f"and is not expected to match these values"})
        if not ok:
            bad.append(f"{arm}: (a) off by {da:+.6f}, (b) off by {db:+.6f}")
    if bad:
        raise SystemExit(
            "a33 STOPPED. The image-level reproduction of t21_age_confound.csv does not "
            "agree:\n  " + "\n  ".join(bad) +
            "\nThe loading path here differs from a21's, so the decomposition would rest "
            "on numbers a21 does not agree with. Fix that before running this script.")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="Intervals on the composition decomposition.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"outer participant-clustered resamples (default {N_BOOT})")
    ap.add_argument("--draws", type=int, default=N_DRAWS,
                    help=f"counterfactual draws on the observed cohort (default {N_DRAWS})")
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
    n_draws = QUICK_BOOT if args.quick else int(args.draws)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples and {n_draws} draws. Intervals are a smoke "
              f"test, not results.")

    print(f"a33  composition decomposition with intervals, {n_boot:,} outer resamples, "
          f"{n_draws:,} draws, base seed {BASE_SEED}")

    full_dir = find_full_tree(args.full_from)
    if full_dir is None:
        raise SystemExit(
            "a33 needs the full-cohort runs and no experiments folder holding them was "
            "found. Pass --full-from pointing at the tree that holds exp01, exp02, exp07 "
            "and exp08.")
    full = Cohort("full", FULL_EXPERIMENTS, full_dir)
    restricted = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     full:       {len(full.arms)} arms, {full.n_participants} participants, "
          f"{full.n_images:,} images, from {full_dir}")
    print(f"     restricted: defined by the exp10 to exp13 runs, "
          f"{restricted.n_participants} participants, from {EXPERIMENTS_DIR}")

    keep = set(restricted.participants)
    in_restricted = np.isin(full.participants, restricted.participants)
    if int(in_restricted.sum()) != restricted.n_participants:
        raise SystemExit(
            f"a33 STOPPED. {int(in_restricted.sum())} of the {restricted.n_participants} "
            f"restricted participants were found inside the full cohort. The restricted "
            f"cohort must be a subset of the full one for this decomposition to mean "
            f"anything.")

    cells = cell_vector(full.labels, in_restricted)
    counts = np.bincount(cells, minlength=4)
    print(f"     cells: {counts[0]} restricted controls, {counts[1]} other controls, "
          f"{counts[2]} CDR 0.5, {counts[3]} CDR >= 1")
    if tuple(counts) != (85, 181, 58, 23):
        warn(f"the four cells are {tuple(counts)}, not the expected (85, 181, 58, 23). "
             f"Every count below is computed from the data, not from the expectation, but "
             f"read the numbers knowing the cohort is not the one this script was written "
             f"against.")
    n_controls = int(counts[0] + counts[1])
    n_take = int(counts[0])

    # -- metadata cross check, so the cells are not just a label count ----------
    meta = pd.read_excel(metadata_xlsx())
    meta["pid"] = meta["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid").reindex(full.participants)
    cdr = pd.to_numeric(meta["CDR"], errors="coerce").to_numpy()
    age = pd.to_numeric(meta["Age"], errors="coerce").to_numpy()
    derived = np.isfinite(cdr) & (age >= 60)
    if not np.array_equal(derived, in_restricted):
        warn(f"the restricted cohort taken from the exp10 to exp13 runs differs from "
             f"'CDR assessed and age at least 60' derived from the metadata, in "
             f"{int((derived != in_restricted).sum())} participant(s). The runs define the "
             f"cohort here; the metadata is only a cross check.")
    else:
        print("     cross check: the exp10 to exp13 cohort is exactly "
              "'CDR assessed and age at least 60' in the metadata")

    # -- a21 reproduction, image level ------------------------------------------
    rows: List[Dict] = check_against_t21(full, keep)
    if rows:
        worst = max(max(abs(r["a_difference"]), abs(r["b_difference"])) for r in rows)
        print(f"     a21 reproduction: {len(rows)} arm(s) checked at the IMAGE level, "
              f"largest absolute difference {worst:.6f} accuracy points, tolerance "
              f"{T21_TOLERANCE}")

    # -- the draws ---------------------------------------------------------------
    idx = stratified_participant_bootstrap(cells, n_boot, "a33:cells", BASE_SEED,
                                           cohort="full")
    sl = stratum_slices(cells)
    # The helper's column layout is a contract this script depends on, so it is
    # checked against the helper rather than trusted.
    for c, s in sl.items():
        got = cells[idx[0, s]]
        if not np.all(got == c):
            raise SystemExit(f"a33 STOPPED. stratified_participant_bootstrap did not lay "
                             f"stratum {c} out at columns {s}; the column layout this "
                             f"script assumes is wrong and every number would be mislabelled.")

    sel_boot = draw_selection(n_controls, n_take, n_boot, "a33:cf_inner", scope="outer")
    sel_obs = draw_selection(n_controls, n_take, n_draws, "a33:cf_observed", scope="observed")
    print(f"\n     {len(full.order())} arms, participant level. Nesting: "
          f"{n_boot:,} outer resamples with one counterfactual draw inside each, "
          f"plus {n_draws:,} draws on the observed cohort.")

    arms = [a for a in ARM_ORDER if a in full.arms]
    order = np.concatenate([np.flatnonzero(cells == c) for c in np.unique(cells)])
    blocks: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    points: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        c_part = correctness(full.part[arm])
        C_boot = c_part[idx]
        # The draw-only spread holds the cohort fixed and varies only the draw, so it
        # is built from the observed correctness laid out in the same stratum column
        # order the bootstrap index matrix uses. Nothing below may mix the two orders.
        C_obs = np.broadcast_to(c_part[order], (n_draws, full.n_participants))
        blocks[arm] = {
            NESTED: decompose(C_boot, sl, n_take, sel_boot),
            SAMPLING_ONLY: decompose(C_boot, sl, n_take, None),
            DRAW_ONLY: decompose(C_obs, sl, n_take, sel_obs),
        }

        one = np.broadcast_to(c_part[order], (1, full.n_participants))
        exact = decompose(one, sl, n_take, None)
        mc = decompose(C_obs, sl, n_take, sel_obs)
        points[arm] = {
            "a_full": float(exact["a_full"][0]),
            "b_restricted": float(exact["b_restricted"][0]),
            "d_counterfactual_mean_of_draws": float(mc["d_counterfactual"].mean()),
            "d_counterfactual_exact": float(exact["d_counterfactual"][0]),
            "total drop": float(exact["total drop"][0]),
            "class rebalancing": float(exact["a_full"][0]
                                       - mc["d_counterfactual"].mean()),
            "age-specific selection": float(mc["d_counterfactual"].mean()
                                            - exact["b_restricted"][0]),
        }
        print(f"       {arm:<16} fitted")

    # -- rows --------------------------------------------------------------------
    alpha_family = REGISTRY.alpha_corrected(TOTAL_DROP_FAMILY)
    for arm in arms:
        p = points[arm]
        mc_error = p["d_counterfactual_mean_of_draws"] - p["d_counterfactual_exact"]
        for component in COMPONENTS:
            for interval, blk in blocks[arm].items():
                d = blk[component]
                lo, hi = percentile_interval(d)
                stats = bootstrap_p(d)
                fam = (TOTAL_DROP_FAMILY
                       if (component == "total drop" and interval == NESTED) else None)
                if fam is not None:
                    # One source of truth for a corrected threshold: the percentiles
                    # come from the registry, never from a number typed here.
                    lo_pct, hi_pct = REGISTRY.ci_percentiles(fam)
                    lo, hi = np.percentile(d, [lo_pct, hi_pct])
                    ci_level = 1.0 - alpha_family
                else:
                    ci_level = 0.95
                rows.append({
                    "kind": "component", "arm": arm, "unit": "participant",
                    "component": component, "interval_basis": interval,
                    "point": p[component], "ci_low": float(lo), "ci_high": float(hi),
                    "ci_level": ci_level,
                    "p_bootstrap": stats["p_bootstrap"],
                    "p_at_resolution_floor": stats["p_at_resolution_floor"],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "family": fam or UNDECLARED,
                    "alpha_used": alpha_family if fam else 0.05,
                    "significant_at_alpha": bool(
                        fam is not None and stats["p_bootstrap"] < alpha_family
                        and (lo > 0 or hi < 0)),
                    "n_bootstrap": n_boot, "n_draws": n_draws,
                    "a_full": p["a_full"], "b_restricted": p["b_restricted"],
                    "d_counterfactual": p["d_counterfactual_mean_of_draws"],
                    "d_counterfactual_exact_expectation": p["d_counterfactual_exact"],
                    "d_monte_carlo_error": mc_error,
                    "n_full": full.n_participants,
                    "n_restricted": int(counts[0] + counts[2] + counts[3]),
                    "n_counterfactual": int(n_take + counts[2] + counts[3]),
                    "n_controls_drawn_from": n_controls,
                    "code_fingerprint": full.fingerprint.get(arm, ""),
                    "note": ("declared family age_cohort_shift, interval at its corrected "
                             "level; the declaration does not name a unit and this row is "
                             "the PARTICIPANT unit" if fam else
                             "no declared family covers a counterfactual decomposition, so "
                             "this is a nominal 95 percent interval and not a "
                             "pre-registered test")})

        # The identity is checked, not assumed, on the primary draws.
        blk = blocks[arm][NESTED]
        resid = np.abs(blk["class rebalancing"] + blk["age-specific selection"]
                       - blk["total drop"]).max()
        rows.append({"kind": "identity_check", "arm": arm, "unit": "participant",
                     "component": "rebalancing plus age-specific minus total drop",
                     "interval_basis": NESTED,
                     "point": float(resid), "family": UNDECLARED,
                     "n_bootstrap": n_boot, "n_draws": n_draws,
                     "note": "largest absolute departure from the identity over every "
                             "outer resample; it must be zero to floating point"})
        if resid > 1e-9:
            raise SystemExit(f"a33 STOPPED. The two components do not sum to the total "
                             f"drop for arm {arm!r}: worst residual {resid:.3e}.")

        # The image-level total drop, reported beside the participant-level one so the
        # unit difference is visible rather than left for a reader to discover.
        a_img, b_img = image_level_ab(full, arm, keep)
        rows.append({"kind": "unit_context", "arm": arm, "unit": "image",
                     "component": "total drop", "interval_basis": "point estimate only",
                     "point": a_img - b_img, "a_full": a_img, "b_restricted": b_img,
                     "family": UNDECLARED, "n_bootstrap": 0, "n_draws": 0,
                     "code_fingerprint": full.fingerprint.get(arm, ""),
                     "note": "image unit, for comparison with t21 only; the decomposition "
                             "is reported at the participant unit"})

    table = pd.DataFrame(rows)
    lead = ["kind", "arm", "unit", "component", "interval_basis", "point", "ci_low",
            "ci_high", "ci_level", "p_bootstrap", "excludes_zero", "family",
            "alpha_used", "significant_at_alpha"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen --------------------------------------------------------------
    elapsed = time.time() - t_start
    comp = table[table["kind"] == "component"]
    primary = comp[comp["interval_basis"] == NESTED]
    print("\n" + "=" * 100)
    print("a33  HOW MUCH OF THE ACCURACY DROP IS ARITHMETIC, AND HOW MUCH IS AGE?")
    print("=" * 100)
    print(f"  Full cohort {full.n_participants} participants ({counts[0] + counts[1]} / "
          f"{counts[2]} / {counts[3]}). Restricted "
          f"{int(counts[0] + counts[2] + counts[3])} ({counts[0]} / {counts[2]} / "
          f"{counts[3]}).")
    print(f"  Counterfactual {int(n_take + counts[2] + counts[3])}: {n_take} controls drawn "
          f"at random from all {n_controls}, plus every impaired participant.")
    print(f"  Participant level, all {len(arms)} full-cohort arms. Accuracy points. "
          f"PRIMARY interval is the nested one.")
    print(f"\n  {'arm':<16}{'(a) full':>9}{'(b) restr':>10}{'(d) cf':>9}"
          f"{'total':>8}{'rebalance [95 pct]':>26}{'age-specific [95 pct]':>28}")
    for arm in arms:
        g = primary[primary["arm"] == arm]
        reb = g[g["component"] == "class rebalancing"].iloc[0]
        age_c = g[g["component"] == "age-specific selection"].iloc[0]
        tot = g[g["component"] == "total drop"].iloc[0]
        print(f"  {arm:<16}{tot['a_full']:>9.2f}{tot['b_restricted']:>10.2f}"
              f"{tot['d_counterfactual']:>9.2f}{tot['point']:>8.2f}"
              f"{reb['point']:>9.2f} [{reb['ci_low']:+.2f}, {reb['ci_high']:+.2f}]"
              f"{age_c['point']:>11.2f} [{age_c['ci_low']:+.2f}, {age_c['ci_high']:+.2f}]")

    age_rows = primary[primary["component"] == "age-specific selection"]
    n_excl = int(age_rows["excludes_zero"].sum())
    reb_rows = primary[primary["component"] == "class rebalancing"]
    share = (reb_rows["point"].to_numpy()
             / primary[primary["component"] == "total drop"]["point"].to_numpy())
    print(f"\n  AGE-SPECIFIC COMPONENT: the interval excludes zero for {n_excl} of "
          f"{len(age_rows)} arms.")
    print(f"  Range across arms {age_rows['point'].min():+.2f} to "
          f"{age_rows['point'].max():+.2f} accuracy points.")
    print(f"  CLASS REBALANCING: {reb_rows['point'].min():+.2f} to "
          f"{reb_rows['point'].max():+.2f} points, which is "
          f"{100 * share.min():.0f} to {100 * share.max():.0f} percent of the drop.")
    if n_excl < len(age_rows):
        print(f"  !! For {len(age_rows) - n_excl} arm(s) the age-specific interval includes "
              f"zero. For those arms the paper cannot say the drop is age; it must say the "
              f"drop is rebalancing arithmetic.")

    print(f"\n  TOTAL DROP against the declared family {TOTAL_DROP_FAMILY}, corrected alpha "
          f"{alpha_family:.6g}:")
    tot_rows = primary[primary["component"] == "total drop"]
    for _, r in tot_rows.iterrows():
        print(f"    {r['arm']:<16}{r['point']:>8.2f}  "
              f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] at "
              f"{100 * r['ci_level']:.4g} percent   p {r['p_bootstrap']:.4g}   "
              f"{'significant' if r['significant_at_alpha'] else 'NOT significant'}")
    print(f"    The registry's audit lists this family as NOT RUN. It is realised here at "
          f"the PARTICIPANT unit,")
    print(f"    {len(tot_rows)} comparisons against a declared size of "
          f"{REGISTRY.family(TOTAL_DROP_FAMILY).n_declared}.")

    print("\n  THE THREE INTERVALS, and what each one covers:")
    for basis in (NESTED, SAMPLING_ONLY, DRAW_ONLY):
        g = comp[(comp["interval_basis"] == basis) &
                 (comp["component"] == "age-specific selection")]
        width = (g["ci_high"] - g["ci_low"]).mean()
        print(f"    {basis}")
        print(f"      mean width on the age-specific component: {width:.3f} accuracy points")
    print("    The nested interval is the primary one. It is the widest because it carries "
          "both sources.")

    mcerr = comp["d_monte_carlo_error"].abs().max()
    print(f"\n  MONTE CARLO CHECK: the {n_draws:,} draw mean of the counterfactual differs "
          f"from its exact")
    print(f"  expectation by at most {mcerr:.4f} accuracy points across all arms. "
          f"The expectation is closed")
    print(f"  form because accuracy is linear in per-participant correctness.")

    rep = table[table["kind"] == "a21_reproduction"]
    if not rep.empty:
        print(f"\n  a21 REPRODUCTION, image unit, {len(rep)} of {len(arms)} arms have a t21 "
              f"row:")
        print(f"    every checked arm reproduces t21 to within {T21_TOLERANCE} accuracy "
              f"points on both (a) and (b).")
        print(f"    The participant-level (a) and (b) printed above are a DIFFERENT UNIT "
              f"and do not match t21.")
        miss = [a for a in arms if a not in set(rep["arm"])]
        if miss:
            print(f"    No t21 row exists for {', '.join(miss)}; those arms are unchecked "
                  f"against a21.")

    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "the full-to-restricted accuracy drop split into class rebalancing and age-specific "
        "selection for all seven full-cohort arms, using a counterfactual cohort of 85 "
        "controls drawn at random from all 266 plus every impaired participant, nested "
        "inside a participant-clustered bootstrap stratified on four cohort cells",
        "participant-level accuracy, percentage points",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
