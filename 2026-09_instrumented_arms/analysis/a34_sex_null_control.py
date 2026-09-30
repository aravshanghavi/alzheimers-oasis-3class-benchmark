#!/usr/bin/env python3
"""a34: does the sex shortcut survive when the training labels are destroyed?

    python analysis/a34_sex_null_control.py
    python analysis/a34_sex_null_control.py --quick      # 200 resamples
    python analysis/a34_sex_null_control.py --boot 10000

THE MECHANISM UNDER TEST
------------------------
Among the 85 cognitively normal participants of the restricted cohort, every one
of the eight objectives assigns an impaired class to men more often than to
women. The risk differences run from +0.317 to +0.438. The explanation the
manuscript offers is a base rate: CDR 0.5 makes up 0.482 of male participants and
0.286 of female participants in this cohort, and the networks are reproducing
that sex-specific share from the training labels.

That explanation makes a prediction that can be checked with runs that already
exist. exp14 trained the same architecture on PERMUTED labels. Permuting the
labels destroys the association between sex and CDR 0.5 in the training set, so
if the base rate is the pathway, the disparity must shrink toward zero in the
permuted runs. If it does not shrink, the base rate is not the pathway and the
networks are reading something sex-related out of the image itself, which is a
different and more serious claim.

This is a falsification test, and it is the whole reason the script exists. It
can only come out one of two ways and both ways change the paper.

FIVE REPLICATES, AND WHAT THAT ALLOWS
--------------------------------------
There are five permuted-label replicates, each pooled over its five test folds.
Five numbers cannot support a confidence interval on the null's mean disparity,
and none is offered. What five numbers CAN support is a range and a count: the
observed spread of the five, and how many of the eight real arms sit outside it.
The replicate count is printed beside every number in this file so that nobody
reads the null's spread as though it came from a large sample.

Each replicate DOES get a participant-clustered interval of its own, because that
interval answers a different question: how precisely is THIS replicate's disparity
measured on 85 people. That is a within-replicate sampling question and 85
participants is the sample size for it. It is not an interval on the null.

THE SMOKE RUN IS EXCLUDED BY CONFIG, NOT BY NAME
-------------------------------------------------
run_permnull.py --smoke writes a real COMPLETE run with a real seed and fold into
the same outputs directory: two epochs, 64-pixel images, no pretrained weights.
Selecting runs by filename would either keep it or need a hand-maintained list of
names. The permutation tree's own src/runfilter.py decides production-ness from
the resolved config, and a23's collect_null already calls it, so that is what is
used here. Every excluded run is printed with the reason the filter gave.

THE BOOTSTRAP
-------------
Participants are resampled WITHIN sex among the 85 controls, so the 23 and the 62
stay fixed. An unstratified draw can return 15 men, and a risk difference
estimated on 15 men is not an estimate of the same quantity at a different
precision, it is a different design. Every arm and every replicate is scored on
the SAME draw, so any difference taken between them is paired.

MULTIPLICITY
------------
No family in analysis/_registry.py covers a sex disparity, under the null or
otherwise. Every row carries the literal string "undeclared", every interval is a
nominal 95 percent one, and nothing here is a pre-registered test. Nothing is
written to the registry: the changelog entry is PRINTED for the first author to
paste.
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
                                    bootstrap_p, changelog_entry,
                                    null_replicates, percentile_interval,
                                    stratified_participant_bootstrap)
from analysis.a23_protocol_audit import (ARM_ORDER, PERMNULL_TREE,            # noqa: E402
                                         collect_null, participant_frame,
                                         prob_frame, warn)
from analysis.a25_increment_over_metadata import multiplicity                 # noqa: E402

SOURCE = "analysis/a34_sex_null_control.py"
EXPERIMENT_ID = "a34"
TABLE = "t34_sex_null_control"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

REAL = "trained arm"
NULL = "permuted-label replicate"

# The range the manuscript reports for the eight real arms. It is printed as the
# reference the null is being compared against, and the script recomputes it here
# rather than quoting it, so a drift between the two is visible.
REPORTED_REAL_RANGE = (0.317, 0.438)


def sex_vector(participants: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """1 for male, 0 for female, NaN when the field is absent."""
    raw = meta.reindex(participants)["M/F"]
    v = np.where(raw.astype(str).str.upper().str.startswith("M"), 1.0, 0.0)
    v[raw.isna().to_numpy()] = np.nan
    return v


def sex_stats(frame: pd.DataFrame, keep: np.ndarray, male: np.ndarray,
              W: np.ndarray) -> Dict[str, object]:
    """Disparity among the kept participants, point estimates and bootstrap draws.

    `keep` selects the true CDR 0 participants in the frame's own row order, and
    `male` is the sex indicator over those same rows. W is the shared
    sex-stratified multiplicity matrix, so every model here is scored on one draw.
    """
    probs = frame[[f"p{c}" for c in range(N_CLASSES)]].to_numpy(dtype=np.float64)[keep]
    pred = frame["pred"].to_numpy().astype(int)[keep]
    m = male.astype(bool)
    f = ~m
    n_m, n_f = int(m.sum()), int(f.sum())

    impaired = (pred > 0).astype(np.float64)
    severe = (pred == 2).astype(np.float64)
    p1 = probs[:, 1]
    p2 = probs[:, 2]

    def split(v: np.ndarray) -> Tuple[float, float, np.ndarray]:
        vm = float(v[m].mean()) if n_m else float("nan")
        vf = float(v[f].mean()) if n_f else float("nan")
        dm = (W[:, m] @ v[m]) / n_m
        df = (W[:, f] @ v[f]) / n_f
        return vm, vf, dm - df

    imp_m, imp_f, imp_d = split(impaired)
    sev_m, sev_f, sev_d = split(severe)
    p1_m, p1_f, p1_d = split(p1)
    p2_m, p2_f, p2_d = split(p2)

    lo, hi = percentile_interval(imp_d)
    stats = bootstrap_p(imp_d)
    lo1, hi1 = percentile_interval(p1_d)
    return {
        "n_male": n_m, "n_female": n_f,
        "share_impaired_male": imp_m, "share_impaired_female": imp_f,
        "risk_difference": imp_m - imp_f,
        "risk_difference_ci_low": lo, "risk_difference_ci_high": hi,
        "risk_difference_p_bootstrap": stats["p_bootstrap"],
        "risk_difference_p_at_resolution_floor": stats["p_at_resolution_floor"],
        "risk_difference_excludes_zero": bool(lo > 0 or hi < 0),
        "n_impaired_male": int(impaired[m].sum()), "n_impaired_female": int(impaired[f].sum()),
        "share_cdr_ge1_male": sev_m, "share_cdr_ge1_female": sev_f,
        "share_cdr_ge1_difference": sev_m - sev_f,
        "mean_p_cdr05_male": p1_m, "mean_p_cdr05_female": p1_f,
        "mean_p_cdr05_difference": p1_m - p1_f,
        "mean_p_cdr05_difference_ci_low": lo1, "mean_p_cdr05_difference_ci_high": hi1,
        "mean_p_cdr_ge1_male": p2_m, "mean_p_cdr_ge1_female": p2_f,
    }


def load_null_frames(tree: Path) -> Tuple[Dict[int, pd.DataFrame], List[Tuple[str, str]]]:
    """Participant-level frames for each permuted replicate, five folds pooled.

    collect_null is a23's, and it calls the permutation tree's own
    production_reason, so the smoke run is dropped by its resolved config rather
    than by anything about its name. A replicate that does not have exactly five
    folds is dropped and named, because four folds is not this cohort.
    """
    by_seed, excluded = collect_null(tree)
    out: Dict[int, pd.DataFrame] = {}
    for seed, dirs in sorted(by_seed.items()):
        if len(dirs) != 5:
            excluded.append((f"seed {seed}", f"{len(dirs)} folds, need 5"))
            continue
        parts = []
        for d in dirs:
            z = np.load(d / "predictions_test.npz", allow_pickle=False)
            parts.append(prob_frame(z["probabilities"].astype(np.float64),
                                    z["participant_id"].astype(str), z["labels"]))
        img = pd.concat(parts, ignore_index=True)
        out[seed] = participant_frame(img)
    return out, excluded


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="Sex disparity under permuted training labels.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"paired bootstrap resamples (default {N_BOOT})")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--permnull-from", default=None, metavar="TREE",
                    help=f"root of the {PERMNULL_TREE} tree")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"a34  sex disparity under the permuted-label null, {n_boot:,} resamples, "
          f"base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images")

    meta = pd.read_excel(metadata_xlsx())
    meta["pid"] = meta["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid")

    sex = sex_vector(coh.participants, meta)
    if np.isnan(sex).any():
        raise SystemExit(f"a34 STOPPED. Sex is missing for {int(np.isnan(sex).sum())} of the "
                         f"{coh.n_participants} restricted participants, so the disparity "
                         f"cannot be computed on the cohort it claims to describe.")
    controls = coh.labels == 0
    male = sex[controls] > 0.5
    n_m, n_f = int(male.sum()), int((~male).sum())
    print(f"     true CDR 0 participants: {int(controls.sum())} ({n_m} men, {n_f} women)")
    if (int(controls.sum()), n_m, n_f) != (85, 23, 62):
        warn(f"the control counts are ({int(controls.sum())}, {n_m}, {n_f}), not the "
             f"expected (85, 23, 62). Every number below is computed from the data, but "
             f"read them knowing the cohort is not the one this script was written against.")

    # -- the base rate the mechanism claims the networks are copying -------------
    base_male = float(((coh.labels == 1) & (sex > 0.5)).sum() / max((sex > 0.5).sum(), 1))
    base_female = float(((coh.labels == 1) & (sex <= 0.5)).sum() / max((sex <= 0.5).sum(), 1))
    print(f"     cohort base rate of CDR 0.5: {base_male:.3f} of men, "
          f"{base_female:.3f} of women. That is the quantity the mechanism says the")
    print(f"     networks reproduce, and permuting the training labels destroys it.")

    # -- the permuted replicates --------------------------------------------------
    tree = Path(args.permnull_from).expanduser() if args.permnull_from \
        else REPO_ROOT.parent / PERMNULL_TREE
    null_frames: Dict[int, pd.DataFrame] = {}
    excluded: List[Tuple[str, str]] = []
    n_null_runs = 0
    if not tree.is_dir():
        warn(f"no permutation tree at {tree}. The null half cannot run and only the eight "
             f"real arms are reported.")
    else:
        null_metrics, excl_metrics, n_null_runs = null_replicates(tree)
        null_frames, excluded = load_null_frames(tree)
        n_rep = len(null_frames)
        print(f"\n     permuted-label null: {n_rep} replicate(s) from {n_null_runs} "
              f"production runs, in {tree}")
        for name, why in excluded:
            print(f"       excluded by src/runfilter.py: {name}: {why}")
        if not null_metrics.empty:
            mf = null_metrics[null_metrics["unit"] == "participant"]["macro_f1"]
            print(f"       context, from null_replicates: participant-level macro-F1 across "
                  f"the {n_rep} replicate(s) runs {mf.min():.4f} to {mf.max():.4f}")

    n_rep = len(null_frames)

    # -- one shared bootstrap draw ------------------------------------------------
    idx = stratified_participant_bootstrap(male.astype(int), n_boot, "a34:sex", BASE_SEED,
                                           cohort="restricted")
    W = multiplicity(idx, int(controls.sum()))

    rows: List[Dict] = []
    # stratified_participant_bootstrap groups its strata contiguously, but it indexes
    # into the ORIGINAL positions, so multiplicity() returns counts in the original
    # participant order and no reordering is needed here. The check below says so
    # rather than leaving it to be rediscovered.
    if not np.all(W.sum(axis=1) == int(controls.sum())):
        raise SystemExit("a34 STOPPED. The bootstrap multiplicities do not sum to the "
                         "control count; the draw is not a resample of these people.")
    if not (np.all(W[:, male].sum(axis=1) == n_m) and np.all(W[:, ~male].sum(axis=1) == n_f)):
        raise SystemExit(f"a34 STOPPED. The bootstrap did not hold the sexes at {n_m} men "
                         f"and {n_f} women, so the risk difference is not the declared "
                         f"estimand.")

    for arm in coh.order():
        frame = coh.part[arm]
        s = sex_stats(frame, controls, male, W)
        rows.append({"kind": REAL, "model": arm, "perm_seed": np.nan,
                     "n_null_replicates": n_rep, "n_bootstrap": n_boot,
                     "unit": "participant",
                     "base_rate_cdr05_male": base_male,
                     "base_rate_cdr05_female": base_female,
                     "family": UNDECLARED, "ci_level": 0.95,
                     "code_fingerprint": coh.fingerprint.get(arm, ""),
                     "note": "trained on the real labels; interval is the within-arm "
                             "participant-clustered one, sexes held at their observed "
                             "counts", **s})

    for seed, frame in sorted(null_frames.items()):
        aligned = frame.set_index("participant_id").reindex(coh.participants).reset_index()
        if aligned[[f"p{c}" for c in range(N_CLASSES)]].isna().any().any():
            raise SystemExit(f"a34 STOPPED. Permuted replicate seed {seed} does not cover "
                             f"the same {coh.n_participants} participants as the trained "
                             f"arms, so it cannot be compared with them.")
        s = sex_stats(aligned, controls, male, W)
        rows.append({"kind": NULL, "model": f"permuted seed {seed}", "perm_seed": seed,
                     "n_null_replicates": n_rep, "n_bootstrap": n_boot,
                     "unit": "participant",
                     "base_rate_cdr05_male": base_male,
                     "base_rate_cdr05_female": base_female,
                     "family": UNDECLARED, "ci_level": 0.95,
                     "code_fingerprint": "",
                     "note": f"training labels permuted; this interval is the WITHIN "
                             f"replicate sampling interval over {int(controls.sum())} "
                             f"participants, not an interval on the null, which "
                             f"{n_rep} replicates cannot support", **s})

    table = pd.DataFrame(rows)
    real = table[table["kind"] == REAL]
    null = table[table["kind"] == NULL]

    # -- every arm against the measured null, a23's R2 pattern ---------------------
    if not null.empty:
        rd_real = real["risk_difference"].to_numpy(dtype=float)
        rd_null = null["risk_difference"].to_numpy(dtype=float)
        p_floor = 1.0 / (n_rep + 1)
        extra = []
        for _, r in real.iterrows():
            v = float(r["risk_difference"])
            ge = int((rd_null >= v).sum())
            extra.append({
                "kind": "arm_vs_null", "model": r["model"], "unit": "participant",
                "n_null_replicates": n_rep, "n_bootstrap": n_boot,
                "risk_difference": v,
                "null_risk_difference_mean": float(rd_null.mean()),
                "null_risk_difference_min": float(rd_null.min()),
                "null_risk_difference_max": float(rd_null.max()),
                "null_ge_arm": ge,
                "outside_observed_null_range": bool(v > rd_null.max()),
                "p_permutation": (1 + ge) / (1 + n_rep),
                "p_permutation_floor": p_floor,
                "family": UNDECLARED,
                "note": f"the directed quantity is tested, male minus female, because the "
                        f"mechanism predicts a shift toward zero and not a shift toward a "
                        f"smaller magnitude. With {n_rep} replicates the smallest "
                        f"reportable permutation p is {p_floor:.4f}, so p < 0.05 is not "
                        f"available at any effect size"})
        margin = float(rd_real.min() - rd_null.max())
        extra.append({
            "kind": "comparison", "model": "real arms against permuted replicates",
            "n_null_replicates": n_rep, "n_bootstrap": n_boot, "unit": "participant",
            "real_risk_difference_min": float(rd_real.min()),
            "real_risk_difference_max": float(rd_real.max()),
            "null_risk_difference_min": float(rd_null.min()),
            "null_risk_difference_max": float(rd_null.max()),
            "null_risk_difference_mean": float(rd_null.mean()),
            "n_real_arms_above_every_replicate": int((rd_real > rd_null.max()).sum()),
            "n_real_arms": int(len(rd_real)),
            "margin_min_real_minus_max_null": margin,
            "null_range_width": float(rd_null.max() - rd_null.min()),
            "p_permutation_floor": p_floor,
            "family": UNDECLARED,
            "note": f"{n_rep} replicates. No interval is placed on the null's mean: five "
                    f"numbers cannot support one, so a range, a count and a permutation p "
                    f"floored at {p_floor:.4f} are what is reported instead"}) 
        table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)

    lead = ["kind", "model", "perm_seed", "unit", "n_male", "n_female",
            "share_impaired_male", "share_impaired_female", "risk_difference",
            "risk_difference_ci_low", "risk_difference_ci_high",
            "risk_difference_excludes_zero", "mean_p_cdr05_male", "mean_p_cdr05_female",
            "n_null_replicates", "family"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 100)
    print("a34  DOES THE SEX DISPARITY SURVIVE WHEN THE TRAINING LABELS ARE PERMUTED?")
    print("=" * 100)
    print(f"  Restricted cohort, true CDR 0 only: {n_m} men and {n_f} women. Participant "
          f"level.")
    print(f"  Risk difference = share of men given an impaired class minus the share of "
          f"women. Positive means men.")
    print(f"  Cohort base rate of CDR 0.5: {base_male:.3f} male, {base_female:.3f} female. "
          f"Permuting destroys it.")
    print(f"\n  {'model':<26}{'men imp':>9}{'women imp':>11}{'risk diff':>11}"
          f"  {'[95 pct]':<22}{'mean P(0.5) M':>14}{'F':>8}")
    for _, r in real.iterrows():
        print(f"  {r['model']:<26}{r['share_impaired_male']:>9.3f}"
              f"{r['share_impaired_female']:>11.3f}{r['risk_difference']:>+11.3f}"
              f"  [{r['risk_difference_ci_low']:+.3f}, {r['risk_difference_ci_high']:+.3f}]"
              f"    {r['mean_p_cdr05_male']:>12.3f}{r['mean_p_cdr05_female']:>8.3f}")
    if null.empty:
        print("\n  NO PERMUTED REPLICATES WERE LOADED. The falsification test did not run.")
    else:
        print(f"\n  permuted-label replicates, {n_rep} of them:")
        for _, r in null.iterrows():
            print(f"  {r['model']:<26}{r['share_impaired_male']:>9.3f}"
                  f"{r['share_impaired_female']:>11.3f}{r['risk_difference']:>+11.3f}"
                  f"  [{r['risk_difference_ci_low']:+.3f}, "
                  f"{r['risk_difference_ci_high']:+.3f}]"
                  f"    {r['mean_p_cdr05_male']:>12.3f}{r['mean_p_cdr05_female']:>8.3f}")

        rd_real = real["risk_difference"].to_numpy(dtype=float)
        rd_null = null["risk_difference"].to_numpy(dtype=float)
        p_floor = 1.0 / (n_rep + 1)
        n_above = int((rd_real > rd_null.max()).sum())
        margin = float(rd_real.min() - rd_null.max())
        print(f"\n  THE ANSWER, with {n_rep} replicates behind every null number:")
        print(f"    real arms      {len(rd_real)} of them, risk difference "
              f"{rd_real.min():+.3f} to {rd_real.max():+.3f} "
              f"(the manuscript reports {REPORTED_REAL_RANGE[0]:+.3f} to "
              f"{REPORTED_REAL_RANGE[1]:+.3f})")
        print(f"    permuted null  {n_rep} of them, risk difference "
              f"{rd_null.min():+.3f} to {rd_null.max():+.3f}, mean {rd_null.mean():+.3f}, "
              f"width {rd_null.max() - rd_null.min():.3f}")
        n_excl = int(null["risk_difference_excludes_zero"].sum())
        print(f"    {n_excl} of the {n_rep} permuted replicates have a within-replicate "
              f"interval that excludes zero.")
        print(f"    {n_above} of the {len(rd_real)} real arms sit above EVERY permuted "
              f"replicate. Smallest real arm minus")
        print(f"    largest replicate: {margin:+.3f}. Permutation p for every arm that "
              f"clears all {n_rep}: {p_floor:.4f}, the floor.")
        if n_above == len(rd_real):
            print("\n    THE DISPARITY SHRINKS UNDER THE NULL. Every real arm is above "
                  "every permuted replicate,")
            print("    and the permuted replicates centre near zero. That is what the "
                  "base-rate mechanism predicts,")
            print("    and the mechanism sentence in the paper now has a direct test "
                  "behind it.")
            if margin < 0.10 or (rd_null.max() - rd_null.min()) > 0.4:
                print("\n    READ THAT WITH THE MARGIN IN VIEW. The separation is narrow "
                      "and the null's own spread is")
                print("    wide, so the test is weak, not decisive. With five replicates "
                      "the strongest sentence")
                print(f"    available is that no replicate reached the real range, at a "
                      f"permutation p of {p_floor:.4f}.")
        elif n_above == 0:
            print("\n    THE DISPARITY PERSISTS UNDER THE NULL. Every real arm is matched "
                  "or exceeded by a permuted")
            print("    replicate whose training labels carried no sex-specific base rate. "
                  "The base-rate mechanism")
            print("    does not explain the disparity, and the paper must state the "
                  "alternative instead: the")
            print("    networks are reading something sex-related out of the image itself.")
        else:
            print(f"\n    PARTIAL. {n_above} of {len(rd_real)} real arms clear every "
                  f"permuted replicate and the rest do not.")
            print("    The base rate explains part of the disparity and not all of it, and "
                  "the paper has to say")
            print("    which arms are which rather than reporting one verdict for all "
                  "eight.")
        print(f"\n    FIVE REPLICATES CANNOT SUPPORT AN INTERVAL. No interval is placed on "
              f"the null's mean")
        print(f"    disparity anywhere in this output. The per-replicate intervals printed "
              f"above are")
        print(f"    within-replicate sampling intervals over {int(controls.sum())} "
              f"participants and say nothing")
        print(f"    about how much a sixth replicate would move. With {n_rep} replicates "
              f"the strongest available")
        print(f"    sentence is a range and a count, and that is what is written above.")

    print("\n  EVERY comparison here is UNDECLARED. No registry family covers a sex "
          "disparity, no p-value")
    print("  here is corrected, and the intervals are nominal 95 percent.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "male minus female risk difference in predicted impairment among true CDR 0 "
        "participants, and mean predicted P(CDR 0.5) by sex, for the eight trained "
        "restricted arms against the permuted-label replicates",
        "risk difference and mean predicted probability",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
