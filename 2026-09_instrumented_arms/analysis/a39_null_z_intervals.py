#!/usr/bin/env python3
"""a39: every z against the permuted-label null is quoted without its own error bar.

    python analysis/a39_null_z_intervals.py
    python analysis/a39_null_z_intervals.py --quick      # 200 resamples
    python analysis/a39_null_z_intervals.py --boot 10000

THE PROBLEM
-----------
The paper separates arms from chance by reporting how many null standard
deviations above the null mean each arm sits. That number is a ratio, and its
denominator is estimated from FIVE permuted-label replicates. The standard
deviation of five numbers has a standard error of roughly SD / sqrt(2(n-1)),
which is SD / 2.83, so the denominator is itself uncertain by about a third of
its own size.

A quantity whose denominator is uncertain by a third is not a point estimate. Every
z in the manuscript is currently printed as though it were one. a23 already prints
the standard error of the null SD beside each z, which is the right instinct and
does not go far enough: a reader cannot turn SD / 2.83 into an interval on z in
their head, because the map from the SD to the z is nonlinear and the two ends are
not symmetric. So the interval is computed here and travels with the z.

HOW THE INTERVAL IS BUILT, AND WHY IT IS WIDE
---------------------------------------------
The arm's own value is held fixed. The five replicate values are resampled with
replacement, the null mean and the null SD are recomputed inside each resample,
and z is recomputed from those. The percentile interval of those recomputed z
values is the reported interval.

Five points cannot support a precise interval, and this file does not pretend
otherwise. A bootstrap over five values has at most C(9,4) = 126 distinct
resamples, so the interval below takes at most 126 distinct values no matter how
large --boot is. Raising --boot makes the percentile smoother, never the interval
narrower or better founded. Resamples in which all five draws happen to be the
same replicate have zero SD and an undefined z; those are counted and reported
rather than dropped quietly.

The interval is therefore WIDE BY CONSTRUCTION. That width is the honest object.
It is what five replicates buy. The alternative on offer is not a narrower
interval, it is a point estimate whose uncertainty nobody has written down.

THREE OTHER NUMBERS THAT BELONG BESIDE EVERY Z
----------------------------------------------
THE PERMUTATION P FLOOR. With five replicates the smallest attainable
permutation p is 1 / (5 + 1) = 0.1667. No effect size can produce p < 0.05 from
this null. Any sentence of the form "significantly above chance" that leans on
the permutation test is unavailable at any effect size, and a z of 9 does not
change that.

THE MINIMUM DETECTABLE EFFECT. The smallest value that clears every observed
replicate, expressed as a distance above the null mean. An arm below it has not
separated from the measured null however large its z looks.

THE Z OF THE NULL MAXIMUM. a23's PASS rule for this requirement is "above every
replicate", which in z units is (null max minus null mean) / null SD. An arm
whose z interval reaches down BELOW that value has a separation the same data
could plausibly fail to show, and those are flagged. That bar is internal to the
existing rule, not a new convention invented here, which is why it is the one
used for the flag.

THE NULL IS THE RESTRICTED COHORT'S NULL
----------------------------------------
exp14 permuted the labels of the age-60 cohort, so the null describes that cohort
and the arms compared against it are the eight restricted-cohort arms. There is no
permuted-label null for the full cohort and none is invented here.

MULTIPLICITY
------------
a23 already reports its R2 rows as undeclared. Nothing changes that. Every row
here carries the literal string "undeclared", every interval is a nominal 95
percent one, and nothing here is a pre-registered test. Nothing is written to the
registry: the changelog entry is PRINTED for the first author to paste.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from math import comb
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import save_table                                       # noqa: E402
from analysis._determinism import seeded_rng                                  # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                              # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, RESTRICTED_EXPERIMENTS,           # noqa: E402
                                    changelog_entry, confusion, null_replicates,
                                    per_class_f1, percentile_interval)
from analysis.a23_protocol_audit import ARM_ORDER, PERMNULL_TREE, SHORT       # noqa: E402

SOURCE = "analysis/a39_null_z_intervals.py"
EXPERIMENT_ID = "a39"
TABLE = "t39_null_z_intervals"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

UNITS = ["participant", "image"]
QUANTITIES = ["macro_f1"] + [f"f1_{s}" for s in SHORT]

BOOTSTRAP = ("resample of the permuted-label replicates themselves, with the arm's own "
             "value held fixed")


def z_interval(value: float, null_vals: np.ndarray, n_boot: int,
               tag_parts: Dict[str, str]) -> Dict[str, float]:
    """Point z and a percentile interval on z from resampling the replicates.

    The arm's value does not move. What moves is the null mean and the null SD, and
    those are exactly the two estimated quantities the current point z treats as
    known. Resampling the arm as well would mix a different question, the sampling
    error of the arm's own metric, into an interval that is about the null.
    """
    n_rep = len(null_vals)
    mean = float(null_vals.mean())
    sd = float(null_vals.std(ddof=1)) if n_rep > 1 else float("nan")
    point = (value - mean) / sd if sd > 0 else float("nan")
    rng = seeded_rng("a39:z", BASE_SEED, **tag_parts)
    draws = null_vals[rng.integers(0, n_rep, size=(n_boot, n_rep))]
    m = draws.mean(axis=1)
    s = draws.std(axis=1, ddof=1)
    z = np.full(n_boot, np.nan)
    ok = s > 0
    z[ok] = (value - m[ok]) / s[ok]
    lo, hi = percentile_interval(z)
    return {"z": point, "z_ci_low": lo, "z_ci_high": hi,
            "null_mean": mean, "null_sd": sd,
            "null_sd_standard_error": sd / np.sqrt(2.0 * (n_rep - 1)) if n_rep > 1
            else float("nan"),
            "n_degenerate_draws": int((~ok).sum())}


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Intervals on the z-scores against the permuted-label null.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"resamples of the replicate set (default {N_BOOT})")
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

    print(f"a39  intervals on the null z-scores, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    tree = Path(args.permnull_from).expanduser() if args.permnull_from \
        else REPO_ROOT.parent / PERMNULL_TREE
    if not tree.is_dir():
        raise SystemExit(f"a39 CANNOT RUN. No permutation tree at {tree}. Every number in "
                         f"this file is a function of the permuted-label replicates, so "
                         f"there is nothing to compute without them. Pass --permnull-from.")
    null, excluded, n_runs = null_replicates(tree)
    if null.empty:
        raise SystemExit(f"a39 CANNOT RUN. No usable permuted-label replicates under {tree}.")
    n_rep = int(null["perm_seed"].nunique())
    p_floor = 1.0 / (n_rep + 1)
    distinct = comb(2 * n_rep - 1, n_rep - 1)
    print(f"     null: {n_rep} replicates from {n_runs} production runs at {tree}")
    if excluded:
        print(f"     {len(excluded)} run(s) excluded by the permutation tree's own filter:")
        for name, why in excluded[:6]:
            print(f"       {name}: {why}")
    print(f"     permutation p floor 1/({n_rep}+1) = {p_floor:.4f}. No effect size can "
          f"produce p < 0.05 from this null.")
    print(f"     a bootstrap over {n_rep} values has at most {distinct} distinct resamples, "
          f"so every interval below takes at most {distinct} distinct values.")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images")
    arms = [a for a in ARM_ORDER if a in coh.arms] + \
           [a for a in sorted(coh.arms) if a not in ARM_ORDER]

    rows: List[Dict] = []
    for unit in UNITS:
        sub = null[null["unit"] == unit]
        if sub.empty:
            print(f"  !! no replicate rows at unit {unit}; skipping that unit")
            continue
        for arm in arms:
            frame = coh.unit(unit)[arm]
            cm = confusion(frame["pred"].to_numpy(), frame["label"].to_numpy())
            f1 = per_class_f1(cm)
            values = {"macro_f1": float(f1.mean()),
                      **{f"f1_{SHORT[c]}": float(f1[c]) for c in range(N_CLASSES)}}
            for q in QUANTITIES:
                nv = sub[q].to_numpy(dtype=float)
                v = values[q]
                st = z_interval(v, nv, n_boot, {"unit": unit, "arm": arm, "metric": q})
                null_max = float(nv.max())
                mde = null_max - st["null_mean"]
                z_null_max = (mde / st["null_sd"]) if st["null_sd"] > 0 else float("nan")
                outside = bool(v > null_max)
                ge = int((nv >= v).sum())
                survives = bool(np.isfinite(st["z_ci_low"]) and np.isfinite(z_null_max)
                                and st["z_ci_low"] > z_null_max)
                rows.append({
                    "unit": unit, "arm": arm, "metric": q,
                    "arm_value": v,
                    "z": st["z"], "z_ci_low": st["z_ci_low"], "z_ci_high": st["z_ci_high"],
                    "ci_level": 0.95,
                    "z_interval_width": st["z_ci_high"] - st["z_ci_low"],
                    "null_mean": st["null_mean"], "null_sd": st["null_sd"],
                    "null_sd_standard_error": st["null_sd_standard_error"],
                    "null_min": float(nv.min()), "null_max": null_max,
                    "min_detectable_effect": mde,
                    "z_of_null_maximum": z_null_max,
                    "currently_claimed_separation": outside,
                    "survives_own_z_interval_lower_end": survives,
                    "null_ge_arm": ge,
                    "p_permutation": (1 + ge) / (1 + n_rep),
                    "p_permutation_floor": p_floor,
                    "n_null_replicates": n_rep, "n_null_runs": n_runs,
                    "n_distinct_resamples_possible": distinct,
                    "n_degenerate_draws": st["n_degenerate_draws"],
                    "bootstrap": BOOTSTRAP, "n_bootstrap": n_boot, "family": UNDECLARED,
                    "code_fingerprint": coh.fingerprint.get(arm, ""),
                    "note": (f"the interval on z propagates the sampling uncertainty of a "
                             f"null SD estimated from {n_rep} replicates. It is wide by "
                             f"construction and that width is what {n_rep} replicates buy. "
                             f"The flag compares the interval's lower end against a23's own "
                             f"PASS bar, the z of the null maximum.")})
        print(f"     {unit}: {len(arms) * len(QUANTITIES)} z intervals computed")

    table = pd.DataFrame(rows)
    lead = ["unit", "arm", "metric", "arm_value", "z", "z_ci_low", "z_ci_high",
            "z_of_null_maximum", "currently_claimed_separation",
            "survives_own_z_interval_lower_end"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 104)
    print("a39  EVERY Z AGAINST THE PERMUTED-LABEL NULL, WITH THE INTERVAL IT NEVER HAD")
    print("=" * 104)
    print(f"  {n_rep} permuted-label replicates. Null SD standard error is about SD/"
          f"{np.sqrt(2.0 * (n_rep - 1)):.2f}, a third of the SD itself.")
    print(f"  Permutation p floor {p_floor:.4f}: 'significant against chance' is not "
          f"available from this null at any effect size.")
    print(f"  A bootstrap over {n_rep} values has at most {distinct} distinct resamples, "
          f"so these intervals are coarse and wide by construction.")

    for unit in UNITS:
        t = table[table["unit"] == unit]
        if t.empty:
            continue
        print(f"\n  -- unit: {unit}")
        print(f"     {'arm':<16}" + "".join(f"{q:>22}" for q in QUANTITIES))
        for arm in arms:
            cells = []
            for q in QUANTITIES:
                r = t[(t["arm"] == arm) & (t["metric"] == q)].iloc[0]
                cells.append(f"{r['z']:>6.1f} [{r['z_ci_low']:.1f},{r['z_ci_high']:.1f}]")
            print(f"     {arm:<16}" + "".join(f"{c:>22}" for c in cells))
        print(f"     a23's PASS bar in z units (z of the null maximum) is about "
              f"{t['z_of_null_maximum'].min():.2f} to {t['z_of_null_maximum'].max():.2f}")

    claimed = table[table["currently_claimed_separation"]]
    fails = claimed[~claimed["survives_own_z_interval_lower_end"]]
    print(f"\n  WHICH CLAIMED SEPARATIONS SURVIVE THE LOWER END OF THEIR OWN Z INTERVAL")
    print(f"     {len(claimed)} of {len(table)} arm-by-metric-by-unit cells are currently "
          f"claimed as separated (arm above every replicate).")
    print(f"     {len(claimed) - len(fails)} of those {len(claimed)} still clear a23's own "
          f"PASS bar at the BOTTOM of their z interval.")
    if len(fails):
        print(f"     The following {len(fails)} do NOT, so the separation is one this same "
              f"data could plausibly fail to show:")
        for _, r in fails.sort_values(["unit", "arm", "metric"]).iterrows():
            print(f"       {r['unit']:<12}{r['arm']:<16}{r['metric']:<12}"
                  f"z {r['z']:>6.1f}  interval [{r['z_ci_low']:.1f}, {r['z_ci_high']:.1f}]  "
                  f"bar {r['z_of_null_maximum']:.2f}")
    else:
        print(f"     None fail, so the separations are robust to the uncertainty in the "
              f"null SD, at this bar.")
    print(f"\n  HOW TO WRITE THIS UP. Report the z WITH its interval, or report the simpler "
          f"and stronger")
    print(f"  sentence the null actually supports: the arm is above all {n_rep} permuted "
          f"replicates. The second")
    print(f"  claim needs no SD at all and cannot be attacked through the denominator.")
    print(f"\n  EVERY row here is UNDECLARED, as a23's R2 rows already are, and every "
          f"interval is nominal 95 percent.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "every restricted-cohort arm's z against the permuted-label null, for macro-F1 and "
        "each class's F1 at both units, with an interval on z obtained by resampling the "
        "five replicates, beside the minimum detectable effect and the permutation p floor",
        "z against the null SD, with interval",
        "participant and image"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
