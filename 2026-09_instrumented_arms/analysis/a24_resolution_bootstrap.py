#!/usr/bin/env python3
"""a24: is FA-FL's resolution advantage real, or is it the binning?

    python analysis/a24_resolution_bootstrap.py
    python analysis/a24_resolution_bootstrap.py --quick     # 200 resamples

THE CLAIM THIS SETTLES
----------------------
Of everything the paper says about FA-FL, one thing has kept surviving: its
probabilities look better sorted than the other objectives', so the Brier score
splits in its favour even where accuracy does not. Murphy's decomposition names
the three pieces:

    Brier  =  Reliability  -  Resolution  +  Uncertainty

Reliability is how far the predicted probabilities sit from the observed rates
inside each bin, and lower is better. Resolution is how far those observed
rates sit from the base rate, and higher is better: it is the part that says
the model separates people rather than repeating the prior. Uncertainty is a
property of the labels alone and is identical for every arm on a given cohort,
so it cannot favour anybody.

The catch is that Resolution computed from bins is a BIASED estimator. Bins
with few points have noisy observed rates, and that noise inflates Resolution.
Finer bins inflate it more. An arm whose predictions happen to be more spread
out therefore picks up more of that inflation, and can look better sorted
without being better sorted.

So this script does three things the single-number version cannot.

  1. It checks the identity closes and prints the leftover per arm. The
     identity is exact only when the score is rebuilt from the binned means.
     With the actual probabilities there is a leftover made of two within-bin
     terms, the spread of the predicted probabilities and its covariance with
     the outcome, so the leftover can come out either sign and usually comes out
     negative. If it is large the decomposition is not describing the score.

  2. It puts a participant-clustered interval on EVERY component, not just on
     the Brier score. A resolution gap with an interval spanning zero is not a
     finding, however large the point estimate.

  3. It recomputes everything at 10, 15 and 20 bins. If FA-FL's ordering
     against the next-best arm moves when the bin count moves, the ordering was
     the binning.

Temperature scaling is included because it changes reliability by construction
and should leave resolution close to untouched: scaling is monotone per class,
so it cannot reorder anything within a class. A resolution gap that appears or
vanishes under scaling is a numerical artefact of re-binning, and seeing that
is the point of running both halves.

NO REGISTRY FAMILY COVERS RESOLUTION
------------------------------------
The declared families cover ECE, accuracy, macro-F1, QWK, AUROC, the increment
result, leakage and the multislice null. None covers a Murphy component. So
every comparison here is marked undeclared and carries an ordinary 95 percent
interval, and none of them may be described as a pre-registered test. Declaring
a family after seeing this table would be exactly the post-hoc selection the
registry exists to stop.

WHY THE BOOTSTRAP IS A MATRIX MULTIPLY
--------------------------------------
Every ingredient of the decomposition is a sum of per-image contributions, so
it is additive over participants. Each participant is reduced once to a small
fixed block of bin counts, and a resample becomes a vector of participant
multiplicities times that block. 2000 resamples of 86,437 images for seven arms
at three bin counts is then a handful of small matrix products rather than
several hundred million rebinning operations. This is the same idea as
SuffStats in _common.py, applied to a statistic SuffStats does not carry.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import analysis.discover as dsc                                              # noqa: E402
from analysis._common import N_CLASSES, paired_difference, save_table         # noqa: E402
from analysis._determinism import seeded_rng                                  # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                              # noqa: E402

from analysis.a23_protocol_audit import (ARM_ORDER, FA, FULL_EXPERIMENTS,     # noqa: E402
                                         Cohort, find_full_tree, warn)
from analysis._protocol_lib import (RESTRICTED_EXPERIMENTS,                    # noqa: E402
                                    set_include_new_arms)

SOURCE = "analysis/a24_resolution_bootstrap.py"
N_BOOT = 2000
QUICK_BOOT = 200
BASE_SEED = 12345
BIN_COUNTS = (10, 15, 20)
UNDECLARED = "undeclared"
COMPONENTS = ["reliability", "resolution", "uncertainty", "brier", "residual"]
# Higher resolution is better. Everything else here is lower-is-better, except
# uncertainty, which is a property of the labels and is not a score at all.
HIGHER_BETTER = {"resolution": True, "reliability": False, "brier": False}


class MurphyStats:
    """Per-participant bin blocks for one arm at one bin count.

    Three blocks of shape (participants, classes x bins): the count of images
    in each bin, the sum of the predicted probability in each bin, and the sum
    of the class indicator in each bin. Plus each participant's Brier sum and
    image count. Everything else is arithmetic on weighted sums of these, which
    is what makes the clustered bootstrap a matrix product.
    """

    def __init__(self, frame: pd.DataFrame, n_bins: int, participants: np.ndarray):
        pid = frame["participant_id"].to_numpy()
        index = {p: i for i, p in enumerate(participants)}
        pi = np.fromiter((index[p] for p in pid), dtype=np.int64, count=len(pid))
        n_p = len(participants)
        probs = frame[[f"p{c}" for c in range(N_CLASSES)]].to_numpy(dtype=np.float64)
        y = frame["label"].to_numpy().astype(int)
        onehot = np.eye(N_CLASSES)[y]

        self.n_bins = n_bins
        self.width = N_CLASSES * n_bins
        edges = np.linspace(0.0, 1.0, n_bins + 1)

        self.count = np.zeros((n_p, self.width))
        self.sum_p = np.zeros((n_p, self.width))
        self.sum_o = np.zeros((n_p, self.width))
        for c in range(N_CLASSES):
            b = np.clip(np.digitize(probs[:, c], edges[1:-1], right=True), 0, n_bins - 1)
            flat = pi * self.width + c * n_bins + b
            size = n_p * self.width
            self.count += np.bincount(flat, minlength=size).reshape(n_p, self.width)
            self.sum_p += np.bincount(flat, weights=probs[:, c],
                                      minlength=size).reshape(n_p, self.width)
            self.sum_o += np.bincount(flat, weights=onehot[:, c],
                                      minlength=size).reshape(n_p, self.width)

        self.brier = np.bincount(pi, weights=((probs - onehot) ** 2).sum(axis=1),
                                 minlength=n_p)
        self.n_img = np.bincount(pi, minlength=n_p).astype(float)

    def decompose(self, weights: np.ndarray) -> Dict[str, np.ndarray]:
        """Murphy components for one or many participant-multiplicity vectors.

        `weights` is (n_p,) or (n_draws, n_p). Returns arrays of matching
        leading shape, so the point estimate and the whole bootstrap come out
        of the same code path and cannot drift apart.
        """
        W = np.atleast_2d(np.asarray(weights, dtype=float))
        cnt = W @ self.count                       # (d, width)
        sp = W @ self.sum_p
        so = W @ self.sum_o
        total = W @ self.n_img                     # (d,)
        brier = (W @ self.brier) / total

        nz = cnt > 0
        tot = total[:, None]
        # Reliability: (pbar - obar)^2 weighted by bin share, summed over class and bin.
        rel = np.where(nz, (sp - so) ** 2 / np.where(nz, cnt, 1.0), 0.0).sum(axis=1) / total

        d, K, B = cnt.shape[0], N_CLASSES, self.n_bins
        so3 = so.reshape(d, K, B)
        cnt3 = cnt.reshape(d, K, B)
        base = so3.sum(axis=2) / total[:, None]    # (d, K) observed class base rate
        nz3 = cnt3 > 0
        obar = np.where(nz3, so3 / np.where(nz3, cnt3, 1.0), 0.0)
        res = (np.where(nz3, cnt3 * (obar - base[:, :, None]) ** 2, 0.0).sum(axis=(1, 2))
               / total)
        unc = (base * (1.0 - base)).sum(axis=1)

        out = {"reliability": rel, "resolution": res, "uncertainty": unc,
               "brier": brier, "residual": brier - (rel - res + unc)}
        if np.ndim(weights) == 1:
            out = {k: v[0] for k, v in out.items()}
        return out


def multiplicity_matrix(n_p: int, n_boot: int, tag: str, **parts) -> np.ndarray:
    """Participant-clustered resample weights, shared by every arm in a block.

    One matrix per (cohort, unit, scaling) so that all arms and all bin counts
    see the SAME draws, which is what keeps every difference paired. The seed
    names its call site, per _determinism.py, so adding a cohort or a bin count
    does not move anyone else's interval.
    """
    rng = seeded_rng(tag, BASE_SEED, n_p=n_p, n_boot=n_boot, **parts)
    M = np.zeros((n_boot, n_p))
    for b in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        M[b] = np.bincount(idx, minlength=n_p)
    return M


def block(stats: Dict[str, MurphyStats], M: np.ndarray, component: str) -> Dict[str, Dict]:
    """The shape paired_difference expects, so its CI and p logic is reused."""
    out = {}
    ones = np.ones(M.shape[1])
    for arm, st in stats.items():
        draws = st.decompose(M)[component]
        point = float(st.decompose(ones)[component])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[arm] = {"point": point, "ci_low": float(lo), "ci_high": float(hi),
                    "boot_mean": float(draws.mean()), "boot_sd": float(draws.std(ddof=1)),
                    "_draws": draws}
    return out


def favours(d: Dict, higher_is_better: bool) -> bool:
    if not d["excludes_zero"]:
        return False
    return d["ci_low"] > 0 if higher_is_better else d["ci_high"] < 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Murphy decomposition with clustered intervals.")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--bins", type=int, nargs="+", default=list(BIN_COUNTS))
    ap.add_argument("--full-from", default=None, metavar="EXPERIMENTS_DIR")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)

    n_boot = QUICK_BOOT if args.quick else args.boot
    bins = sorted(set(int(b) for b in args.bins))
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")
    print(f"a24  Murphy decomposition, {n_boot:,} resamples, bins {bins}, base seed {BASE_SEED}")

    restricted = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     restricted: {len(restricted.arms)} arms, {restricted.n_participants} "
          f"participants, {restricted.n_images:,} images")
    full: Optional[Cohort] = None
    full_dir = find_full_tree(args.full_from)
    if full_dir is None:
        warn("no full-cohort experiments folder found; running the restricted half only.")
    else:
        try:
            full = Cohort("full", FULL_EXPERIMENTS, full_dir)
            print(f"     full:       {len(full.arms)} arms, {full.n_participants} participants, "
                  f"{full.n_images:,} images, from {full_dir}")
        except Exception as exc:                                    # noqa: BLE001
            warn(f"full cohort unusable ({type(exc).__name__}: {exc}); continuing without it.")

    cohorts = [c for c in (full, restricted) if c is not None]
    for c in cohorts:
        if c.missing_scaled:
            warn(f"{c.name}: predictions_test_temperature_scaled.npz missing for "
                 f"{', '.join(c.missing_scaled)}. The temperature-scaled half is SKIPPED for "
                 f"this cohort; the unscaled half still runs.")

    rows: List[Dict] = []
    verdicts: List[Dict] = []

    for c in cohorts:
        scalings = [("raw", c.arms)]
        if not c.missing_scaled and c.scaled:
            scalings.append(("temperature_scaled", c.scaled))
        for scaling, source in scalings:
            for unit in ("participant", "image"):
                if unit == "participant":
                    from analysis.a23_protocol_audit import participant_frame
                    frames = {a: participant_frame(f) for a, f in source.items()}
                else:
                    frames = dict(source)
                arms = ([a for a in ARM_ORDER if a in frames]
                        + [a for a in sorted(frames) if a not in ARM_ORDER])
                M = multiplicity_matrix(c.n_participants, n_boot, "a24:murphy",
                                        cohort=c.name, unit=unit, scaling=scaling)
                print(f"     {c.name:<11} {scaling:<19} {unit:<12} {len(arms)} arms ...")
                for n_bins in bins:
                    stats = {a: MurphyStats(frames[a], n_bins, c.participants) for a in arms}
                    blocks = {comp: block(stats, M, comp) for comp in COMPONENTS}

                    for a in arms:
                        row = {"cohort": c.name, "unit": unit, "scaling": scaling,
                               "n_bins": n_bins, "arm": a,
                               "code_fingerprint": c.fingerprint.get(a, ""),
                               "n_participants": c.n_participants,
                               "n_images": c.n_images}
                        for comp in COMPONENTS:
                            b = blocks[comp][a]
                            row[comp] = b["point"]
                            row[f"{comp}_ci_low"] = b["ci_low"]
                            row[f"{comp}_ci_high"] = b["ci_high"]
                        # The leftover is never zero, so a test of "residual == 0"
                        # would read False on every row and say nothing. What
                        # matters is whether the leftover is small enough that the
                        # three components still describe the score, so the flag
                        # is the leftover as a share of the Brier score.
                        row["decomposition_describes_score"] = bool(
                            abs(row["residual"]) < 0.02 * row["brier"])
                        rows.append(row)

                    # FA-FL against every other arm, on every component that is a score.
                    if FA in stats:
                        others = [a for a in arms if a != FA]
                        # Two comparators, because "the strongest" and "the closest"
                        # are different questions and a verdict that holds for one and
                        # not the other is a fact about the choice, not about FA-FL.
                        nearest = max(others,
                                      key=lambda a: blocks["resolution"][a]["point"]) \
                            if others else None
                        nearest_val = min(
                            others,
                            key=lambda a: abs(blocks["resolution"][a]["point"]
                                              - blocks["resolution"][FA]["point"])) \
                            if others else None
                        for a in others:
                            for comp in ("resolution", "reliability", "brier"):
                                d = paired_difference(blocks[comp], FA, a)
                                rows.append({
                                    "cohort": c.name, "unit": unit, "scaling": scaling,
                                    "n_bins": n_bins, "arm": f"{FA} minus {a}",
                                    "comparison_component": comp,
                                    "difference": d["diff"], "diff_ci_low": d["ci_low"],
                                    "diff_ci_high": d["ci_high"], "ci_level": d["ci_level"],
                                    "p_bootstrap": d["p_bootstrap"],
                                    "p_at_resolution_floor": d["p_at_resolution_floor"],
                                    "favours_FA_FL": favours(d, HIGHER_BETTER[comp]),
                                    "is_strongest_other_arm": bool(a == nearest),
                                    "is_nearest_in_value_arm": bool(a == nearest_val),
                                    "family": UNDECLARED,
                                    "code_fingerprint": c.fingerprint.get(a, ""),
                                    "note": "no declared registry family covers a Murphy "
                                            "component; interval is a nominal 95 percent one "
                                            "and this is not a pre-registered test"})
                        if nearest is not None:
                            d = paired_difference(blocks["resolution"], FA, nearest)
                            d2 = paired_difference(blocks["resolution"], FA, nearest_val)
                            verdicts.append({
                                "cohort": c.name, "unit": unit, "scaling": scaling,
                                "n_bins": n_bins, "strongest_other_arm": nearest,
                                "fa_fl_resolution": blocks["resolution"][FA]["point"],
                                "strongest_other_resolution": blocks["resolution"][nearest]["point"],
                                "difference": d["diff"], "ci_low": d["ci_low"],
                                "ci_high": d["ci_high"], "p_bootstrap": d["p_bootstrap"],
                                "excludes_zero_favouring_FA_FL": favours(d, True),
                                "fa_fl_rank_by_resolution": int(
                                    1 + sum(blocks["resolution"][a]["point"]
                                            > blocks["resolution"][FA]["point"]
                                            for a in arms)),
                                "nearest_in_value_arm": nearest_val,
                                "nearest_in_value_resolution":
                                    blocks["resolution"][nearest_val]["point"],
                                "difference_vs_nearest_in_value": d2["diff"],
                                "ci_low_vs_nearest_in_value": d2["ci_low"],
                                "ci_high_vs_nearest_in_value": d2["ci_high"],
                                "p_bootstrap_vs_nearest_in_value": d2["p_bootstrap"],
                                "excludes_zero_favouring_FA_FL_vs_nearest_in_value":
                                    favours(d2, True),
                                "verdict_depends_on_comparator":
                                    bool(favours(d, True) != favours(d2, True))})

    table = pd.DataFrame(rows)
    save_table(table, "t24_resolution", float_fmt="%.6f")
    vdf = pd.DataFrame(verdicts)
    vdf.to_csv(OUT / "t24_resolution_verdict.csv", index=False, float_format="%.6f")
    print(f"    -> t24_resolution_verdict.csv  ({len(vdf)} verdict rows)")

    # -- identity check ---------------------------------------------------------
    point_rows = table          # every row is one CNN arm, see the arm list above
    worst = point_rows.reindex(point_rows["residual"].abs().sort_values(ascending=False).index)
    print("\n" + "=" * 96)
    print("IDENTITY CHECK. Brier minus (Reliability - Resolution + Uncertainty).")
    print("The leftover is what the binned form drops: the within-bin spread of the predicted")
    print("probabilities, minus twice its within-bin covariance with the outcome. It takes")
    print("either sign, so a negative leftover is not an error. A LARGE one, either sign,")
    print("means the decomposition is not describing this arm's score and nothing below it")
    print("should be read.")
    print("=" * 96)
    print(f"  largest absolute leftover: {worst['residual'].abs().iloc[0]:.6f}  "
          f"({worst['cohort'].iloc[0]}/{worst['unit'].iloc[0]}/"
          f"{worst['scaling'].iloc[0]}/{worst['n_bins'].iloc[0]} bins/{worst['arm'].iloc[0]})")
    print(f"  median absolute leftover:  {point_rows['residual'].abs().median():.6f}")
    print(f"  as a share of Brier:       "
          f"{(point_rows['residual'].abs() / point_rows['brier']).max():.4%} at worst")

    # -- the verdict ------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"DOES {FA}'s RESOLUTION ADVANTAGE HOLD? One row per cohort, unit, scaling and bin "
          f"count.")
    print("=" * 96)
    if vdf.empty:
        print("  no verdict rows: FA-FL was not present in any loaded arm set.")
    else:
        panels = (
            ("PRIMARY COMPARATOR: the arm with the HIGHEST resolution other than FA-FL. "
             "The hardest available test.",
             "strongest_other_arm", "difference", "ci_low", "ci_high",
             "excludes_zero_favouring_FA_FL"),
            ("SECONDARY COMPARATOR: the arm NEAREST to FA-FL in resolution. Reported so "
             "the verdict can be read against the choice of comparator, never in place "
             "of the primary.",
             "nearest_in_value_arm", "difference_vs_nearest_in_value",
             "ci_low_vs_nearest_in_value", "ci_high_vs_nearest_in_value",
             "excludes_zero_favouring_FA_FL_vs_nearest_in_value"),
        )
        for label, arm_c, diff_c, lo_c, hi_c, ok_c in panels:
            print()
            print(f"  {label}")
            print(f"  {'cohort':<11}{'unit':<12}{'scaling':<19}{'bins':>5}{'rank':>5}  "
                  f"{'comparator':<17}{'difference':>11}  {'interval':<22} verdict")
            for _, r in vdf.iterrows():
                v = "EXCLUDES ZERO" if r[ok_c] else "no"
                interval = f"[{r[lo_c]:+.5f}, {r[hi_c]:+.5f}]"
                print(f"  {r['cohort']:<11}{r['unit']:<12}{r['scaling']:<19}{r['n_bins']:>5}"
                      f"{r['fa_fl_rank_by_resolution']:>5}  {r[arm_c]:<17}"
                      f"{r[diff_c]:>+11.5f}  {interval:<22} {v}")

        print("\n  STABILITY ACROSS BIN COUNTS")
        for (coh, unit, scaling), g in vdf.groupby(["cohort", "unit", "scaling"], sort=False):
            same_nearest = g["strongest_other_arm"].nunique() == 1
            same_rank = g["fa_fl_rank_by_resolution"].nunique() == 1
            all_excl = bool(g["excludes_zero_favouring_FA_FL"].all())
            none_excl = not bool(g["excludes_zero_favouring_FA_FL"].any())
            state = ("advantage excludes zero at EVERY bin count" if all_excl else
                     "advantage excludes zero at NO bin count" if none_excl else
                     "advantage excludes zero at SOME bin counts only, which is binning noise")
            print(f"    {coh:<11}{unit:<12}{scaling:<19} FA-FL rank stable: "
                  f"{'yes' if same_rank else ' NO'}   best other arm stable: "
                  f"{'yes' if same_nearest else ' NO'}   {state}")

        n_pass = int(vdf["excludes_zero_favouring_FA_FL"].sum())
        n_pass2 = int(vdf["excludes_zero_favouring_FA_FL_vs_nearest_in_value"].sum())
        n_split = int(vdf["verdict_depends_on_comparator"].sum())
        print(f"\n  BOTTOM LINE: against the STRONGEST other arm, {n_pass} of {len(vdf)} cells "
              f"show a resolution")
        print(f"  advantage for {FA} whose interval excludes zero. Against the NEAREST-IN-VALUE "
              f"arm, {n_pass2} of {len(vdf)}.")
        print(f"  The two comparators disagree in {n_split} of {len(vdf)} cells, so the verdict "
              f"{'DOES' if n_split else 'does not'} depend on which one is chosen.")
        if n_pass == 0 and n_pass2 == 0:
            print(f"  {FA} does not hold a resolution advantage over either comparator "
                  f"anywhere in this table.")

    print("\n  No comparison in this script belongs to a declared registry family, so none of")
    print("  them carries a corrected threshold and none may be reported as a registered test.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
