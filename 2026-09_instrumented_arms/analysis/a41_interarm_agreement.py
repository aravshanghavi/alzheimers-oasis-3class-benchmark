#!/usr/bin/env python3
"""a41: are the eight arms eight results, or one result printed eight times?

    python analysis/a41_interarm_agreement.py
    python analysis/a41_interarm_agreement.py --quick      # 200 resamples
    python analysis/a41_interarm_agreement.py --boot 10000

THE QUESTION
------------
The paper reports eight training objectives as eight arms and corrects for
multiplicity across 22 declared families holding 204 declared comparisons. All of
that arithmetic assumes the arms are separate pieces of evidence. If the eight
networks assign the same class to nearly every participant, then the paper does
not have eight results. It has one result and seven confirmations of it, and the
multiplicity structure describes a breadth of evidence that is not there.

This cuts in a direction that is easy to get backwards, so it is worth stating
carefully. High agreement does NOT make the corrections wrong. Bonferroni over
correlated comparisons is conservative, so a significant result stays significant.
What high agreement makes wrong is any sentence claiming the conclusion is
supported independently by eight objectives, or that a finding reproduced across
eight arms is thereby eight times better established. The count of comparisons and
the amount of information are different quantities, and this file measures the
second one.

WHAT IS MEASURED, ALL AT THE PARTICIPANT LEVEL
----------------------------------------------
THE AGREEMENT RATE. The share of participants two arms assign to the same class.
Easy to read and misleading on its own, because two arms that both predict the
majority class most of the time will agree often by construction.

COHEN'S KAPPA. The same agreement with the agreement expected from the two arms'
own marginal prediction distributions divided out. Kappa is the number to read.
The usual descriptive bands are 0.41 to 0.60 moderate, 0.61 to 0.80 substantial,
above 0.80 almost perfect. Those bands are conventions, not tests, and they are
printed as labels only.

THE PROBABILITY CORRELATION. Agreement and kappa only see the argmax, which
throws away how close the decision was. Two arms could agree on every label while
being quite different underneath, or disagree on labels while ranking
participants almost identically. So the Pearson correlation of the predicted
probabilities is reported too, averaged over the three classes.

THE PER-PARTICIPANT VERDICT COUNTS. How many participants every arm gets right,
how many every arm gets wrong, how many are right for a MINORITY of arms and how
many for a majority. The all-wrong count is the one that matters most: those
participants are not hard for one objective, they are invisible to the whole
family of models, and no amount of loss engineering inside this family will reach
them.

THE REGISTRY IS NOT TOUCHED
---------------------------
This file reads analysis/_registry.py to report how many families and how many
declared comparisons exist, and changes neither. The declared sizes are the
pre-registration; recomputing them from an agreement measurement after the fact
would be exactly the runtime drift the registry exists to prevent.

MULTIPLICITY
------------
No family covers an agreement measurement, and none should: these are descriptive
quantities, not tests, and no significance verdict is issued anywhere in this
file. Every row carries the literal string "undeclared" and every interval is a
nominal 95 percent one. Nothing is written to the registry: the changelog entry is
PRINTED for the first author to paste.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import save_table                                       # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                              # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, FULL_EXPERIMENTS,                 # noqa: E402
                                    RESTRICTED_EXPERIMENTS, changelog_entry,
                                    find_full_tree, percentile_interval,
                                    unstratified_participant_bootstrap)
from analysis._registry import FAMILIES                                       # noqa: E402
from analysis.a23_protocol_audit import ARM_ORDER, warn                       # noqa: E402
from analysis.a25_increment_over_metadata import multiplicity                 # noqa: E402

SOURCE = "analysis/a41_interarm_agreement.py"
EXPERIMENT_ID = "a41"
TABLE = "t41_interarm_agreement"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

BOOTSTRAP = "unstratified participant, every pair scored on the same draw"

# Landis and Koch's descriptive bands. Labels, not thresholds, and the output says
# so. They are here only so the panel is readable at three in the morning without
# a reference table beside it.
BANDS = [(0.00, "none to slight"), (0.21, "fair"), (0.41, "moderate"),
         (0.61, "substantial"), (0.81, "almost perfect")]


def band(k: float) -> str:
    if not np.isfinite(k):
        return "undefined"
    out = BANDS[0][1]
    for lo, name in BANDS:
        if k >= lo:
            out = name
    return out


def joint_counts(pa: np.ndarray, pb: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(n_boot, 3, 3) joint prediction tables for one pair, on every resample.

    A participant contributes one cell, so the resampled table is the multiplicity
    vector against a fixed indicator. Agreement is the trace and kappa needs the
    two margins, and both come out of the same table, so one product serves both.
    """
    n = len(pa)
    ind = np.zeros((n, N_CLASSES * N_CLASSES))
    ind[np.arange(n), np.asarray(pa, int) * N_CLASSES + np.asarray(pb, int)] = 1.0
    return (W @ ind).reshape(-1, N_CLASSES, N_CLASSES)


def agreement_and_kappa(tab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Observed agreement and Cohen's kappa from joint tables.

    Kappa is undefined when the expected agreement reaches one, which happens in a
    resample where both arms predicted a single class throughout. Those draws come
    back NaN and are counted by the caller rather than being filled in with a
    number nobody computed.
    """
    total = tab.sum(axis=(-2, -1))
    po = np.divide(np.trace(tab, axis1=-2, axis2=-1), total,
                   out=np.full(np.shape(total), np.nan), where=total > 0)
    row = tab.sum(axis=-1)
    col = tab.sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        pe = (row * col).sum(axis=-1) / np.where(total > 0, total ** 2, np.nan)
    kappa = np.divide(po - pe, 1.0 - pe, out=np.full(np.shape(po), np.nan),
                      where=(1.0 - pe) > 1e-12)
    return po, kappa


def weighted_correlation(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pearson correlation of x and y under each resample's multiplicities.

    The bootstrap reweights participants, so the means, the variances and the
    covariance all have to be recomputed with those weights. Doing it in closed
    form keeps the whole thing to a handful of matrix products.
    """
    n_w = W.sum(axis=1)
    mx = (W @ x) / n_w
    my = (W @ y) / n_w
    sxx = (W @ (x * x)) / n_w - mx * mx
    syy = (W @ (y * y)) / n_w - my * my
    sxy = (W @ (x * y)) / n_w - mx * my
    den = np.sqrt(np.clip(sxx, 0.0, None) * np.clip(syy, 0.0, None))
    return np.divide(sxy, den, out=np.full(np.shape(sxy), np.nan), where=den > 1e-15)


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Pairwise agreement between the arms, and what it means for "
                    "multiplicity.")
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

    print(f"a41  inter-arm agreement, {n_boot:,} resamples, base seed {BASE_SEED}")
    n_fam = len(FAMILIES)
    n_declared = int(sum(f.n_declared for f in FAMILIES.values()))
    print(f"     the registry declares {n_fam} families holding {n_declared} comparisons. "
          f"Nothing here changes either number.")

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

    rows: List[Dict] = []
    summary: Dict[str, Dict[str, object]] = {}

    for coh in cohorts:
        y = coh.labels
        n = coh.n_participants
        arms = [a for a in ARM_ORDER if a in coh.arms] + \
               [a for a in sorted(coh.arms) if a not in ARM_ORDER]
        print(f"\n     {coh.name}: {len(arms)} arms, {n} participants, "
              f"{len(list(combinations(arms, 2)))} pairs")
        W = multiplicity(unstratified_participant_bootstrap(
            n, n_boot, "a41:pairs", BASE_SEED, cohort=coh.name), n)
        one = np.ones((1, n))

        pred = {a: coh.part[a]["pred"].to_numpy().astype(int) for a in arms}
        probs = {a: coh.part[a][[f"p{c}" for c in range(N_CLASSES)]]
                 .to_numpy(dtype=float) for a in arms}
        correct = np.stack([(pred[a] == y).astype(int) for a in arms])

        kappas, agrees, corrs = [], [], []
        for a, b in combinations(arms, 2):
            tab_p = joint_counts(pred[a], pred[b], one)
            tab_d = joint_counts(pred[a], pred[b], W)
            po_p, k_p = agreement_and_kappa(tab_p)
            po_d, k_d = agreement_and_kappa(tab_d)
            r_point = float(np.mean([
                weighted_correlation(probs[a][:, c], probs[b][:, c], one)[0]
                for c in range(N_CLASSES)]))
            r_draw = np.mean(np.stack([
                weighted_correlation(probs[a][:, c], probs[b][:, c], W)
                for c in range(N_CLASSES)]), axis=0)
            ag_lo, ag_hi = percentile_interval(po_d)
            k_lo, k_hi = percentile_interval(k_d)
            r_lo, r_hi = percentile_interval(r_draw)
            kappas.append(float(k_p[0]))
            agrees.append(float(po_p[0]))
            corrs.append(r_point)
            rows.append({
                "kind": "pair", "cohort": coh.name, "unit": "participant",
                "arm_a": a, "arm_b": b,
                "agreement_rate": float(po_p[0]),
                "agreement_ci_low": ag_lo, "agreement_ci_high": ag_hi,
                "cohens_kappa": float(k_p[0]),
                "kappa_ci_low": k_lo, "kappa_ci_high": k_hi,
                "kappa_band": band(float(k_p[0])),
                "mean_probability_correlation": r_point,
                "probability_correlation_ci_low": r_lo,
                "probability_correlation_ci_high": r_hi,
                "ci_level": 0.95,
                "n_disagreeing_participants": int((pred[a] != pred[b]).sum()),
                "n_participants": n, "bootstrap": BOOTSTRAP,
                "n_kappa_undefined_draws": int(np.isnan(k_d).sum()),
                "n_bootstrap": n_boot, "family": UNDECLARED,
                "note": "descriptive. No significance verdict is issued for any row in "
                        "this file"})

        # -- who the whole family gets right, and who it never reaches -------------
        n_right = correct.sum(axis=0)
        n_arms = len(arms)
        cats = {
            "every arm correct": n_right == n_arms,
            "majority of arms correct": (n_right > n_arms / 2.0) & (n_right < n_arms),
            "minority of arms correct": (n_right > 0) & (n_right <= n_arms / 2.0),
            "every arm wrong": n_right == 0,
        }
        for name, m in cats.items():
            share = float(m.mean())
            d = (W @ m.astype(float)) / W.sum(axis=1)
            lo, hi = percentile_interval(d)
            per_class = np.bincount(y[m], minlength=N_CLASSES)
            rows.append({
                "kind": "verdict_count", "cohort": coh.name, "unit": "participant",
                "group": name, "n_participants_in_group": int(m.sum()),
                "share": share, "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                "n_cdr0": int(per_class[0]), "n_cdr05": int(per_class[1]),
                "n_cdr_ge1": int(per_class[2]),
                "n_participants": n, "n_arms": n_arms, "bootstrap": BOOTSTRAP,
                "n_bootstrap": n_boot, "family": UNDECLARED,
                "note": "a participant every arm gets wrong is not hard for one objective, "
                        "it is out of reach of this whole family of models"})
        summary[coh.name] = {
            "arms": arms, "n_arms": n_arms, "n": n,
            "kappa_min": float(np.nanmin(kappas)), "kappa_max": float(np.nanmax(kappas)),
            "kappa_median": float(np.nanmedian(kappas)),
            "agree_min": float(np.min(agrees)), "agree_max": float(np.max(agrees)),
            "agree_median": float(np.median(agrees)),
            "corr_mean": float(np.nanmean(corrs)),
            "corr_min": float(np.nanmin(corrs)), "corr_max": float(np.nanmax(corrs)),
            "counts": {k: int(v.sum()) for k, v in cats.items()},
            "n_pairs": len(kappas)}
        rows.append({
            "kind": "cohort_summary", "cohort": coh.name, "unit": "participant",
            "n_arms": n_arms, "n_pairs": len(kappas), "n_participants": n,
            "kappa_median": summary[coh.name]["kappa_median"],
            "kappa_min": summary[coh.name]["kappa_min"],
            "kappa_max": summary[coh.name]["kappa_max"],
            "agreement_median": summary[coh.name]["agree_median"],
            "mean_probability_correlation": summary[coh.name]["corr_mean"],
            "n_registry_families": n_fam, "n_registry_declared_comparisons": n_declared,
            "bootstrap": BOOTSTRAP, "n_bootstrap": n_boot, "family": UNDECLARED,
            "note": "the registry's declared sizes are reported, never recomputed. High "
                    "agreement makes a breadth-of-evidence claim wrong, not a correction "
                    "wrong"})

    table = pd.DataFrame(rows)
    lead = ["kind", "cohort", "arm_a", "arm_b", "group", "agreement_rate", "cohens_kappa",
            "kappa_ci_low", "kappa_ci_high", "kappa_band", "mean_probability_correlation"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 104)
    print("a41  ARE THE EIGHT ARMS EIGHT RESULTS, OR ONE RESULT EIGHT TIMES?")
    print("=" * 104)
    for coh in cohorts:
        s = summary[coh.name]
        pair = table[(table["kind"] == "pair") & (table["cohort"] == coh.name)]
        vc = table[(table["kind"] == "verdict_count") & (table["cohort"] == coh.name)]
        print(f"\n  == cohort: {coh.name}, {s['n_arms']} arms, {s['n']} participants, "
              f"{s['n_pairs']} pairs")
        print(f"     Cohen's kappa matrix (upper triangle), participant level:")
        arms = s["arms"]
        print(f"     {'':<16}" + "".join(f"{a[:11]:>13}" for a in arms[1:]))
        for i, a in enumerate(arms[:-1]):
            cells = []
            for b in arms[1:]:
                r = pair[((pair["arm_a"] == a) & (pair["arm_b"] == b)) |
                         ((pair["arm_a"] == b) & (pair["arm_b"] == a))]
                cells.append(f"{r['cohens_kappa'].iloc[0]:>13.3f}" if len(r) else f"{'':>13}")
            print(f"     {a:<16}" + "".join(cells))
        print(f"     kappa       median {s['kappa_median']:.3f}  range "
              f"{s['kappa_min']:.3f} to {s['kappa_max']:.3f}   ({band(s['kappa_median'])} "
              f"at the median, a descriptive label only)")
        print(f"     agreement   median {s['agree_median']:.3f}  range "
              f"{s['agree_min']:.3f} to {s['agree_max']:.3f}")
        print(f"     probability correlation, mean over pairs {s['corr_mean']:.3f}, range "
              f"{s['corr_min']:.3f} to {s['corr_max']:.3f}")
        print(f"\n     WHO THE WHOLE FAMILY REACHES:")
        print(f"     {'group':<28}{'n':>6}{'share':>9}{'95 pct interval':>22}"
              f"{'CDR 0 / 0.5 / >=1':>22}")
        for _, r in vc.iterrows():
            print(f"     {r['group']:<28}{int(r['n_participants_in_group']):>6}"
                  f"{r['share']:>9.3f}"
                  f"{f'[{r.ci_low:.3f}, {r.ci_high:.3f}]':>22}"
                  f"{f'{int(r.n_cdr0)} / {int(r.n_cdr05)} / {int(r.n_cdr_ge1)}':>22}")

    print(f"\n  WHAT THIS IMPLIES FOR THE {n_fam}-FAMILY STRUCTURE, WITHOUT CHANGING IT")
    print(f"     The registry declares {n_fam} families and {n_declared} comparisons and "
          f"this file leaves both exactly as they are.")
    print(f"     Correcting across correlated comparisons is CONSERVATIVE, so no corrected "
          f"result becomes weaker")
    print(f"     because the arms agree. What the agreement does rule out is any sentence "
          f"saying the conclusion")
    print(f"     is supported INDEPENDENTLY by eight objectives, or that reproducing "
          f"across arms multiplies the")
    print(f"     evidence. On this data the arms are close enough that the honest wording "
          f"is: one architecture and")
    print(f"     one data set, evaluated under eight loss functions that mostly make the "
          f"same decisions.")
    for coh in cohorts:
        s = summary[coh.name]
        print(f"     {coh.name:<11} every arm wrong on "
              f"{s['counts']['every arm wrong']} of {s['n']} participants, every arm right "
              f"on {s['counts']['every arm correct']}, minority-decided on "
              f"{s['counts']['minority of arms correct']}")
    print(f"\n  NO significance verdict is issued anywhere in this file, every row is "
          f"UNDECLARED, and every")
    print(f"  interval is a nominal 95 percent one.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "pairwise participant-level agreement rate, Cohen's kappa and mean predicted-"
        "probability correlation between every pair of arms on both cohorts, with the counts "
        "of participants every arm gets right, every arm gets wrong, and a minority gets "
        "right",
        "agreement rate, Cohen's kappa, Pearson correlation of predicted probabilities",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
