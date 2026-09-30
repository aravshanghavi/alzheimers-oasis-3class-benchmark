#!/usr/bin/env python3
"""a30: does the one surviving FA-FL calibration result depend on the estimator?

    python analysis/a30_ece_estimator_robustness.py
    python analysis/a30_ece_estimator_robustness.py --quick      # 200 resamples
    python analysis/a30_ece_estimator_robustness.py --boot 10000

THE QUESTION
------------
Exactly one registered calibration comparison in this paper clears its
family-corrected threshold: FA-FL against focal loss at gamma 1.37 on the
restricted cohort, p = 0.009 against an alpha of 0.0125 (a20, t20_comparisons.csv).
Every other ECE comparison in the gamma-control family misses.

That one result is computed with the POOLED estimator: the five test folds are
concatenated and the confidence bins are built once over the whole cohort. This
project has already shown that pooling inflates FA-FL's apparent calibration
advantage by a factor of 2.09, because pooling lets fold-to-fold differences in
mean confidence masquerade as within-bin calibration error, and FA-FL's folds
happen to agree with each other more closely than the focal arms' do.

So the question this script settles is narrow and answerable: recompute the same
registered comparisons with an estimator that never pools folds, and see whether
the verdict survives.

THE TWO ESTIMATORS
------------------
POOLED          concatenate the five test folds, bin once, one ECE.
MEAN OF FOLDS   compute ECE inside each test fold separately, then average the
                five values with equal weight. A fold-level difference in mean
                confidence cannot contribute, because each fold is calibrated
                against its own bins and its own accuracy.

Both are computed at 10, 15 and 20 equal-width bins and at equal-mass
(quantile) bins, because an ECE claim that moves with the bin count was the bin
count. Equal-mass edges are fixed at the observed sample's quantiles and are NOT
recomputed inside each bootstrap draw: re-estimating the edges per draw would
make the estimator itself a random function of the resample, and the quantity
under test would stop being the quantity that was reported.

THE BOOTSTRAP
-------------
Participants are resampled WITHIN their own test fold, so every fold keeps its
observed size. That is what makes a mean-of-folds estimator well defined on a
resample: an unrestricted draw can leave a fold with two participants, and the
average of five fold ECEs one of which rests on two people is not the estimator
anybody reported. Every arm is scored on the SAME draw, so differences stay
paired, and the point estimate and the bootstrap come out of one code path.

The reference pooled estimator is ALSO run under an ordinary unstratified
participant bootstrap, so that a change of verdict can be attributed to the
estimator rather than to the resampling scheme. If the verdict moved because of
the scheme, that row shows it.

exp07 AND exp08
---------------
a11_ece_robustness.py globs `exp0[12]*/outputs/*/manifest.json`. That pattern
matches exp01_main_sweep and exp02_class_balanced and nothing else, so the two
focal gamma-control arms, exp07_focal_gamma137 and exp08_focal_gamma200, have
never appeared in a11's binning-robustness table. This script verifies that glob
against the tree it is actually reading and prints what it found. Both arms are
included here through Cohort, which loads FULL_EXPERIMENTS.

MULTIPLICITY
------------
Every comparison recomputed here belongs to a family declared in
analysis/_registry.py, and the interval comes from paired_difference(family=...)
at that family's corrected level. Nothing is written to the registry: the
changelog entry is PRINTED for the first author to paste.

The declared families were realised at the IMAGE unit (a20's constant predictor
sits at 51.8349 percent, the image-level majority share, not the participant
level 51.2048), except objective_ece_participant which was declared at the
participant unit. A comparison computed at the other unit would add to a family
size the declaration fixes, so it carries the literal string "undeclared" and a
nominal 95 percent interval instead.

ONE KNOWN REGISTRY OVERLAP, REPORTED NOT FIXED
----------------------------------------------
On the full cohort, objective_ece declares six comparisons, one per non-FA-FL
objective, and three of those objectives are the focal arms that gamma_control_ece
also declares. The same comparison therefore sits in two families, which is the
thing the age60 declarations were corrected to avoid. This script does not change
either declaration. It computes both as declared and prints the overlap, because
silently dropping a declared comparison is a worse fault than reporting a
declaration that needs amending.
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

from analysis._common import paired_difference, save_table                     # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                               # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, FULL_EXPERIMENTS,                  # noqa: E402
                                    RESTRICTED_EXPERIMENTS, changelog_entry,
                                    find_full_tree, percentile_interval,
                                    stratified_participant_bootstrap,
                                    unstratified_participant_bootstrap)
from analysis._registry import REGISTRY                                        # noqa: E402
from analysis.a23_protocol_audit import ARM_ORDER, FA, warn                    # noqa: E402
from analysis.a25_increment_over_metadata import multiplicity                  # noqa: E402

SOURCE = "analysis/a30_ece_estimator_robustness.py"
EXPERIMENT_ID = "a30"
TABLE = "t30_ece_estimator_robustness"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
UNDECLARED = "undeclared"

EQUAL_WIDTH = "equal width"
EQUAL_MASS = "equal mass"
POOLED = "pooled"
MEAN_OF_FOLDS = "mean of folds"

FOLD_CLUSTERED = "fold clustered, participants resampled within fold"
PLAIN_PARTICIPANT = "participant, unstratified"

# (estimator, binning, n_bins). The reference is the estimator the paper used:
# pooled, equal width, 15 bins, which is _common.ECE_BINS.
REFERENCE = (POOLED, EQUAL_WIDTH, 15)
ESTIMATORS: List[Tuple[str, str, int]] = [
    (POOLED, EQUAL_WIDTH, 10), (POOLED, EQUAL_WIDTH, 15), (POOLED, EQUAL_WIDTH, 20),
    (POOLED, EQUAL_MASS, 15),
    (MEAN_OF_FOLDS, EQUAL_WIDTH, 10), (MEAN_OF_FOLDS, EQUAL_WIDTH, 15),
    (MEAN_OF_FOLDS, EQUAL_WIDTH, 20), (MEAN_OF_FOLDS, EQUAL_MASS, 15),
]

UNITS = ["image", "participant"]

# Which declared family owns a comparison, keyed by (cohort, unit, kind).
# "objective" is every non-FA-FL arm against FA-FL. "gamma" is FA-FL against the
# focal ladder. A cell that is absent gets UNDECLARED and a nominal interval.
FAMILY_OF = {
    ("full", "image", "objective"): "objective_ece",
    ("full", "participant", "objective"): "objective_ece_participant",
    ("full", "image", "gamma"): "gamma_control_ece",
    ("restricted", "image", "objective"): "age60_objective_ece",
    ("restricted", "image", "gamma"): "age60_gamma_control_ece",
}

# Which arms each declared family covers, as the declaration words it.
GAMMA_ARMS = {"full": ["Focal g=1.37", "Focal g=2.00", "Focal g=3.00"],
              "restricted": ["Focal g=1.37", "Focal g=2.00", "Focal g=2.69",
                             "Focal g=3.00"]}
# age60_objective_ece declares three comparisons and its note says every focal
# comparison belongs to the gamma family and to that family only. objective_ece
# on the full cohort declares six, which is every non-FA-FL arm.
OBJECTIVE_ARMS = {"full": None,                       # None means every non-FA arm
                  "restricted": ["Weighted CE", "LDAM", "Class-Balanced"]}


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def edges_for(conf: np.ndarray, binning: str, n_bins: int) -> np.ndarray:
    """Bin edges, fixed once from the observed sample.

    Equal-mass edges collapse when many confidences tie, which they do here: a
    network that reports the same top probability for thousands of near-duplicate
    slices puts them all on one quantile. np.unique drops the duplicates, so the
    realised bin count can be below n_bins, and the realised count travels with
    every row rather than being assumed.
    """
    if binning == EQUAL_WIDTH:
        return np.linspace(0.0, 1.0, n_bins + 1)
    e = np.unique(np.quantile(np.asarray(conf, dtype=float),
                              np.linspace(0.0, 1.0, n_bins + 1)))
    if len(e) < 2:
        return np.linspace(0.0, 1.0, n_bins + 1)
    e[0], e[-1] = 0.0, 1.0
    return e


def blocks(conf: np.ndarray, correct: np.ndarray, pidx: np.ndarray, n_p: int,
           edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per participant, per bin: count, summed confidence, summed correctness.

    ECE is a function of these three sums alone, and they are ADDITIVE over
    participants, so a participant-clustered resample is a matrix product against
    them rather than a rebinning of 86,437 rows. This is the SuffStats idea from
    _common.py, kept here because a30 needs the bins to change and SuffStats fixes
    them at construction.
    """
    n_bins = len(edges) - 1
    b = np.clip(np.digitize(np.asarray(conf, dtype=float), edges[1:-1], right=True),
                0, n_bins - 1)
    flat = np.asarray(pidx, dtype=np.int64) * n_bins + b
    size = n_p * n_bins
    cnt = np.bincount(flat, minlength=size).reshape(n_p, n_bins).astype(np.float64)
    sc = np.bincount(flat, weights=np.asarray(conf, dtype=float),
                     minlength=size).reshape(n_p, n_bins)
    so = np.bincount(flat, weights=np.asarray(correct, dtype=float),
                     minlength=size).reshape(n_p, n_bins)
    return cnt, sc, so


def ece_from_blocks(W: np.ndarray, cnt: np.ndarray, sc: np.ndarray,
                    so: np.ndarray) -> np.ndarray:
    """Top-label ECE for every row of the multiplicity matrix W at once."""
    n = W @ cnt
    c = W @ sc
    k = W @ so
    total = n.sum(axis=1)
    acc = np.divide(k, n, out=np.zeros_like(n), where=n > 0)
    con = np.divide(c, n, out=np.zeros_like(n), where=n > 0)
    gap = np.abs(acc - con)
    num = (n * gap).sum(axis=1)
    return np.divide(num, total, out=np.zeros_like(num), where=total > 0)


class ArmEce:
    """One arm, one cohort, one unit: everything needed to evaluate both estimators.

    Built once per (arm, unit) and reused across all eight estimators, because
    the confidences and the correctness flags do not depend on the binning and
    recomputing them eight times is eight times the reading for no new number.
    """

    def __init__(self, frame: pd.DataFrame, participants: np.ndarray,
                 folds: np.ndarray, n_classes: int = 3):
        index = {p: i for i, p in enumerate(participants)}
        pid = frame["participant_id"].to_numpy()
        self.pidx = np.fromiter((index[p] for p in pid), dtype=np.int64, count=len(pid))
        probs = frame[[f"p{c}" for c in range(n_classes)]].to_numpy(dtype=np.float64)
        y = frame["label"].to_numpy().astype(int)
        self.conf = probs.max(axis=1)
        self.correct = (probs.argmax(axis=1) == y).astype(np.float64)
        self.n_p = len(participants)
        self.folds = np.asarray(folds, dtype=int)
        self.fold_values = np.unique(self.folds)
        # Row masks per fold, so the mean-of-folds estimator never has to scan.
        self.fold_cols = {f: np.flatnonzero(self.folds == f) for f in self.fold_values}
        self.fold_rows = {}
        self.fold_local = {}
        for f, cols in self.fold_cols.items():
            local = -np.ones(self.n_p, dtype=np.int64)
            local[cols] = np.arange(len(cols))
            rows = np.flatnonzero(np.isin(self.pidx, cols))
            self.fold_rows[f] = rows
            self.fold_local[f] = local[self.pidx[rows]]
        self._cache: Dict[Tuple[str, int], object] = {}

    def _pooled_blocks(self, binning: str, n_bins: int):
        key = ("pooled", binning, n_bins)
        if key not in self._cache:
            e = edges_for(self.conf, binning, n_bins)
            self._cache[key] = (e, blocks(self.conf, self.correct, self.pidx,
                                          self.n_p, e))
        return self._cache[key]

    def _fold_blocks(self, binning: str, n_bins: int):
        key = ("folds", binning, n_bins)
        if key not in self._cache:
            out = {}
            for f in self.fold_values:
                r = self.fold_rows[f]
                e = edges_for(self.conf[r], binning, n_bins)
                out[f] = (e, blocks(self.conf[r], self.correct[r], self.fold_local[f],
                                    len(self.fold_cols[f]), e))
            self._cache[key] = out
        return self._cache[key]

    def realised_bins(self, estimator: str, binning: str, n_bins: int) -> int:
        """Bins the estimator actually used, which equal-mass can shrink."""
        if estimator == POOLED:
            e, _ = self._pooled_blocks(binning, n_bins)
            return len(e) - 1
        fb = self._fold_blocks(binning, n_bins)
        return int(min(len(e) - 1 for e, _ in fb.values()))

    def draws(self, W: np.ndarray, estimator: str, binning: str,
              n_bins: int) -> np.ndarray:
        if estimator == POOLED:
            e, (cnt, sc, so) = self._pooled_blocks(binning, n_bins)
            return ece_from_blocks(W, cnt, sc, so)
        fb = self._fold_blocks(binning, n_bins)
        acc = np.zeros(W.shape[0])
        for f in self.fold_values:
            e, (cnt, sc, so) = fb[f]
            acc += ece_from_blocks(W[:, self.fold_cols[f]], cnt, sc, so)
        return acc / len(self.fold_values)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def comparisons_for(cohort: str, arms: Sequence[str]) -> List[Tuple[str, str, str]]:
    """(kind, arm_a, arm_b). Direction is FA-FL minus the other arm, always.

    a20 and t20_comparisons.csv print every ECE difference as FA-FL minus X.
    Flipping the sign for the objective family because its description reads
    "each objective against FA-FL" would give this project two sign conventions
    for one quantity, which is how a table and a figure end up disagreeing.
    """
    out: List[Tuple[str, str, str]] = []
    obj = OBJECTIVE_ARMS[cohort]
    objective_arms = [a for a in arms if a != FA] if obj is None else \
        [a for a in obj if a in arms]
    for a in objective_arms:
        out.append(("objective", FA, a))
    for a in GAMMA_ARMS[cohort]:
        if a in arms:
            out.append(("gamma", FA, a))
    return out


def block_of(draws: Dict[str, np.ndarray], points: Dict[str, float]) -> Dict[str, Dict]:
    """The shape paired_difference expects, so its CI and p logic is reused."""
    out = {}
    for arm, d in draws.items():
        lo, hi = percentile_interval(d)
        out[arm] = {"point": float(points[arm]), "ci_low": lo, "ci_high": hi,
                    "boot_mean": float(np.mean(d)), "boot_sd": float(np.std(d, ddof=1)),
                    "_draws": d}
    return out


def verdict_label(sig_here: bool, sig_reference: bool) -> str:
    """One word a person can read at three in the morning."""
    if sig_reference and sig_here:
        return "HOLDS"
    if sig_reference and not sig_here:
        return "FALLS"
    if not sig_reference and sig_here:
        return "APPEARS"
    return "NULL BOTH"


def check_a11_glob(tree: Path) -> Tuple[List[str], List[str]]:
    """What a11's own pattern matches under this tree, and what it leaves out."""
    matched = sorted({p.parts[-4] for p in tree.glob("exp0[12]*/outputs/*/manifest.json")})
    present = sorted({p.parts[-4] for p in tree.glob("*/outputs/*/manifest.json")})
    return matched, [e for e in present if e not in matched]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="ECE estimator and bin-count robustness.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"paired bootstrap resamples (default {N_BOOT})")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--full-from", default=None, metavar="EXPERIMENTS_DIR",
                    help="experiments folder holding the full-cohort runs")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"a30  ECE estimator robustness, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    restricted = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     restricted: {len(restricted.arms)} arms, {restricted.n_participants} "
          f"participants, {restricted.n_images:,} images")

    full_dir = find_full_tree(args.full_from)
    full: Optional[Cohort] = None
    if full_dir is None:
        warn("no full-cohort experiments folder found. Only the restricted families run. "
             "Pass --full-from to point at one.")
    else:
        full = Cohort("full", FULL_EXPERIMENTS, full_dir)
        print(f"     full:       {len(full.arms)} arms, {full.n_participants} participants, "
              f"{full.n_images:,} images, from {full_dir}")

    cohorts = [c for c in (full, restricted) if c is not None]

    # -- the a11 glob, verified rather than asserted ----------------------------
    print("\n     a11_ece_robustness.py globs exp0[12]*/outputs/*/manifest.json. "
          "What that matches:")
    glob_rows = []
    for label, tree in [("full tree", full_dir), ("EXPERIMENTS_DIR", EXPERIMENTS_DIR)]:
        if tree is None:
            continue
        matched, missed = check_a11_glob(Path(tree))
        print(f"       {label:<16} matched {matched}")
        print(f"       {label:<16} EXCLUDED {missed}")
        glob_rows.append({"kind": "a11_glob_check", "tree_label": label, "tree": str(tree),
                          "a11_glob_matched": "; ".join(matched),
                          "a11_glob_excluded": "; ".join(missed),
                          "exp07_excluded_by_a11": "exp07_focal_gamma137" in missed,
                          "exp08_excluded_by_a11": "exp08_focal_gamma200" in missed,
                          "family": UNDECLARED,
                          "note": "a30 loads FULL_EXPERIMENTS through Cohort, so exp07 and "
                                  "exp08 are included here"})

    # -- bootstrap draw matrices -------------------------------------------------
    rows: List[Dict] = list(glob_rows)
    verdict_rows: List[Dict] = []

    for coh in cohorts:
        arms = coh.order()
        idx_fold = stratified_participant_bootstrap(coh.folds, n_boot, "a30:foldclustered",
                                                    BASE_SEED, cohort=coh.name)
        idx_plain = unstratified_participant_bootstrap(coh.n_participants, n_boot,
                                                       "a30:plain", BASE_SEED,
                                                       cohort=coh.name)
        # stratified_participant_bootstrap lays each stratum out contiguously in the
        # order np.unique gives, so the column order of the returned index matrix is
        # NOT participant order. multiplicity() counts per participant, which is
        # order free, so the two schemes stay comparable.
        W_fold = multiplicity(idx_fold, coh.n_participants)
        W_plain = multiplicity(idx_plain, coh.n_participants)
        ones = np.ones((1, coh.n_participants))

        for unit in UNITS:
            frames = coh.unit(unit)
            print(f"\n     {coh.name}/{unit}: {len(arms)} arms, "
                  f"{len(ESTIMATORS)} estimators ...")
            armece = {a: ArmEce(frames[a], coh.participants, coh.folds) for a in arms}

            schemes = [(FOLD_CLUSTERED, W_fold)] + \
                      [(PLAIN_PARTICIPANT, W_plain)]
            for estimator, binning, n_bins in ESTIMATORS:
                for scheme, W in schemes:
                    # The plain scheme is run for the reference estimator only. Its
                    # job is to show that a changed verdict came from the estimator
                    # and not from the resampling, and running it everywhere would
                    # double a table nobody would read twice.
                    if scheme == PLAIN_PARTICIPANT and \
                            (estimator, binning, n_bins) != REFERENCE:
                        continue
                    draws, points, realised = {}, {}, {}
                    for a in arms:
                        draws[a] = armece[a].draws(W, estimator, binning, n_bins)
                        points[a] = float(armece[a].draws(ones, estimator, binning,
                                                          n_bins)[0])
                        realised[a] = armece[a].realised_bins(estimator, binning, n_bins)
                    boot = block_of(draws, points)

                    for a in arms:
                        rows.append({
                            "kind": "arm_ece", "cohort": coh.name, "unit": unit,
                            "arm": a, "estimator": estimator, "binning": binning,
                            "n_bins_requested": n_bins,
                            "n_bins_realised": realised[a],
                            "bootstrap": scheme, "n_bootstrap": n_boot,
                            "ece": boot[a]["point"], "ci_low": boot[a]["ci_low"],
                            "ci_high": boot[a]["ci_high"], "ci_level": 0.95,
                            "family": UNDECLARED,
                            "n_participants": coh.n_participants,
                            "n_images": coh.n_images,
                            "code_fingerprint": coh.fingerprint.get(a, ""),
                            "note": "point ECE with a nominal 95 percent interval; the "
                                    "corrected intervals live on the comparison rows"})

                    for kind, a, b in comparisons_for(coh.name, arms):
                        fam = FAMILY_OF.get((coh.name, unit, kind))
                        d = paired_difference(boot, a, b, family=fam)
                        alpha = (REGISTRY.alpha_corrected(fam) if fam else 0.05)
                        sig = bool(d["p_bootstrap"] < alpha and d["excludes_zero"])
                        rows.append({
                            "kind": "comparison", "cohort": coh.name, "unit": unit,
                            "comparison_kind": kind,
                            "comparison": f"{a} minus {b}", "arm": a, "reference": b,
                            "estimator": estimator, "binning": binning,
                            "n_bins_requested": n_bins,
                            "n_bins_realised": min(realised[a], realised[b]),
                            "bootstrap": scheme, "n_bootstrap": n_boot,
                            "ece": boot[a]["point"], "reference_ece": boot[b]["point"],
                            "difference": d["diff"], "ci_low": d["ci_low"],
                            "ci_high": d["ci_high"], "ci_level": d["ci_level"],
                            "p_bootstrap": d["p_bootstrap"],
                            "p_at_resolution_floor": d["p_at_resolution_floor"],
                            "family": fam or UNDECLARED,
                            "alpha_used": alpha,
                            "significant_at_alpha": sig,
                            "excludes_zero": d["excludes_zero"],
                            "favours_FA_FL": bool(d["excludes_zero"] and d["ci_high"] < 0),
                            "n_participants": coh.n_participants,
                            "n_images": coh.n_images,
                            "code_fingerprint": coh.fingerprint.get(b, ""),
                            "note": ("declared family, interval at the corrected level"
                                     if fam else
                                     "no declared family covers this comparison at this "
                                     "unit, so the interval is a nominal 95 percent one "
                                     "and this is not a pre-registered test")})

    table = pd.DataFrame(rows)

    # -- verdict grid, every registered comparison against the reference estimator
    comp = table[table["kind"] == "comparison"].copy()
    ref = comp[(comp["estimator"] == REFERENCE[0]) & (comp["binning"] == REFERENCE[1]) &
               (comp["n_bins_requested"] == REFERENCE[2]) &
               (comp["bootstrap"] == FOLD_CLUSTERED)]
    # The family belongs in this key. On the full cohort one comparison sits in two
    # declared families at two different corrected thresholds, and a key without the
    # family would hand both of them whichever reference row happened to be written
    # last, which turns a threshold difference into a phantom change of verdict.
    ref_sig = {(r["cohort"], r["unit"], r["comparison"], r["family"]):
               bool(r["significant_at_alpha"]) for _, r in ref.iterrows()}
    comp["reference_significant"] = [
        ref_sig.get((r["cohort"], r["unit"], r["comparison"], r["family"]), False)
        for _, r in comp.iterrows()]
    comp["verdict"] = [verdict_label(bool(r["significant_at_alpha"]),
                                     bool(r["reference_significant"]))
                       for _, r in comp.iterrows()]
    table = pd.concat([table[table["kind"] != "comparison"], comp], ignore_index=True)

    lead = ["kind", "cohort", "unit", "comparison", "arm", "reference", "estimator",
            "binning", "n_bins_requested", "n_bins_realised", "bootstrap", "verdict",
            "significant_at_alpha", "difference", "ci_low", "ci_high", "ci_level",
            "p_bootstrap", "alpha_used", "family"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen --------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 100)
    print("a30  DOES THE ONE SIGNIFICANT FA-FL CALIBRATION RESULT SURVIVE A "
          "NON-POOLING ESTIMATOR?")
    print("=" * 100)
    print("  Sign convention: NEGATIVE means FA-FL has the LOWER ECE, which is better "
          "for FA-FL.")
    print("  HOLDS / FALLS is measured against the reference estimator, pooled, equal "
          "width, 15 bins,")
    print("  under the same fold-clustered bootstrap. APPEARS means significant here and "
          "not there.")

    declared = comp[comp["family"] != UNDECLARED]
    for (coh_name, unit), g in declared.groupby(["cohort", "unit"], sort=False):
        print(f"\n  cohort {coh_name}, unit {unit}")
        print(f"    {'comparison':<32}{'family':<26}{'alpha':>9}  "
              f"{'estimator':<16}{'bins':>10}{'diff':>10}{'p':>9}  verdict")
        # Grouped by (comparison, family), not by comparison alone: on the full
        # cohort the focal arms sit in two declared families at once, with two
        # different corrected thresholds, and collapsing them into one block would
        # print each row twice with no way to tell which family it belonged to.
        for (cname, fname), gg in g.groupby(["comparison", "family"], sort=False):
            first = True
            for _, r in gg.iterrows():
                lab = cname if first else ""
                famlab = fname if first else ""
                alab = f"{r['alpha_used']:.6g}" if first else ""
                first = False
                tag = "EW" if r["binning"] == EQUAL_WIDTH else "EM"
                bins = f"{tag} {int(r['n_bins_realised'])}"
                est = r["estimator"] if r["bootstrap"] == FOLD_CLUSTERED \
                    else r["estimator"] + " *"
                print(f"    {lab:<32}{famlab:<26}{alab:>9}  {est:<16}{bins:>10}"
                      f"{r['difference']:>+10.4f}{r['p_bootstrap']:>9.4f}  {r['verdict']}")
    print("\n    * = the reference estimator re-run under an ordinary unstratified "
          "participant bootstrap,")
    print("      so a changed verdict can be attributed to the estimator rather than "
          "to the resampling.")

    n_falls = int((declared["verdict"] == "FALLS").sum())
    n_holds = int((declared["verdict"] == "HOLDS").sum())
    n_app = int((declared["verdict"] == "APPEARS").sum())
    n_null = int((declared["verdict"] == "NULL BOTH").sum())
    print(f"\n  ACROSS {len(declared)} declared comparison-by-estimator cells: "
          f"{n_holds} HOLDS, {n_falls} FALLS, {n_app} APPEARS, {n_null} NULL BOTH.")

    target = declared[(declared["cohort"] == "restricted") &
                      (declared["unit"] == "image") &
                      (declared["comparison"] == f"{FA} minus Focal g=1.37")]
    if not target.empty:
        print(f"\n  THE RESULT THIS SCRIPT EXISTS FOR: {FA} minus Focal g=1.37, restricted "
              f"cohort, image unit,")
        print(f"  family age60_gamma_control_ece, corrected alpha "
              f"{target['alpha_used'].iloc[0]:.6g}. a20 reported "
              f"p = 0.009.")
        for _, r in target.iterrows():
            mark = "*" if r["bootstrap"] == PLAIN_PARTICIPANT else " "
            print(f"    {r['estimator']:<16}{r['binning']:<12}"
                  f"{int(r['n_bins_realised']):>3} bins {mark}  "
                  f"diff {r['difference']:>+9.4f}  "
                  f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                  f"p {r['p_bootstrap']:.4f}  ->  {r['verdict']}")
        mof = target[(target["estimator"] == MEAN_OF_FOLDS) &
                     (target["binning"] == EQUAL_WIDTH)]
        if not mof.empty:
            survived = bool(mof["significant_at_alpha"].all())
            print(f"    Under MEAN OF FOLDS at every equal-width bin count, the result "
                  f"{'SURVIVES' if survived else 'DOES NOT survive'}.")

    print("\n  UNDECLARED CELLS. Every comparison computed at a unit its family was not "
          "realised at")
    print("  carries the literal string 'undeclared', a nominal 95 percent interval, and "
          "no verdict")
    print("  that may be called pre-registered. They are in the CSV, not printed here.")
    print("\n  REGISTRY OVERLAP, reported and not fixed: on the full cohort the three "
          "focal arms sit in")
    print("  both objective_ece and gamma_control_ece as those families are currently "
          "declared. The")
    print("  age60 declarations were corrected to forbid exactly that. Amending the "
          "full-cohort")
    print("  declaration is the first author's call and belongs in REGISTRY_CHANGELOG.md.")

    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "every registered ECE comparison on both cohorts recomputed with a mean-of-folds "
        "estimator under a fold-clustered bootstrap, and with pooled ECE at 10, 15 and 20 "
        "equal-width bins and at equal-mass bins, exp07 and exp08 included",
        "expected calibration error, top label",
        "image and participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
