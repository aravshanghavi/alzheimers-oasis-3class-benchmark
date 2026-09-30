#!/usr/bin/env python3
"""a25: does the network know anything MMSE or four spreadsheet columns do not?

    python analysis/a25_increment_over_metadata.py
    python analysis/a25_increment_over_metadata.py --quick      # 200 resamples
    python analysis/a25_increment_over_metadata.py --boot 10000

THE QUESTION
------------
Every imaging result in this paper is worth exactly what the answer to one
question is worth: once a clinician already has the things that cost nothing,
does the network add anything. The things that cost nothing are a bedside
cognitive score (MMSE) and four numbers that fall out of the scan header and the
standard segmentation (age, sex, eTIV, nWBV). If a model built on those alone
matches a model that also sees the network, then the network is an expensive way
of restating them.

So on the restricted cohort (166 participants, the people a clinician actually
assessed), four cross-validated models are fitted on the SAME grouped folds as
the networks:

    (a) MMSE
    (b) MMSE + stacked CNN score
    (c) age + sex + eTIV + nWBV
    (d) age + sex + eTIV + nWBV + stacked CNN score

and one more, without which the increment test cannot be falsified:

    (e) stacked CNN score alone

BOTH DIRECTIONS, ALWAYS
-----------------------
(b) minus (a) asks whether the stacked CNN score adds to MMSE. (b) minus (e)
asks whether MMSE adds to the stacked CNN score. If the first is null and the
second is large, the network's information sits inside MMSE's. If both are null,
the two carry the same thing. Reporting only the forward direction makes the
asymmetry unfalsifiable, which is the exact failure the registry's paired
increment families were written to prevent. So all four contrasts are computed
and all four are in the table.

WHAT THE STACKED MODEL IS, AND WHAT IT IS NOT
---------------------------------------------
The stacked CNN score is the pair log(p1/p0), log(p2/p0) taken from the
network's participant-level mean probabilities and entered as two predictors in
a multinomial logistic regression. It is a two number summary of what the
network said. It is NOT the network. No row of this table may be described as
the network beating or failing to beat a spreadsheet. The label printed
everywhere in the output is "stacked CNN score" for that reason.

The stacking is the ordinary out of fold arrangement, the same one a18 uses. A
held out participant's own score comes from a network that never saw that
participant. The regression's coefficients are fitted on training fold scores
which were themselves produced out of fold, so they were produced by networks
that did see the held out fold. That is standard stacked generalisation and it
is stated here rather than discovered later.

RESTRICTED COHORT ONLY, AND WHY MMSE IS NOT RUN ON THE FULL ONE
---------------------------------------------------------------
MMSE is complete on these 166 participants, which is the only reason the MMSE
half of this experiment exists. On the full 347 it is missing for 147 people,
42 percent, so a median imputed MMSE model there is mostly its own median and
would measure the imputation. The MMSE models are therefore not run on the full
cohort. That is a decision, not an omission.

WHY THE VECTORISED BOOTSTRAP LIVES IN THIS FILE
------------------------------------------------
B = 10000 paired resamples over 8 arms, 5 models and 5 metrics is roughly a
million AUC evaluations. Looping midrank_auc over them costs over an hour, and
an experiment that has to run overnight without a person watching cannot afford
that. Three primitives below turn the whole bootstrap into a handful of matrix
products, because every metric used here is a function of resample
MULTIPLICITIES against a fixed score vector. a26 imports them from here rather
than keeping a second copy: _protocol_lib is the shared module and it is not
edited by this night's work, so the alternative to one import is two copies that
drift, and this project has already paid for that once.

Each primitive is checked against the canonical midrank_auc at startup. If the
matrix form and the reference form ever disagree, the script stops.

MULTIPLICITY
------------
No family declared in analysis/_registry.py covers a metadata increment on the
restricted cohort. cnn_increment_over_nwbv and nwbv_increment_over_cnn are
declared over nWBV on the full cohort and cannot absorb these comparisons
without breaking the declared sizes they exist to hold. So every row carries the
literal string "undeclared", every interval is a nominal 95 percent interval,
nothing here is a pre-registered test, and the script PRINTS the changelog entry
for the author to paste rather than editing the registry itself.
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

from analysis._common import save_table                                       # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx               # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, MetaFitter, RESTRICTED_EXPERIMENTS,  # noqa: E402
                                    bootstrap_p, changelog_entry, cnn_features,
                                    midrank_auc, null_replicates,
                                    percentile_interval,
                                    stratified_participant_bootstrap,
                                    unstratified_participant_bootstrap)
from analysis.a23_protocol_audit import PERMNULL_TREE, SHORT, per_class_f1     # noqa: E402

SOURCE = "analysis/a25_increment_over_metadata.py"
EXPERIMENT_ID = "a25"
TABLE = "t25_increment_over_metadata"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

STACKED = "stacked CNN score"
CNN_COLS = ["cnn_log_ratio_1", "cnn_log_ratio_2"]

MMSE_ONLY = "MMSE"
MMSE_CNN = f"MMSE + {STACKED}"
META4 = "age+sex+eTIV+nWBV"
META4_CNN = f"age+sex+eTIV+nWBV + {STACKED}"
CNN_ONLY = f"{STACKED} alone"

# Short tags, used only in the console panel. The full names are what the CSV
# carries; a 60 character model name repeated four times per row turns the one
# screen summary into three screens, which defeats its purpose.
SHORT_TAG = {MMSE_ONLY: "(a) MMSE", MMSE_CNN: "(b) MMSE+CNN", META4: "(c) 4 columns",
             META4_CNN: "(d) 4 columns+CNN", CNN_ONLY: "(e) CNN alone"}

MODELS: List[Tuple[str, List[str], bool]] = [
    (MMSE_ONLY, ["MMSE"], False),
    (MMSE_CNN, ["MMSE"] + CNN_COLS, True),
    (META4, ["Age", "M/F", "eTIV", "nWBV"], False),
    (META4_CNN, ["Age", "M/F", "eTIV", "nWBV"] + CNN_COLS, True),
    (CNN_ONLY, list(CNN_COLS), True),
]

# Forward first, then the reverse of each. The reverses are not decoration: the
# argument the paper wants to make is an ASYMMETRY, and an asymmetry with one
# side missing is not a claim anyone can check.
CONTRASTS: List[Tuple[str, str, str]] = [
    (MMSE_CNN, MMSE_ONLY, "forward, does the stacked CNN score add to MMSE"),
    (META4_CNN, META4, "forward, does the stacked CNN score add to the four columns"),
    (MMSE_CNN, CNN_ONLY, "reverse, does MMSE add to the stacked CNN score"),
    (META4_CNN, CNN_ONLY, "reverse, do the four columns add to the stacked CNN score"),
]

# The three clinically readable separations, at the participant level.
PAIRWISE = [
    ("AUC CDR>=1 vs CDR 0", 2, 0),
    ("AUC any impairment vs CDR 0", None, 0),
    ("AUC CDR>=1 vs CDR 0.5", 2, 1),
]
AUC_METRICS = [name for name, _, _ in PAIRWISE]
CM_METRICS = ["macro F1", "accuracy percent"]
ALL_METRICS = AUC_METRICS + CM_METRICS

STRATIFIED = "stratified participant, resampled within class"
UNSTRATIFIED = "unstratified participant"
BOOTSTRAP_FOR = {**{m: STRATIFIED for m in AUC_METRICS},
                 **{m: UNSTRATIFIED for m in CM_METRICS}}


# ---------------------------------------------------------------------------
# Vectorised paired bootstrap primitives
# ---------------------------------------------------------------------------

def multiplicity(idx: np.ndarray, n: int) -> np.ndarray:
    """(n_boot, n) counts of how many times each row was drawn into each resample.

    Every metric here is a function of these counts against a FIXED score vector,
    so once the counts exist the whole bootstrap is linear algebra. bincount on a
    flattened row-offset index is used instead of np.add.at, which is roughly
    thirty times slower on a matrix this shape.
    """
    n_boot = int(idx.shape[0])
    offset = (np.arange(n_boot, dtype=np.int64) * n)[:, None]
    flat = (idx.astype(np.int64) + offset).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(np.float64)


def auc_draws(score: np.ndarray, positive: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Midrank AUC on every resample at once, from the win/tie matrix.

    AUC is the fraction of positive/negative pairs the score orders correctly,
    counting a tie as half. Resampling multiplies each pair's contribution by the
    product of its two multiplicities, so the whole vector of resampled AUCs is
    one matrix product. This is the same quantity midrank_auc returns, not an
    approximation of it, and _self_check below proves that on the real data.
    """
    s = np.asarray(score, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    sp, sn = s[pos], s[~pos]
    win = (sp[:, None] > sn[None, :]).astype(np.float64)
    win += 0.5 * (sp[:, None] == sn[None, :])
    cp, cn = mult[:, pos], mult[:, ~pos]
    denom = cp.sum(axis=1) * cn.sum(axis=1)
    num = ((cp @ win) * cn).sum(axis=1)
    out = np.full(len(denom), np.nan)
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    return out


def cm_draws(pred: np.ndarray, label: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """(n_boot, 3, 3) confusion matrices, one per resample.

    A participant contributes a single cell, so the resampled confusion matrix is
    the multiplicity vector times a fixed indicator matrix.
    """
    n = len(label)
    ind = np.zeros((n, N_CLASSES * N_CLASSES))
    ind[np.arange(n), np.asarray(label, int) * N_CLASSES + np.asarray(pred, int)] = 1.0
    return (mult @ ind).reshape(-1, N_CLASSES, N_CLASSES)


def accuracy_percent(cm: np.ndarray) -> np.ndarray:
    """Percent correct, vectorised over a leading resample axis."""
    cm = np.atleast_3d(cm) if cm.ndim == 3 else cm
    total = cm.sum(axis=(-2, -1))
    tr = np.trace(cm, axis1=-2, axis2=-1)
    return np.where(total > 0, 100.0 * tr / np.where(total > 0, total, 1.0), 0.0)


def macro_f1(cm: np.ndarray) -> np.ndarray:
    """Unweighted mean of the three per class F1 scores, vectorised."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return f1.mean(axis=-1)


def _self_check(score: np.ndarray, positive: np.ndarray, pred: np.ndarray,
                label: np.ndarray) -> None:
    """The matrix forms must agree with the canonical helpers on the real data.

    Run on the first arm's first contrast rather than on toy numbers, because a
    unit test on toy numbers cannot catch an indexing mistake that only shows up
    at this cohort's tie structure.
    """
    n = len(score)
    identity = np.ones((1, n))
    got = float(auc_draws(score, positive, identity)[0])
    want = float(midrank_auc(score, positive))
    if not np.isfinite(got) or abs(got - want) > 1e-12:
        raise SystemExit(f"auc_draws disagrees with midrank_auc ({got!r} vs {want!r}); "
                         f"refusing to run a bootstrap on a primitive that is wrong")
    cm = cm_draws(pred, label, np.ones((1, len(label))))[0]
    ref = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(ref, (np.asarray(label, int), np.asarray(pred, int)), 1.0)
    if not np.array_equal(cm, ref):
        raise SystemExit("cm_draws disagrees with a direct confusion count")


# ---------------------------------------------------------------------------
# Scores and metrics for one probability matrix
# ---------------------------------------------------------------------------

def contrast_score(probs: np.ndarray, higher: Optional[int], lower: int) -> np.ndarray:
    """P(higher class), or one minus P(CDR 0) when "higher" spans two classes.

    One convention is applied to all five models so the comparison between them is
    like for like, which is all an increment test needs. a26 is the script that
    asks whether the choice of convention changes anything.
    """
    return probs[:, higher] if higher is not None else 1.0 - probs[:, lower]


def point_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, higher, lower in PAIRWISE:
        mask = (y > 0) if higher is None else (y == higher)
        mask = mask | (y == lower)
        pos = (y[mask] > 0) if higher is None else (y[mask] == higher)
        out[name] = float(midrank_auc(contrast_score(probs, higher, lower)[mask], pos))
    cm = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(cm, (y, probs.argmax(axis=1)), 1.0)
    out["macro F1"] = float(macro_f1(cm))
    out["accuracy percent"] = float(accuracy_percent(cm[None, ...])[0])
    return out


# ---------------------------------------------------------------------------
# The pre-declared arm subset for the summary
# ---------------------------------------------------------------------------

SEPARATION_RULE = ("an arm SEPARATED FROM THE NULL if its F1 is above every one of the "
                   "measured permuted-label replicates on all four quantities (macro-F1 "
                   "and each class's F1) at BOTH the participant unit and the image unit")


def arms_separated_from_null(coh: Cohort, tree: Path) -> Tuple[List[str], str]:
    """Which arms clear the measured permuted-label null on every quantity.

    The rule is fixed above, before any increment number below is looked at, so
    the summary's arm set cannot be chosen to flatter the increment. When the
    permutation tree is missing the summary falls back to every arm and says so,
    rather than quietly reporting a median over a different set.
    """
    if not tree.is_dir():
        return list(coh.order()), (f"FALLBACK, every arm, because no permutation tree was "
                                   f"found at {tree}")
    null, _excluded, _n_runs = null_replicates(tree)
    if null.empty:
        return list(coh.order()), ("FALLBACK, every arm, because no usable permuted-label "
                                   "replicates were found")
    quantities = ["macro_f1"] + [f"f1_{s}" for s in SHORT]
    keep: List[str] = []
    for arm in coh.order():
        ok = True
        for unit in ("participant", "image"):
            sub = null[null["unit"] == unit]
            if sub.empty:
                ok = False
                continue
            frame = coh.unit(unit)[arm]
            cm = np.zeros((N_CLASSES, N_CLASSES))
            np.add.at(cm, (frame["label"].to_numpy().astype(int),
                           frame["pred"].to_numpy().astype(int)), 1.0)
            f1 = per_class_f1(cm)
            values = {"macro_f1": float(f1.mean()),
                      **{f"f1_{SHORT[c]}": float(f1[c]) for c in range(N_CLASSES)}}
            for q in quantities:
                if not values[q] > float(sub[q].to_numpy(dtype=float).max()):
                    ok = False
        if ok:
            keep.append(arm)
    n_rep = int(null["perm_seed"].nunique())
    return keep, f"measured permuted-label null, {n_rep} replicates"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="Increment of a stacked CNN score over metadata.")
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

    print(f"a25  increment over metadata, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    y = coh.labels
    counts = np.bincount(y, minlength=N_CLASSES)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images, classes CDR 0 / 0.5 / >=1 = "
          f"{counts[0]} / {counts[1]} / {counts[2]}")

    meta_raw = pd.read_excel(metadata_xlsx())
    meta_raw["pid"] = meta_raw["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta_raw = meta_raw.drop_duplicates("pid").set_index("pid")
    base_meta = meta_raw.reindex(coh.participants)
    n_missing_mmse = int(pd.to_numeric(base_meta["MMSE"], errors="coerce").isna().sum())
    print(f"     MMSE missing for {n_missing_mmse} of {coh.n_participants} on this cohort. "
          f"The full cohort is NOT run here: MMSE is missing for 147 of 347 there.")
    if n_missing_mmse:
        print(f"  !! MMSE was expected to be complete on the restricted cohort and is not. "
              f"Every MMSE row below is partly an imputation and must be read that way.")

    # -- bootstrap index matrices, one per quantity family, shared by every model --
    subsets: Dict[str, Dict[str, np.ndarray]] = {}
    for name, higher, lower in PAIRWISE:
        mask = ((y > 0) if higher is None else (y == higher)) | (y == lower)
        pos = (y[mask] > 0) if higher is None else (y[mask] == higher)
        idx = stratified_participant_bootstrap(pos.astype(int), n_boot, "a25:auc",
                                               BASE_SEED, contrast=name)
        subsets[name] = {"mask": mask, "pos": pos, "higher": higher, "lower": lower,
                         "mult": multiplicity(idx, int(mask.sum()))}
    cm_mult = multiplicity(
        unstratified_participant_bootstrap(coh.n_participants, n_boot, "a25:cm", BASE_SEED),
        coh.n_participants)

    # -- fit every model on every arm -------------------------------------------
    print(f"\n     fitting {len(MODELS)} cross-validated models on each of {len(coh.arms)} "
          f"arms, same grouped folds as the networks, preprocessing from training folds only")
    probs: Dict[Tuple[str, str], np.ndarray] = {}
    imputed: Dict[str, Dict[str, int]] = {}
    checked = False
    for arm in coh.order():
        aug = base_meta.copy()
        feats = cnn_features(coh.part[arm][[f"p{c}" for c in range(N_CLASSES)]].to_numpy())
        aug[CNN_COLS[0]] = feats[:, 0]
        aug[CNN_COLS[1]] = feats[:, 1]
        fitter = MetaFitter(coh.participants, y, coh.folds, aug)
        for model, cols, _uses_cnn in MODELS:
            Q, miss = fitter.fit(cols)
            probs[(arm, model)] = Q
            imputed[model] = {k: int(v) for k, v in miss.items()}
            if not checked:
                s = contrast_score(Q, 2, 0)
                m = subsets[AUC_METRICS[0]]["mask"]
                _self_check(s[m], subsets[AUC_METRICS[0]]["pos"], Q.argmax(axis=1), y)
                checked = True
        print(f"       {arm:<16} fitted")

    # -- point estimates ---------------------------------------------------------
    rows: List[Dict] = []
    points: Dict[Tuple[str, str], Dict[str, float]] = {}
    for arm in coh.order():
        for model, cols, uses_cnn in MODELS:
            pm = point_metrics(probs[(arm, model)], y)
            points[(arm, model)] = pm
            for metric in ALL_METRICS:
                rows.append({"kind": "point", "arm": arm, "model": model,
                             "uses_stacked_cnn_score": uses_cnn,
                             "predictors": " + ".join(cols), "metric": metric,
                             "value": pm[metric], "contrast": "", "direction": "",
                             "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                             "ci_level": np.nan, "p_bootstrap": np.nan,
                             "p_at_resolution_floor": False,
                             "bootstrap": "", "n_bootstrap": n_boot,
                             "family": UNDECLARED,
                             "median_imputed": str(imputed.get(model, {})),
                             "n_participants": coh.n_participants,
                             "code_fingerprint": coh.fingerprint.get(arm, ""),
                             "note": ""})

    # -- bootstrap draws, every model on the same resamples ----------------------
    print(f"\n     paired bootstrap, {n_boot:,} resamples. "
          f"AUC uses the {STRATIFIED} draw; macro-F1 and accuracy use the {UNSTRATIFIED} draw.")
    draws: Dict[Tuple[str, str, str], np.ndarray] = {}
    for arm in coh.order():
        for model, _cols, _u in MODELS:
            Q = probs[(arm, model)]
            for name in AUC_METRICS:
                s = subsets[name]
                sc = contrast_score(Q, s["higher"], s["lower"])[s["mask"]]
                draws[(arm, model, name)] = auc_draws(sc, s["pos"], s["mult"])
            cms = cm_draws(Q.argmax(axis=1), y, cm_mult)
            draws[(arm, model, "macro F1")] = macro_f1(cms)
            draws[(arm, model, "accuracy percent")] = accuracy_percent(cms)

    for arm in coh.order():
        for a, b, direction in CONTRASTS:
            for metric in ALL_METRICS:
                d = draws[(arm, a, metric)] - draws[(arm, b, metric)]
                lo, hi = percentile_interval(d, alpha=0.05)
                stats = bootstrap_p(d)
                rows.append({"kind": "difference", "arm": arm, "model": a,
                             "uses_stacked_cnn_score": True,
                             "predictors": "", "metric": metric,
                             "value": points[(arm, a)][metric],
                             "contrast": f"{a} minus {b}", "direction": direction,
                             "difference": points[(arm, a)][metric] - points[(arm, b)][metric],
                             "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                             "p_bootstrap": stats["p_bootstrap"],
                             "p_at_resolution_floor": stats["p_at_resolution_floor"],
                             "bootstrap": BOOTSTRAP_FOR[metric], "n_bootstrap": n_boot,
                             "family": UNDECLARED,
                             "median_imputed": "",
                             "n_participants": coh.n_participants,
                             "code_fingerprint": coh.fingerprint.get(arm, ""),
                             "note": ("nominal 95 percent, no declared family covers a "
                                      "metadata increment on the restricted cohort")})

    # -- the pre-declared summary ------------------------------------------------
    tree = REPO_ROOT.parent / PERMNULL_TREE
    separated, basis = arms_separated_from_null(coh, tree)
    excluded = [a for a in coh.order() if a not in separated]

    summary: List[Dict] = []
    for a, b, direction in CONTRASTS:
        for metric in ALL_METRICS:
            vals = {arm: points[(arm, a)][metric] - points[(arm, b)][metric]
                    for arm in separated}
            if not vals:
                continue
            best_arm = max(vals, key=lambda k: vals[k])
            med = float(np.median(list(vals.values())))
            summary.append({"contrast": f"{a} minus {b}", "direction": direction,
                            "metric": metric,
                            "n_arms_separated": len(separated),
                            "median_over_separated_arms": med,
                            "best_arm": best_arm, "best_arm_difference": vals[best_arm],
                            "arms_separated": "; ".join(separated),
                            "arms_excluded": "; ".join(excluded) or "none",
                            "separation_basis": basis})
            rows.append({"kind": "summary", "arm": f"median over {len(separated)} separated arms",
                         "model": a, "uses_stacked_cnn_score": True, "predictors": "",
                         "metric": metric, "value": np.nan,
                         "contrast": f"{a} minus {b}", "direction": direction,
                         "difference": med, "ci_low": np.nan, "ci_high": np.nan,
                         "ci_level": np.nan, "p_bootstrap": np.nan,
                         "p_at_resolution_floor": False,
                         "bootstrap": BOOTSTRAP_FOR[metric], "n_bootstrap": n_boot,
                         "family": UNDECLARED, "median_imputed": "",
                         "n_participants": coh.n_participants, "code_fingerprint": "",
                         "note": f"pre-declared summary. {SEPARATION_RULE}. Basis: {basis}"})
            rows.append({"kind": "summary", "arm": f"best arm: {best_arm}", "model": a,
                         "uses_stacked_cnn_score": True, "predictors": "",
                         "metric": metric, "value": np.nan,
                         "contrast": f"{a} minus {b}", "direction": direction,
                         "difference": vals[best_arm], "ci_low": np.nan, "ci_high": np.nan,
                         "ci_level": np.nan, "p_bootstrap": np.nan,
                         "p_at_resolution_floor": False,
                         "bootstrap": BOOTSTRAP_FOR[metric], "n_bootstrap": n_boot,
                         "family": UNDECLARED, "median_imputed": "",
                         "n_participants": coh.n_participants, "code_fingerprint": "",
                         "note": "pre-declared summary, single best arm on this contrast"})

    table = pd.DataFrame(rows)
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen --------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 96)
    print("a25  DOES THE STACKED CNN SCORE ADD ANYTHING TO WHAT IS ALREADY ON THE CHART?")
    print("=" * 96)
    print(f"  restricted cohort, {coh.n_participants} participants, participant level, "
          f"{len(coh.arms)} arms, {n_boot:,} paired resamples, nominal 95 percent intervals.")
    print(f"  The stacked CNN score is two log ratios of the network's probabilities. "
          f"It is not the network.")
    print("\n  SIGN convention: a POSITIVE difference means the model on the left is better.")
    pool = separated or list(coh.order())
    for metric in ALL_METRICS:
        print(f"\n  -- {metric}")
        print(f"     {'contrast':<40}{'median':>10}{'best arm':>16}{'best':>10}"
              f"{'arms sig':>12}")
        for a, b, _direction in CONTRASTS:
            sub = table[(table["kind"] == "difference") & (table["metric"] == metric) &
                        (table["contrast"] == f"{a} minus {b}")]
            sig = int(((sub["ci_low"] > 0) | (sub["ci_high"] < 0)).sum())
            vals = {arm: points[(arm, a)][metric] - points[(arm, b)][metric] for arm in pool}
            best_arm = max(vals, key=lambda k: vals[k])
            label = f"{SHORT_TAG[a]} minus {SHORT_TAG[b]}"
            print(f"     {label:<40}{np.median(list(vals.values())):>+10.4f}"
                  f"{best_arm:>16}{vals[best_arm]:>+10.4f}{sig:>8} of {len(coh.arms)}")
    print(f"\n  SUMMARY BASIS. {SEPARATION_RULE}.")
    print(f"     basis: {basis}")
    print(f"     {len(separated)} of {len(coh.arms)} arms separated: {', '.join(separated)}")
    print(f"     excluded: {', '.join(excluded) if excluded else 'none'}")
    if len(separated) != 6:
        print(f"  !! The brief for this experiment expected SIX arms to have separated from "
              f"the null. On this data the rule above selects {len(separated)}. The count is "
              f"printed rather than forced; if six is the intended set, the rule needs "
              f"restating before the summary row is quoted.")
    print("\n  HOW TO READ THE FOUR CONTRASTS")
    print("     forward rows ask whether the stacked CNN score adds to the cheap variables.")
    print("     reverse rows ask whether the cheap variables add to the stacked CNN score.")
    print("     Both are needed. One direction alone cannot be falsified, which is why the")
    print("     registry declares its increment families in pairs.")
    print("\n  EVERY comparison here is UNDECLARED. Nothing in this table is a pre-registered")
    print("  test, no p-value here is corrected, and the intervals are nominal 95 percent.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting anything.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "stacked CNN score added to MMSE, and to age+sex+eTIV+nWBV, both directions, "
        "restricted cohort",
        "pairwise midrank AUC (three contrasts), macro-F1, accuracy",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
