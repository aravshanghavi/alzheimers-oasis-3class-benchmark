#!/usr/bin/env python3
"""a27: the metadata reference table, recomputed with the AUC defect fixed.

    python analysis/a27_metadata_reference_fixed.py
    python analysis/a27_metadata_reference_fixed.py --quick   # 200 resamples
    python analysis/a27_metadata_reference_fixed.py --boot 2000

THIS IS A CORRECTNESS PASS, NOT A NEW QUESTION
-----------------------------------------------
a17 already reports the metadata reference models. Two things about how it does
it are wrong or out of date, and both of them move numbers that the manuscript
quotes.

Defect D3, TIES. a17 ranks tied scores as if they were distinct, which is why
its constant majority predictor scores a macro one-vs-rest AUC of 0.4958 rather
than 0.5000. A predictor that assigns every participant the same three
probabilities cannot order anybody, so its AUC is 0.5 by definition. Anything
else is a ranking of array position. Every AUC here uses midranks, and the
constant predictor is ASSERTED to come out at exactly 0.5000. If it does not,
this script stops, because a table whose own null is wrong cannot be used to
judge anything else in it.

Transductive preprocessing. a17 imputes and standardises once over the whole
cohort, so the held out participants help choose the fill value and the scaling
of the very rows the model is then scored on. a23's MetaFitter does it inside
the training folds. That fitter is reused here rather than copied.

WHAT THIS PRODUCES
------------------
t27_metadata_reference_fixed  every model, both cohorts, ten metrics
t27_diff_vs_a17_a23           how far each number moved, and from which source

The second table is the point of the exercise. A correctness pass whose output
cannot be diffed against what it replaces is not a correctness pass, it is a
second opinion.

MODELS
------
age, sex, age+sex, nWBV, eTIV, age+sex+eTIV+nWBV on both cohorts. MMSE and
age+sex+eTIV+nWBV+MMSE on the restricted cohort only: MMSE is missing for 147 of
the 347 full cohort participants, so a full cohort MMSE model is mostly its own
imputed median and measures the imputation.

THE ORDERING IN THE ABSTRACT
----------------------------
On the full cohort a17 puts nWBV alone at a macro one-vs-rest AUC of 0.8833 and
the best network at 0.8828. Five ten-thousandths, and it is in the abstract. Both
sides are recomputed here with midranks and the script says in one line whether
the ordering survives. That line is the reason this file exists.

MULTIPLICITY
------------
Only one comparison here maps to a declared family: a CNN arm against
age+sex+eTIV+nWBV on the full cohort is arm_vs_metadata, which is the same
mapping a23 uses. Everything else carries the literal string "undeclared" and a
nominal 95 percent interval. Nothing is written to the registry.
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

from analysis._common import (SuffStats, clustered_bootstrap, paired_difference,  # noqa: E402
                              save_table)
from analysis._determinism import seed_for                                     # noqa: E402
from analysis._paths import CACHE_NPZ, EXPERIMENTS_DIR, OUT, metadata_xlsx      # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, MetaFitter, changelog_entry,        # noqa: E402
                                    constant_frame, find_full_tree,
                                    FULL_EXPERIMENTS, midrank_auc,
                                    participant_frame, prob_frame,
                                    RESTRICTED_EXPERIMENTS)

SOURCE = "analysis/a27_metadata_reference_fixed.py"
EXPERIMENT_ID = "a27"
TABLE = "t27_metadata_reference_fixed"
DIFF_TABLE = "t27_diff_vs_a17_a23"
N_BOOT = 2000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"
CONSTANT = "Constant majority"

# The threshold the brief sets for "this number moved". Applied on each metric's
# own natural scale, which is why accuracy is carried as a FRACTION in the diff
# table: 0.005 on a percent scale would flag one participant in two hundred as a
# change worth investigating, and nothing would ever pass.
MOVED = 0.005

META_MODELS: List[Tuple[str, List[str], bool]] = [
    ("META age only", ["Age"], True),
    ("META sex only", ["M/F"], True),
    ("META age+sex", ["Age", "M/F"], True),
    ("META nWBV only", ["nWBV"], True),
    ("META eTIV only", ["eTIV"], True),
    ("META age+sex+eTIV+nWBV", ["Age", "M/F", "eTIV", "nWBV"], True),
    ("META MMSE only", ["MMSE"], False),
    ("META age+sex+eTIV+nWBV+MMSE", ["Age", "M/F", "eTIV", "nWBV", "MMSE"], False),
]
META_REFERENCE = "META age+sex+eTIV+nWBV"

PAIRWISE = [("AUC CDR>=1 vs CDR 0", 2, 0),
            ("AUC any impairment vs CDR 0", None, 0),
            ("AUC CDR>=1 vs CDR 0.5", 2, 1)]
BOOT_METRICS = ["accuracy fraction", "macro F1"]
METRICS = (["accuracy fraction", "macro F1", "F1 CDR 0", "F1 CDR 0.5", "F1 CDR>=1",
            "QWK", "macro one-vs-rest AUC"] + [n for n, _, _ in PAIRWISE])

# a17 spells its cohorts and its models differently. Mapping them here, once,
# beats three ad hoc renames scattered through the diff builder.
A17_COHORT = {"full": "full", "restricted": "age60"}
A17_METRIC = {"accuracy fraction": "accuracy", "macro F1": "macroF1", "QWK": "QWK",
              "macro one-vs-rest AUC": "macroAUROC"}
A17_MODEL = {n: n for n, _, _ in META_MODELS}
A17_MODEL[CONSTANT] = "BASELINE constant"
A23_METRIC = {"accuracy fraction": "accuracy", "macro F1": "macro_f1"}


def qwk(cm: np.ndarray) -> float:
    """Quadratic weighted kappa from a confusion matrix.

    The (i-j)^2 weights make a CDR 0 called CDR >= 1 four times as costly as a
    neighbour error, which is the ordering the clinical scale actually has and
    which plain accuracy throws away.
    """
    n = cm.sum()
    if n == 0:
        return float("nan")
    w = np.array([[(i - j) ** 2 for j in range(N_CLASSES)] for i in range(N_CLASSES)], float)
    po = (cm * w).sum() / n
    pe = (np.outer(cm.sum(axis=1), cm.sum(axis=0)) * w).sum() / (n * n)
    return float("nan") if pe == 0 else float(1.0 - po / pe)


def all_metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """The ten numbers this table reports, for one probability matrix."""
    pred = probs.argmax(axis=1)
    cm = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(cm, (y, pred), 1.0)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(N_CLASSES), where=denom > 0)
    out = {"accuracy fraction": float(np.trace(cm) / cm.sum()),
           "macro F1": float(f1.mean()),
           "F1 CDR 0": float(f1[0]), "F1 CDR 0.5": float(f1[1]), "F1 CDR>=1": float(f1[2]),
           "QWK": qwk(cm),
           "macro one-vs-rest AUC": float(np.mean(
               [midrank_auc(probs[:, c], y == c) for c in range(N_CLASSES)]))}
    for name, higher, lower in PAIRWISE:
        mask = ((y > 0) if higher is None else (y == higher)) | (y == lower)
        pos = (y[mask] > 0) if higher is None else (y[mask] == higher)
        score = probs[:, higher] if higher is not None else 1.0 - probs[:, lower]
        out[name] = float(midrank_auc(score[mask], pos))
    return out


def assert_constant_is_half(probs: np.ndarray, y: np.ndarray) -> None:
    """The constant predictor must score exactly 0.5 on every AUC in this table.

    This is the defect being fixed, so it gets an assertion rather than a comment.
    a17 prints 0.4958 for the same predictor because it breaks ties by array
    position. A table that cannot get its own null right is not evidence about
    anything else it contains, so a failure here stops the script.
    """
    m = all_metrics(probs, y)
    auc_names = ["macro one-vs-rest AUC"] + [n for n, _, _ in PAIRWISE]
    bad = {k: m[k] for k in auc_names if not abs(m[k] - 0.5) < 1e-12}
    if bad:
        raise SystemExit(
            "FATAL. The constant majority predictor did not score exactly 0.5000 on "
            f"{sorted(bad)}: {bad}. Every participant gets the same three probabilities, "
            "so no ordering exists and 0.5 is the only defensible value. Getting anything "
            "else means the AUC is breaking ties by position, which is defect D3 and is "
            "the thing this script exists to fix. Nothing below this point is usable.")


def load_a17() -> Optional[pd.DataFrame]:
    """a17's point estimates, if that script has been run in this repository."""
    f = OUT / "a17_participant_contrasts.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    return d[d["kind"] == "point"].copy()


def load_a23() -> Optional[pd.DataFrame]:
    """a23's R1 and R4 reference values, which are the same models at 2000 resamples."""
    f = OUT / "t23_protocol_audit_long.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    keep = d[(d["unit"] == "participant") & (d["requirement"].isin(["R1", "R4"]))]
    return keep[["cohort", "metric", "reference", "reference_value"]].drop_duplicates(
        ["cohort", "metric", "reference"]).copy()


def a17_folds_match(cohort: Cohort) -> Optional[bool]:
    """Did a17 fit this cohort's metadata models on the SAME folds this script uses?

    On the full cohort it did. On the restricted cohort it did not: a17 masks the
    full cohort's five-fold assignment down to the 166 assessed participants, while
    this script uses the restricted cohort's own five folds. Numbers that move for
    that reason are not a defect being fixed, and the diff table has to say which
    is which or the whole comparison is misleading.
    """
    if not CACHE_NPZ.is_file():
        return None
    c = np.load(CACHE_NPZ, allow_pickle=False)
    upid = c["upid"].astype(str)
    pf = c["pf"].astype(int)
    pos = {p: i for i, p in enumerate(upid)}
    if not set(cohort.participants).issubset(pos):
        return False
    theirs = np.array([pf[pos[p]] for p in cohort.participants])
    return bool(np.array_equal(theirs, cohort.folds))


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="Metadata reference models, AUC defect fixed.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"paired bootstrap resamples (default {N_BOOT}, matching a23)")
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

    print(f"a27  metadata reference, midrank AUC, {n_boot:,} resamples, base seed {BASE_SEED}")
    restricted = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    print(f"     restricted: {len(restricted.arms)} arms, {restricted.n_participants} "
          f"participants, {restricted.n_images:,} images")
    full_dir = find_full_tree(args.full_from)
    full: Optional[Cohort] = None
    if full_dir is None:
        print("  !! no full-cohort experiments folder found. The full half of this table, "
              "including the abstract's nWBV ordering, is SKIPPED.")
    else:
        full = Cohort("full", FULL_EXPERIMENTS, full_dir)
        print(f"     full:       {len(full.arms)} arms, {full.n_participants} participants, "
              f"{full.n_images:,} images, from {full_dir}")
    cohorts = [c for c in (full, restricted) if c is not None]

    meta_raw = pd.read_excel(metadata_xlsx())
    meta_raw["pid"] = meta_raw["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta_raw = meta_raw.drop_duplicates("pid").set_index("pid")

    rows: List[Dict] = []
    points: Dict[Tuple[str, str], Dict[str, float]] = {}
    imputed: Dict[Tuple[str, str], Dict[str, int]] = {}
    ordering_line = ""

    for coh in cohorts:
        y = coh.labels
        print(f"\n     {coh.name}: fitting metadata models on the same grouped folds as the "
              f"networks, preprocessing from training folds only")
        fitter = MetaFitter(coh.participants, y, coh.folds, meta_raw)
        probs: Dict[str, np.ndarray] = {}
        for name, cols, both in META_MODELS:
            if not both and coh.name != "restricted":
                print(f"       {name:<30} SKIPPED on the full cohort, MMSE is missing for "
                      f"147 of 347 there")
                continue
            Q, miss = fitter.fit(cols)
            probs[name] = Q
            imputed[(coh.name, name)] = {k: int(v) for k, v in miss.items()}
            bad = {k: v for k, v in miss.items() if v}
            print(f"       {name:<30} fitted" + (f", imputed {bad}" if bad else ""))

        template = coh.part[coh.order()[0]]
        const_frame, maj, prior = constant_frame(template)
        const_probs = const_frame[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
        assert_constant_is_half(const_probs, y)
        print(f"       {CONSTANT:<30} majority class {maj}, prior "
              f"{np.round(prior, 4).tolist()}, AUC checked at exactly 0.5000")

        scored: Dict[str, np.ndarray] = dict(probs)
        scored[CONSTANT] = const_probs
        for arm in coh.order():
            scored[arm] = coh.part[arm][[f"p{c}" for c in range(N_CLASSES)]].to_numpy()

        for name, P in scored.items():
            m = all_metrics(P, y)
            points[(coh.name, name)] = m
            kind = ("CNN arm" if name in coh.arms else
                    "constant baseline" if name == CONSTANT else "metadata reference")
            for metric in METRICS:
                rows.append({"kind": "point", "cohort": coh.name, "model": name,
                             "model_kind": kind, "metric": metric, "value": m[metric],
                             "reference": "", "difference": np.nan,
                             "ci_low": np.nan, "ci_high": np.nan, "ci_level": np.nan,
                             "p_bootstrap": np.nan, "p_at_resolution_floor": False,
                             "family": UNDECLARED, "alpha_used": np.nan,
                             "n_participants": coh.n_participants, "n_bootstrap": n_boot,
                             "median_imputed": str(imputed.get((coh.name, name), {})),
                             "code_fingerprint": coh.fingerprint.get(name, ""),
                             "note": ""})

        # -- paired bootstrap on accuracy and macro-F1, a23's engine and a23's B ---
        stats: Dict[str, SuffStats] = {}
        for name, P in scored.items():
            if name in coh.arms:
                stats[name] = SuffStats(coh.part[name])
            else:
                stats[name] = SuffStats(prob_frame(P, coh.participants, y))
        for metric, key in (("accuracy fraction", "accuracy"), ("macro F1", "macro_f1")):
            boot = clustered_bootstrap(stats, key, n_boot=n_boot,
                                       seed=seed_for(BASE_SEED, "a27:main",
                                                     cohort=coh.name, metric=key))
            scale = 0.01 if key == "accuracy" else 1.0
            for ref in [n for n in scored if n not in coh.arms]:
                for arm in coh.order():
                    fam = ("arm_vs_metadata"
                           if (coh.name == "full" and ref == META_REFERENCE) else None)
                    d = paired_difference(boot, arm, ref, family=fam)
                    rows.append({"kind": "difference", "cohort": coh.name, "model": arm,
                                 "model_kind": "CNN arm", "metric": metric,
                                 "value": points[(coh.name, arm)][metric],
                                 "reference": ref,
                                 "difference": d["diff"] * scale,
                                 "ci_low": d["ci_low"] * scale, "ci_high": d["ci_high"] * scale,
                                 "ci_level": d["ci_level"], "p_bootstrap": d["p_bootstrap"],
                                 "p_at_resolution_floor": d["p_at_resolution_floor"],
                                 "family": fam or UNDECLARED,
                                 "alpha_used": (1.0 - d["ci_level"]),
                                 "n_participants": coh.n_participants, "n_bootstrap": n_boot,
                                 "median_imputed": "",
                                 "code_fingerprint": coh.fingerprint.get(arm, ""),
                                 "note": "arm minus reference, positive favours the arm"})

    table = pd.DataFrame(rows)
    save_table(table, TABLE, float_fmt="%.6f")

    # -- the abstract's ordering -------------------------------------------------
    if full is not None:
        nwbv = points[("full", "META nWBV only")]["macro one-vs-rest AUC"]
        arm_auc = {a: points[("full", a)]["macro one-vs-rest AUC"] for a in full.order()}
        best = max(arm_auc, key=lambda k: arm_auc[k])
        held = nwbv > arm_auc[best]
        ordering_line = (
            f"full cohort macro one-vs-rest AUC, midrank: nWBV alone {nwbv:.4f}, best network "
            f"{best} {arm_auc[best]:.4f}, margin {nwbv - arm_auc[best]:+.4f}. "
            f"a17 reported 0.8833 against 0.8828, margin +0.0005. The ordering "
            f"{'HOLDS' if held else 'REVERSES'} under the fixed AUC.")

    # -- diff table --------------------------------------------------------------
    a17 = load_a17()
    a23 = load_a23()
    diff_rows: List[Dict] = []
    fold_match = {c.name: a17_folds_match(c) for c in cohorts}
    for coh in cohorts:
        same_folds = fold_match[coh.name]
        for name in [n for n, _, _ in META_MODELS] + [CONSTANT]:
            if (coh.name, name) not in points:
                continue
            for metric in METRICS:
                got = points[(coh.name, name)][metric]
                row = {"cohort": coh.name, "model": name, "metric": metric,
                       "a27_value": got,
                       "a17_value": np.nan, "a17_delta": np.nan, "a17_moved": False,
                       "a23_value": np.nan, "a23_delta": np.nan, "a23_moved": False,
                       "a17_same_folds": same_folds,
                       "comparable_to_a17": bool(same_folds) if same_folds is not None else None,
                       "note": ""}
                if a17 is not None and metric in A17_METRIC:
                    sel = a17[(a17["cohort"] == A17_COHORT[coh.name]) &
                              (a17["model"] == A17_MODEL.get(name, name))]
                    if not sel.empty:
                        v = float(sel.iloc[0][A17_METRIC[metric]])
                        row["a17_value"] = v
                        row["a17_delta"] = got - v
                        row["a17_moved"] = bool(abs(got - v) > MOVED)
                if a23 is not None and metric in A23_METRIC:
                    sel = a23[(a23["cohort"] == coh.name) &
                              (a23["metric"] == A23_METRIC[metric]) &
                              (a23["reference"] == name)]
                    if not sel.empty:
                        v = float(sel.iloc[0]["reference_value"])
                        v = v * 0.01 if metric == "accuracy fraction" else v
                        row["a23_value"] = v
                        row["a23_delta"] = got - v
                        row["a23_moved"] = bool(abs(got - v) > MOVED)
                notes = []
                if same_folds is False:
                    notes.append("a17 fitted this cohort on the FULL cohort's fold assignment "
                                 "masked down to these participants, not on this cohort's own "
                                 "five folds, so an a17 delta here is not the defect fix")
                if same_folds is None:
                    notes.append("analysis/outputs/_cache.npz is absent, so whether a17 used "
                                 "these folds could not be checked")
                if metric in A17_METRIC and a17 is None:
                    notes.append("a17_participant_contrasts.csv absent")
                if metric in A23_METRIC and a23 is None:
                    notes.append("t23_protocol_audit_long.csv absent")
                row["note"] = "; ".join(notes)
                diff_rows.append(row)
    diffs = pd.DataFrame(diff_rows)
    save_table(diffs, DIFF_TABLE, float_fmt="%.6f")

    # -- one screen --------------------------------------------------------------
    elapsed = time.time() - t_start
    moved17 = diffs[diffs["a17_moved"]]
    moved23 = diffs[diffs["a23_moved"]]
    print("\n" + "=" * 96)
    print("a27  METADATA REFERENCE TABLE, RECOMPUTED WITH TIES HANDLED CORRECTLY")
    print("=" * 96)
    print(f"  The constant predictor now scores exactly 0.5000 on every AUC, asserted in code.")
    print(f"  a17 printed 0.4958 for the same predictor. That was the defect.")
    # Printed whatever the threshold says. This one number IS the defect, and a
    # reader should not have to work out from a flag column whether it moved.
    for coh in cohorts:
        sel = diffs[(diffs["cohort"] == coh.name) & (diffs["model"] == CONSTANT) &
                    (diffs["metric"] == "macro one-vs-rest AUC")]
        if not sel.empty and np.isfinite(sel.iloc[0]["a17_value"]):
            r = sel.iloc[0]
            print(f"     cohort {coh.name:<11} constant macro one-vs-rest AUC "
                  f"{r['a17_value']:.4f} in a17 -> {r['a27_value']:.4f} here "
                  f"({r['a17_delta']:+.4f})")
    if ordering_line:
        print(f"\n  THE ORDERING IN THE ABSTRACT")
        print(f"     {ordering_line}")
    for coh in cohorts:
        print(f"\n  -- cohort {coh.name}, n = {coh.n_participants}")
        print(f"     {'model':<30}{'acc':>8}{'macroF1':>9}{'QWK':>8}{'mOVR AUC':>10}"
              f"{'>=1 vs 0':>10}{'any vs 0':>10}{'>=1 vs .5':>11}")
        names = [n for n, _, _ in META_MODELS if (coh.name, n) in points] + \
                [CONSTANT] + list(coh.order())
        for name in names:
            m = points[(coh.name, name)]
            print(f"     {name:<30}{m['accuracy fraction']:>8.4f}{m['macro F1']:>9.4f}"
                  f"{m['QWK']:>8.4f}{m['macro one-vs-rest AUC']:>10.4f}"
                  f"{m['AUC CDR>=1 vs CDR 0']:>10.4f}{m['AUC any impairment vs CDR 0']:>10.4f}"
                  f"{m['AUC CDR>=1 vs CDR 0.5']:>11.4f}")
    print(f"\n  WHAT MOVED, threshold {MOVED} on each metric's own scale "
          f"(accuracy is a FRACTION here, so {MOVED} is half a percentage point)")
    print(f"     against a17: {len(moved17)} of {int(diffs['a17_value'].notna().sum())} "
          f"comparable numbers moved by more than {MOVED}")
    for _, r in moved17.iterrows():
        tag = "" if r["comparable_to_a17"] else "   [different folds, not the defect fix]"
        print(f"       {r['cohort']:<11}{r['model']:<30}{r['metric']:<24}"
              f"{r['a17_value']:>9.4f} -> {r['a27_value']:>8.4f}  "
              f"{r['a17_delta']:>+8.4f}{tag}")
    print(f"     against a23: {len(moved23)} of {int(diffs['a23_value'].notna().sum())} "
          f"comparable numbers moved by more than {MOVED}")
    for _, r in moved23.iterrows():
        print(f"       {r['cohort']:<11}{r['model']:<30}{r['metric']:<24}"
              f"{r['a23_value']:>9.4f} -> {r['a27_value']:>8.4f}  {r['a23_delta']:>+8.4f}")
    for coh in cohorts:
        if fold_match[coh.name] is False:
            print(f"\n  !! cohort {coh.name}: a17 used a DIFFERENT fold assignment for its "
                  f"metadata models. Deltas against a17 on this cohort mix the tie fix, the "
                  f"fold-honest preprocessing and the change of folds, and cannot be read as "
                  f"the defect fix alone. The a23 column is the like-for-like one.")
        if fold_match[coh.name] is None:
            print(f"\n  !! cohort {coh.name}: _cache.npz absent, so the fold check could not "
                  f"run. Treat every a17 delta on this cohort as unverified.")
    print("\n  Only one comparison in this file maps to a declared family: a CNN arm against")
    print(f"  {META_REFERENCE} on the full cohort is arm_vs_metadata, the same mapping a23")
    print("  uses. Everything else is undeclared at a nominal 95 percent.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting anything.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "metadata reference models and the constant predictor recomputed with midrank AUC "
        "and fold-honest preprocessing, both cohorts, diffed against a17 and a23",
        "accuracy, macro-F1, per-class F1, QWK, macro one-vs-rest AUC, three pairwise AUCs",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
