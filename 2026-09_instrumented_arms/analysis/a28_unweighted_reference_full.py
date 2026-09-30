#!/usr/bin/env python3
"""a28: what happens on the full cohort with NO imbalance correction at all.

    python analysis/a28_unweighted_reference_full.py
    python analysis/a28_unweighted_reference_full.py --quick   # 200 resamples
    python analysis/a28_unweighted_reference_full.py --boot 2000

THE HOLE THIS FILLS
-------------------
The paper compares seven imbalance-aware objectives against each other. It never
shows the one thing a reviewer will ask for first: what the same network does
when nothing is done about the imbalance. Without that row, "FA-FL handles
imbalance well" has no floor to be measured against, and the reader cannot tell
whether the seven arms differ from each other by more or less than they differ
from doing nothing.

That run already exists and was never used. exp02 was first run at beta = 0.999,
then superseded by beta = 0.9999. The Cui class-balanced weight is
(1 - beta) / (1 - beta^n). At beta = 0.999 and this cohort's training counts,
every class's effective number has already saturated, so the three weights come
out at roughly 0.987, 0.987 and 1.025. Numerically that is an UNWEIGHTED network.
The superseded arm is therefore the missing reference, and it costs nothing
because it is already on disk.

WHAT THIS SCRIPT CHECKS BEFORE IT USES ANYTHING
-----------------------------------------------
The folder layout is verified rather than assumed: the five fold runs whose
directory name ends in __splitfixed-init42__8378f120 are used, and the older
__split42-init42__4c22b5e5 fold-0 run is excluded by name and reported, because
it came from the session-level split that this project abandoned. Pooling it in
would mix two different splits under one arm label and nothing downstream would
notice.

The weights are read from the run artefacts and printed. A caveat worth knowing
in advance: manifest.json carries data.class_weights, and on these runs that
field holds the normalised INVERSE-FREQUENCY alpha, roughly [0.05, 0.26, 0.69],
which the class-balanced loss never uses. The weights this loss actually applied
live in metrics.json under criterion.weights. Both are printed, and the
class-balanced weights are independently recomputed from data.train_class_counts
and the manifest's beta so that the near-uniform claim rests on arithmetic this
script did, not on a field whose meaning has to be taken on trust.

WHAT IT THEN RUNS
-----------------
Requirements R1 to R4 of the a23 protocol, at both units, on this reference and
on every imbalance-aware full-cohort arm, using a23's own functions rather than
a second implementation. R2 is NOT AVAILABLE: the permuted-label replicates were
trained on the restricted cohort, so there is no measured null for these 347
people, and the cell says so instead of borrowing a null from a different cohort.

Then the manuscript's Table 4 metrics, the calibration block, and a paired
participant-clustered bootstrap of every imbalance-aware arm against this
reference on accuracy, macro-F1, F1 for CDR >= 1 and QWK.

ON THE FITTED TEMPERATURE
-------------------------
The temperature reported here is the one each fold already fitted on its own
VALIDATION split, read from metrics.json, together with the ECE the stored
temperature-scaled predictions give. No new temperature is fitted on the test
split. Fitting a scalar on the data you then score with it is leakage, it always
improves ECE, and a calibration number obtained that way is not comparable with
the paper's.

MULTIPLICITY
------------
Nothing here is declared. The declared arm-vs-constant and arm-vs-metadata
families were sized for the arms that existed when they were written, and this
reference was not one of them, so folding it in would break the counts those
declarations exist to hold. Every row carries the literal string "undeclared" at
a nominal 95 percent level, and the script prints the changelog entry rather than
writing it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import (SHORT, SuffStats, clustered_bootstrap,           # noqa: E402
                              paired_difference, save_table)
from analysis._determinism import seed_for                                     # noqa: E402
from analysis._paths import OUT, metadata_xlsx                                 # noqa: E402
from analysis._protocol_lib import (Cohort, MetaFitter, changelog_entry,       # noqa: E402
                                    find_full_tree, FULL_EXPERIMENTS,
                                    participant_frame, per_class_f1)
from analysis.a23_protocol_audit import (ARM_ORDER, METRICS as A23_METRICS,    # noqa: E402
                                         UNITS, build_boots, requirement_1,
                                         requirement_3, requirement_4)
from analysis.discover import load_predictions                                 # noqa: E402

SOURCE = "analysis/a28_unweighted_reference_full.py"
EXPERIMENT_ID = "a28"
TABLE = "t28_unweighted_reference_full"
N_BOOT = 2000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

COHORT_NAME = "full_unweighted_reference"
REFERENCE_ARM = "Class-Balanced b=0.999 uniform"

FULL_TREE = "2026-08_srep_revision_participant_level"
REFERENCE_SUBDIR = Path("exp02_class_balanced") / "_superseded_beta0999"
KEEP_SUFFIX = "__splitfixed-init42__8378f120"
EXCLUDE_SUFFIX = "__split42-init42__4c22b5e5"

# Near enough to [1, 1, 1] that the arm is an unweighted network for any purpose
# this paper has. Stated as a number so the claim is falsifiable rather than a
# matter of opinion about the word "near".
UNIFORM_TOL = 0.05

BOOT_METRICS = [("accuracy", "accuracy", None),
                ("macro_f1", "macro F1", None),
                ("f1_Dem", "F1 CDR>=1", 2),
                ("qwk", "QWK", None)]


def qwk_from_cm(cm: np.ndarray) -> float:
    """Quadratic weighted kappa. The (i-j)^2 weights respect the CDR ordering."""
    n = cm.sum()
    if n == 0:
        return float("nan")
    w = np.array([[(i - j) ** 2 for j in range(N_CLASSES)] for i in range(N_CLASSES)], float)
    po = (cm * w).sum() / n
    pe = (np.outer(cm.sum(axis=1), cm.sum(axis=0)) * w).sum() / (n * n)
    return float("nan") if pe == 0 else float(1.0 - po / pe)


def qwk_bootstrap(stats: Dict[str, SuffStats], n_boot: int, seed: int) -> Dict[str, Dict]:
    """QWK on the same resamples _common.clustered_bootstrap would draw.

    _common has no QWK, and adding one to a shared module the rest of this night's
    work depends on is not a change to make at midnight. So the loop is repeated
    here EXACTLY as clustered_bootstrap writes it, one rng.integers call of size
    n_p per resample from the same seed. That is what makes a QWK difference
    paired with the accuracy difference beside it rather than merely similar.
    """
    losses = list(stats)
    n_p = stats[losses[0]].n_p
    rng = np.random.default_rng(seed)
    draws = {L: np.empty(n_boot) for L in losses}
    for b in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        for L in losses:
            draws[L][b] = qwk_from_cm(stats[L].metrics(idx)["confusion_matrix"])
    out = {}
    for L in losses:
        point = qwk_from_cm(stats[L].metrics()["confusion_matrix"])
        lo, hi = np.percentile(draws[L], [2.5, 97.5])
        out[L] = {"point": float(point), "ci_low": float(lo), "ci_high": float(hi),
                  "boot_mean": float(draws[L].mean()),
                  "boot_sd": float(draws[L].std(ddof=1)), "_draws": draws[L]}
    return out


def class_balanced_weights(counts: List[int], beta: float) -> np.ndarray:
    """Cui et al. weights, recomputed here from the recorded training counts.

    src/losses.py computes (1-beta)/(1-beta^n) and then renormalises to sum to the
    number of classes. Repeating that here means the "numerically uniform" claim
    is checked against arithmetic rather than against a stored field.
    """
    c = np.asarray(counts, dtype=np.float64)
    w = (1.0 - beta) / (1.0 - np.power(beta, c))
    return w / w.sum() * len(c)


def inspect_reference_folder(root: Path) -> Tuple[List[Path], List[Tuple[str, str]], List[Dict]]:
    """Find the five fold runs, name everything excluded, and read their weights.

    The layout is checked rather than assumed. If it differs from what this script
    was told to expect, the difference is printed and the run stops, because
    adapting silently to an unexpected layout is how the wrong five folders get
    pooled under the right-looking arm name.
    """
    if not root.is_dir():
        raise SystemExit(f"FATAL. {root} does not exist. Pass --reference-from with the folder "
                         f"holding the superseded beta=0.999 runs.")
    keep, excluded, weights = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.endswith(EXCLUDE_SUFFIX):
            excluded.append((d.name, "old session-level split, excluded by name"))
            continue
        if not d.name.endswith(KEEP_SUFFIX):
            excluded.append((d.name, f"name does not end in {KEEP_SUFFIX}"))
            continue
        missing = [f for f in ("manifest.json", "metrics.json", "predictions_test.npz",
                               "_COMPLETE") if not (d / f).exists()]
        if missing:
            excluded.append((d.name, f"missing {', '.join(missing)}"))
            continue
        keep.append(d)
    for d in keep:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        beta = float(man["loss"]["params"]["beta"])
        counts = list(man["data"]["train_class_counts"])
        recomputed = class_balanced_weights(counts, beta)
        crit = met.get("criterion", {}).get("weights")
        ts = met["test"].get("temperature_scaling", {})
        weights.append({
            "run_dir": str(d), "run_id": man["run_id"], "fold": int(man["split"]["fold"]),
            "loss": man["loss"]["type"], "beta": beta,
            "code_fingerprint": man["code_fingerprint"][:8],
            "split_strategy": man["split"]["strategy"], "group_by": man["split"]["group_by"],
            "group_disjoint": man["data"].get("group_disjoint_assert"),
            "train_class_counts": counts,
            "criterion_weights_from_metrics": crit,
            "criterion_weights_recomputed": [round(float(v), 6) for v in recomputed],
            "manifest_data_class_weights": [round(float(v), 6)
                                            for v in man["data"]["class_weights"]],
            "max_abs_deviation_from_uniform": float(np.abs(recomputed - 1.0).max()),
            "temperature_fitted_on_validation": ts.get("temperature"),
            "ece_test_raw": met["test"].get("ece"),
            "ece_test_after_temperature": ts.get("ece_after"),
            "has_temperature_scaled_predictions":
                (d / "predictions_test_temperature_scaled.npz").exists(),
        })
    return keep, excluded, weights


def pool_reference(dirs: List[Path]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Concatenate the five test folds into one image-level frame, disjointness checked."""
    parts, scaled_parts = [], []
    for d in dirs:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        f = load_predictions(d, "test")
        f = f[["participant_id", "label"]].copy()
        probs = load_predictions(d, "test")[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
        f["pred"] = probs.argmax(axis=1)
        for c in range(N_CLASSES):
            f[f"p{c}"] = probs[:, c]
        f["fold"] = int(man["split"]["fold"])
        parts.append(f)
        sp = d / "predictions_test_temperature_scaled.npz"
        if sp.exists():
            z = np.load(sp, allow_pickle=False)
            p = z["probabilities"].astype(np.float64)
            g = pd.DataFrame({"participant_id": z["participant_id"].astype(str),
                              "label": z["labels"].astype(int)})
            g["pred"] = p.argmax(axis=1)
            for c in range(N_CLASSES):
                g[f"p{c}"] = p[:, c]
            g["fold"] = int(man["split"]["fold"])
            scaled_parts.append(g)
    pooled = pd.concat(parts, ignore_index=True)
    dupes = pooled.groupby("participant_id")["fold"].nunique()
    if (dupes > 1).any():
        raise SystemExit(f"FATAL. {int((dupes > 1).sum())} participant(s) appear in more than "
                         f"one test fold of the reference arm. The folds are not disjoint and "
                         f"pooling them would count those people twice.")
    scaled = (pd.concat(scaled_parts, ignore_index=True)
              if len(scaled_parts) == len(dirs) else None)
    return pooled, scaled


def merged_cohort(full: Cohort, ref_img: pd.DataFrame, ref_part: pd.DataFrame,
                  fingerprint: str) -> Cohort:
    """The full cohort's arms plus the unweighted reference, as one Cohort object.

    Built by hand rather than through Cohort.__init__ because the reference runs
    live under _superseded_beta0999/ and discover.load_runs only looks in
    outputs/. Everything else about the object is the real thing, so a23's own
    requirement functions run on it unchanged instead of being reimplemented.

    The cohort is deliberately NOT called "full". a23 maps ("full", metric) onto
    the declared arm-vs-constant and arm-vs-metadata families, and this arm was
    not in any of those declarations, so inheriting their corrected thresholds
    here would silently misdescribe what was tested.
    """
    c = Cohort.__new__(Cohort)
    c.name = COHORT_NAME
    c.experiments_dir = full.experiments_dir
    c.arms = dict(full.arms)
    c.part = dict(full.part)
    c.scaled = dict(full.scaled)
    c.fingerprint = dict(full.fingerprint)
    c.run_ids = list(full.run_ids)
    c.missing_scaled = list(full.missing_scaled)
    c.runs = full.runs
    c.arms[REFERENCE_ARM] = ref_img
    c.part[REFERENCE_ARM] = ref_part
    c.fingerprint[REFERENCE_ARM] = fingerprint

    ref_ids = set(full.participants)
    for arm, frame in c.arms.items():
        if set(frame["participant_id"]) != ref_ids:
            raise SystemExit(f"FATAL. {arm} covers a different participant set from the full "
                             f"cohort, so nothing here can be paired against it.")
    c.participants = np.array(sorted(ref_ids))
    c.n_participants = len(c.participants)
    c.n_images = len(next(iter(c.arms.values())))
    first = full.arms[full.order()[0]]
    g = first.groupby("participant_id", sort=True)
    c.labels = g["label"].first().to_numpy().astype(int)
    c.folds = g["fold"].first().to_numpy().astype(int)
    c.image_counts = g.size().to_numpy().astype(int)

    ref_g = ref_img.groupby("participant_id", sort=True)
    if not np.array_equal(ref_g["label"].first().to_numpy().astype(int), c.labels):
        raise SystemExit("FATAL. The reference arm's labels disagree with the full cohort's.")
    if not np.array_equal(ref_g["fold"].first().to_numpy().astype(int), c.folds):
        raise SystemExit("FATAL. The reference arm's fold assignment disagrees with the full "
                         "cohort's, so the two were not cross-validated on the same splits.")
    return c


def ece_of(frame: pd.DataFrame) -> float:
    return float(SuffStats(frame).metrics()["ece"])


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="The unweighted full-cohort reference arm.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"paired bootstrap resamples (default {N_BOOT}, matching a23)")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--full-from", default=None, metavar="EXPERIMENTS_DIR",
                    help="experiments folder holding the full-cohort runs")
    ap.add_argument("--reference-from", default=None, metavar="DIR",
                    help="folder holding the superseded beta=0.999 fold runs")
    args = ap.parse_args()
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"a28  unweighted reference on the full cohort, {n_boot:,} resamples, "
          f"base seed {BASE_SEED}")

    # -- step 1, verify the folder before using anything ------------------------
    full_dir = find_full_tree(args.full_from)
    if full_dir is None:
        raise SystemExit("FATAL. No full-cohort experiments folder found. This experiment is "
                         "about the full cohort and cannot be run without it. Pass --full-from.")
    ref_root = (Path(args.reference_from).expanduser() if args.reference_from
                else Path(full_dir) / REFERENCE_SUBDIR)
    print(f"\n  STEP 1. Verifying the reference folder before reading anything from it.")
    print(f"     {ref_root}")
    dirs, excluded, info = inspect_reference_folder(ref_root)
    for name, why in excluded:
        print(f"     EXCLUDED  {name}")
        print(f"               {why}")
    for r in info:
        print(f"     USED      fold {r['fold']}  {r['run_id'][:60]}")
    if len(dirs) != 5:
        raise SystemExit(f"FATAL. Expected 5 fold runs ending in {KEEP_SUFFIX} and found "
                         f"{len(dirs)}. The layout is not what this script was told to expect. "
                         f"Nothing has been read. Inspect the folder and rerun with "
                         f"--reference-from if the runs moved.")
    folds = sorted(r["fold"] for r in info)
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"FATAL. The five runs cover folds {folds}, not 0 through 4.")

    # -- step 2, the weights ----------------------------------------------------
    print(f"\n  STEP 2. Class weights, read from the run artefacts.")
    print(f"     manifest.json data.class_weights holds the normalised INVERSE-FREQUENCY alpha.")
    print(f"     The class-balanced loss never uses that field. Its own weights are in")
    print(f"     metrics.json criterion.weights, and are recomputed below from")
    print(f"     data.train_class_counts and beta to check the stored value.")
    print(f"     {'fold':<6}{'beta':>8}   {'manifest alpha (UNUSED by this loss)':<42}"
          f"{'criterion.weights':<30}{'recomputed':<30}{'max |w-1|':>10}")
    worst = 0.0
    for r in info:
        worst = max(worst, r["max_abs_deviation_from_uniform"])
        print(f"     {r['fold']:<6}{r['beta']:>8.4f}   "
              f"{str(r['manifest_data_class_weights']):<42}"
              f"{str(r['criterion_weights_from_metrics']):<30}"
              f"{str(r['criterion_weights_recomputed']):<30}"
              f"{r['max_abs_deviation_from_uniform']:>10.4f}")
        stored = r["criterion_weights_from_metrics"]
        if stored is not None:
            gap = float(np.abs(np.array(stored) -
                               np.array(r["criterion_weights_recomputed"])).max())
            if gap > 1e-4:
                print(f"  !! fold {r['fold']}: the stored criterion weights and the recomputed "
                      f"ones differ by {gap:.6f}. The near-uniform claim is not safe.")
    if worst > UNIFORM_TOL:
        print(f"\n  !! The largest departure from a weight of 1.0 is {worst:.4f}, above the "
              f"{UNIFORM_TOL} this script calls uniform. This arm is NOT an unweighted "
              f"reference and must not be described as one.")
    else:
        print(f"\n     Largest departure from 1.0 across all five folds: {worst:.4f}, within "
              f"{UNIFORM_TOL}. This arm applied numerically uniform weights and is the "
              f"no-imbalance-correction reference.")

    # -- step 3, pool and merge -------------------------------------------------
    ref_img, ref_scaled = pool_reference(dirs)
    ref_part = participant_frame(ref_img)
    full = Cohort("full", FULL_EXPERIMENTS, full_dir)
    print(f"\n  STEP 3. Cohorts.")
    print(f"     full cohort arms: {len(full.arms)} from {full_dir}")
    coh = merged_cohort(full, ref_img, ref_part, info[0]["code_fingerprint"])
    print(f"     merged: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images. Reference arm labelled '{REFERENCE_ARM}'.")
    aware = [a for a in coh.order() if a != REFERENCE_ARM]
    if ref_scaled is None:
        print("  !! the reference arm has no temperature-scaled predictions for every fold; "
              "the post-temperature calibration columns are reported as NaN")

    # -- metadata references for R4 ---------------------------------------------
    meta_raw = pd.read_excel(metadata_xlsx())
    meta_raw["pid"] = meta_raw["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta_raw = meta_raw.drop_duplicates("pid").set_index("pid")
    fitter = MetaFitter(coh.participants, coh.labels, coh.folds, meta_raw)
    meta_probs: Dict[str, np.ndarray] = {}
    meta_miss: Dict[str, Dict[str, int]] = {}
    for name, cols in [("META age only", ["Age"]), ("META sex only", ["M/F"]),
                       ("META nWBV only", ["nWBV"]), ("META eTIV only", ["eTIV"]),
                       ("META age+sex+eTIV+nWBV", ["Age", "M/F", "eTIV", "nWBV"])]:
        Q, miss = fitter.fit(cols)
        meta_probs[name] = Q
        meta_miss[name] = miss
    print(f"     metadata references for R4: {', '.join(meta_probs)} "
          f"(MMSE models omitted: missing for 147 of 347 on this cohort)")

    # -- R1, R3, R4 via a23's own functions -------------------------------------
    print(f"\n  STEP 4. Protocol requirements R1, R3 and R4, using a23's own functions.")
    long: List[Dict] = []
    boots = build_boots(coh, meta_probs, n_boot)
    r1 = requirement_1(coh, boots, long)
    r3, unit_flags = requirement_3(coh, boots, long)
    r4, r4_counted, r4_dropped = requirement_4(coh, meta_probs, meta_miss, boots, long)
    print(f"     R2 is NOT AVAILABLE. The permuted-label replicates were trained on the "
          f"RESTRICTED cohort, so no measured null exists for these 347 participants. "
          f"Borrowing the restricted null would compare against a different label "
          f"distribution and a different task.")
    long_df = pd.DataFrame(long)
    # The cohort is not called "full", so a23's family lookups should already have
    # returned nothing. Checked rather than overwritten: silently stamping every row
    # "undeclared" would hide it if one of them had picked up a declared threshold.
    stray = sorted(set(long_df["family"].dropna().unique()) - {UNDECLARED})
    if stray:
        raise SystemExit(f"FATAL. Rows in this exploratory experiment picked up declared "
                         f"families {stray}. Those declarations were sized before this "
                         f"reference arm existed and cannot absorb it.")
    long_df["cohort"] = COHORT_NAME
    long_df.to_csv(OUT / f"{TABLE}_long.csv", index=False, float_format="%.6f")
    print(f"    -> {TABLE}_long.csv  ({len(long_df):,} comparisons)")

    # -- Table 4 metrics and calibration ----------------------------------------
    print(f"\n  STEP 5. Table 4 metrics and calibration, both units.")
    temps = [r["temperature_fitted_on_validation"] for r in info]
    table4: List[Dict] = []
    for arm in coh.order():
        for unit in UNITS:
            frame = coh.unit(unit)[arm]
            m = SuffStats(frame).metrics()
            f1 = m["per_class_f1"]
            if arm == REFERENCE_ARM:
                sc = ref_scaled if unit == "image" else (
                    participant_frame(ref_scaled) if ref_scaled is not None else None)
                mean_t = float(np.mean([t for t in temps if t is not None])) if any(
                    t is not None for t in temps) else float("nan")
            else:
                raw_sc = coh.scaled.get(arm)
                sc = raw_sc if unit == "image" else (
                    participant_frame(raw_sc) if raw_sc is not None else None)
                sub = full.runs[full.runs["arm"] == arm] if "arm" in full.runs else None
                mean_t = float("nan")
                if sub is not None and not sub.empty:
                    vals = []
                    for d in sub["run_dir"]:
                        try:
                            mt = json.loads((Path(d) / "metrics.json").read_text(
                                encoding="utf-8"))
                            vals.append(mt["test"]["temperature_scaling"]["temperature"])
                        except Exception:                                  # noqa: BLE001
                            pass
                    mean_t = float(np.mean(vals)) if vals else float("nan")
            table4.append({
                "arm": arm, "unit": unit,
                "is_unweighted_reference": arm == REFERENCE_ARM,
                "accuracy_percent": m["accuracy"], "macro_f1": m["macro_f1"],
                **{f"f1_{SHORT[c]}": float(f1[c]) for c in range(N_CLASSES)},
                "qwk": qwk_from_cm(SuffStats(frame).metrics()["confusion_matrix"]),
                "ece": m["ece"], "mce": m["mce"], "brier": m["brier"],
                "ece_after_temperature": (ece_of(sc) if sc is not None else float("nan")),
                "brier_after_temperature": (float(SuffStats(sc).metrics()["brier"])
                                            if sc is not None else float("nan")),
                "mean_temperature_fitted_on_validation": mean_t,
                "n": m["n_participants"] if unit == "participant" else m["n_images"],
                "code_fingerprint": coh.fingerprint.get(arm, ""),
                "family": UNDECLARED,
            })
    t4 = pd.DataFrame(table4)

    # -- the headline comparison -------------------------------------------------
    print(f"\n  STEP 6. Paired bootstrap of every imbalance-aware arm against the reference.")
    stats = {a: SuffStats(coh.part[a]) for a in coh.order()}
    comp: List[Dict] = []
    for key, label, per_class in BOOT_METRICS:
        seed = seed_for(BASE_SEED, "a28:compare", metric=key, unit="participant")
        if key == "qwk":
            boot = qwk_bootstrap(stats, n_boot, seed)
        else:
            boot = clustered_bootstrap(stats, "macro_f1" if per_class is not None else key,
                                       n_boot=n_boot, seed=seed, per_class=per_class)
        for arm in aware:
            d = paired_difference(boot, arm, REFERENCE_ARM, family=None)
            comp.append({"arm": arm, "reference": REFERENCE_ARM, "unit": "participant",
                         "metric": label, "arm_value": boot[arm]["point"],
                         "reference_value": boot[REFERENCE_ARM]["point"],
                         "difference": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                         "ci_level": d["ci_level"], "p_bootstrap": d["p_bootstrap"],
                         "p_at_resolution_floor": d["p_at_resolution_floor"],
                         "excludes_zero": d["excludes_zero"], "n_bootstrap": n_boot,
                         "family": UNDECLARED,
                         "code_fingerprint": coh.fingerprint.get(arm, ""),
                         "note": "arm minus the unweighted reference, positive favours the arm"})
    comp_df = pd.DataFrame(comp)

    # -- one output table --------------------------------------------------------
    t4_out = t4.copy()
    t4_out.insert(0, "kind", "table4")
    c_out = comp_df.copy()
    c_out.insert(0, "kind", "comparison")
    combined = pd.concat([t4_out, c_out], ignore_index=True)
    combined.insert(0, "cohort", COHORT_NAME)
    save_table(combined, TABLE, float_fmt="%.6f")
    save_table(pd.DataFrame(info).drop(columns=["run_dir"]), f"{TABLE}_provenance",
               float_fmt="%.6f")

    # -- one screen --------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 100)
    print("a28  WHAT THE SAME NETWORK DOES WITH NO IMBALANCE CORRECTION AT ALL")
    print("=" * 100)
    print(f"  Reference arm: exp02 at beta = 0.999, weights {info[0]['criterion_weights_recomputed']}, "
          f"largest departure from 1.0 is {worst:.4f}.")
    print(f"  {coh.n_participants} participants, {coh.n_images:,} images, same folds and same "
          f"people as every other full-cohort arm.")

    print(f"\n  TABLE 4 METRICS, participant unit")
    print(f"     {'arm':<38}{'acc%':>8}{'macroF1':>9}{'F1>=1':>8}{'QWK':>8}{'ECE':>8}"
          f"{'ECE post-T':>12}{'mean T':>8}")
    for _, r in t4[t4["unit"] == "participant"].iterrows():
        mark = "  <== reference" if r["is_unweighted_reference"] else ""
        print(f"     {r['arm']:<38}{r['accuracy_percent']:>8.2f}{r['macro_f1']:>9.4f}"
              f"{r['f1_Dem']:>8.4f}{r['qwk']:>8.4f}{r['ece']:>8.4f}"
              f"{r['ece_after_temperature']:>12.4f}{r['mean_temperature_fitted_on_validation']:>8.3f}"
              f"{mark}")
    print(f"\n  TABLE 4 METRICS, image unit")
    print(f"     {'arm':<38}{'acc%':>8}{'macroF1':>9}{'F1>=1':>8}{'QWK':>8}{'ECE':>8}"
          f"{'ECE post-T':>12}")
    for _, r in t4[t4["unit"] == "image"].iterrows():
        mark = "  <== reference" if r["is_unweighted_reference"] else ""
        print(f"     {r['arm']:<38}{r['accuracy_percent']:>8.2f}{r['macro_f1']:>9.4f}"
              f"{r['f1_Dem']:>8.4f}{r['qwk']:>8.4f}{r['ece']:>8.4f}"
              f"{r['ece_after_temperature']:>12.4f}{mark}")

    print(f"\n  EVERY IMBALANCE-AWARE ARM AGAINST THE UNWEIGHTED REFERENCE, participant unit")
    print(f"  Positive means the imbalance-aware arm is better. * marks an interval "
          f"excluding zero.")
    print(f"     {'arm':<20}" + "".join(f"{lab:>26}" for _, lab, _ in BOOT_METRICS))
    for arm in aware:
        cells = []
        for _k, lab, _p in BOOT_METRICS:
            r = comp_df[(comp_df["arm"] == arm) & (comp_df["metric"] == lab)].iloc[0]
            star = "*" if r["excludes_zero"] else " "
            cells.append(f"{r['difference']:>+9.4f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{star}")
        print(f"     {arm:<20}" + "".join(f"{c:>26}" for c in cells))

    n_sig = int(comp_df["excludes_zero"].sum())
    print(f"\n  BOTTOM LINE. {n_sig} of {len(comp_df)} arm-by-metric comparisons have an "
          f"interval excluding zero.")
    for _k, lab, _p in BOOT_METRICS:
        sub = comp_df[comp_df["metric"] == lab]
        better = int((sub["difference"] > 0).sum())
        sig = int(sub["excludes_zero"].sum())
        print(f"     {lab:<12} {better} of {len(sub)} arms score above the unweighted "
              f"reference, {sig} of them with an interval excluding zero")

    print(f"\n  PROTOCOL REQUIREMENTS for the reference arm (a23's own functions, both units)")
    for metric in ("accuracy", "macro_f1"):
        for unit in UNITS:
            v = r1.get((REFERENCE_ARM, f"{unit}:{metric}"), False)
            print(f"     R1 {metric:<9} {unit:<12} {'PASS' if v else 'FAIL'} against the "
                  f"constant majority predictor")
    print(f"     R2 {'NA':<9} {'both':<12} no measured permuted-label null exists for the "
          f"full cohort")
    print(f"     R3 {'':<9} {'both':<12} "
          f"{'PASS' if r3.get(REFERENCE_ARM, False) else 'FAIL'} units agree on the sign "
          f"against the constant predictor")
    print(f"     R4 {'':<9} {'participant':<12} "
          f"{'PASS' if r4.get(REFERENCE_ARM, False) else 'FAIL'} against every interpretable "
          f"metadata reference on macro-F1")
    if r4_dropped:
        print(f"        references excluded as not interpretable: {', '.join(r4_dropped)}")
    if unit_flags:
        print(f"\n  UNIT DISAGREEMENTS ({len(unit_flags)}), first few:")
        for f in unit_flags[:6]:
            print(f"     {f}")

    print(f"\n  ON THE TEMPERATURE. Every temperature above was fitted on that fold's own")
    print(f"  VALIDATION split and is read from metrics.json. Nothing was refitted on the test")
    print(f"  split here. A temperature fitted on the data it is then scored with always")
    print(f"  improves ECE and is not comparable with the paper's numbers.")
    print(f"\n  EVERYTHING in this file is UNDECLARED. The declared arm-vs-constant and")
    print(f"  arm-vs-metadata families were sized before this reference arm existed.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting anything.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "every imbalance-aware full-cohort arm against the superseded beta=0.999 "
        "class-balanced arm, whose weights are numerically uniform, as a "
        "no-imbalance-correction reference",
        "accuracy, macro-F1, F1 for CDR>=1, QWK, plus Table 4 metrics and calibration",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
