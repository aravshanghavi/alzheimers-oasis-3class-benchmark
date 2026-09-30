#!/usr/bin/env python3
"""Pre-flight check for the permuted-label null. No GPU, no training, ~1 minute.

Run this in the conda env BEFORE launching run_permnull.py. It exercises the
real scan, the real cohort filter, the real splitter and the real permutation
module, then asserts every invariant the null depends on.

What is asserted as exact:
  - the cohort is the 166-participant age-60 cohort at 85 / 58 / 23
  - splits are participant-disjoint and the five test folds partition the cohort
  - test-fold labels are untouched
  - PARTICIPANT-level class marginals of train and val are preserved exactly
  - one label per participant survives the permutation
  - the same seed reproduces, different seeds differ
  - an invalid scope is refused

What is reported but NOT asserted as exact:
  - IMAGE-level class marginals. A participant-level permutation cannot hold
    these, because participants contribute unequal image counts (183, 244 or
    366 here), so exchanging two participants' labels moves images between
    classes. A few percent of drift is expected and correct. It is gated at a
    loose tolerance to catch a genuine bug, and the realised values are printed
    so you can see them.

    conda activate alzheimers
    python tools\\verify_permutation.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_experiment_config, validate_config        # noqa: E402
from src.cohort import apply_cohort_filter                            # noqa: E402
from src.data import scan_dataset, make_splits, assert_group_disjoint  # noqa: E402
from src.permute import (apply_label_permutation, PermutationError,    # noqa: E402
                         IMAGE_DRIFT_TOLERANCE)
from src.runner import config_for_spec, RunSpec                        # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("verify")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    cfg = load_experiment_config("exp14_permnull_wce.yaml")
    spec = RunSpec(loss="WCE", fold=0, init_seed=42)
    cfg = config_for_spec(cfg, spec)
    cfg.setdefault("permutation", {})["seed"] = 1
    validate_config(cfg, require_data_root=True)
    n_classes = len(cfg["data"]["new_classes"])

    print("\nScanning dataset (this walks the image tree once)...")
    df = scan_dataset(cfg["data"]["root"], cfg["data"]["original_classes"], cfg["data"]["class_map"])
    print(f"  scanned {len(df):,} images, {df['participant_id'].nunique()} participants")

    df, _cohort_info = apply_cohort_filter(df, cfg, log)
    n_part = int(df["participant_id"].nunique())
    counts = df.groupby("participant_id")["class_3"].first().value_counts().sort_index().tolist()
    print(f"\nCohort after restriction: {n_part} participants, {len(df):,} images, classes {counts}")
    check("cohort is the 166-participant age-60 cohort", n_part == 166, f"got {n_part}")
    check("class counts are 85 / 58 / 23", counts == [85, 58, 23], f"got {counts}")

    print(f"\nPer-fold permutation checks"
          f"  (image-drift tolerance {IMAGE_DRIFT_TOLERANCE:.0%}, drift is expected):")
    all_test_parts: list[set] = []
    worst_drift = 0.0

    for fold in range(cfg["split"]["n_splits"]):
        cfg["split"]["fold"] = fold
        splits = make_splits(df, cfg)
        ar = assert_group_disjoint(df, splits, cfg["split"]["group_by"], allow_leakage=False)
        check(f"fold {fold}: splits are participant-disjoint", ar == "PASS",
              "" if ar == "PASS" else ar.splitlines()[0])

        before_all = df["class_3"].to_numpy().copy()
        try:
            pdf, info = apply_label_permutation(df, splits, cfg, log)
        except PermutationError as e:
            check(f"fold {fold}: permutation applied", False, str(e).splitlines()[0])
            continue
        after_all = pdf["class_3"].to_numpy()

        te = np.asarray(splits["test"], dtype=int)
        check(f"fold {fold}: test labels untouched",
              bool(np.array_equal(before_all[te], after_all[te])))

        for sname in ("train", "val"):
            idx = np.asarray(splits[sname], dtype=int)
            sub_b = df.iloc[idx].groupby("participant_id")["class_3"].first().to_numpy()
            sub_a = pdf.iloc[idx].groupby("participant_id")["class_3"].first().to_numpy()
            pb = np.bincount(sub_b, minlength=n_classes)
            pa = np.bincount(sub_a, minlength=n_classes)
            check(f"fold {fold}: {sname} PARTICIPANT class marginal preserved",
                  bool(np.array_equal(pb, pa)), f"{pb.tolist()} -> {pa.tolist()}")

            nun = pdf.iloc[idx].groupby("participant_id")["class_3"].nunique()
            check(f"fold {fold}: {sname} one label per participant after permutation",
                  bool((nun == 1).all()))

            d = info["per_split"][sname]
            worst_drift = max(worst_drift, d["image_drift_max_rel"])
            print(f"        {sname:<5} images {d['image_counts_before']} -> "
                  f"{d['image_counts_after']}, drift {d['image_drift_max_rel']:.3f}; "
                  f"unchanged {d['labels_unchanged']}/{d['participants']} "
                  f"({d['labels_unchanged_fraction']:.3f}, predicts "
                  f"{d['expected_unchanged_fraction']:.3f})")

        changed = int((before_all != after_all).sum())
        check(f"fold {fold}: some labels actually changed", changed > 0,
              f"{changed:,} image rows changed")

        all_test_parts.append(set(pdf.iloc[te]["participant_id"].unique()))

    if all_test_parts:
        union = set().union(*all_test_parts)
        total = sum(len(s) for s in all_test_parts)
        check("the five test folds exactly partition the cohort",
              len(union) == 166 and total == 166, f"union={len(union)} sum={total}")

    print(f"\n  worst image-level drift across all folds and splits: {worst_drift:.3f} "
          f"(tolerance {IMAGE_DRIFT_TOLERANCE:.2f})")
    print("  This is expected. Participants contribute 183, 244 or 366 images, so exchanging")
    print("  two participants' labels moves images between classes while leaving the")
    print("  participant counts exactly as they were. Class weights in a permuted run")
    print("  therefore differ from its real counterpart by about this much, which is")
    print("  recorded in each manifest and is immaterial for a noise floor.")

    print("\nDeterminism check (same seed twice, different seeds differ):")
    cfg["split"]["fold"] = 0
    splits = make_splits(df, cfg)
    cfg["permutation"]["seed"] = 1
    a1, _ = apply_label_permutation(df, splits, cfg, log)
    a2, _ = apply_label_permutation(df, splits, cfg, log)
    cfg["permutation"]["seed"] = 2
    b1, _ = apply_label_permutation(df, splits, cfg, log)
    check("seed 1 is reproducible",
          bool(np.array_equal(a1["class_3"].to_numpy(), a2["class_3"].to_numpy())))
    check("seed 2 differs from seed 1",
          not bool(np.array_equal(a1["class_3"].to_numpy(), b1["class_3"].to_numpy())))

    print("\nRefusal checks:")
    cfg["permutation"]["scope"] = "all"
    try:
        apply_label_permutation(df, splits, cfg, log)
        check("invalid scope is refused", False, "it was accepted, which is wrong")
    except PermutationError:
        check("invalid scope is refused", True, "PermutationError")
    cfg["permutation"]["scope"] = "train_val"

    saved = cfg["permutation"].pop("seed", None)
    try:
        apply_label_permutation(df, splits, cfg, log)
        check("missing seed is refused", False, "it was accepted, which is wrong")
    except PermutationError:
        check("missing seed is refused", True, "PermutationError")
    if saved is not None:
        cfg["permutation"]["seed"] = saved

    cfg["permutation"]["enabled"] = False
    noop_df, noop_info = apply_label_permutation(df, splits, cfg, log)
    check("disabled permutation is a true no-op",
          noop_info == {"enabled": False}
          and bool(np.array_equal(noop_df["class_3"].to_numpy(), df["class_3"].to_numpy())))
    cfg["permutation"]["enabled"] = True

    print()
    if FAILURES:
        print(f"NOT READY. {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nDo not launch run_permnull.py until these pass.")
        return 1
    print("READY. All checks passed. You can launch:  python run_permnull.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
