"""Participant-level label permutation, for measuring the pipeline's noise floor.

WHY THIS EXISTS

The headline result of this project is a null: on the CDR-assessed age-60 cohort
no training objective's accuracy is distinguishable from a constant
majority-class predictor, while macro-F1 sits well above it. A null result needs
a measured floor. Without one the paper asserts how much apparent signal the
pipeline produces from nothing, rather than measuring it.

Permuting the model's OUTPUT predictions is not the same test. That holds the
trained model fixed and cannot capture the optimism injected by the pipeline
itself: five-fold cross-validation, checkpoint selection on validation accuracy,
temperature fitting on the same validation split, and early stopping. This
module permutes the LABELS the model learns from, and then the complete
unmodified pipeline runs on top, so the resulting distribution absorbs all of it.

THREE DESIGN DECISIONS, EACH OF WHICH MATTERS

1. Permutation is at the PARTICIPANT level, never the image level. Slices from
   one participant are near-duplicates (r = 0.988 between acquisitions in this
   dataset). Permuting per image would hand the model a bag of contradictory
   labels for visually identical inputs, which is a different and much easier
   task to fail at. Every image of a participant carries that participant's
   permuted label.

2. Only TRAIN and VAL are permuted. The test fold keeps its true labels, so the
   metric is still computed against reality and is directly comparable to the
   real runs. This is asserted, not assumed: the function refuses to return if a
   single test-fold label moved.

3. Train and val are permuted SEPARATELY, each within itself, so the
   PARTICIPANT-level class marginal of each split is preserved exactly. That is
   asserted below and it is the invariant a participant-level permutation
   actually guarantees.

WHAT IS NOT PRESERVED, AND WHY THAT IS CORRECT

The IMAGE-level class marginal shifts slightly. It has to. Participants in this
cohort contribute unequal numbers of images (183, 244 or 366), so exchanging the
labels of a 183-image participant and a 244-image participant moves 61 images
between classes even though the participant counts are untouched. Observed drift
across the five folds is a few percent of a class's image count.

This has one consequence worth stating plainly, because an earlier version of
this file's documentation got it wrong. `compute_class_weights` in this codebase
is computed from IMAGE counts in the training fold, so a permuted run does NOT
see class weights identical to its real counterpart. They differ by roughly the
same few percent. For a noise floor that is immaterial, and it is recorded in
the manifest (`permutation.per_split.<split>.image_counts_before/after` and
`image_drift_max_rel`) rather than being left for a reader to discover.

Forcing the image marginal to hold exactly would mean permuting only within
blocks of equal image count. That would tie a participant's label to its
acquisition count and stop the result being a permutation null. It is not done.

The permutation is a derangement only by chance. Some participants keep their
true label, at the rate the class marginal implies. That is correct: forcing a
derangement would make the null harder than chance rather than equal to it. The
realised fraction of unchanged labels is recorded next to the value the marginal
predicts, so the reader can confirm it sits where it should.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# A participant-level permutation cannot hold the image marginal exactly (see the
# module docstring). This gate exists to catch a genuine bug, not to police the
# expected drift, so it is deliberately loose. Observed values are a few percent.
IMAGE_DRIFT_TOLERANCE = 0.25


class PermutationError(RuntimeError):
    pass


def permutation_requested(cfg: Dict) -> bool:
    return bool((cfg.get("permutation") or {}).get("enabled", False))


def apply_label_permutation(df: pd.DataFrame, splits: Dict[str, np.ndarray],
                            cfg: Dict, log) -> Tuple[pd.DataFrame, Dict]:
    """Return (df with permuted train/val labels, provenance dict for the manifest).

    A no-op returning enabled=False when permutation is not requested, so the
    call site is unconditional and a non-permuted run in this tree behaves
    exactly as it would in a tree without this module.
    """
    pcfg = cfg.get("permutation") or {}
    if not pcfg.get("enabled", False):
        return df, {"enabled": False}

    if "seed" not in pcfg or pcfg["seed"] is None:
        raise PermutationError(
            "permutation.enabled is true but permutation.seed is unset. "
            "A permutation null with an unrecorded seed is not reproducible.")
    seed = int(pcfg["seed"])
    scope = pcfg.get("scope", "train_val")
    if scope != "train_val":
        raise PermutationError(
            f"permutation.scope must be 'train_val', got {scope!r}. Permuting the test "
            "fold would measure nothing, because the metric would then be computed "
            "against permuted labels rather than against reality.")

    for required in ("train", "val", "test"):
        if required not in splits:
            raise PermutationError(f"splits has no {required!r} key; cannot permute.")

    n_classes = len(cfg["data"]["new_classes"])
    pid_all = df["participant_id"].to_numpy()
    lab_all = df["class_3"].to_numpy()
    new_lab = lab_all.copy()

    rng = np.random.default_rng(seed)
    per_split: Dict[str, Dict] = {}

    for split_name in ("train", "val"):
        rows = np.asarray(splits[split_name], dtype=int)
        if rows.size == 0:
            raise PermutationError(f"{split_name} split is empty; cannot permute.")
        sel = np.zeros(len(df), dtype=bool)
        sel[rows] = True

        sub = df.iloc[rows]
        nun = sub.groupby("participant_id")["class_3"].nunique()
        if (nun > 1).any():
            bad = nun[nun > 1].index.tolist()[:5]
            raise PermutationError(
                f"{int((nun > 1).sum())} participant(s) in {split_name} carry more than one "
                f"class label, e.g. {bad}. Permuting at participant level assumes one label "
                "per participant. Resolve the label conflict before running a permuted arm.")

        part_lab = sub.groupby("participant_id")["class_3"].first()
        parts = part_lab.index.to_numpy()
        labs = part_lab.to_numpy()

        shuffled = labs[rng.permutation(len(labs))]

        # The invariant a participant-level permutation guarantees. Exact.
        p_before = np.bincount(labs, minlength=n_classes)
        p_after = np.bincount(shuffled, minlength=n_classes)
        if not np.array_equal(p_before, p_after):
            raise PermutationError(
                f"{split_name} PARTICIPANT class marginal changed under permutation: "
                f"{p_before.tolist()} -> {p_after.tolist()}. This is a bug in this module, "
                "not a data problem; a permutation cannot change a marginal.")

        img_before = np.bincount(lab_all[rows], minlength=n_classes)
        for p, lab in zip(parts, shuffled):
            new_lab[sel & (pid_all == p)] = lab
        img_after = np.bincount(new_lab[rows], minlength=n_classes)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(img_after - img_before) / np.maximum(img_before, 1)
        drift = float(rel.max())
        if drift > IMAGE_DRIFT_TOLERANCE:
            raise PermutationError(
                f"{split_name} IMAGE class marginal moved by {drift:.1%}, above the "
                f"{IMAGE_DRIFT_TOLERANCE:.0%} tolerance: {img_before.tolist()} -> "
                f"{img_after.tolist()}. Some drift is expected because participants "
                "contribute unequal image counts, but this much suggests a bug.")

        unchanged = int((shuffled == labs).sum())
        per_split[split_name] = {
            "participants": int(len(parts)),
            "images": int(rows.size),
            "participant_counts_before": p_before.tolist(),
            "participant_counts_after": p_after.tolist(),
            "image_counts_before": img_before.tolist(),
            "image_counts_after": img_after.tolist(),
            "image_drift_max_rel": round(drift, 6),
            "labels_unchanged": unchanged,
            "labels_unchanged_fraction": round(unchanged / max(len(parts), 1), 4),
            "expected_unchanged_fraction": round(
                float(sum((c / len(parts)) ** 2 for c in p_before)), 4),
        }

    test_rows = np.asarray(splits["test"], dtype=int)
    if not np.array_equal(new_lab[test_rows], lab_all[test_rows]):
        raise PermutationError("test-fold labels were modified. Refusing to continue.")

    out = df.copy()
    out["class_3"] = new_lab

    info = {
        "enabled": True,
        "seed": seed,
        "scope": scope,
        "unit": "participant",
        "per_split": per_split,
        "test_labels_untouched": True,
        "note": ("Participant-level class marginals are preserved exactly. Image-level "
                 "marginals drift by a few percent because participants contribute "
                 "unequal image counts, so class weights differ slightly from the "
                 "corresponding real run."),
    }

    log.info("PERMUTATION  labels permuted at participant level, seed=%d, scope=%s", seed, scope)
    for name, d in per_split.items():
        log.info("  %-5s %3d participants, %6d images, participant counts %s (preserved), "
                 "unchanged %d/%d (%.3f, marginal predicts %.3f)",
                 name, d["participants"], d["images"], d["participant_counts_before"],
                 d["labels_unchanged"], d["participants"],
                 d["labels_unchanged_fraction"], d["expected_unchanged_fraction"])
        log.info("        image counts %s -> %s, max relative drift %.3f (expected, see module docstring)",
                 d["image_counts_before"], d["image_counts_after"], d["image_drift_max_rel"])
    log.info("  test fold retains TRUE labels: %d images untouched", len(test_rows))
    return out, info
