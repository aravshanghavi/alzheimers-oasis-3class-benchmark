"""Cohort restriction, applied after scanning and before splitting.

WHY THIS EXISTS
---------------
Every "age60" number produced so far came from a model TRAINED on the full
cohort -- including the 147 non-demented participants with no recorded CDR and
the 181 aged under 60 -- and merely EVALUATED on the clean subset. That is a
confound inside the confound analysis: it cannot separate "the signal survives
restriction" from "a model that learned an age contrast still ranks the old
participants correctly".

Training on the restricted cohort is the honest comparison, and it also changes
the imbalance regime, which makes it a mechanism test for FA-FL. On the full
cohort the class split is roughly 77/17/7 and FA-FL's frequency-weighted
effective gamma is about 1.86. Restricting to CDR-assessed participants aged 60
and over leaves 85/58/23, so the majority falls to about 51% and the effective
gamma rises. If FA-FL is simply focal loss at its effective gamma, the focal arm
it most resembles should MOVE with the cohort. That is a falsifiable prediction
and this module is what makes it testable.

DESIGN
------
src/data.py is left byte-identical to the 2026-08 version so that any change in
results is attributable to the cohort and the configuration rather than to a
quietly edited data path. The filter therefore lives here and is applied to the
scanned frame by the runner, before make_splits, so class weights, splits and
every downstream count follow automatically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_METADATA = "SREP_METADATA_XLSX"

# OASIS ids in the metadata are per-session (OAS1_0001_MR1); the cohort is keyed
# on the participant (OAS1_0001).
_SESSION_SUFFIX = r"_MR\d+$"


class CohortError(RuntimeError):
    pass


def find_metadata() -> Path:
    env = os.environ.get(ENV_METADATA)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise CohortError(f"{ENV_METADATA} is set to '{env}' but that file does not exist.")
        return p.resolve()
    for c in [REPO_ROOT / "oasis_cross-sectional.xlsx",
              REPO_ROOT / "oasis_cross-sectional.csv",
              *sorted(REPO_ROOT.parent.glob("*/oasis_cross-sectional.xlsx"))]:
        if c.is_file():
            return c.resolve()
    raise CohortError(
        "cohort restriction needs oasis_cross-sectional.xlsx (Age and CDR) and it was not "
        f"found in {REPO_ROOT} or one level up. Set {ENV_METADATA} to its full path.")


def load_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path else find_metadata()
    raw = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    idcol = next((c for c in raw.columns if c.strip().upper() in ("ID", "SUBJECT ID", "SUBJECT_ID")),
                 raw.columns[0])
    agecol = next((c for c in raw.columns if c.strip().upper() == "AGE"), None)
    cdrcol = next((c for c in raw.columns if c.strip().upper() == "CDR"), None)
    if agecol is None or cdrcol is None:
        raise CohortError(f"{path} lacks an Age or CDR column; found {list(raw.columns)}")
    out = pd.DataFrame({
        "participant_id": raw[idcol].astype(str).str.replace(_SESSION_SUFFIX, "", regex=True),
        "age": pd.to_numeric(raw[agecol], errors="coerce"),
        "cdr": pd.to_numeric(raw[cdrcol], errors="coerce"),
    })
    # A participant with two sessions appears twice; the demographics are the same.
    return out.drop_duplicates("participant_id").reset_index(drop=True)


def apply_cohort_filter(df: pd.DataFrame, cfg: Dict,
                        log=None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Restrict the scanned frame to the configured cohort.

    Returns (filtered_frame, info). `info` goes verbatim into the manifest so the
    cohort a run was trained on is recoverable months later without rerunning
    anything.
    """
    ccfg = (cfg.get("data", {}) or {}).get("cohort") or {}
    mode = str(ccfg.get("filter", "none")).lower()

    before_p = int(df["participant_id"].nunique())
    before_i = int(len(df))
    info: Dict[str, object] = {
        "filter": mode,
        "participants_before": before_p, "images_before": before_i,
        "participants_after": before_p, "images_after": before_i,
        "participants_removed": 0, "images_removed": 0,
    }

    if mode in ("none", "", "full"):
        if log:
            log.info("COHORT  no restriction: %d participants, %s images",
                     before_p, f"{before_i:,}")
        return df, info

    if mode != "cdr_age":
        raise CohortError(f"unknown data.cohort.filter {mode!r}; expected 'none' or 'cdr_age'")

    min_age = float(ccfg.get("min_age", 60))
    require_cdr = bool(ccfg.get("require_cdr", True))
    meta_path = ccfg.get("metadata_path")
    meta = load_metadata(Path(meta_path) if meta_path else None)

    present = df[["participant_id", "class_3"]].drop_duplicates("participant_id")
    merged = present.merge(meta, on="participant_id", how="left")

    missing = merged["age"].isna().sum()
    if missing:
        raise CohortError(
            f"{missing} of {len(merged)} participants in the scanned cohort have no row in the "
            f"metadata table, so an age restriction cannot be applied to them without silently "
            f"dropping people. Resolve the id mismatch before running a restricted arm.")

    keep_mask = merged["age"] >= min_age
    if require_cdr:
        keep_mask &= merged["cdr"].notna()
    keep = set(merged.loc[keep_mask, "participant_id"])

    out = df[df["participant_id"].isin(keep)].reset_index(drop=True)
    if out.empty:
        raise CohortError(f"cohort filter {mode!r} removed every participant")

    per_class_before = present["class_3"].value_counts().sort_index().to_dict()
    per_class_after = (out.drop_duplicates("participant_id")["class_3"]
                       .value_counts().sort_index().to_dict())
    counts = np.array([per_class_after.get(c, 0) for c in sorted(per_class_before)], dtype=float)
    majority_share = float(counts.max() / counts.sum()) if counts.sum() else float("nan")

    info.update({
        "min_age": min_age, "require_cdr": require_cdr,
        "metadata": str(meta_path) if meta_path else str(find_metadata()),
        "participants_after": int(out["participant_id"].nunique()),
        "images_after": int(len(out)),
        "participants_removed": before_p - int(out["participant_id"].nunique()),
        "images_removed": before_i - int(len(out)),
        "participants_per_class_before": {int(k): int(v) for k, v in per_class_before.items()},
        "participants_per_class_after": {int(k): int(v) for k, v in per_class_after.items()},
        "majority_participant_share_after": majority_share,
    })

    if log:
        log.info("COHORT  filter=%s min_age=%g require_cdr=%s", mode, min_age, require_cdr)
        log.info("  participants %d -> %d   (removed %d)",
                 before_p, info["participants_after"], info["participants_removed"])
        log.info("  images       %s -> %s", f"{before_i:,}", f"{info['images_after']:,}")
        log.info("  per class    %s -> %s", per_class_before, per_class_after)
        log.info("  majority participant share after restriction: %.4f", majority_share)
        log.info("  NOTE: class weights, and therefore FA-FL's gamma_t, are computed from the "
                 "TRAINING FOLD of this restricted cohort and are logged per run.")
    return out, info
