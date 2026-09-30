#!/usr/bin/env python3
"""Discovery layer: turn run folders into a DataFrame.

Manifests are the source of truth. There is no shared mutable index file -- a
single append-only CSV written by 35 sequential runs will eventually be
corrupted by a crash mid-write, whereas a per-run manifest cannot be damaged by
another run. RUNS_INDEX.csv is regenerated from manifests as a browsable view.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
OUTPUTS = REPO_ROOT / "analysis" / "outputs"


class FingerprintMismatch(UserWarning):
    pass


def load_runs(experiment: Optional[str] = None, status: str = "COMPLETE",
              strict_fingerprint: bool = False) -> pd.DataFrame:
    pattern = f"{experiment}/outputs/*/manifest.json" if experiment else "*/outputs/*/manifest.json"
    rows: List[Dict] = []
    for mf in sorted(EXPERIMENTS.glob(pattern)):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] unreadable manifest {mf}: {exc}", file=sys.stderr)
            continue
        if status and m.get("status") != status:
            continue
        row = {
            "run_id": m["run_id"], "run_dir": str(mf.parent),
            "experiment": m.get("experiment"), "loss": m.get("loss", {}).get("type"),
            "split_strategy": m.get("split", {}).get("strategy"),
            "group_by": m.get("split", {}).get("group_by"),
            "fold": m.get("split", {}).get("fold"),
            "split_seed": m.get("split", {}).get("split_seed"),
            "init_seed": m.get("init_seed"),
            "context_slices": m.get("context_slices"),
            "code_fingerprint": m.get("code_fingerprint"),
            "config_fingerprint": m.get("config_fingerprint"),
            "duration_seconds": m.get("duration_seconds"),
            "status": m.get("status"),
            "group_disjoint": m.get("data", {}).get("group_disjoint_assert"),
            "test_participants": m.get("data", {}).get("participants", {}).get("test"),
            "test_images": m.get("data", {}).get("images", {}).get("test"),
            "epochs_run": m.get("training", {}).get("epochs_run"),
            "best_epoch": m.get("training", {}).get("best_epoch"),
        }
        row.update(m.get("headline", {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    fps = df["code_fingerprint"].dropna().unique()
    if len(fps) > 1:
        msg = ["", "  !! runs span %d different CODE fingerprints "
                   "(config differences are expected and not flagged):" % len(fps)]
        for fp in fps:
            sub = df[df["code_fingerprint"] == fp]
            msg.append(f"     {fp[:8]}  {len(sub):>3} run(s)  {sorted(sub['experiment'].unique())}")
        msg.append("     Aggregating across differing code is how silent inconsistencies")
        msg.append("     enter a paper. Re-run or filter before building tables.")
        print("\n".join(msg), file=sys.stderr)
        if strict_fingerprint:
            raise RuntimeError("refusing to aggregate across differing code fingerprints")
    return df.sort_values(["experiment", "loss", "fold", "init_seed"]).reset_index(drop=True)


def load_predictions(run_dir: str | Path, split: str = "test") -> pd.DataFrame:
    """One row per image, with the identifiers needed for participant-clustered
    resampling. Never resample these rows directly -- group by participant first."""
    d = np.load(Path(run_dir) / f"predictions_{split}.npz", allow_pickle=False)
    probs = d["probabilities"]
    out = pd.DataFrame({
        "participant_id": d["participant_id"].astype(str),
        "session_id": d["session_id"].astype(str),
        "image_path": d["image_path"].astype(str),
        "label": d["labels"].astype(int),
        "pred": probs.argmax(axis=1).astype(int),
    })
    for c in range(probs.shape[1]):
        out[f"p{c}"] = probs[:, c]
    return out


def load_metrics(run_dir: str | Path) -> Dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def check_completeness(df: pd.DataFrame) -> List[str]:
    """Compare what is on disk against what each experiment config declares."""
    import yaml
    problems = []
    for cfg_path in sorted((REPO_ROOT / "config").glob("exp*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        name, sweep = cfg.get("experiment"), cfg.get("sweep")
        if not sweep:
            continue
        expected = {(l, f, s) for l in sweep["losses"] for f in sweep["folds"] for s in sweep["init_seeds"]}
        got = set(map(tuple, df[df["experiment"] == name][["loss", "fold", "init_seed"]].values)) \
            if not df.empty else set()
        missing = sorted(expected - got)
        if missing:
            problems.append(f"{name}: {len(missing)}/{len(expected)} missing -> " +
                            ", ".join(f"{l}:fold{f}:init{s}" for l, f, s in missing[:12]) +
                            (" ..." if len(missing) > 12 else ""))
    return problems


def write_index(df: pd.DataFrame) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "RUNS_INDEX.csv"
    df.to_csv(path, index=False)
    return path


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Inspect completed runs")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--check", action="store_true", help="fingerprints and completeness only")
    ap.add_argument("--all-status", action="store_true", help="include FAILED and RUNNING")
    args = ap.parse_args()

    df = load_runs(args.experiment, status=None if args.all_status else "COMPLETE")
    if df.empty:
        print("no runs found"
              + (" (try --all-status)" if not args.all_status else ""))
        return 1

    if args.summary or not (args.missing or args.check):
        cols = ["run_id", "experiment", "loss", "fold", "init_seed", "status",
                "group_disjoint", "test_participants", "test_images",
                "test_accuracy", "test_macro_f1", "test_ece", "duration_seconds"]
        cols = [c for c in cols if c in df.columns]
        with pd.option_context("display.width", 220, "display.max_columns", 40,
                               "display.max_colwidth", 46):
            print(df[cols].to_string(index=False))
        print(f"\n{len(df)} run(s). index -> {write_index(df)}")

    if args.missing or args.check:
        problems = check_completeness(df)
        print("\nCOMPLETENESS")
        if problems:
            for p in problems:
                print("  MISSING  " + p)
        else:
            print("  all configured runs present")

    stale = [d.parent.name for d in EXPERIMENTS.glob("*/outputs/*/_RUNNING")]
    if stale:
        print(f"\nSTALE\n  {len(stale)} run folder(s) still marked _RUNNING (killed or crashed "
              f"before the handler ran):")
        for s in stale[:10]:
            print("    " + s)
        print("  These are excluded from all analysis. Delete them or re-run those specs.")

    if args.check:
        # exp05 is the leakage ablation: it is SUPPOSED to report leakage, and says
        # so in its own manifest. Only unexpected leakage is an integrity failure.
        bad = df[(df["group_disjoint"] != "PASS") &
                 (~df["experiment"].astype(str).str.startswith("exp05"))]
        declared = df[(df["group_disjoint"] != "PASS") &
                      (df["experiment"].astype(str).str.startswith("exp05"))]
        print("\nINTEGRITY")
        print(f"  group-disjointness PASS on "
              f"{len(df) - len(bad) - len(declared)}/{len(df)} runs")
        if len(declared):
            print(f"  {len(declared)} run(s) carry DECLARED leakage (exp05 ablation, expected)")
        if len(bad):
            print("  !! UNEXPECTED leakage in:", list(bad["run_id"]))
        print(f"  distinct code fingerprints:   {df['code_fingerprint'].nunique()} "
              f"(must be 1 across a results table)")
        print(f"  distinct config fingerprints: {df['config_fingerprint'].nunique()} "
              f"(one per unique run configuration; {len(df)} runs)")
        # Missing runs are reported above but are NOT an integrity failure: an
        # optional experiment you chose not to run must not block the analysis.
        integrity_failed = bool(len(bad)) or df["code_fingerprint"].nunique() > 1
        if problems and not integrity_failed:
            print("\n  (missing runs are reported for information; they do not fail this check)")
        return 1 if integrity_failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
