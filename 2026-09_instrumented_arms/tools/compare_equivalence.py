#!/usr/bin/env python3
"""Is this tree's training path numerically identical to the 2026-08 tree's?

    python tools\\compare_equivalence.py

WHY THIS EXISTS
---------------
a21 compares models trained on the full cohort against models trained on the
age-60 cohort. Those two sides live in different trees under different code
fingerprints, and a fingerprint difference is a refusal condition, because
attributing a code difference to the cohort is the error the provenance layer
exists to prevent.

The difference between the trees is confined to bookkeeping. src/data.py and
src/losses.py are byte-identical. src/train.py gained per-class accumulators
that run under no_grad on detached outputs after the optimizer step, and
src/runner.py gained manifest fields. Neither touches the optimizer, the loss,
the data order or the seeding. But "confined to bookkeeping by inspection" is an
argument, and a reviewer is entitled to a measurement.

WHY ALL FIVE OBJECTIVES AND NOT JUST ONE
----------------------------------------
The family a21 computes has one comparison per objective, and each rests on a
different loss. src/diagnostics.py branches on the criterion type, so a
demonstration for FA-FL does not cover weighted cross-entropy. Attesting one
loss and then computing five comparisons would leave four of them resting on an
assumption. Each untested objective is listed as NOT TESTED rather than passed
over, and a21 refuses to register a comparison whose arm is not attested.

WHAT TO RUN
-----------
For each objective still listed NOT TESTED, about 29 minutes each:

    python experiments\\exp01_main_sweep\\run.py --only WCE:fold0
    python experiments\\exp01_main_sweep\\run.py --only LDAM:fold0
    python experiments\\exp01_main_sweep\\run.py --only FocalLoss:fold0
    python experiments\\exp01_main_sweep\\run.py --only FA_FL:fold0
    python experiments\\exp02_class_balanced\\run.py --only ClassBalanced:fold0

THE OUTPUT
----------
analysis/outputs/equivalence_attestation.json, which a21 reads. It records both
run ids, both fingerprints and the per-metric differences for every spec tested,
so the justification for a cross-tree comparison is a file on disk rather than a
sentence in a paper.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OTHER = REPO_ROOT.parent / "2026-08_srep_revision_participant_level"
ATTESTATION = REPO_ROOT / "analysis" / "outputs" / "equivalence_attestation.json"

# (experiment, loss type, the arm label a21 uses)
SPECS = [("exp01_main_sweep", "WCE", "Weighted CE"),
         ("exp01_main_sweep", "LDAM", "LDAM"),
         ("exp01_main_sweep", "FocalLoss", "Focal g=3.00"),
         ("exp01_main_sweep", "FA_FL", "FA-FL"),
         ("exp02_class_balanced", "ClassBalanced", "Class-Balanced")]
FOLD = 0

# Bit-identity is the claim. These names exist to report HOW far off a mismatch
# is, never to wave one through: any non-zero difference fails.
KEYS = ["test_accuracy", "test_macro_f1", "test_ece", "test_brier",
        "val_accuracy", "val_macro_f1", "temperature"]


def find(tree: Path, experiment: str, loss: str, fold: int) -> Optional[Dict]:
    out = None
    outputs = tree / "experiments" / experiment / "outputs"
    if not outputs.is_dir():
        return None
    for mf in sorted(outputs.glob("*/manifest.json")):
        if not (mf.parent / "_COMPLETE").exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        if (m.get("loss", {}).get("type") == loss
                and m.get("split", {}).get("fold") == fold
                and m.get("init_seed") == 42
                and (m.get("cohort") or {}).get("filter", "none") == "none"):
            out = m                       # last wins: the most recent COMPLETE run
    return out


def compare(here: Dict, there: Dict) -> Dict:
    metrics, worst = {}, 0.0
    for key in KEYS:
        x, y = here["headline"].get(key), there["headline"].get(key)
        if x is None or y is None:
            metrics[key] = {"here": x, "there": y, "diff": None}
            continue
        d = abs(float(x) - float(y))
        worst = max(worst, d)
        metrics[key] = {"here": float(x), "there": float(y), "diff": d}
    return {"max_abs_difference": worst, "metrics": metrics,
            "status": "IDENTICAL" if worst == 0.0 else "DIFFERENT"}


def main() -> int:
    rows: List[Dict] = []
    fps_here, fps_there = set(), set()

    for experiment, loss, arm in SPECS:
        here = find(REPO_ROOT, experiment, loss, FOLD)
        there = find(OTHER, experiment, loss, FOLD)
        row = {"arm": arm, "experiment": experiment, "loss": loss, "fold": FOLD}
        if there is None:
            row.update({"status": "NO COUNTERPART",
                        "note": f"no full-cohort {loss}:fold{FOLD} run in {OTHER.name}"})
        elif here is None:
            row.update({"status": "NOT TESTED",
                        "note": f"train it: python experiments\\{experiment}\\run.py "
                                f"--only {loss}:fold{FOLD}"})
        else:
            row.update(compare(here, there))
            row["run_id_here"], row["run_id_there"] = here["run_id"], there["run_id"]
            fps_here.add(here["code_fingerprint"])
            fps_there.add(there["code_fingerprint"])
        rows.append(row)

    print(f"  {'arm':<16} {'status':<14} {'|max difference|':>17}")
    for r in rows:
        d = r.get("max_abs_difference")
        print(f"  {r['arm']:<16} {r['status']:<14} "
              f"{('%.2e' % d) if d is not None else '--':>17}")
        if r.get("note"):
            print(f"      {r['note']}")

    tested = [r for r in rows if r["status"] in ("IDENTICAL", "DIFFERENT")]
    different = [r for r in rows if r["status"] == "DIFFERENT"]
    untested = [r for r in rows if r["status"] not in ("IDENTICAL", "DIFFERENT")]

    if different:
        verdict = "DIFFERENT"
    elif untested:
        verdict = "INCOMPLETE"
    else:
        verdict = "IDENTICAL"

    ATTESTATION.parent.mkdir(parents=True, exist_ok=True)
    ATTESTATION.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "this_tree": {"root": str(REPO_ROOT), "fingerprints": sorted(fps_here)},
        "other_tree": {"root": str(OTHER), "fingerprints": sorted(fps_there)},
        "fold": FOLD, "init_seed": 42, "metrics_compared": KEYS,
        "verdict": verdict,
        "arms_attested": sorted(r["arm"] for r in rows if r["status"] == "IDENTICAL"),
        "specs": rows,
    }, indent=2), encoding="utf-8")

    print(f"\n  attestation written to {ATTESTATION}")
    print(f"  verdict: {verdict}   ({len(tested)} of {len(SPECS)} objectives tested)")
    if verdict == "IDENTICAL":
        print("\n  Every objective reproduces the 2026-08 run exactly. The instrumentation")
        print("  does not perturb training for any loss, so a21 may compute its registered")
        print("  family across the two trees, citing this file.")
        return 0
    if verdict == "INCOMPLETE":
        print("\n  The objectives tested so far match exactly, but a21 will refuse to")
        print("  register a comparison for any arm still listed above as untested.")
        for r in untested:
            print(f"      {r['arm']:<16} {r.get('note', '')}")
        return 1
    print("\n  NOT IDENTICAL. Something in this tree changes training. Do not compare across")
    print("  the trees: rebuild the full-cohort arms here, and find out what moved before")
    print("  trusting anything else in this folder.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
