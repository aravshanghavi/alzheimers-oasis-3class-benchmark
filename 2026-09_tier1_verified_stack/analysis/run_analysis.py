#!/usr/bin/env python3
"""Run the whole analysis layer with one command.

    python analysis/run_analysis.py

Order matters: tables are built before verification, because verification
re-derives the published tables from raw predictions and diffs them.
Grad-CAM is optional and its failure does not stop the rest.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import logging_utils as lg          # noqa: E402
from src.provenance import timestamp_slug    # noqa: E402

# a11-a18 were unreachable from this file in the 2026-08 stack: roughly half the
# revision's evidence -- participant-level metrics, the baseline panel, the age
# sensitivity, the gamma control, the nWBV increment -- was produced by scripts
# nothing orchestrated and nothing verified. They are wired in here, after
# _cache_build.py, which a17 and a18 depend on.
STEPS = [
    ("a00_preflight.py",       [],           True,  "preflight -- fail fast, before any work"),
    ("discover.py",            ["--check"],  True,  "integrity and completeness"),
    ("_cache_build.py",        [],           True,  "build _cache.npz (prerequisite for a17/a18)"),
    ("a01_tables.py",          [],           True,  "Tables 1-4"),
    ("a02_calibration.py",     [],           True,  "calibration figures"),
    ("a03_bootstrap.py",       [],           True,  "participant-clustered bootstrap"),
    ("a05_variance_decomp.py", [],           True,  "variance decomposition"),
    ("a06_pooled_cv.py",       [],           True,  "pooled confusion / ROC / PR"),
    ("a08_leakage_ablation.py", [],          True,  "leakage ablation (exp05 vs exp01)"),
    ("a09_leakage_within_model.py", [],      True,  "powered leakage contrasts (within-model + DiD)"),
    ("a10_multislice.py",      [],           True,  "2.5D multi-slice vs 2D (exp04 vs exp01)"),
    ("a11_ece_robustness.py",  [],           True,  "ECE under two binnings; offset vs shape"),
    ("a12_ldam_scale.py",      [],           True,  "LDAM logit-scale convention"),
    ("a13_cohort_age_check.py", [],          True,  "cohort composition: CDR-blank and under-60"),
    ("a14_gamma_control.py",   [],           True,  "FA-FL vs focal at gamma 1.37 / 2.0 / 3.0"),
    ("a15_final_free.py",      [],           True,  "QWK, kappa, MCC, error structure, AURC"),
    ("a16_remediation.py",     [],           True,  "participant-level battery, baselines, age curve"),
    ("a17_participant_baseline.py", [],      True,  "arms vs constant and metadata regressions"),
    ("a18_increment_and_structure.py", [],   True,  "CNN increment over nWBV, inter-arm agreement"),
    ("a04_gradcam_quant.py",   [],           False, "quantitative Grad-CAM (optional, needs GPU)"),
    ("a07_verify.py",          [],           True,  "VERIFICATION -- must pass"),
]


def clean_outputs(log) -> int:
    """Move everything currently in analysis/outputs/ aside, so a run starts clean.

    Moved into _superseded/<timestamp>/, never deleted -- the same rule the runner
    uses for superseded run folders. A partial or failed pass leaves artefacts
    behind, and a later pass that only overwrites SOME of them produces an output
    directory that is a mixture of two runs with nothing recording which is which.
    That is the failure this flag exists to prevent, and it is the analysis-layer
    version of the stale-output problem in experiments/.
    """
    out = REPO_ROOT / "analysis" / "outputs"
    if not out.is_dir():
        return 0
    movable = [p for p in out.iterdir() if p.name != "_superseded"]
    if not movable:
        return 0
    dest = out / "_superseded" / timestamp_slug()
    dest.mkdir(parents=True, exist_ok=True)
    for p in movable:
        shutil.move(str(p), str(dest / p.name))
    log.info("cleaned: moved %d item(s) to %s", len(movable), dest)
    return len(movable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gradcam", action="store_true")
    ap.add_argument("--only", default=None, help="substring of a script name")
    ap.add_argument("--clean", action="store_true",
                    help="move existing analysis/outputs/ aside before starting, so the run "
                         "cannot mix fresh results with leftovers from a partial pass. "
                         "Moves into _superseded/<timestamp>/; nothing is deleted.")
    args = ap.parse_args()

    log_path = REPO_ROOT / "console_logs" / f"{timestamp_slug()}__analysis.log"
    log = lg.init_queue_logging(log_path, console="full")
    log.info(lg.RULE)
    log.info("ANALYSIS LAYER")
    log.info("  outputs -> %s", REPO_ROOT / "analysis" / "outputs")
    log.info("  log     -> %s", log_path)
    log.info(lg.RULE)

    if args.clean:
        clean_outputs(log)

    results, t0 = [], time.time()
    for script, extra, required, desc in STEPS:
        if args.only and args.only not in script:
            continue
        if args.skip_gradcam and "gradcam" in script:
            continue
        log.info("")
        log.info("-" * 96)
        log.info(">>> %-24s %s", script, desc)
        log.info("-" * 96)
        t = time.time()
        rc = subprocess.run([sys.executable, str(REPO_ROOT / "analysis" / script), *extra]).returncode
        results.append((script, rc, time.time() - t, required))
        if rc != 0 and required:
            if script == "a00_preflight.py":
                # Everything downstream reads the same paths, packages and runs that
                # preflight just found wanting. Carrying on would turn one legible
                # failure into eighteen, and you would come back to a log full of
                # cascading tracebacks instead of one line naming the cause.
                log.error("PREFLIGHT FAILED -- stopping before any work. Fix what it "
                          "listed above and re-run; nothing downstream can succeed.")
                break
            if script == "_cache_build.py":
                log.error("_cache_build failed: a17 and a18 will fail too (they read "
                          "_cache.npz). Everything else is unaffected and continues.")
            elif script == "a07_verify.py":
                log.error("VERIFICATION FAILED -- do not use these tables. See "
                          "analysis/outputs/verification_FAILED.txt")
            else:
                log.error("%s failed with exit %d; later steps may be inconsistent", script, rc)

    log.info("")
    log.info(lg.RULE)
    log.info("ANALYSIS FINISHED in %s", lg._fmt_duration(time.time() - t0))
    for script, rc, dt, required in results:
        status = "OK" if rc == 0 else ("FAILED" if required else "skipped/optional")
        log.info("  %-26s %-18s %s", script, status, lg._fmt_duration(dt))
    log.info(lg.RULE)

    hard = [s for s, rc, _, req in results if rc != 0 and req]
    if hard:
        log.error("%d required step(s) failed: %s", len(hard), ", ".join(hard))
        return 1
    log.info("All required steps passed. Tables and figures are in analysis/outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
