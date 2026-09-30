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
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import logging_utils as lg          # noqa: E402
from src.provenance import timestamp_slug    # noqa: E402

STEPS = [
    ("discover.py",            ["--check"],  True,  "integrity and completeness"),
    ("a01_tables.py",          [],           True,  "Tables 1-4"),
    ("a02_calibration.py",     [],           True,  "calibration figures"),
    ("a03_bootstrap.py",       [],           True,  "participant-clustered bootstrap"),
    ("a05_variance_decomp.py", [],           True,  "variance decomposition"),
    ("a06_pooled_cv.py",       [],           True,  "pooled confusion / ROC / PR"),
    ("a08_leakage_ablation.py", [],          True,  "leakage ablation (exp05 vs exp01)"),
    ("a09_leakage_within_model.py", [],      True,  "powered leakage contrasts (within-model + DiD)"),
    ("a10_multislice.py",      [],           True,  "2.5D multi-slice vs 2D (exp04 vs exp01)"),
    ("a04_gradcam_quant.py",   [],           False, "quantitative Grad-CAM (optional, needs GPU)"),
    ("a07_verify.py",          [],           True,  "VERIFICATION -- must pass"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gradcam", action="store_true")
    ap.add_argument("--only", default=None, help="substring of a script name")
    args = ap.parse_args()

    log_path = REPO_ROOT / "console_logs" / f"{timestamp_slug()}__analysis.log"
    log = lg.init_queue_logging(log_path, console="full")
    log.info(lg.RULE)
    log.info("ANALYSIS LAYER")
    log.info("  outputs -> %s", REPO_ROOT / "analysis" / "outputs")
    log.info("  log     -> %s", log_path)
    log.info(lg.RULE)

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
            if script == "a07_verify.py":
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
