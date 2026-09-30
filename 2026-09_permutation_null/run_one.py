#!/usr/bin/env python3
"""Execute exactly ONE run, then exit.

This exists so the queue can offer --isolate: each run gets its own process, so
VRAM reclamation is enforced by the operating system instead of relying on
Python reference counting and the CUDA caching allocator. It is also useful on
its own for re-running a single failed spec.

The last line of stdout is a JSON result object, which the parent parses.

    python run_one.py --config exp01_main_sweep.yaml --loss FA_FL --fold 3 --init-seed 42
"""
from __future__ import annotations

import argparse
import json
import os
# Must be set before torch initialises CUDA. Reduces allocator fragmentation
# across a long sequence of runs; the log warns when it is unset.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from src import logging_utils as lg  # noqa: E402
from src.config import load_experiment_config, validate_config  # noqa: E402
from src.data import scan_dataset  # noqa: E402
from src.runner import RunSpec, config_for_spec, execute_run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute a single run")
    ap.add_argument("--config", required=True, help="experiment config filename, e.g. exp01_main_sweep.yaml")
    ap.add_argument("--loss", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--init-seed", type=int, required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--scan-cache", default=None,
                    help="CSV written by the parent queue, to skip re-scanning 86k files")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--console", choices=("full", "progress", "silent"), default="full")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--perm-seed", type=int, default=None,
                    help="permutation.seed for a permuted-label null run")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    if args.data_root:      cfg["data"]["root"] = args.data_root
    if args.device:         cfg["device"] = args.device
    if args.epochs:         cfg["train"]["total_epochs"] = args.epochs
    if args.warmup_epochs is not None: cfg["train"]["warmup_epochs"] = args.warmup_epochs
    if args.batch_size:     cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:   cfg["train"]["num_workers"] = args.num_workers
    if args.image_size:     cfg["data"]["image_size"] = [args.image_size, args.image_size]
    if args.no_pretrained:  cfg["model"]["pretrained"] = False
    if args.perm_seed is not None:
        cfg.setdefault("permutation", {})["seed"] = args.perm_seed

    spec = RunSpec(loss=args.loss, fold=args.fold, init_seed=args.init_seed)
    run_cfg = config_for_spec(cfg, spec)
    validate_config(run_cfg, require_data_root=True)

    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "console_logs"
    lg.init_queue_logging(log_dir / f"run_one__{cfg['experiment']}.log", console=args.console)

    if args.scan_cache and Path(args.scan_cache).exists():
        scan = pd.read_csv(args.scan_cache)
    else:
        scan = scan_dataset(run_cfg["data"]["root"], run_cfg["data"]["original_classes"],
                            run_cfg["data"]["class_map"])

    outputs_dir = REPO_ROOT / "experiments" / cfg["experiment"] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    result = execute_run(run_cfg, spec, outputs_dir,
                         f"run_one.py --config {args.config}", scan)

    print(json.dumps(result), flush=True)   # parseable last line for the parent
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
