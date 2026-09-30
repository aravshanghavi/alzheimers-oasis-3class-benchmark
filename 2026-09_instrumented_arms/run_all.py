#!/usr/bin/env python3
"""Run every experiment back to back, unattended.

    python run_all.py                       # exp10 -> exp11 -> exp12 -> exp13 (40 runs)
    python run_all.py --experiments exp10_age60_main
    python run_all.py --resume              # skip anything already COMPLETE
    python run_all.py --isolate             # each run in its own subprocess
    python run_all.py --smoke               # 2 epochs, tiny images, 1 run each

Nothing of consequence goes to the terminal: each experiment writes a full log
to console_logs/<timestamp>__<experiment>.log, every individual run keeps its
own run.log, and this script writes a master log tying them together. The
terminal shows one heartbeat line per run so you can glance at it, and that is
all -- close the window, come back tomorrow, read the files.

Between experiments the GPU allocator is drained and the peak counters reset, so
one experiment's footprint cannot push the next one into an out-of-memory state.
"""
from __future__ import annotations

import argparse
import subprocess
import os
# Must be set before torch initialises CUDA. Reduces allocator fragmentation
# across a long sequence of runs; the log warns when it is unset.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import gpu, logging_utils as lg  # noqa: E402
from src.provenance import timestamp_slug  # noqa: E402

# This repository exists for the train-matched age-60 arms and nothing else.
# Running exp01-exp05 here would duplicate work that is already COMPLETE in the
# 2026-08 tree under a different code fingerprint, which is exactly the kind of
# ambiguity the provenance layer is meant to prevent. Those configs are kept so
# the comparison is reproducible from one checkout, but they are not the default.
# To run them deliberately:  python run_all.py --experiments exp01_main_sweep
DEFAULT_ORDER = ["exp10_age60_main", "exp11_age60_gamma137", "exp12_age60_gamma200",
                 "exp13_age60_gamma269"]
OPTIONAL = []
LEGACY = ["exp01_main_sweep", "exp02_class_balanced", "exp03_init_variance",
          "exp04_multislice_25d", "exp05_leakage_ablation",
          "exp07_focal_gamma137", "exp08_focal_gamma200"]

REQUIRED_MODULES = ["torch", "torchvision", "numpy", "pandas", "sklearn", "PIL", "yaml"]


def preflight(data_root: str | None, log) -> list[str]:
    """Check the interpreter and the data path BEFORE spawning any child.

    Without this a missing dependency costs one subprocess launch per experiment
    and reports only "exit 1", which is useless when you come back in the morning.
    """
    problems = []

    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        problems.append(f"missing module(s): {', '.join(missing)}")
        problems.append(f"  interpreter in use: {sys.executable}")
        problems.append("  if you use conda, the environment is probably not active:")
        problems.append("      conda activate alzheimers")
        problems.append("  otherwise:  pip install -r requirements.txt")

    if data_root is None:
        try:
            import yaml
            base = yaml.safe_load((REPO_ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
            data_root = base["data"]["root"]
        except Exception as exc:
            problems.append(f"could not read config/base.yaml: {exc!r}")
            data_root = None
    if data_root and not Path(data_root).is_dir():
        problems.append(f"data.root does not exist: {data_root}")
        problems.append("  set it in config/base.yaml or pass --data-root")

    if not problems:
        try:
            import torch
            log.info("preflight OK | %s | torch %s | CUDA %s",
                     sys.executable, torch.__version__,
                     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available")
        except Exception:
            pass
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all experiments unattended")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help=f"override the order (default: {' '.join(DEFAULT_ORDER)})")
    ap.add_argument("--with-optional", action="store_true", help="also run " + ", ".join(OPTIONAL))
    ap.add_argument("--skip", nargs="*", default=[], help="experiment names to skip")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--supersede-previous", action="store_true",
                    help="after a spec COMPLETEs, move any other COMPLETE run of the same "
                         "spec+code into outputs/_superseded/ (moved, never deleted). Without "
                         "this, a rerun without --resume leaves duplicate run folders that the "
                         "analysis layer will refuse to pool from.")
    ap.add_argument("--console", choices=("full", "progress", "silent"), default="progress")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet weights (offline wiring checks only)")
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="abort the batch if an experiment reports any failed run "
                         "(default: carry on to the next experiment)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast wiring check: 2 epochs, 64px, one run per experiment")
    args = ap.parse_args()

    experiments = args.experiments or (DEFAULT_ORDER + (OPTIONAL if args.with_optional else []))
    experiments = [e for e in experiments if e not in args.skip]

    stamp = timestamp_slug()
    master_log = REPO_ROOT / "console_logs" / f"{stamp}__run_all.log"
    log = lg.init_queue_logging(master_log, console=args.console)

    lg.progress(lg.RULE)
    lg.progress("RUN ALL -- %d experiment(s): %s", len(experiments), ", ".join(experiments))
    lg.progress("  master log: %s", master_log)
    lg.progress("  per-experiment logs: %s", REPO_ROOT / "console_logs")
    if args.smoke:
        lg.progress("  SMOKE MODE: 2 epochs, 64px, one run each -- results are meaningless")
    log.info("%s", gpu.preflight_env_hint())
    if gpu.cuda_available():
        log.info("GPU at batch start: %s", gpu.format_snapshot(gpu.memory_snapshot()))
    problems = preflight(args.data_root, log)
    if problems:
        lg.progress("")
        lg.progress("PREFLIGHT FAILED -- nothing was launched:")
        for line in problems:
            lg.progress("  %s", line)
        lg.progress(lg.RULE)
        return 2
    lg.progress(lg.RULE)

    common = ["--console", args.console]
    if args.resume:    common.append("--resume")
    if args.isolate:   common.append("--isolate")
    if args.supersede_previous: common.append("--supersede-previous")
    if args.data_root: common += ["--data-root", args.data_root]
    if args.device:    common += ["--device", args.device]
    if args.no_pretrained: common.append("--no-pretrained")
    if args.smoke:
        common += ["--epochs", "2", "--warmup-epochs", "1", "--image-size", "64", "--limit", "1"]

    results, t0 = [], time.time()
    for i, exp in enumerate(experiments, 1):
        driver = REPO_ROOT / "experiments" / exp / "run.py"
        if not driver.exists():
            log.error("no driver at %s -- skipping %s", driver, exp)
            results.append((exp, "NO_DRIVER", 0.0))
            continue

        lg.progress("")
        lg.progress(">>> [%d/%d] %s", i, len(experiments), exp)
        t_exp = time.time()
        # Child stdout carries the progress heartbeat and passes straight through
        # to the terminal. Child stderr is captured: the full text goes to the
        # master log, and on failure the tail is echoed to the terminal too --
        # an unattended batch that dies must say why without needing the log file.
        completed = subprocess.run([sys.executable, str(driver), *common],
                                   stderr=None if args.console == "full" else subprocess.PIPE,
                                   text=True)
        stderr_text = completed.stderr or ""
        if stderr_text.strip():
            with open(master_log, "a", encoding="utf-8") as err_sink:
                err_sink.write(f"\n----- stderr of {exp} -----\n{stderr_text}\n")
        if completed.returncode != 0 and stderr_text.strip():
            tail = [ln for ln in stderr_text.strip().splitlines() if ln.strip()][-6:]
            lg.progress("  error from %s:", exp)
            for line in tail:
                lg.progress("    %s", line.rstrip())
        elapsed = time.time() - t_exp
        status = "OK" if completed.returncode == 0 else f"FAILURES (exit {completed.returncode})"
        results.append((exp, status, elapsed))
        lg.progress("<<< [%d/%d] %s -- %s in %s", i, len(experiments), exp, status,
                    lg._fmt_duration(elapsed))

        # Drain the allocator so this experiment cannot squeeze the next one.
        # In --isolate mode the children have already exited and this is a no-op,
        # which is exactly why --isolate is the safer choice for a long batch.
        gpu.release_between_runs(log=log)

        if completed.returncode != 0 and args.stop_on_failure:
            lg.progress("stopping: --stop-on-failure and %s reported failures", exp)
            break

    lg.progress("")
    lg.progress(lg.RULE)
    lg.progress("RUN ALL FINISHED in %s", lg._fmt_duration(time.time() - t0))
    for exp, status, elapsed in results:
        lg.progress("  %-24s %-22s %s", exp, status, lg._fmt_duration(elapsed))
    lg.progress(lg.RULE)
    lg.progress("next: python analysis/discover.py --check")

    return 0 if all(s == "OK" for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
