"""Logging. Two handlers: stdout and <run_dir>/run.log.

Redirecting stdout to a file is therefore optional rather than required -- every
run keeps its own complete log regardless of how it was launched.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RULE = "=" * 96
THIN = "-" * 96


PROGRESS = logging.getLogger("progress")   # console heartbeat; also propagates to files

_QUEUE_HANDLERS: List[logging.Handler] = []


def init_queue_logging(queue_log_path: Optional[Path] = None,
                       console: str = "progress",
                       level: int = logging.INFO) -> logging.Logger:
    """Configure logging for a whole queue. Call ONCE per process.

    console:
      full      everything to the terminal as well as to files
      progress  only queue-level heartbeat lines and warnings to the terminal;
                full detail still goes to the log files (default -- this is what
                makes an unattended overnight queue readable afterwards)
      silent    nothing to the terminal

    Whatever the mode, the complete stream is written to the queue log file and,
    once a run starts, duplicated into that run's own run.log.
    """
    global _QUEUE_HANDLERS
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in list(PROGRESS.handlers):
        PROGRESS.removeHandler(h)
        h.close()
    _QUEUE_HANDLERS = []

    fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    if queue_log_path is not None:
        queue_log_path = Path(queue_log_path)
        queue_log_path.parent.mkdir(parents=True, exist_ok=True)
        qh = logging.FileHandler(queue_log_path, mode="a", encoding="utf-8")
        qh.setFormatter(fmt)
        root.addHandler(qh)
        _QUEUE_HANDLERS.append(qh)

    PROGRESS.propagate = True          # progress lines land in the files too
    if console == "full":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        _QUEUE_HANDLERS.append(sh)
    elif console == "progress":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", DATE_FORMAT))
        PROGRESS.addHandler(sh)
        wh = logging.StreamHandler(sys.stderr)          # never hide warnings
        wh.setFormatter(fmt)
        wh.setLevel(logging.WARNING)
        root.addHandler(wh)
        _QUEUE_HANDLERS.append(wh)
    elif console != "silent":
        raise ValueError(f"unknown console mode {console!r}")

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger("srep")


def attach_run_log(run_dir: Path) -> logging.Handler:
    """Add a per-run file handler on top of the queue handlers."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logging.getLogger().addHandler(fh)
    return fh


def detach_run_log(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    handler.close()


def progress(message: str, *args) -> None:
    """One-line heartbeat: shown on the terminal in progress mode, always logged."""
    PROGRESS.info(message, *args)


def setup_logging(run_dir: Optional[Path] = None, name: str = "srep",
                  level: int = logging.INFO) -> logging.Logger:
    """Backwards-compatible single-run setup (stdout + run.log)."""
    init_queue_logging(None, console="full", level=level)
    if run_dir is not None:
        attach_run_log(run_dir)
    return logging.getLogger(name)


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def log_banner(log, *, run_id: str, experiment: str, driver: str, config_file: str,
               fingerprint: str, run_dir: Path, env: Dict, deterministic: bool,
               strict: bool, started: str) -> None:
    log.info(RULE)
    log.info("RUN %s", run_id)
    log.info(RULE)
    log.info("  experiment       %s", experiment)
    log.info("  driver           %s", driver)
    log.info("  config           %s", config_file)
    log.info("  code fingerprint %s", fingerprint)
    log.info("  output dir       %s", run_dir)
    log.info("  host / GPU       %s / %s", env.get("hostname", "?"), env.get("gpu_name", "cpu"))
    log.info("  torch / CUDA     %s / %s   deterministic=%s strict=%s",
             env.get("torch", "?"), env.get("cuda", "?"), deterministic, strict)
    log.info("  python / OS      %s / %s", env.get("python", "?"), env.get("platform", "?"))
    log.info("  started          %s", started)
    log.info(RULE)


def log_config(log, cfg: Dict) -> None:
    log.info("RESOLVED CONFIG %s", THIN[15:])
    for line in json.dumps(cfg, sort_keys=True, indent=2, default=str).splitlines():
        log.info("  %s", line)


def log_cohort(log, summary: Dict) -> None:
    log.info("COHORT %s", THIN[7:])
    log.info("  images discovered ... %s", f"{summary['images']:,}")
    log.info("  participants ........ %-6d sessions ... %d",
             summary["participants"], summary["sessions"])
    log.info("  participants with >1 session ... %d", summary["participants_multi_session"])
    log.info("  slice positions ..... %d  (index %d-%d)",
             summary["slice_positions"], *summary["slice_range"])
    log.info("  acquisitions/session  %s", summary["acquisitions_per_session"])
    log.info("  participants/class .. %s", summary["participants_per_class"])
    log.info("  sessions/class ...... %s", summary["sessions_per_class"])


def log_split(log, split_df, cfg: Dict, class_names: Sequence[str],
              class_weights: np.ndarray, gammas: Optional[Sequence[float]],
              assert_result: str) -> None:
    sp = cfg["split"]
    log.info("DATA %s", THIN[5:])
    log.info("  root ................ %s", cfg["data"]["root"])
    log.info("  strategy ............ %s(%d) group_by=%s fold %d of %d  shuffle=%s%s",
             sp["strategy"], sp["n_splits"], sp["group_by"], sp["fold"], sp["n_splits"],
             sp.get("shuffle", False),
             f" split_seed={sp['split_seed']}" if sp.get("shuffle", False)
             else "  (deterministic partition, no split seed)")
    log.info("  context slices ...... %d %s", cfg["data"]["context_slices"],
             "(2.5D)" if cfg["data"]["context_slices"] == 3 else "(2D)")
    log.info("")
    header = f"  {'split':<8}{'particip.':>11}{'sessions':>10}{'images':>11}   " + \
             "".join(f"{n[:9]:>10}" for n in class_names)
    log.info(header)
    for _, r in split_df.iterrows():
        cells = "".join(f"{int(r[f'participants_c{c}']):>10}" for c in range(len(class_names)))
        log.info("  %-8s%11d%10d%11s   %s", r["split"], r["participants"], r["sessions"],
                 f"{int(r['images']):,}", cells)
    log.info("            (participant counts; image counts in fold_participants.csv)")
    log.info("")
    log.info("  class weights (normalised inverse freq): %s",
             "[" + ", ".join(f"{w:.4f}" for w in class_weights) + "]")
    if gammas is not None:
        log.info("  effective gamma per class:               %s",
                 "[" + ", ".join(f"{g:.3f}" for g in gammas) + "]")
    log.info("  ASSERT no participant in >1 split ....... %s", assert_result)


def log_epoch(log, *, stage: str, epoch: int, total: int, lr: float,
              train_loss: float, train_acc: float, val_loss: float, val_acc: float,
              epoch_seconds: float, elapsed: float, eta: float,
              improved: bool, best: float, metric_name: str) -> None:
    log.info("EPOCH %02d/%02d [%s] lr=%.3e", epoch, total, stage, lr)
    log.info("  train  loss %.4f  acc %6.2f%%   |   val  loss %.4f  acc %6.2f%%",
             train_loss, train_acc, val_loss, val_acc)
    log.info("  epoch %s   elapsed %s   eta %s",
             _fmt_duration(epoch_seconds), _fmt_duration(elapsed), _fmt_duration(eta))
    if improved:
        log.info("  >> %s improved to %.4f, checkpoint saved", metric_name, best)


def log_confusion_matrix(log, cm: np.ndarray, class_names: Sequence[str], title: str) -> None:
    """Confusion matrix rendered into the log, with recall and precision margins."""
    cm = np.asarray(cm, dtype=np.int64)
    width = max(10, max(len(n) for n in class_names) + 2)
    log.info("  %s (rows = true, cols = predicted)", title)
    head = " " * 16 + "".join(f"{n[:width-1]:>{width}}" for n in class_names) + \
           f"   |{'total':>9}{'recall':>10}"
    log.info("  %s", head)
    for i, name in enumerate(class_names):
        row_total = cm[i].sum()
        recall = 100.0 * cm[i, i] / row_total if row_total else 0.0
        cells = "".join(f"{cm[i, j]:>{width},}" for j in range(len(class_names)))
        log.info("  %-16s%s   |%9s%9.2f%%", name[:16], cells, f"{row_total:,}", recall)
    log.info("  %s", " " * 16 + "-" * (width * len(class_names) + 22))
    prec_cells = ""
    for j in range(len(class_names)):
        col_total = cm[:, j].sum()
        prec = 100.0 * cm[j, j] / col_total if col_total else 0.0
        prec_cells += f"{prec:>{width}.2f}"
    log.info("  %-16s%s", "precision %", prec_cells)


def log_metrics(log, metrics: Dict, class_names: Sequence[str]) -> None:
    log.info("  accuracy %.2f%%   macro-F1 %.4f   weighted-F1 %.4f",
             metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
    log.info("  per-class F1   %s",
             "   ".join(f"{n[:12]} {f:.4f}" for n, f in zip(class_names, metrics["per_class_f1"])))
    log.info("  ECE (%d bin, %s) %.4f   Brier %.4f",
             metrics["ece_bins"], metrics["ece_binning"], metrics["ece"], metrics["brier"])


def log_footer(log, *, status: str, duration: float, peak_gpu_gb: Optional[float],
               run_dir: Path, artifacts: Sequence[str]) -> None:
    log.info(RULE)
    gpu = f"   peak GPU {peak_gpu_gb:.2f} GB" if peak_gpu_gb is not None else ""
    log.info("%s  duration %s%s", status, _fmt_duration(duration), gpu)
    log.info("  wrote %d artifacts to %s", len(artifacts), run_dir)
    log.info("  %s", "  ".join(sorted(artifacts)))
    log.info(RULE)
