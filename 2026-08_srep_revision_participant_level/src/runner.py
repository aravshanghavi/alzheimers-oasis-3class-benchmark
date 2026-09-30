"""Single-run orchestration and the experiment queue driver.

experiments/*/run.py are thin: they name a config and hand control to
``main``. All shared behaviour lives here, which is what prevents the
five-near-identical-39KB-scripts problem the previous codebase had.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import evaluate as ev
from . import figures as figs
from . import gpu
from . import logging_utils as lg
from . import provenance as pv
from .config import (REPO_ROOT, load_experiment_config, resolved_json,
                     validate_config)
from .data import (assert_group_disjoint, build_dataloaders, class_counts,
                   cohort_summary, compute_class_weights, make_eval_loader,
                   make_splits, scan_dataset, split_summary)
from .losses import build_criterion, describe_criterion
from .model import DementiaModel, count_parameters
from .seeding import make_dataloader_rng, resolve_device, set_all_seeds
from .train import train_model


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM surfaces as several different types and messages, including from
    the pin-memory thread where it is wrapped in a plain RuntimeError."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


# Progressively safer evaluation settings. Each step removes a source of
# page-locked or parallel memory pressure. NONE of them changes the numbers:
# inference runs under no_grad with BatchNorm in eval mode, so batch size and
# worker count are numerically irrelevant. That is precisely why evaluation may
# be degraded automatically, while a training OOM may not be -- silently halving
# the training batch size would change the protocol and invalidate the run.
EVAL_FALLBACKS = (
    {"label": "configured"},
    {"label": "no workers, no pinning", "num_workers": 0, "pin_memory": False},
    {"label": "no workers, half batch", "num_workers": 0, "pin_memory": False, "batch_div": 2},
    {"label": "no workers, batch 8", "num_workers": 0, "pin_memory": False, "batch": 8},
)


def evaluate_with_fallback(model, df, idx, cfg, device, split: str, log):
    """Run inference, degrading loader settings on out-of-memory and retrying.

    An unattended queue must not lose a completed 30-minute training run because
    a transient allocation failed during the final forward pass.
    """
    last_exc = None
    for attempt, plan in enumerate(EVAL_FALLBACKS):
        batch = plan.get("batch")
        if batch is None and "batch_div" in plan:
            batch = max(1, cfg["train"]["eval_batch_size"] // plan["batch_div"])
        loader = None
        try:
            loader = make_eval_loader(df, idx, cfg,
                                      num_workers=plan.get("num_workers"),
                                      pin_memory=plan.get("pin_memory"),
                                      batch_size=batch)
            if attempt:
                log.warning("  retrying %s evaluation with fallback %d/%d (%s)",
                            split, attempt, len(EVAL_FALLBACKS) - 1, plan["label"])
            pred = ev.predict(model, loader, device)
            if attempt:
                log.warning("  %s evaluation succeeded on fallback %d (%s). Results are "
                            "unaffected: batch size and worker count do not change "
                            "inference output.", split, attempt, plan["label"])
            return pred
        except BaseException as exc:                                  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt) or not _is_oom(exc):
                raise
            last_exc = exc
            log.warning("  OUT OF MEMORY during %s evaluation (%s): %s",
                        split, plan["label"], str(exc).splitlines()[0][:120])
            log.warning("  %s", gpu.format_snapshot(gpu.memory_snapshot()))
        finally:
            if loader is not None:
                gpu.shutdown_dataloaders([loader])
                del loader
            gpu.release_between_runs(log=None)
    raise RuntimeError(
        f"{split} evaluation failed with out-of-memory on all "
        f"{len(EVAL_FALLBACKS)} fallback configurations"
    ) from last_exc


@dataclass(frozen=True)
class RunSpec:
    loss: str
    fold: int
    init_seed: int

    def label(self) -> str:
        return f"{self.loss}:fold{self.fold}:init{self.init_seed}"


# ---------------------------------------------------------------------------

def expand_sweep(cfg: Dict) -> List[RunSpec]:
    sweep = cfg.get("sweep")
    if not sweep:
        raise ValueError("experiment config has no 'sweep' block")
    return [RunSpec(loss=loss, fold=fold, init_seed=seed)
            for loss in sweep["losses"]
            for fold in sweep["folds"]
            for seed in sweep["init_seeds"]]


def config_for_spec(base_cfg: Dict, spec: RunSpec) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["loss"]["type"] = spec.loss
    cfg["split"]["fold"] = spec.fold
    cfg["init_seed"] = spec.init_seed
    cfg.pop("sweep", None)          # not part of a single run's identity
    cfg.pop("description", None)
    validate_config(cfg, require_data_root=False)
    return cfg


def find_existing(outputs_dir: Path, spec: RunSpec, code_fp: str, config_fp: str) -> Optional[Path]:
    """A completed run with the same spec, the same code AND the same config."""
    pattern = f"*__{spec.loss}__fold{spec.fold}__split*-init{spec.init_seed}__{code_fp[:8]}"
    for candidate in sorted(outputs_dir.glob(pattern)):
        if not (candidate / "_COMPLETE").exists():
            continue
        try:
            m = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if m.get("config_fingerprint") == config_fp:
            return candidate
    return None


# ---------------------------------------------------------------------------

def _install_sigterm_handler() -> None:
    """Turn SIGTERM into KeyboardInterrupt so a killed run is recorded as FAILED
    rather than left with a stale _RUNNING marker."""
    def handler(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        pass


def execute_run(cfg: Dict, spec: RunSpec, outputs_dir: Path, driver: str,
                scan_cache: Optional[pd.DataFrame] = None) -> Dict:
    """Execute one run end to end. Returns a summary dict; never raises
    except on KeyboardInterrupt, which propagates so the queue can stop."""
    _install_sigterm_handler()
    cfg_json = resolved_json(cfg)
    fingerprint = pv.code_fingerprint()
    config_fp = pv.config_fingerprint(cfg_json)
    timestamp = pv.timestamp_slug()
    split_seed_effective = cfg["split"]["split_seed"] if cfg["split"].get("shuffle", False) else None
    run_id = pv.build_run_id(timestamp=timestamp, experiment=cfg["experiment"], loss=spec.loss,
                             fold=spec.fold, split_seed=split_seed_effective,
                             init_seed=spec.init_seed, fingerprint=fingerprint)
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pv.set_status(run_dir, "RUNNING")

    run_handler = lg.attach_run_log(run_dir)
    log = logging.getLogger("srep")
    model = criterion = loaders = None
    env = pv.capture_environment()
    started = pv.utc_now_iso()
    t0 = time.time()

    manifest: Dict = {
        "run_id": run_id, "started_utc": started, "finished_utc": None,
        "duration_seconds": None, "status": "RUNNING",
        "experiment": cfg["experiment"], "driver_script": driver,
        "config_file": cfg.get("_config_file"), "code_fingerprint": fingerprint,
        "config_fingerprint": config_fp,
        "src_file_hashes": pv.src_file_hashes(),
        "loss": {"type": spec.loss, "params": cfg["loss"]["params"][spec.loss]},
        "split": {**{k: cfg["split"][k] for k in ("strategy", "group_by", "n_splits", "fold",
                                                  "val_n_splits")},
                  "shuffle": cfg["split"].get("shuffle", False),
                  "split_seed": split_seed_effective,
                  "allow_participant_leakage": cfg["split"].get("allow_participant_leakage", False)},
        "init_seed": spec.init_seed,
        "deterministic": cfg["deterministic"],
        "strict_deterministic": cfg["strict_deterministic"],
        "context_slices": cfg["data"]["context_slices"],
        "environment": env,
        "resolved_config": cfg,
    }

    try:
        lg.log_banner(log, run_id=run_id, experiment=cfg["experiment"], driver=driver,
                      config_file=str(cfg.get("_config_file")), fingerprint=fingerprint,
                      run_dir=run_dir, env=env, deterministic=cfg["deterministic"],
                      strict=cfg["strict_deterministic"], started=started)
        lg.log_config(log, cfg)
        if gpu.cuda_available():
            log.info("  GPU at run start: %s", gpu.format_snapshot(gpu.memory_snapshot()))
            log.info("  %s", gpu.preflight_env_hint())

        pv.write_environment_txt(run_dir, env)
        pv.snapshot_code(run_dir, cfg_json)

        set_all_seeds(spec.init_seed, cfg["deterministic"], cfg["strict_deterministic"])
        device = resolve_device(cfg["device"])

        df = scan_cache if scan_cache is not None else scan_dataset(
            cfg["data"]["root"], cfg["data"]["original_classes"], cfg["data"]["class_map"])
        lg.log_cohort(log, cohort_summary(df))

        splits = make_splits(df, cfg)
        assert_result = assert_group_disjoint(
            df, splits, cfg["split"]["group_by"],
            allow_leakage=cfg["split"].get("allow_participant_leakage", False))
        if assert_result != "PASS":
            log.warning("  THIS RUN CONTAINS DELIBERATE LEAKAGE: %s",
                        assert_result.splitlines()[0])
        ssum = split_summary(df, splits, len(cfg["data"]["new_classes"]))

        train_labels = df.iloc[splits["train"]]["class_3"].to_numpy()
        weights_np = compute_class_weights(train_labels, len(cfg["data"]["new_classes"]))
        counts = class_counts(train_labels, len(cfg["data"]["new_classes"]))
        gammas = None
        if spec.loss == "FA_FL":
            p = cfg["loss"]["params"]["FA_FL"]
            gammas = [p["gamma_base"] + p["lambda_val"] * float(w) for w in weights_np]
        lg.log_split(log, ssum, cfg, cfg["data"]["new_classes"], weights_np, gammas, assert_result)

        parts = pd.concat([df.iloc[idx].assign(split=name)[
            ["participant_id", "session_id", "class_3", "split"]] for name, idx in splits.items()])
        parts.drop_duplicates().to_csv(run_dir / "fold_participants.csv", index=False)

        generator, worker_init = make_dataloader_rng(spec.init_seed)
        loaders = build_dataloaders(df, splits, cfg, generator, worker_init)

        model = DementiaModel(num_classes=len(cfg["data"]["new_classes"]),
                              pretrained=cfg["model"]["pretrained"],
                              head_hidden=cfg["model"]["head_hidden"],
                              dropout=cfg["model"]["dropout"])
        params = count_parameters(model)
        log.info("MODEL  %s  parameters total=%s trainable=%s",
                 cfg["model"]["arch"], f"{params['total']:,}", f"{params['trainable']:,}")

        criterion = build_criterion(spec.loss, cfg["loss"]["params"][spec.loss],
                                    torch.tensor(weights_np), counts, device)
        crit_desc = describe_criterion(criterion)
        log.info("LOSS   %s  %s", spec.loss, json.dumps(crit_desc, default=str))

        def on_epoch(**kw):
            lg.log_epoch(log, **kw)

        train_info = train_model(model, loaders, criterion, cfg, device, run_dir, log, on_epoch)

        # Training is finished: release its workers and pinned buffers BEFORE any
        # evaluation loader is built. Holding them open through evaluation is what
        # exhausted page-locked memory previously.
        gpu.shutdown_dataloaders(loaders)
        loaders = None
        gpu.release_between_runs(log=log)

        metrics: Dict[str, Dict] = {}
        preds: Dict[str, Dict] = {}
        for split in ("val", "test"):
            log.info(lg.THIN)
            log.info("%s EVALUATION", split.upper())
            pred = evaluate_with_fallback(model, df, splits[split], cfg, device, split, log)
            preds[split] = pred
            ev.save_predictions(run_dir, split, pred)
            m = ev.compute_metrics(pred["probabilities"], pred["labels"],
                                   cfg["data"]["new_classes"], cfg["eval"]["ece_bins"],
                                   cfg["eval"]["ece_binning"])
            m["n_participants"] = int(len(set(pred["participant_id"].tolist())))
            metrics[split] = m
            ev.save_confusion_matrix_csv(run_dir, split, m["confusion_matrix"], cfg["data"]["new_classes"])
            ev.save_classification_report(run_dir, split, pred["probabilities"], pred["labels"],
                                          cfg["data"]["new_classes"])
            lg.log_confusion_matrix(log, np.array(m["confusion_matrix"]),
                                    cfg["data"]["new_classes"], f"Confusion matrix ({split})")
            lg.log_metrics(log, m, cfg["data"]["new_classes"])
            # Temperature is fitted on VALIDATION and applied to TEST -- never
            # fitted on the split it is evaluated on.
            temp_payload = None
            if split == "test" and "val" in preds:
                fit = ev.fit_temperature(preds["val"]["probabilities"], preds["val"]["labels"])
                scaled = ev.apply_temperature(pred["probabilities"], fit["temperature"])
                ece_after, mce_after = ev.expected_calibration_error(
                    scaled, pred["labels"], cfg["eval"]["ece_bins"], cfg["eval"]["ece_binning"])
                brier_after = ev.brier_scores(scaled, pred["labels"], len(cfg["data"]["new_classes"]))
                m["temperature_scaling"] = {
                    **fit, "ece_after": ece_after, "mce_after": mce_after,
                    "brier_sum_after": brier_after["brier_sum"],
                    "ece_before": m["ece"], "brier_sum_before": m["brier_sum"],
                    "accuracy_unchanged": float(100.0 * (scaled.argmax(1) == pred["labels"]).mean()),
                }
                if fit.get("at_search_boundary"):
                    log.warning("  temperature hit the search boundary (T=%.4f, bounds %s). The "
                                "model is extremely miscalibrated; treat the scaled numbers with "
                                "care and widen the bounds if this recurs on real runs.",
                                fit["temperature"], fit["search_bounds"])
                log.info("  temperature scaling: T=%.4f fitted on val | test ECE %.4f -> %.4f "
                         "| Brier %.4f -> %.4f | accuracy unchanged at %.2f%%",
                         fit["temperature"], m["ece"], ece_after,
                         m["brier_sum"], brier_after["brier_sum"],
                         m["temperature_scaling"]["accuracy_unchanged"])
                np.savez_compressed(run_dir / "predictions_test_temperature_scaled.npz",
                                    probabilities=scaled.astype(np.float32),
                                    labels=pred["labels"],
                                    participant_id=pred["participant_id"].astype("U32"),
                                    temperature=np.array([fit["temperature"]]))
                temp_payload = {"probs_scaled": scaled, "ece_after": ece_after,
                                "temperature": fit["temperature"]}

            written = figs.write_run_figures(
                run_dir, split, pred["probabilities"], pred["labels"],
                m["confusion_matrix"], cfg["data"]["new_classes"], m["ece"],
                cfg["eval"]["ece_bins"],
                history=train_info["history"] if split == "test" else None,
                temperature=temp_payload)
            log.info("  figures: %s", ", ".join(written) if written else "none written")

        metrics_payload = {
            "run_id": run_id, "loss": spec.loss, "fold": spec.fold,
            "split_seed": cfg["split"]["split_seed"], "init_seed": spec.init_seed,
            "class_names": cfg["data"]["new_classes"],
            "class_weights": [float(w) for w in weights_np],
            "train_class_counts": counts,
            "criterion": crit_desc,
            "model_parameters": params,
            "training": {k: v for k, v in train_info.items() if k != "history"},
            "split_summary": ssum.to_dict(orient="records"),
            "val": metrics["val"], "test": metrics["test"],
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)

        duration = time.time() - t0
        mem = gpu.memory_snapshot()
        peak_gpu = mem["peak_allocated_gb"] if mem else None
        manifest.update({
            "finished_utc": pv.utc_now_iso(), "duration_seconds": round(duration, 1),
            "status": "COMPLETE",
            "data": {
                "root": cfg["data"]["root"],
                "participants": {r["split"]: r["participants"] for r in metrics_payload["split_summary"]},
                "sessions": {r["split"]: r["sessions"] for r in metrics_payload["split_summary"]},
                "images": {r["split"]: r["images"] for r in metrics_payload["split_summary"]},
                "class_weights": [float(w) for w in weights_np],
                "train_class_counts": counts,
                "group_disjoint_assert": assert_result,
            },
            "training": {k: v for k, v in train_info.items() if k != "history"},
            "gpu_memory": mem,
            "headline": {
                "test_accuracy": round(metrics["test"]["accuracy"], 4),
                "test_macro_f1": round(metrics["test"]["macro_f1"], 6),
                "test_ece": round(metrics["test"]["ece"], 6),
                "test_brier": round(metrics["test"]["brier"], 6),
                "test_ece_temp_scaled": round(
                    metrics["test"].get("temperature_scaling", {}).get("ece_after", float("nan")), 6),
                "temperature": round(
                    metrics["test"].get("temperature_scaling", {}).get("temperature", float("nan")), 4),
                "val_accuracy": round(metrics["val"]["accuracy"], 4),
                "val_macro_f1": round(metrics["val"]["macro_f1"], 6),
            },
        })
        pv.write_manifest(run_dir, manifest)
        pv.set_status(run_dir, "COMPLETE")
        lg.log_footer(log, status="COMPLETE", duration=duration, peak_gpu_gb=peak_gpu,
                      run_dir=run_dir, artifacts=pv.list_artifacts(run_dir))
        return {"status": "COMPLETE", "run_id": run_id, "run_dir": str(run_dir),
                **manifest["headline"]}

    except BaseException as exc:                                  # noqa: BLE001
        duration = time.time() - t0
        log.error("RUN FAILED after %.1fs: %r", duration, exc)
        for line in traceback.format_exc().splitlines():
            log.error("  %s", line)
        manifest.update({"finished_utc": pv.utc_now_iso(), "duration_seconds": round(duration, 1),
                         "status": "FAILED", "error": repr(exc),
                         "traceback": traceback.format_exc()})
        pv.write_manifest(run_dir, manifest)
        pv.set_status(run_dir, "FAILED")
        lg.log_footer(log, status="FAILED", duration=duration, peak_gpu_gb=None,
                      run_dir=run_dir, artifacts=pv.list_artifacts(run_dir))
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {"status": "FAILED", "run_id": run_id, "run_dir": str(run_dir), "error": repr(exc)}

    finally:
        # Runs share one process, so every large object must be dropped here or
        # its VRAM stays held for the rest of the queue.
        #
        # Order matters. Handing the objects to release_between_runs would only
        # rebind ITS parameters -- these locals would still reference the model
        # while gc.collect() ran, and nothing would actually be freed. So: shut
        # the workers down while the loaders still exist, then null every local,
        # then collect.
        gpu.shutdown_dataloaders(loaders)
        model = criterion = loaders = None          # noqa: F841
        gpu.release_between_runs(log=log)
        lg.detach_run_log(run_handler)


# ---------------------------------------------------------------------------

def build_arg_parser(experiment: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Run {experiment}")
    p.add_argument("--dry-run", action="store_true", help="list planned runs, train nothing")
    p.add_argument("--only", type=str, default=None,
                   help="restrict to LOSS[:foldK][:initN], e.g. FA_FL:fold3")
    p.add_argument("--resume", action="store_true",
                   help="skip specs already COMPLETE with the same code fingerprint")
    p.add_argument("--data-root", type=str, default=None, help="override data.root")
    p.add_argument("--device", type=str, default=None, help="auto | cuda | cpu")
    p.add_argument("--epochs", type=int, default=None, help="override train.total_epochs (smoke tests)")
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None,
                   help="override data.image_size (smoke tests only; production is 224)")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline tests)")
    p.add_argument("--limit", type=int, default=None, help="run at most N specs")
    p.add_argument("--console", choices=("full", "progress", "silent"), default="progress",
                   help="terminal verbosity. Full detail always goes to the log files. "
                        "Default 'progress': one heartbeat line per run plus warnings.")
    p.add_argument("--log-dir", type=str, default=None,
                   help="where the queue log is written (default: console_logs/)")
    p.add_argument("--isolate", action="store_true",
                   help="run each spec in its own subprocess. Slower by ~20s per run but "
                        "guarantees the OS reclaims all VRAM, immune to fragmentation.")
    p.add_argument("--min-free-gb", type=float, default=3.0,
                   help="warn if free VRAM drops below this before a run")
    return p


def _apply_cli(cfg: Dict, args: argparse.Namespace) -> Dict:
    if args.data_root:      cfg["data"]["root"] = args.data_root
    if args.device:         cfg["device"] = args.device
    if args.epochs:         cfg["train"]["total_epochs"] = args.epochs
    if args.warmup_epochs is not None: cfg["train"]["warmup_epochs"] = args.warmup_epochs
    if args.batch_size:     cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:   cfg["train"]["num_workers"] = args.num_workers
    if args.image_size:     cfg["data"]["image_size"] = [args.image_size, args.image_size]
    if args.no_pretrained:  cfg["model"]["pretrained"] = False
    return cfg


def _matches(spec: RunSpec, only: str) -> bool:
    for token in only.split(":"):
        if token.startswith("fold"):
            if spec.fold != int(token[4:]):
                return False
        elif token.startswith("init"):
            if spec.init_seed != int(token[4:]):
                return False
        elif token and token != spec.loss:
            return False
    return True


def _run_isolated(cfg: Dict, spec: RunSpec, args, config_name: str,
                  scan_cache_path: Path, log) -> Dict:
    """Execute one spec in a subprocess so the OS reclaims everything on exit."""
    cmd = [sys.executable, str(REPO_ROOT / "run_one.py"),
           "--config", config_name, "--loss", spec.loss,
           "--fold", str(spec.fold), "--init-seed", str(spec.init_seed),
           "--data-root", cfg["data"]["root"], "--device", cfg["device"],
           "--scan-cache", str(scan_cache_path), "--console", "silent"]
    for flag, value in (("--epochs", args.epochs), ("--warmup-epochs", args.warmup_epochs),
                        ("--batch-size", args.batch_size), ("--num-workers", args.num_workers),
                        ("--image-size", args.image_size)):
        if value is not None:
            cmd += [flag, str(value)]
    if args.no_pretrained:
        cmd.append("--no-pretrained")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.stdout.strip():
        for line in completed.stdout.strip().splitlines()[-15:]:
            log.info("  [child] %s", line)
    if completed.returncode != 0:
        for line in (completed.stderr or "").strip().splitlines()[-25:]:
            log.error("  [child] %s", line)
        return {"status": "FAILED", "error": f"subprocess exit {completed.returncode}"}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {"status": "FAILED", "error": "child produced no parseable result line"}


def main(config_name: str, driver_file: str) -> int:
    driver_path = Path(driver_file).resolve()
    outputs_dir = driver_path.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    args = build_arg_parser(config_name).parse_args()
    base_cfg = _apply_cli(load_experiment_config(config_name), args)
    experiment = base_cfg["experiment"]

    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "console_logs"
    queue_log = log_dir / f"{pv.timestamp_slug()}__{experiment}.log"
    log = lg.init_queue_logging(queue_log, console=args.console)

    specs = expand_sweep(base_cfg)
    if args.only:
        specs = [s for s in specs if _matches(s, args.only)]
    if args.limit:
        specs = specs[: args.limit]
    if not specs:
        lg.progress("no runs match the given filters")
        return 1

    lg.progress("QUEUE %s -- %d run(s) planned", experiment, len(specs))
    lg.progress("  log file: %s", queue_log)
    log.info("planned runs:")
    for i, spec in enumerate(specs, 1):
        log.info("  %2d/%d  %s", i, len(specs), spec.label())
    if args.dry_run:
        lg.progress("dry run: nothing executed")
        return 0

    validate_config(config_for_spec(base_cfg, specs[0]), require_data_root=True)
    log.info("%s", gpu.preflight_env_hint())
    if gpu.cuda_available():
        lg.progress("  GPU: %s", gpu.format_snapshot(gpu.memory_snapshot()))

    lg.progress("  scanning dataset ...")
    scan = scan_dataset(base_cfg["data"]["root"], base_cfg["data"]["original_classes"],
                        base_cfg["data"]["class_map"])
    lg.progress("  %s images, %d participants, %d sessions",
                f"{len(scan):,}", scan["participant_id"].nunique(), scan["session_id"].nunique())

    scan_cache_path = outputs_dir / ".scan_cache.csv"
    if args.isolate:
        scan.to_csv(scan_cache_path, index=False)   # children reuse it, saving a rescan each
        log.info("scan cached for subprocesses at %s", scan_cache_path)

    driver_rel = f"experiments/{driver_path.parent.name}/run.py"
    results, t_queue = [], time.time()

    for i, spec in enumerate(specs, 1):
        cfg = config_for_spec(base_cfg, spec)
        if args.resume:
            existing = find_existing(outputs_dir, spec, pv.code_fingerprint(),
                                     pv.config_fingerprint(resolved_json(cfg)))
            if existing:
                lg.progress("[%d/%d] %s -- already COMPLETE, skipping", i, len(specs), spec.label())
                results.append({"status": "SKIPPED", "spec": spec.label()})
                continue

        gpu.assert_headroom(args.min_free_gb, log)
        t_run = time.time()
        lg.progress("[%d/%d] %s -- started", i, len(specs), spec.label())

        if args.isolate:
            outcome = _run_isolated(cfg, spec, args, config_name, scan_cache_path, log)
        else:
            outcome = execute_run(cfg, spec, outputs_dir, driver_rel, scan)

        results.append({"spec": spec.label(), **outcome})
        if outcome.get("status") == "COMPLETE":
            lg.progress("[%d/%d] %s -- COMPLETE in %s | acc %.2f%% macroF1 %.4f ECE %.4f",
                        i, len(specs), spec.label(), lg._fmt_duration(time.time() - t_run),
                        outcome.get("test_accuracy", float("nan")),
                        outcome.get("test_macro_f1", float("nan")),
                        outcome.get("test_ece", float("nan")))
        else:
            lg.progress("[%d/%d] %s -- FAILED after %s (queue continues)",
                        i, len(specs), spec.label(), lg._fmt_duration(time.time() - t_run))

        if gpu.cuda_available():
            log.info("  post-run GPU: %s", gpu.format_snapshot(gpu.memory_snapshot()))

    if args.isolate and scan_cache_path.exists():
        scan_cache_path.unlink()

    n_ok = sum(r.get("status") == "COMPLETE" for r in results)
    n_fail = sum(r.get("status") == "FAILED" for r in results)
    n_skip = sum(r.get("status") == "SKIPPED" for r in results)

    log.info(lg.RULE)
    log.info("QUEUE FINISHED  %d complete, %d failed, %d skipped, wall %s",
             n_ok, n_fail, n_skip, lg._fmt_duration(time.time() - t_queue))
    for r in results:
        if r.get("status") == "COMPLETE":
            log.info("  OK      %-28s acc %6.2f%%  macroF1 %.4f  ECE %.4f",
                     r["spec"], r["test_accuracy"], r["test_macro_f1"], r["test_ece"])
        elif r.get("status") == "FAILED":
            log.info("  FAILED  %-28s %s", r["spec"], str(r.get("error"))[:110].replace("\n", " "))
    log.info(lg.RULE)

    lg.progress("QUEUE FINISHED %s -- %d complete, %d failed, %d skipped, wall %s",
                experiment, n_ok, n_fail, n_skip, lg._fmt_duration(time.time() - t_queue))
    lg.progress("  full log: %s", queue_log)
    return 1 if n_fail else 0
