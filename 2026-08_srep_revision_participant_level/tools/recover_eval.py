#!/usr/bin/env python3
"""Finish a run that trained successfully but failed during evaluation.

A run that reaches TEST EVALUATION has already saved checkpoint_best.pt, so a
failure there costs the whole training run for nothing. This reloads that
checkpoint, redoes evaluation with the OOM-degrading fallbacks, writes every
missing artifact into the SAME run folder, and flips its status to COMPLETE.

    python tools/recover_eval.py --run-dir experiments/exp01_main_sweep/outputs/2026...__6dbc408c
    python tools/recover_eval.py --all          # every _FAILED run with a checkpoint

The recovered manifest records recovered_from_failure: true, so nothing is
silently passed off as a clean first-pass run.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import evaluate as ev, figures as figs, gpu, logging_utils as lg, provenance as pv  # noqa: E402
from src.config import _normalise_class_map  # noqa: E402
from src.data import (assert_group_disjoint, class_counts, compute_class_weights,  # noqa: E402
                      make_splits, scan_dataset, split_summary)
from src.model import DementiaModel  # noqa: E402
from src.runner import evaluate_with_fallback  # noqa: E402
from src.seeding import resolve_device, set_all_seeds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def recover(run_dir: Path, log, scan: pd.DataFrame | None = None,
            data_root: str | None = None) -> bool:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["resolved_config"]
    # JSON has no integer keys, so class_map came back as {"0": 0, ...}.
    _normalise_class_map(cfg)
    if data_root:
        cfg["data"]["root"] = data_root
    ckpt = run_dir / "checkpoint_best.pt"
    if not ckpt.exists():
        log.error("%s has no checkpoint_best.pt -- training never finished; re-run it",
                  run_dir.name)
        return False

    log.info(lg.RULE)
    log.info("RECOVERING %s", run_dir.name)
    log.info("  original status %s | failed after %.0fs",
             manifest.get("status"), manifest.get("duration_seconds") or 0)
    t0 = time.time()

    set_all_seeds(manifest["init_seed"], cfg["deterministic"], cfg["strict_deterministic"])
    device = resolve_device(cfg["device"])

    df = scan if scan is not None else scan_dataset(
        cfg["data"]["root"], cfg["data"]["original_classes"], cfg["data"]["class_map"])
    splits = make_splits(df, cfg)
    assert_result = assert_group_disjoint(
        df, splits, cfg["split"]["group_by"],
        allow_leakage=cfg["split"].get("allow_participant_leakage", False))
    log.info("  splits rebuilt, disjointness: %s", assert_result.splitlines()[0])

    model = DementiaModel(num_classes=len(cfg["data"]["new_classes"]),
                          pretrained=False,
                          head_hidden=cfg["model"]["head_hidden"],
                          dropout=cfg["model"]["dropout"])
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    log.info("  checkpoint loaded from epoch %s", manifest.get("training", {}).get("best_epoch"))

    train_labels = df.iloc[splits["train"]]["class_3"].to_numpy()
    weights_np = compute_class_weights(train_labels, len(cfg["data"]["new_classes"]))
    counts = class_counts(train_labels, len(cfg["data"]["new_classes"]))
    ssum = split_summary(df, splits, len(cfg["data"]["new_classes"]))

    history = []
    curve = run_dir / "training_curve.csv"
    if curve.exists():
        history = pd.read_csv(curve).to_dict(orient="records")

    metrics, preds = {}, {}
    for split in ("val", "test"):
        log.info(lg.THIN)
        log.info("%s EVALUATION", split.upper())
        pred = evaluate_with_fallback(model, df, splits[split], cfg, device, split, log)
        preds[split] = pred
        ev.save_predictions(run_dir, split, pred)
        m = ev.compute_metrics(pred["probabilities"], pred["labels"], cfg["data"]["new_classes"],
                               cfg["eval"]["ece_bins"], cfg["eval"]["ece_binning"])
        m["n_participants"] = int(len(set(pred["participant_id"].tolist())))
        metrics[split] = m
        ev.save_confusion_matrix_csv(run_dir, split, m["confusion_matrix"], cfg["data"]["new_classes"])
        ev.save_classification_report(run_dir, split, pred["probabilities"], pred["labels"],
                                      cfg["data"]["new_classes"])
        lg.log_confusion_matrix(log, np.array(m["confusion_matrix"]),
                                cfg["data"]["new_classes"], f"Confusion matrix ({split})")
        lg.log_metrics(log, m, cfg["data"]["new_classes"])

        temp_payload = None
        if split == "test" and "val" in preds:
            fit = ev.fit_temperature(preds["val"]["probabilities"], preds["val"]["labels"])
            scaled = ev.apply_temperature(pred["probabilities"], fit["temperature"])
            ece_after, mce_after = ev.expected_calibration_error(
                scaled, pred["labels"], cfg["eval"]["ece_bins"], cfg["eval"]["ece_binning"])
            brier_after = ev.brier_scores(scaled, pred["labels"], len(cfg["data"]["new_classes"]))
            m["temperature_scaling"] = {
                **fit, "ece_after": ece_after, "mce_after": mce_after,
                "brier_sum_after": brier_after["brier_sum"], "ece_before": m["ece"],
                "brier_sum_before": m["brier_sum"],
                "accuracy_unchanged": float(100.0 * (scaled.argmax(1) == pred["labels"]).mean())}
            log.info("  temperature scaling: T=%.4f | test ECE %.4f -> %.4f",
                     fit["temperature"], m["ece"], ece_after)
            np.savez_compressed(run_dir / "predictions_test_temperature_scaled.npz",
                                probabilities=scaled.astype(np.float32), labels=pred["labels"],
                                participant_id=pred["participant_id"].astype("U32"),
                                temperature=np.array([fit["temperature"]]))
            temp_payload = {"probs_scaled": scaled, "ece_after": ece_after,
                            "temperature": fit["temperature"]}

        figs.write_run_figures(run_dir, split, pred["probabilities"], pred["labels"],
                               m["confusion_matrix"], cfg["data"]["new_classes"], m["ece"],
                               cfg["eval"]["ece_bins"],
                               history=history if split == "test" else None,
                               temperature=temp_payload)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"run_id": manifest["run_id"], "loss": manifest["loss"]["type"],
                   "fold": manifest["split"]["fold"], "split_seed": manifest["split"]["split_seed"],
                   "init_seed": manifest["init_seed"],
                   "class_names": cfg["data"]["new_classes"],
                   "class_weights": [float(w) for w in weights_np],
                   "train_class_counts": counts,
                   "training": manifest.get("training", {}),
                   "split_summary": ssum.to_dict(orient="records"),
                   "val": metrics["val"], "test": metrics["test"]}, fh, indent=2)

    manifest.update({
        "status": "COMPLETE",
        "recovered_from_failure": True,
        "recovery_utc": pv.utc_now_iso(),
        "recovery_seconds": round(time.time() - t0, 1),
        "original_error": manifest.get("error"),
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
            "val_macro_f1": round(metrics["val"]["macro_f1"], 6)},
    })
    pv.write_manifest(run_dir, manifest)
    pv.set_status(run_dir, "COMPLETE")
    log.info("RECOVERED in %s -- acc %.2f%% macroF1 %.4f ECE %.4f",
             lg._fmt_duration(time.time() - t0), metrics["test"]["accuracy"],
             metrics["test"]["macro_f1"], metrics["test"]["ece"])
    log.info(lg.RULE)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--all", action="store_true", help="every _FAILED run holding a checkpoint")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    log = lg.init_queue_logging(REPO_ROOT / "console_logs" /
                                f"{pv.timestamp_slug()}__recover_eval.log", console="full")

    if args.all:
        targets = sorted(p.parent for p in
                         (REPO_ROOT / "experiments").glob("*/outputs/*/_FAILED"))
    elif args.run_dir:
        targets = [Path(args.run_dir).resolve()]
    else:
        ap.error("pass --run-dir or --all")

    targets = [t for t in targets if (t / "checkpoint_best.pt").exists()]
    if not targets:
        log.info("nothing to recover (no failed run holds a checkpoint)")
        return 0
    log.info("%d run(s) to recover", len(targets))

    scan = None
    ok = 0
    for run_dir in targets:
        cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["resolved_config"]
        _normalise_class_map(cfg)
        if args.data_root:
            cfg["data"]["root"] = args.data_root
        if scan is None:
            scan = scan_dataset(cfg["data"]["root"], cfg["data"]["original_classes"],
                                cfg["data"]["class_map"])
        try:
            ok += bool(recover(run_dir, log, scan, args.data_root))
        except Exception as exc:                                     # noqa: BLE001
            log.error("recovery failed for %s: %r", run_dir.name, exc)
        gpu.release_between_runs(log=None)
    log.info("recovered %d of %d", ok, len(targets))
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
