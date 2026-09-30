"""Two-stage training. Protocol unchanged from the submitted version.

Stage 1  warm-up      5 epochs, backbone frozen, head only, AdamW lr 1e-4
Stage 2  fine-tuning 15 epochs, all layers, AdamW lr 1e-5 wd 1e-4,
                     cosine annealing to 1e-7, AMP, grad accumulation 2,
                     grad clipping at max-norm 1.0, early stopping patience 5

One inherited quirk is preserved deliberately: early stopping monitors
validation LOSS while checkpoint selection uses validation ACCURACY. The
submitted manuscript describes both as accuracy-based, which is wrong. The
behaviour is kept for comparability and both metrics are now configurable and
recorded, so the paper can describe what the code actually does.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from . import diagnostics as dg


def make_amp(device: torch.device, enabled: bool):
    """AMP context and scaler, tolerant of torch>=2.4 and older APIs."""
    use = bool(enabled) and device.type == "cuda"
    try:
        from torch.amp import GradScaler, autocast  # torch >= 2.4
        return (lambda: autocast(device_type=device.type, enabled=use)), GradScaler(device.type, enabled=use)
    except Exception:
        from torch.cuda.amp import GradScaler, autocast  # legacy
        return (lambda: autocast(enabled=use)), GradScaler(enabled=use)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be min or max")
        self.patience, self.mode = patience, mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter, self.early_stop = 0, False

    def _better(self, value: float) -> bool:
        return value < self.best if self.mode == "min" else value > self.best

    def __call__(self, value: float) -> None:
        if self._better(value):
            self.best, self.counter = value, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


@torch.no_grad()
def _validate(model, loader, criterion, device, amp_ctx,
              stats: Optional[dg.EpochStats] = None) -> Tuple[float, float]:
    """Numerics unchanged. When `stats` is supplied, per-class diagnostics are
    accumulated from the same forward pass, so instrumentation costs one mirror
    computation per batch and no additional passes over the data."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx():
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += images.size(0)
        if stats is not None:
            ps, _ = dg.per_sample_loss(criterion, outputs.float(), labels)
            stats.update(outputs, labels, ps)
    return total_loss / max(seen, 1), 100.0 * correct / max(seen, 1)


def train_model(model, loaders: Dict, criterion, cfg: Dict, device: torch.device,
                run_dir: Path, log, epoch_callback: Optional[Callable] = None) -> Dict:
    tcfg = cfg["train"]
    ckpt_path = run_dir / "checkpoint_best.pt"
    model.to(device)

    # Per-class diagnostics. On by default in this stack; they are the markers
    # that explain WHY two arms diverge rather than only that they did. They are
    # computed on detached outputs under no_grad and never enter the graph, so
    # training numerics are identical with them on or off.
    diag = bool(tcfg.get("diagnostics", True))
    n_classes = len(cfg["data"]["new_classes"])
    mirror_diff = float("nan")

    amp_ctx, scaler = make_amp(device, tcfg["amp"])
    history: List[Dict] = []
    t_start = time.time()
    if diag:
        log.info("DIAG   per-class diagnostics ON: per-epoch loss share, recall, confidence "
                 "and ECE for train and val, written to training_curve.csv")

    # ---------------- Stage 1: warm-up, classifier head only ----------------
    for p in model.model.parameters():
        p.requires_grad = False
    optimizer = optim.AdamW(model.classifier.parameters(), lr=tcfg["lr_warmup"])
    n_warmup = tcfg["warmup_epochs"]
    log.info("STAGE 1: WARM-UP -- head only, %d epochs, lr=%.1e", n_warmup, tcfg["lr_warmup"])

    for epoch in range(1, n_warmup + 1):
        t0 = time.time()
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        tr_stats = dg.EpochStats(n_classes) if diag else None
        checked = False
        for images, labels in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += images.size(0)
            if diag:
                # Diagnostics only. Detached, no_grad, after the optimizer step,
                # so nothing here can influence the weights that get trained.
                with torch.no_grad():
                    det = outputs.detach().float()
                    if not checked:
                        mirror_diff = dg.verify_mirror(criterion, det, labels)
                        checked = True
                    ps, _ = dg.per_sample_loss(criterion, det, labels)
                    tr_stats.update(det, labels, ps)

        tr_loss, tr_acc = running_loss / max(seen, 1), 100.0 * correct / max(seen, 1)
        va_stats = dg.EpochStats(n_classes) if diag else None
        va_loss, va_acc = _validate(model, loaders["val"], criterion, device, amp_ctx, va_stats)
        dt, elapsed = time.time() - t0, time.time() - t_start
        remaining = (n_warmup - epoch) + (tcfg["total_epochs"] - n_warmup)
        row = {"epoch": epoch, "phase": "warmup", "lr": tcfg["lr_warmup"],
               "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc, "epoch_seconds": dt}
        if diag:
            row.update(tr_stats.as_dict("tr"))
            row.update(va_stats.as_dict("va"))
            row["mirror_abs_diff"] = mirror_diff
        history.append(row)
        if epoch_callback:
            epoch_callback(stage="warmup", epoch=epoch, total=tcfg["total_epochs"],
                           lr=tcfg["lr_warmup"], train_loss=tr_loss, train_acc=tr_acc,
                           val_loss=va_loss, val_acc=va_acc, epoch_seconds=dt,
                           elapsed=elapsed, eta=dt * remaining, improved=False,
                           best=float("nan"), metric_name=tcfg["checkpoint_metric"])

    # ---------------- Stage 2: full fine-tuning ----------------
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=tcfg["lr_finetune"],
                            weight_decay=tcfg["weight_decay"])
    n_finetune = tcfg["total_epochs"] - n_warmup
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_finetune, eta_min=tcfg["eta_min"])

    stop_metric, ckpt_metric = tcfg["early_stop_metric"], tcfg["checkpoint_metric"]
    stopper = EarlyStopping(tcfg["early_stopping_patience"], mode="min" if stop_metric == "val_loss" else "max")
    best = float("inf") if ckpt_metric == "val_loss" else -float("inf")
    best_epoch, accum = 0, tcfg["grad_accumulation_steps"]

    log.info("STAGE 2: FINE-TUNING -- all layers, %d epochs, lr=%.1e, cosine -> %.1e",
             n_finetune, tcfg["lr_finetune"], tcfg["eta_min"])
    log.info("  early stop on %s (patience %d) | checkpoint on %s",
             stop_metric, tcfg["early_stopping_patience"], ckpt_metric)

    for epoch in range(1, n_finetune + 1):
        t0 = time.time()
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        tr_stats = dg.EpochStats(n_classes) if diag else None
        checked = False

        for i, (images, labels) in enumerate(loaders["train"]):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx():
                outputs = model(images)
                loss = criterion(outputs, labels) / accum
            scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tcfg["grad_max_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * images.size(0) * accum
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += images.size(0)
            if diag:
                # Detached and outside autocast: half-precision logits would make
                # the mirror check fail on precision rather than on drift.
                with torch.no_grad():
                    det = outputs.detach().float()
                    if not checked:
                        mirror_diff = dg.verify_mirror(criterion, det, labels)
                        checked = True
                    ps, _ = dg.per_sample_loss(criterion, det, labels)
                    tr_stats.update(det, labels, ps)

        tr_loss, tr_acc = running_loss / max(seen, 1), 100.0 * correct / max(seen, 1)
        va_stats = dg.EpochStats(n_classes) if diag else None
        va_loss, va_acc = _validate(model, loaders["val"], criterion, device, amp_ctx, va_stats)
        current_lr = optimizer.param_groups[0]["lr"]

        value = va_loss if ckpt_metric == "val_loss" else va_acc
        improved = value < best if ckpt_metric == "val_loss" else value > best
        if improved:
            best, best_epoch = value, n_warmup + epoch
            torch.save(model.state_dict(), ckpt_path)

        dt, elapsed = time.time() - t0, time.time() - t_start
        row = {"epoch": n_warmup + epoch, "phase": "finetune", "lr": current_lr,
               "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc, "epoch_seconds": dt,
               "selected": bool(improved)}
        if diag:
            row.update(tr_stats.as_dict("tr"))
            row.update(va_stats.as_dict("va"))
            row["mirror_abs_diff"] = mirror_diff
        history.append(row)
        if epoch_callback:
            epoch_callback(stage="finetune", epoch=n_warmup + epoch, total=tcfg["total_epochs"],
                           lr=current_lr, train_loss=tr_loss, train_acc=tr_acc,
                           val_loss=va_loss, val_acc=va_acc, epoch_seconds=dt,
                           elapsed=elapsed, eta=dt * (n_finetune - epoch), improved=improved,
                           best=best, metric_name=ckpt_metric)

        scheduler.step()
        stopper(va_loss if stop_metric == "val_loss" else va_acc)
        if stopper.early_stop:
            log.info("  early stopping triggered at epoch %d (%s stopped improving)",
                     n_warmup + epoch, stop_metric)
            break

    pd.DataFrame(history).to_csv(run_dir / "training_curve.csv", index=False)

    if ckpt_path.exists():
        # weights_only=True: this is our own state_dict, and it silences the
        # pickle-security FutureWarning that torch>=2.4 emits.
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:                      # torch < 2.0 has no weights_only
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        log.info("  restored best checkpoint from epoch %d (%s=%.4f)", best_epoch, ckpt_metric, best)
    else:
        log.warning("  no checkpoint was written; evaluating the final-epoch model")

    return {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_value": None if best in (float("inf"), -float("inf")) else float(best),
        "checkpoint_metric": ckpt_metric,
        "early_stop_metric": stop_metric,
        "early_stopped": bool(stopper.early_stop),
        "train_seconds": float(time.time() - t_start),
        "history": history,
    }
