"""Evaluation, metrics, and prediction persistence.

The critical detail here is that predictions are saved WITH participant and
session identifiers aligned to every row. The test set holds ~17,000 images but
only ~70 independent participants, and 3-4 near-duplicate MPRAGE acquisitions of
each anatomical slice (r = 0.988). Any bootstrap or interval must therefore
resample participants, not images -- which is impossible unless the identifiers
are stored at prediction time. This cannot be added retroactively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15, binning: str = "equal_width") -> Tuple[float, float]:
    """Returns (ECE, MCE) over top-label confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    if binning == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif binning == "equal_mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown binning {binning!r}")

    ece, mce, n = 0.0, 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += mask.sum() / n * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_scores(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> Dict[str, float]:
    """Two conventions, both stored, so the analysis layer never has to guess.

    brier_sum  : mean over samples of sum_c (p_c - y_c)^2   (standard multiclass)
    brier_mean : the same divided by n_classes              (mean-per-class variant)
    """
    onehot = np.eye(n_classes, dtype=np.float64)[labels]
    per_sample = ((probs.astype(np.float64) - onehot) ** 2).sum(axis=1)
    return {"brier_sum": float(per_sample.mean()),
            "brier_mean": float(per_sample.mean() / n_classes)}


def compute_metrics(probs: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
                    ece_bins: int = 15, ece_binning: str = "equal_width") -> Dict:
    n_classes = len(class_names)
    preds = probs.argmax(axis=1)
    labels_range = list(range(n_classes))

    cm = confusion_matrix(labels, preds, labels=labels_range)
    per_class_f1 = f1_score(labels, preds, labels=labels_range, average=None, zero_division=0)
    ece, mce = expected_calibration_error(probs, labels, ece_bins, ece_binning)
    brier = brier_scores(probs, labels, n_classes)

    report = classification_report(labels, preds, labels=labels_range,
                                   target_names=list(class_names), output_dict=True,
                                   zero_division=0)
    return {
        "n_images": int(len(labels)),
        "accuracy": float(100.0 * (preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, labels=labels_range, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=labels_range, average="weighted", zero_division=0)),
        "per_class_f1": [float(v) for v in per_class_f1],
        "per_class_precision": [float(report[c]["precision"]) for c in class_names],
        "per_class_recall": [float(report[c]["recall"]) for c in class_names],
        "per_class_support": [int(report[c]["support"]) for c in class_names],
        "confusion_matrix": cm.tolist(),
        "ece": ece, "mce": mce, "ece_bins": int(ece_bins), "ece_binning": ece_binning,
        **brier,
        "brier": brier["brier_sum"],
    }


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Rescale probabilities by temperature T.

    softmax(z/T) == p**(1/T) / sum(p**(1/T)), so this works from stored
    probabilities and no logits need to be persisted.
    """
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    scaled = np.exp(np.log(p) / max(T, 1e-6))
    return (scaled / scaled.sum(axis=1, keepdims=True)).astype(np.float64)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(val_probs: np.ndarray, val_labels: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0) -> Dict[str, float]:
    """Fit a single temperature on the VALIDATION split by minimising NLL.

    Why this is here at all: the manuscript's surviving claim is a calibration
    advantage. The obvious reviewer response is that temperature scaling is a
    one-parameter post-hoc fix that costs nothing, so any well-trained model can
    be calibrated after the fact. Reporting ECE before AND after temperature
    scaling answers that pre-emptively -- and if the advantage does survive, it
    is a much stronger result than the raw number alone.

    Coarse log-spaced sweep then a local refinement; no scipy dependency.
    """
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    losses = [_nll(apply_temperature(val_probs, T), val_labels) for T in grid]
    best = int(np.argmin(losses))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    fine = np.linspace(a, b, 100)
    losses_fine = [_nll(apply_temperature(val_probs, T), val_labels) for T in fine]
    T = float(fine[int(np.argmin(losses_fine))])
    at_bound = bool(T <= lo * 1.02 or T >= hi * 0.98)
    return {"temperature": T,
            "val_nll_before": _nll(val_probs, val_labels),
            "val_nll_after": float(np.min(losses_fine)),
            "search_bounds": [lo, hi],
            "at_search_boundary": at_bound}


@torch.no_grad()
def predict(model, loader, device) -> Dict[str, np.ndarray]:
    """Inference over a loader. Returns probabilities plus aligned identifiers.

    Identifiers come from the Dataset in loader order. shuffle must be False for
    eval loaders, which build_dataloaders guarantees.
    """
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs = F.softmax(model(images).float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int64)
    ds = loader.dataset
    if len(probs) != len(ds):
        raise RuntimeError(f"prediction/dataset length mismatch: {len(probs)} vs {len(ds)}")
    return {
        "probabilities": probs,
        "labels": labels,
        "participant_id": np.array(ds.participants, dtype=object),
        "session_id": np.array(ds.sessions, dtype=object),
        "image_path": np.array(ds.paths, dtype=object),
    }


def save_predictions(run_dir: Path, split: str, pred: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        run_dir / f"predictions_{split}.npz",
        probabilities=pred["probabilities"],
        labels=pred["labels"],
        participant_id=pred["participant_id"].astype("U32"),
        session_id=pred["session_id"].astype("U32"),
        image_path=pred["image_path"].astype("U256"),
    )


def save_confusion_matrix_csv(run_dir: Path, split: str, cm: List[List[int]],
                              class_names: Sequence[str]) -> None:
    df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_names],
                      columns=[f"pred_{c}" for c in class_names])
    df.to_csv(run_dir / f"confusion_matrix_{split}.csv")


def save_classification_report(run_dir: Path, split: str, probs: np.ndarray,
                               labels: np.ndarray, class_names: Sequence[str]) -> None:
    text = classification_report(labels, probs.argmax(axis=1),
                                 labels=list(range(len(class_names))),
                                 target_names=list(class_names), digits=4, zero_division=0)
    (run_dir / f"classification_report_{split}.txt").write_text(text, encoding="utf-8")
