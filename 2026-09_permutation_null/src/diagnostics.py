"""Per-class training diagnostics: the markers that explain WHY arms differ.

WHAT THIS IS FOR
----------------
The existing per-epoch log records aggregate loss and accuracy only. That tells
you where an arm ended up, never where it diverged from another arm or why. The
FA-FL mechanism claim is specifically a claim about where the loss (and hence
the gradient) is concentrated across classes, so the discriminating measurement
is the per-class share of total loss. Nothing in the pipeline measured it.

WHY THE LOSS IS MIRRORED RATHER THAN MODIFIED
---------------------------------------------
Every loss in src/losses.py reduces internally to a scalar, so per-sample values
are not recoverable from the criterion's return value. The obvious fix -- adding
a `reduction` argument to each loss -- would edit the one file whose byte-for-byte
stability makes results comparable with the 2026-08 runs. losses.py is left
alone. `per_sample_loss` re-derives the per-sample vector here instead.

A mirror can drift from its original, so it is checked rather than trusted:
`verify_mirror` reduces the per-sample vector exactly the way the criterion does
and compares against the criterion's own output. Two of the five losses reduce
by a WEIGHTED mean -- `F.cross_entropy(..., weight=w)` divides by the sum of the
weights, not by N -- so a naive `.mean()` comparison would fail for WCE and LDAM
and pass for the other three. That distinction is encoded below, and the check
runs on the first batch of every epoch.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# How each criterion turns its per-sample vector into the scalar it returns.
#   "mean"     -> per_sample.mean()
#   "weighted" -> per_sample.sum() / weight[targets].sum()
REDUCTION = {
    "CrossEntropyLoss": "weighted",        # WCE, nn.CrossEntropyLoss(weight=alpha)
    "LDAMLoss": "weighted",                # F.cross_entropy(s*z, y, weight=w)
    "StandardFocalLoss": "mean",
    "FrequencyAdaptiveFocalLoss": "mean",
    "ClassBalancedLoss": "mean",
}


def per_sample_loss(criterion, inputs: torch.Tensor,
                    targets: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The per-sample loss vector, mirroring src/losses.py exactly.

    Returns (per_sample, class_weight_or_None). The weight is returned because
    the weighted losses need it to reproduce their own reduction.
    """
    name = type(criterion).__name__

    if name == "CrossEntropyLoss":
        w = criterion.weight
        return F.cross_entropy(inputs, targets, weight=w, reduction="none"), w

    if name == "StandardFocalLoss":
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = criterion.alpha.gather(0, targets)
        return alpha_t * (1 - pt) ** criterion.gamma * ce, None

    if name == "FrequencyAdaptiveFocalLoss":
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = criterion.alpha.gather(0, targets)
        gamma_t = criterion.gammas.gather(0, targets)
        return alpha_t * torch.pow(1 - pt, gamma_t) * ce, None

    if name == "LDAMLoss":
        index = torch.zeros_like(inputs, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_m = criterion.m_list[targets].view(-1, 1)
        x_m = inputs - batch_m
        output = torch.where(index, x_m, inputs)
        w = criterion.weight if criterion.weight.numel() else None
        return F.cross_entropy(criterion.s * output, targets, weight=w, reduction="none"), w

    if name == "ClassBalancedLoss":
        ce = F.cross_entropy(inputs, targets, reduction="none")
        if criterion.gamma > 0:
            pt = torch.exp(-ce)
            ce = (1 - pt) ** criterion.gamma * ce
        w_t = criterion.weights.gather(0, targets)
        return w_t * ce, None

    raise TypeError(
        f"no per-sample mirror for criterion {name!r}. Add one here rather than "
        f"changing src/losses.py, and extend REDUCTION with how it reduces.")


def reduce_like(per_sample: torch.Tensor, targets: torch.Tensor,
                criterion, weight: Optional[torch.Tensor]) -> torch.Tensor:
    mode = REDUCTION.get(type(criterion).__name__, "mean")
    if mode == "mean" or weight is None:
        return per_sample.mean()
    return per_sample.sum() / weight.gather(0, targets).sum()


def verify_mirror(criterion, inputs: torch.Tensor, targets: torch.Tensor,
                  rtol: float = 1e-4, atol: float = 1e-6) -> float:
    """Reduce the mirror the way the criterion does and compare. Returns the
    absolute difference; raises if it exceeds tolerance.

    Run once per epoch. If the mirror ever drifts from losses.py, every
    per-class diagnostic derived from it is wrong, and a silently wrong
    diagnostic is worse than none.
    """
    with torch.no_grad():
        ps, w = per_sample_loss(criterion, inputs, targets)
        mine = reduce_like(ps, targets, criterion, w)
        theirs = criterion(inputs, targets)
    diff = float(torch.abs(mine - theirs))
    tol = atol + rtol * float(torch.abs(theirs))
    if diff > tol:
        raise RuntimeError(
            f"per-sample mirror disagrees with {type(criterion).__name__}: "
            f"mirror {float(mine):.8f} vs criterion {float(theirs):.8f} "
            f"(|diff| {diff:.2e} > tol {tol:.2e}). src/diagnostics.py has drifted "
            f"from src/losses.py; fix the mirror before trusting any per-class number.")
    return diff


class EpochStats:
    """Per-class accumulators for one pass over a split.

    Everything here is additive over batches, so it costs one pass and no
    storage proportional to the dataset.
    """

    def __init__(self, n_classes: int, n_bins: int = 15):
        self.k, self.n_bins = n_classes, n_bins
        self.count = np.zeros(n_classes)
        self.correct = np.zeros(n_classes)
        self.conf_sum = np.zeros(n_classes)        # confidence in the PREDICTED class
        self.true_prob_sum = np.zeros(n_classes)   # probability on the TRUE class
        self.loss_sum = np.zeros(n_classes)        # per-class share of total loss
        self.pred_count = np.zeros(n_classes)
        self.bin_n = np.zeros(n_bins)
        self.bin_conf = np.zeros(n_bins)
        self.bin_correct = np.zeros(n_bins)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               per_sample: Optional[torch.Tensor] = None) -> None:
        probs = torch.softmax(logits.float(), dim=1)
        conf, pred = probs.max(dim=1)
        y = targets.detach().cpu().numpy()
        p = pred.detach().cpu().numpy()
        c = conf.detach().cpu().numpy()
        tp = probs.gather(1, targets.view(-1, 1)).squeeze(1).detach().cpu().numpy()
        ok = (p == y).astype(np.float64)

        np.add.at(self.count, y, 1.0)
        np.add.at(self.correct, y, ok)
        np.add.at(self.conf_sum, y, c)
        np.add.at(self.true_prob_sum, y, tp)
        np.add.at(self.pred_count, p, 1.0)
        if per_sample is not None:
            np.add.at(self.loss_sum, y, per_sample.detach().float().cpu().numpy())

        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        b = np.clip(np.digitize(c, edges[1:-1], right=True), 0, self.n_bins - 1)
        np.add.at(self.bin_n, b, 1.0)
        np.add.at(self.bin_conf, b, c)
        np.add.at(self.bin_correct, b, ok)

    def ece(self) -> float:
        total = self.bin_n.sum()
        if total == 0:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = np.where(self.bin_n > 0, self.bin_correct / self.bin_n, 0.0)
            cnf = np.where(self.bin_n > 0, self.bin_conf / self.bin_n, 0.0)
        return float((self.bin_n / total * np.abs(acc - cnf)).sum())

    def as_dict(self, prefix: str) -> Dict[str, float]:
        """Flat columns, ready to append to the training-curve row."""
        out: Dict[str, float] = {}
        n = np.maximum(self.count, 1.0)
        total_loss = self.loss_sum.sum()
        for c in range(self.k):
            out[f"{prefix}_n_c{c}"] = float(self.count[c])
            out[f"{prefix}_recall_c{c}"] = float(self.correct[c] / n[c])
            out[f"{prefix}_conf_c{c}"] = float(self.conf_sum[c] / n[c])
            out[f"{prefix}_trueprob_c{c}"] = float(self.true_prob_sum[c] / n[c])
            out[f"{prefix}_predshare_c{c}"] = float(self.pred_count[c] / max(self.count.sum(), 1.0))
            # The mechanism measurement: what fraction of total loss this class carries.
            out[f"{prefix}_lossshare_c{c}"] = (
                float(self.loss_sum[c] / total_loss) if total_loss > 0 else float("nan"))
            out[f"{prefix}_lossmean_c{c}"] = float(self.loss_sum[c] / n[c])
        out[f"{prefix}_ece"] = self.ece()
        out[f"{prefix}_conf_mean"] = float(self.conf_sum.sum() / max(self.count.sum(), 1.0))
        out[f"{prefix}_acc"] = float(self.correct.sum() / max(self.count.sum(), 1.0))
        return out
