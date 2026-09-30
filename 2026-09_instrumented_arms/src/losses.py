"""Loss functions.

WCE, StandardFocalLoss, LDAMLoss and FrequencyAdaptiveFocalLoss are ported
verbatim from review_experiments/'Loss Function Ablation with Random State'/
random_seed_experiment.py so that corrected results remain comparable with the
submitted ones. They are NOT "improved" here.

ClassBalancedLoss is new: the one appropriate additional imbalance-aware
baseline requested by Reviewer 1 (comment 6).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StandardFocalLoss(nn.Module):
    """Lin et al. 2017. Single global gamma shared across all classes."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = float(gamma)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha.gather(0, targets)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


class FrequencyAdaptiveFocalLoss(nn.Module):
    """FA-FL, the proposed method.

    gamma_t = gamma_base + lambda * alpha_t, with alpha the normalised
    inverse-frequency class weight. With alpha = [0.0555, 0.2732, 0.6714],
    gamma_base = 1.0 and lambda = 7.0 this gives [1.389, 2.912, 5.700].
    """

    def __init__(self, alpha: torch.Tensor, gamma_base: float = 2.0, lambda_val: float = 1.0):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            raise TypeError("alpha must be a torch.Tensor")
        self.register_buffer("alpha", alpha)
        self.register_buffer("gammas", gamma_base + lambda_val * alpha)
        self.gamma_base, self.lambda_val = float(gamma_base), float(lambda_val)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha.gather(0, targets)
        gamma_t = self.gammas.gather(0, targets)
        return (alpha_t * torch.pow(1 - pt, gamma_t) * ce).mean()


class LDAMLoss(nn.Module):
    """Cao et al. 2019, label-distribution-aware margin."""

    def __init__(self, cls_num_list: Sequence[int], max_m: float = 0.5, s: float = 30.0,
                 weight: torch.Tensor | None = None):
        super().__init__()
        counts = np.asarray(cls_num_list, dtype=np.float64)
        if (counts <= 0).any():
            raise ValueError(f"LDAM requires a positive count for every class, got {cls_num_list}")
        m_list = 1.0 / np.sqrt(np.sqrt(counts))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = float(s)
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))

    def forward(self, inputs, targets):
        index = torch.zeros_like(inputs, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_m = self.m_list[targets].view(-1, 1)
        x_m = inputs - batch_m
        output = torch.where(index, x_m, inputs)
        w = self.weight if self.weight.numel() else None
        return F.cross_entropy(self.s * output, targets, weight=w)


class ClassBalancedLoss(nn.Module):
    """Cui et al., CVPR 2019. Weights by effective number (1-beta)/(1-beta^n_c).

    beta is fixed at the configured value and is NOT tuned; that is stated in
    the manuscript. gamma=0 gives class-balanced cross-entropy, which is the
    variant used here.
    """

    def __init__(self, cls_num_list: Sequence[int], beta: float = 0.999, gamma: float = 0.0):
        super().__init__()
        counts = np.asarray(cls_num_list, dtype=np.float64)
        if (counts <= 0).any():
            raise ValueError(f"ClassBalanced requires a positive count per class, got {cls_num_list}")
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.beta, self.gamma = float(beta), float(gamma)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        w_t = self.weights.gather(0, targets)
        if self.gamma > 0:
            pt = torch.exp(-ce)
            ce = (1 - pt) ** self.gamma * ce
        return (w_t * ce).mean()


def build_criterion(loss_type: str, params: Dict, class_weights: torch.Tensor,
                    cls_num_list: List[int], device) -> nn.Module:
    alpha = class_weights.to(device)
    if loss_type == "WCE":
        crit: nn.Module = nn.CrossEntropyLoss(weight=alpha)
    elif loss_type == "FA_FL":
        crit = FrequencyAdaptiveFocalLoss(alpha=alpha, **params)
    elif loss_type == "FocalLoss":
        crit = StandardFocalLoss(alpha=alpha, **params)
    elif loss_type == "LDAM":
        crit = LDAMLoss(cls_num_list=cls_num_list, **params)
    elif loss_type == "ClassBalanced":
        crit = ClassBalancedLoss(cls_num_list=cls_num_list, **params)
    else:
        raise ValueError(f"unknown loss type {loss_type!r}")
    return crit.to(device)


def describe_criterion(criterion: nn.Module) -> Dict[str, object]:
    """Values worth writing into the manifest and the log."""
    out: Dict[str, object] = {"class": type(criterion).__name__}
    for attr in ("gamma", "gamma_base", "lambda_val", "s", "beta"):
        if hasattr(criterion, attr):
            out[attr] = getattr(criterion, attr)
    for buf in ("alpha", "gammas", "weights", "m_list", "weight"):
        if hasattr(criterion, buf):
            t = getattr(criterion, buf)
            if isinstance(t, torch.Tensor) and t.numel():
                out[buf] = [round(float(v), 6) for v in t.detach().cpu().flatten()]
    if hasattr(criterion, "weight") and isinstance(criterion.weight, torch.Tensor) and criterion.weight.numel():
        out["alpha"] = [round(float(v), 6) for v in criterion.weight.detach().cpu().flatten()]
    return out
