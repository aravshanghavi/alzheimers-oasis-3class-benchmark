"""Model definition. Unchanged from the submitted architecture."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class DementiaModel(nn.Module):
    """EfficientNet-B0 backbone with a task-specific head.

    The backbone's own classifier is replaced by Identity so that `model.model`
    is a pure feature extractor and `model.classifier` can be optimised alone
    during the warm-up stage.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 head_hidden: int = 512, dropout: float = 0.7):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features   # 1280
        self.model.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.model(x))

    @property
    def gradcam_target_layers(self):
        return [self.model.features[-1]]


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
