from __future__ import annotations

import torch
import torch.nn as nn


class PriorResidualRefiner(nn.Module):
    """Wrap a dual-input backbone as a residual correction on the prior channel."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        correction = self.backbone(inputs)
        prior = inputs[:, 1:2]
        return prior + correction, correction
