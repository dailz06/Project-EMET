"""Severity regression head: fields -> scalar sigma.

A small CNN (same encoder shape as the autoencoder) predicting the severity
scalar directly from input fields. Used in Phase 4/5 as the crisp scalar-task
shortcut anchor: the field-level FNO collapse can be partially masked by the
parts of u(T) that u0 already determines, whereas sigma regression isolates
exactly the quantity the nuisance is correlated with.
"""

import torch
import torch.nn as nn


class SeverityHead(nn.Module):
    """
    CNN regressor over (B, C, 64, 64) fields -> (B, 1) severity.

    Inputs (constructor):
        in_channels: int, input field channels.
        base_channels: int, first conv width (doubled per block).

    Outputs:
        forward(x) -> (B, 1) severity prediction.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        chans = [base_channels * (2**i) for i in range(4)]  # 32, 64, 128, 256

        layers = []
        prev = in_channels
        for ch in chans:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=ch),
                nn.GELU(),
            ]
            prev = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(chans[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.features(x)).flatten(start_dim=1)
        return self.head(h)
