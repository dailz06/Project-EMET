"""Convolutional autoencoder for featurization (Phase 3).

Role (roadmap variant 1): analysis tool only. The FNO operates on full fields;
the AE's latent z is what the Phase 5 causal audit probes. The AE never feeds
the FNO.

Architecture: 4 stride-2 conv blocks (64 -> 32 -> 16 -> 8 -> 4 spatial),
channels 32-64-128-256, GroupNorm + GELU, flatten -> FC -> z; mirrored
transposed-conv decoder. ~2-4M parameters depending on z_dim.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder over (B, C, 64, 64) fields.

    Inputs (constructor):
        in_channels: int, field channels (1 for u-fields, 2-3 for input stacks).
        z_dim: int, latent dimensionality.
        base_channels: int, channels of the first conv block (doubled per block).

    Outputs:
        forward(x) -> reconstruction (B, C, 64, 64).
        encode(x) -> z (B, z_dim).
        decode(z) -> reconstruction (B, C, 64, 64).
    """

    def __init__(self, in_channels: int = 1, z_dim: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim

        chans = [base_channels * (2**i) for i in range(4)]  # 32, 64, 128, 256
        self._bottleneck_channels = chans[-1]
        self._bottleneck_spatial = 4  # 64 / 2^4

        encoder_layers = []
        prev = in_channels
        for ch in chans:
            encoder_layers += [
                nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=ch),
                nn.GELU(),
            ]
            prev = ch
        self.encoder_conv = nn.Sequential(*encoder_layers)

        flat = self._bottleneck_channels * self._bottleneck_spatial**2
        self.to_latent = nn.Linear(flat, z_dim)
        self.from_latent = nn.Linear(z_dim, flat)

        decoder_layers = []
        rev = list(reversed(chans))
        for i, ch in enumerate(rev):
            out_ch = rev[i + 1] if i + 1 < len(rev) else in_channels
            decoder_layers.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            if i + 1 < len(rev):
                decoder_layers += [
                    nn.GroupNorm(num_groups=8, num_channels=out_ch),
                    nn.GELU(),
                ]
        self.decoder_conv = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.to_latent(h.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(
            -1, self._bottleneck_channels, self._bottleneck_spatial, self._bottleneck_spatial
        )
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
