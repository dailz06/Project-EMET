"""FNO surrogate factory.

Adapted from create_fno_model in phase0_phase1_local_pipeline.py (L1114);
in_channels is configurable because Phase 2 uses (u0, M) while Phase 4 trains
biased (u0, s), control (u0, s), and oracle (u0, M) variants.
"""

import torch.nn as nn
from neuralop.models import FNO


def create_fno_model(config: dict) -> nn.Module:
    """
    Create a 2D Fourier Neural Operator.

    Inputs:
        config: dict with modes, hidden_channels, in_channels, out_channels.

    Outputs:
        model: torch.nn.Module mapping (B, in_channels, nx, ny) to
            (B, out_channels, nx, ny).
    """
    modes = int(config["modes"])

    return FNO(
        n_modes=(modes, modes),
        in_channels=int(config.get("in_channels", 2)),
        out_channels=int(config.get("out_channels", 1)),
        hidden_channels=int(config["hidden_channels"]),
    )
