"""Seeding utilities.

Ported from phase0_phase1_local_pipeline.py (set_seed L284, get_device L309)
with one deliberate upgrade: per-sample rng streams via np.random.SeedSequence
spawn keys. Phase 5 intervention probing must regenerate individual samples
exactly (fix C, vary S and vice versa), which a single sequential rng cannot do.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility (Python, NumPy, torch CPU+CUDA).

    Inputs:
        seed: int.

    Outputs:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Select CUDA if available, otherwise CPU.

    Outputs:
        device: torch.device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Active device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Active device: cpu")
    return device


def rng_for_sample(master_seed: int, sample_index: int) -> np.random.Generator:
    """
    Deterministic, order-independent rng for one dataset sample.

    Inputs:
        master_seed: int, dataset-level seed.
        sample_index: int, sample id.

    Outputs:
        rng: np.random.Generator seeded from
            SeedSequence(master_seed, spawn_key=(sample_index,)).

    Why spawn keys:
        rng_for_sample(seed, i) is identical no matter how many samples exist
        or in what order they are generated, so any single sample (or an
        intervention twin that reuses part of its randomness) can be
        regenerated exactly from (master_seed, i) stored in metadata.
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=master_seed, spawn_key=(sample_index,))
    )
