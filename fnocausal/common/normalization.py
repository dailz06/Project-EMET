"""Channelwise normalization and DataLoader construction.

Ported from phase0_phase1_local_pipeline.py (L939-1107) with paths passed as
arguments instead of module globals. The leakage discipline is unchanged:
statistics are computed on the training split only and reused for every
evaluation set (including, in Phase 4/5, shifted environments - recomputing
per environment would silently launder the distribution shift).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def append_scalar_channel(X: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Append a constant-field channel broadcasting one scalar per sample.

    Inputs:
        X: np.ndarray, shape (N, C, nx, ny).
        values: np.ndarray, shape (N,), one scalar per sample (e.g. the PDE
            parameter epsilon - required input for a parametric surrogate when
            the parameter is not inferable from the other channels).

    Outputs:
        X_out: np.ndarray, shape (N, C+1, nx, ny), float32.
    """
    n, _, nx, ny = X.shape
    channel = np.broadcast_to(
        values.astype(np.float32)[:, np.newaxis, np.newaxis, np.newaxis], (n, 1, nx, ny)
    )
    return np.concatenate([X, channel], axis=1).astype(np.float32)


def compute_channel_stats(array: np.ndarray, eps: float) -> tuple:
    """
    Compute per-channel mean and std for arrays shaped (N, C, nx, ny).

    Inputs:
        array: np.ndarray, shape (N, C, nx, ny).
        eps: float, minimum std.

    Outputs:
        mean: np.ndarray, shape (1, C, 1, 1).
        std: np.ndarray, shape (1, C, 1, 1).
    """
    mean = array.mean(axis=(0, 2, 3), keepdims=True)
    std = array.std(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(std, eps)

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_array(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize an array channelwise.

    Inputs:
        array: np.ndarray, shape (N, C, nx, ny).
        mean: np.ndarray, shape (1, C, 1, 1).
        std: np.ndarray, shape (1, C, 1, 1).

    Outputs:
        normalized: np.ndarray, shape (N, C, nx, ny), dtype float32.
    """
    return ((array - mean) / std).astype(np.float32)


def save_normalizers(path: Path, X_mean, X_std, y_mean, y_std) -> None:
    """
    Save normalization statistics.

    Inputs:
        path: Path, .npz output.
        X_mean, X_std, y_mean, y_std: np.ndarray, shapes (1, C, 1, 1).

    Outputs:
        None.
    """
    np.savez(path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)


def load_normalizers(path: Path) -> dict:
    """
    Load normalization statistics.

    Inputs:
        path: Path, .npz file.

    Outputs:
        stats: dict with X_mean, X_std, y_mean, y_std.
    """
    npz = np.load(path)
    return {key: npz[key] for key in ("X_mean", "X_std", "y_mean", "y_std")}


def make_tensor_loader(
    X_array: np.ndarray,
    y_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader over (X, y) tensors.

    Inputs:
        X_array: np.ndarray, shape (N, C_in, nx, ny).
        y_array: np.ndarray, shape (N, C_out, nx, ny).
        batch_size: int.
        shuffle: bool.
        num_workers: int. Default 0: Windows worker processes re-import the
            package per epoch and cost more than they save at 64x64.
        pin_memory: bool.

    Outputs:
        loader: torch.utils.data.DataLoader.
    """
    dataset = TensorDataset(torch.from_numpy(X_array), torch.from_numpy(y_array))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
