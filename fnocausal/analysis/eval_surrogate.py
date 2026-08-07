"""Surrogate evaluation: relative L2 errors per split + example plots.

Adapted from phase0_phase1_local_pipeline.py (L1466-1813). Extended for the
Phase 2 independence check: the same model is scored against native (SBDF2)
targets and validator (RK4 @128^2, downsampled) targets on identical inputs.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..common.metrics import relative_l2_error_batch


def predict_and_relative_errors(
    model: nn.Module,
    loader: DataLoader,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    eps: float,
) -> np.ndarray:
    """
    Run inference and compute unnormalized relative L2 errors.

    Inputs:
        model: torch.nn.Module.
        loader: DataLoader of (X_norm, y_norm).
        y_mean, y_std: np.ndarray, shape (1, C, 1, 1).
        device: torch.device.
        eps: float.

    Outputs:
        errors: np.ndarray, shape (N,).
    """
    model.eval()
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)

    all_errors = []
    with torch.no_grad():
        for xb, yb_norm in loader:
            xb = xb.to(device, non_blocking=True)
            yb_norm = yb_norm.to(device, non_blocking=True)

            pred = model(xb) * y_std_t + y_mean_t
            target = yb_norm * y_std_t + y_mean_t

            all_errors.append(relative_l2_error_batch(pred, target, eps=eps).cpu().numpy())

    return np.concatenate(all_errors, axis=0)


def predict_fields(
    model: nn.Module,
    X_norm: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run inference and return unnormalized predicted fields.

    Inputs:
        model: torch.nn.Module.
        X_norm: np.ndarray, (N, C_in, nx, ny), already normalized.
        y_mean, y_std: np.ndarray, (1, C_out, 1, 1).
        device: torch.device.
        batch_size: int.

    Outputs:
        preds: np.ndarray, (N, C_out, nx, ny), unnormalized, float32.
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, X_norm.shape[0], batch_size):
            xb = torch.from_numpy(X_norm[start:start + batch_size]).to(device)
            pred = model(xb).cpu().numpy() * y_std + y_mean
            outs.append(pred.astype(np.float32))
    return np.concatenate(outs, axis=0)


def summarize_split_errors(results_df: pd.DataFrame, by: str = "eval_set") -> pd.DataFrame:
    """
    Aggregate per-sample errors.

    Inputs:
        results_df: pd.DataFrame with relative_l2_error and a grouping column.
        by: str, grouping column name.

    Outputs:
        summary: pd.DataFrame with count/mean/median/std per group.
    """
    return (
        results_df.groupby(by)["relative_l2_error"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )


def plot_prediction_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: list,
    output_path: Path,
    input_titles: tuple = ("u0(x)", "M(x)"),
) -> None:
    """
    Plot inputs, target, prediction, and error for a few samples.

    Inputs:
        X: np.ndarray, (N, C_in, nx, ny), unnormalized inputs.
        y_true, y_pred: np.ndarray, (N, 1, nx, ny), unnormalized.
        sample_ids: list of ints, rows to plot.
        output_path: Path.
        input_titles: tuple of channel names for X.

    Outputs:
        None. Saves figure.
    """
    n_inputs = X.shape[1]
    n_cols = n_inputs + 3
    fig, axes = plt.subplots(len(sample_ids), n_cols, figsize=(3 * n_cols, 3 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(sample_ids):
        for c in range(n_inputs):
            im = axes[row, c].imshow(X[idx, c], origin="lower", cmap="viridis")
            axes[row, c].set_title(input_titles[c] if c < len(input_titles) else f"input {c}")
            plt.colorbar(im, ax=axes[row, c], fraction=0.046)

        vmin, vmax = -1.1, 1.1
        im = axes[row, n_inputs].imshow(y_true[idx, 0], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, n_inputs].set_title("u(T) solver")
        plt.colorbar(im, ax=axes[row, n_inputs], fraction=0.046)

        im = axes[row, n_inputs + 1].imshow(y_pred[idx, 0], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, n_inputs + 1].set_title("u(T) FNO")
        plt.colorbar(im, ax=axes[row, n_inputs + 1], fraction=0.046)

        err = y_pred[idx, 0] - y_true[idx, 0]
        im = axes[row, n_inputs + 2].imshow(err, origin="lower", cmap="coolwarm")
        axes[row, n_inputs + 2].set_title("error")
        plt.colorbar(im, ax=axes[row, n_inputs + 2], fraction=0.046)

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved prediction examples: {output_path}")


def plot_training_curve(log_path: Path, output_path: Path, title: str) -> None:
    """
    Plot train/validation loss curves from a CSV log.

    Inputs:
        log_path: Path, CSV with epoch/train_mse_normalized/val_mse_normalized.
        output_path: Path.
        title: str.

    Outputs:
        None.
    """
    log_df = pd.read_csv(log_path)

    plt.figure(figsize=(7, 5))
    plt.semilogy(log_df["epoch"], log_df["train_mse_normalized"], label="train")
    plt.semilogy(log_df["epoch"], log_df["val_mse_normalized"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (normalized target space)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved training curve: {output_path}")
