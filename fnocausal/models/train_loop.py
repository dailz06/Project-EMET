"""Generic supervised training loop.

Ported from phase0_phase1_local_pipeline.py (L1163-1409) with paths passed as
arguments. The same loop drives the FNO surrogates (x -> y fields) and the
autoencoder (x -> x reconstruction): both are (input, target) tensor pairs
under MSE.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def make_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Inputs:
        model: torch.nn.Module.
        config: dict with optimizer and learning_rate.

    Outputs:
        optimizer: torch.optim.Optimizer.
    """
    name = str(config["optimizer"]).lower()
    lr = float(config["learning_rate"])

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)

    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
) -> None:
    """
    Save model checkpoint with its full config.

    Inputs:
        path: Path.
        model, optimizer, epoch, train_loss, val_loss, config: run state.

    Outputs:
        None.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """
    Load a checkpoint robustly across torch versions.

    Inputs:
        path: Path.
        device: torch.device.

    Outputs:
        checkpoint: dict.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> float:
    """
    Run one epoch; trains if an optimizer is given, evaluates otherwise.

    Inputs:
        model, loader, criterion, device, optimizer (None for eval).

    Outputs:
        mean_loss: float.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            if training:
                loss.backward()
                optimizer.step()

            batch = xb.shape[0]
            total_loss += loss.item() * batch
            total_samples += batch

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    best_ckpt_path: Path,
    last_ckpt_path: Path,
    log_path: Path,
    checkpoint_dir: Path = None,
    log_prefix: str = "",
) -> nn.Module:
    """
    Train a model with MSE loss, best-on-val checkpointing, and a CSV log.

    Inputs:
        model: torch.nn.Module (already constructed, not yet on device).
        train_loader, val_loader: DataLoaders of (input, target) pairs.
        config: dict with epochs, learning_rate, optimizer, checkpoint_every.
        device: torch.device.
        best_ckpt_path, last_ckpt_path: Paths for best/last checkpoints.
        log_path: Path for the CSV training log.
        checkpoint_dir: Path for periodic epoch checkpoints (None disables).
        log_prefix: str prepended to console lines (e.g. the model role).

    Outputs:
        model: trained module with the best-validation weights loaded.
    """
    model = model.to(device)
    optimizer = make_optimizer(model, config)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{log_prefix}Trainable parameters: {num_params:,}")

    log_rows = []
    best_val_loss = np.inf
    epochs = int(config["epochs"])
    checkpoint_every = int(config.get("checkpoint_every", 0))

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        log_rows.append(
            {"epoch": epoch, "train_mse_normalized": train_loss, "val_mse_normalized": val_loss}
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"{log_prefix}Epoch {epoch:04d}/{epochs} | "
                f"train MSE: {train_loss:.6e} | val MSE: {val_loss:.6e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, train_loss, val_loss, config)

        if checkpoint_dir is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"{best_ckpt_path.stem}_epoch_{epoch:04d}.pt",
                model, optimizer, epoch, train_loss, val_loss, config,
            )

    save_checkpoint(
        last_ckpt_path, model, optimizer, epochs,
        log_rows[-1]["train_mse_normalized"], log_rows[-1]["val_mse_normalized"], config,
    )
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"{log_prefix}Saved training log: {log_path}")

    best = load_checkpoint(best_ckpt_path, device)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    print(f"{log_prefix}Loaded best checkpoint from epoch {best['epoch']} (val {best['val_loss']:.6e})")

    return model
