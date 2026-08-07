"""Dataset IO and split construction.

make_band_splits generalizes make_correlation_length_splits from
phase0_phase1_local_pipeline.py (L552): the held-out band can be over any
scalar sample property (Phase 2 holds out an epsilon band instead of a
correlation length band). save/load_dataset_npz mirror the Phase 1 format
(npz arrays + CSV metadata + JSON generation config).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_band_splits(
    values: np.ndarray,
    value_range: tuple,
    heldout_fraction: float,
    heldout_side: str,
    id_split_ratios: tuple,
    rng: np.random.Generator,
) -> tuple:
    """
    Split samples into train/val/test/ood_test with a contiguous held-out band.

    Inputs:
        values: np.ndarray, shape (N,), scalar property per sample (e.g. epsilon).
        value_range: tuple(float, float), full sampled range of the property.
        heldout_fraction: float, fraction of the range reserved for OOD.
        heldout_side: str, "high" or "low".
        id_split_ratios: tuple(float, float, float), train/val/test ratios in ID region.
        rng: np.random.Generator.

    Outputs:
        split_labels: np.ndarray, shape (N,), str labels.
        heldout_bounds: tuple(float, float), OOD interval.
    """
    lo, hi = value_range
    width = hi - lo

    if heldout_side == "high":
        threshold = hi - heldout_fraction * width
        ood_mask = values >= threshold
        heldout_bounds = (threshold, hi)
    elif heldout_side == "low":
        threshold = lo + heldout_fraction * width
        ood_mask = values <= threshold
        heldout_bounds = (lo, threshold)
    else:
        raise ValueError("heldout_side must be either 'high' or 'low'.")

    split_labels = np.full(values.shape[0], "unassigned", dtype=object)
    split_labels[ood_mask] = "ood_test"

    id_indices = np.where(~ood_mask)[0]
    rng.shuffle(id_indices)

    train_ratio, val_ratio, test_ratio = id_split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("id_split_ratios must sum to 1.")

    n_id = len(id_indices)
    n_train = min(int(round(train_ratio * n_id)), n_id)
    n_val = min(int(round(val_ratio * n_id)), n_id - n_train)

    split_labels[id_indices[:n_train]] = "train"
    split_labels[id_indices[n_train:n_train + n_val]] = "val"
    split_labels[id_indices[n_train + n_val:]] = "test"

    return split_labels.astype(str), heldout_bounds


def save_dataset_npz(
    dataset_path: Path,
    metadata_csv_path: Path,
    metadata_json_path: Path,
    arrays: dict,
    metadata: pd.DataFrame,
    generation_config: dict,
) -> None:
    """
    Save a dataset in the Phase 1 format: npz arrays + CSV metadata + JSON config.

    Inputs:
        dataset_path: Path, .npz output.
        metadata_csv_path: Path, CSV output.
        metadata_json_path: Path, JSON output.
        arrays: dict of name -> np.ndarray (must include X and y for model data;
            simulation pools may store u0, M, snapshots, etc.).
        metadata: pd.DataFrame, one row per sample.
        generation_config: dict, everything needed to regenerate (incl. seed).

    Outputs:
        None.
    """
    np.savez(dataset_path, **arrays)
    metadata.to_csv(metadata_csv_path, index=False)

    json_safe = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in generation_config.items()
    }
    with open(metadata_json_path, "w") as f:
        json.dump(json_safe, f, indent=2)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metadata CSV: {metadata_csv_path}")
    print(f"Saved generation config JSON: {metadata_json_path}")


def load_dataset_npz(dataset_path: Path, metadata_csv_path: Path) -> dict:
    """
    Load a dataset saved by save_dataset_npz.

    Inputs:
        dataset_path: Path, .npz file.
        metadata_csv_path: Path, CSV metadata file.

    Outputs:
        data: dict of arrays plus "metadata" (pd.DataFrame).
    """
    npz = np.load(dataset_path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    data["metadata"] = pd.read_csv(metadata_csv_path)
    return data
