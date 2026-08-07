"""Phase 2 dataset generation: Allen-Cahn (u0, M) -> u(T) pairs + snapshots.

Inputs are drawn per-sample from rng_for_sample(master_seed, i) so any sample
can be regenerated exactly from its metadata row. Solves are batched through
the spectral SBDF2 solver. The validator test set re-solves the identical
64^2 input fields with the independent RK4 scheme at 128^2 and dt/4
(Fourier-upsampled inputs, Fourier-downsampled outputs) - generator !=
validator by construction.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common.grf import (
    generate_grf,
    lognormal_field_from_grf,
    sample_uniform_range,
    tanh_ic_from_grf,
)
from ..common.io_utils import make_band_splits
from ..common.seeding import rng_for_sample
from .allen_cahn_rk4 import solve_allen_cahn_rk4
from .allen_cahn_spectral import solve_allen_cahn_imex
from .downsample import fourier_downsample, fourier_upsample


def sample_phase2_inputs(config: dict, sample_index: int) -> dict:
    """
    Draw one sample's inputs (u0, M, eps) from its dedicated rng stream.

    Inputs:
        config: dict, phase2 config.
        sample_index: int.

    Outputs:
        sample: dict with u0, mobility (float32 (nx, nx)), eps (float),
            and the drawn hyperparameters for metadata.
    """
    rng = rng_for_sample(int(config["seed"]), sample_index)
    nx = int(config["nx"])
    domain = float(config["domain_size"])

    eps = sample_uniform_range(rng, tuple(config["eps_range"]))

    ic_corr = sample_uniform_range(rng, tuple(config["ic_corr_length_range"]))
    ic_decay = sample_uniform_range(rng, tuple(config["ic_spectral_decay_range"]))
    grf_ic = generate_grf(nx, nx, domain, ic_corr, ic_decay, rng)
    u0 = tanh_ic_from_grf(grf_ic, float(config["ic_interface_width"]))

    mob_corr = sample_uniform_range(rng, tuple(config["mobility_corr_length_range"]))
    mob_decay = sample_uniform_range(rng, tuple(config["mobility_spectral_decay_range"]))
    grf_mob = generate_grf(nx, nx, domain, mob_corr, mob_decay, rng)
    mobility = lognormal_field_from_grf(
        grf_mob,
        float(config["mobility_mean"]),
        float(config["mobility_log_std"]),
        tuple(config["mobility_clip"]),
    )

    return {
        "u0": u0,
        "mobility": mobility,
        "eps": eps,
        "ic_corr_length": ic_corr,
        "ic_spectral_decay": ic_decay,
        "mobility_corr_length": mob_corr,
        "mobility_spectral_decay": mob_decay,
    }


def solve_batched(
    u0: np.ndarray,
    mobility: np.ndarray,
    eps: np.ndarray,
    config: dict,
    batch_size: int,
    desc: str = "Solving",
) -> dict:
    """
    Solve many Allen-Cahn cases in chunks with the primary SBDF2 solver.

    Inputs:
        u0: np.ndarray, (N, nx, nx).
        mobility: np.ndarray, (N, nx, nx).
        eps: np.ndarray, (N,).
        config: dict with t_final, solver_dt, domain_size, g, snapshot_times.
        batch_size: int, samples per batched solve.
        desc: str, progress label.

    Outputs:
        result: dict with u_final (N, nx, nx) and snapshots {t: (N, nx, nx)}.
    """
    n = u0.shape[0]
    snapshot_times = tuple(float(t) for t in config.get("snapshot_times", ()))
    backend = str(config.get("solver_backend", "numpy"))

    finals = []
    snapshots = {t: [] for t in snapshot_times}

    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        result = solve_allen_cahn_imex(
            u0[start:end],
            eps[start:end],
            float(config["t_final"]),
            float(config["solver_dt"]),
            domain_size=float(config["domain_size"]),
            mobility=mobility[start:end],
            g=float(config.get("g", 0.0)),
            snapshot_times=snapshot_times,
            backend=backend,
        )
        finals.append(result["u_final"])
        for t in snapshot_times:
            snapshots[t].append(result["snapshots"][t])

    return {
        "u_final": np.concatenate(finals, axis=0),
        "snapshots": {t: np.concatenate(chunks, axis=0) for t, chunks in snapshots.items()},
    }


def generate_phase2_dataset(config: dict) -> dict:
    """
    Generate the full Phase 2 dataset in memory.

    Inputs:
        config: dict, phase2 config.

    Outputs:
        dataset: dict with
            X: (N, 2, nx, nx) float32, channels (u0, M),
            y: (N, 1, nx, nx) float32, u(T),
            snapshots: dict t -> (N, nx, nx) float32,
            split: (N,) str labels (eps-band holdout),
            metadata: pd.DataFrame,
            heldout_bounds: tuple.
    """
    n = int(config["dataset_n"])
    nx = int(config["nx"])

    u0 = np.empty((n, nx, nx), dtype=np.float32)
    mobility = np.empty((n, nx, nx), dtype=np.float32)
    eps = np.empty(n, dtype=np.float64)
    rows = []

    for i in tqdm(range(n), desc="Sampling inputs"):
        sample = sample_phase2_inputs(config, i)
        u0[i] = sample["u0"]
        mobility[i] = sample["mobility"]
        eps[i] = sample["eps"]
        rows.append(
            {
                "sample_id": i,
                "master_seed": int(config["seed"]),
                "eps": sample["eps"],
                "ic_corr_length": sample["ic_corr_length"],
                "ic_spectral_decay": sample["ic_spectral_decay"],
                "mobility_corr_length": sample["mobility_corr_length"],
                "mobility_spectral_decay": sample["mobility_spectral_decay"],
                "mobility_mean_empirical": float(sample["mobility"].mean()),
            }
        )

    solved = solve_batched(
        u0, mobility, eps, config,
        batch_size=int(config["solver_batch"]),
        desc="Solving Allen-Cahn (SBDF2)",
    )

    metadata = pd.DataFrame(rows)

    split_rng = np.random.default_rng(int(config["seed"]) + 777)
    split_labels, heldout_bounds = make_band_splits(
        values=eps,
        value_range=tuple(config["eps_range"]),
        heldout_fraction=float(config["heldout_fraction"]),
        heldout_side=str(config["heldout_side"]),
        id_split_ratios=tuple(config["id_split_ratios"]),
        rng=split_rng,
    )
    metadata["split"] = split_labels

    X = np.stack([u0, mobility], axis=1)
    y = solved["u_final"][:, np.newaxis, :, :]

    return {
        "X": X,
        "y": y,
        "snapshots": solved["snapshots"],
        "split": split_labels,
        "metadata": metadata,
        "heldout_bounds": heldout_bounds,
    }


def generate_validator_targets(
    X: np.ndarray,
    eps: np.ndarray,
    config: dict,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Re-solve the SAME 64^2 inputs with the independent validator scheme:
    RK4 at validator_nx^2 and solver_dt / validator_dt_factor, results
    Fourier-downsampled back to the native grid.

    Inputs:
        X: np.ndarray, (N, 2, nx, nx), channels (u0, M) - native inputs.
        eps: np.ndarray, (N,).
        config: dict, phase2 config.
        batch_size: int (128^2 RK4 is ~16x the memory of a native solve).

    Outputs:
        y_validator: np.ndarray, (N, 1, nx, nx) float32.
    """
    nx = int(config["nx"])
    fine_nx = int(config["validator_nx"])
    dt = float(config["solver_dt"]) / int(config["validator_dt_factor"])

    n = X.shape[0]
    outputs = []

    for start in tqdm(range(0, n, batch_size), desc="Validator solves (RK4)"):
        end = min(start + batch_size, n)

        u0_fine = fourier_upsample(X[start:end, 0], fine_nx)
        mob_fine = fourier_upsample(X[start:end, 1], fine_nx)
        # Spectral interpolation can slightly undershoot the clip floor;
        # mobility must stay nonnegative for the solver.
        mob_fine = np.maximum(mob_fine, 0.0)

        result = solve_allen_cahn_rk4(
            u0_fine,
            eps[start:end],
            float(config["t_final"]),
            dt,
            domain_size=float(config["domain_size"]),
            mobility=mob_fine,
            g=float(config.get("g", 0.0)),
            backend=str(config.get("solver_backend", "numpy")),
        )
        outputs.append(fourier_downsample(result["u_final"], nx))

    return np.concatenate(outputs, axis=0)[:, np.newaxis, :, :]
