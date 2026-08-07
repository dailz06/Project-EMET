"""Phase 4 simulation pool and environment assembly.

One pool of simulations is generated with the causal factors (m, nucleation
ICs, mobility texture) drawn per-sample from rng_for_sample streams. Because
the nuisance never enters the PDE, environments are assembled AFTER simulation
by assigning s_level per sample - no re-simulation.

The nuisance channel is s(x) = s_level * P(x) + noise, with P(x) the fixed
deterministic pattern from common.grf.nuisance_pattern. The solver API takes
only (u0, mobility, eps, g), so s cannot leak into the dynamics by
construction - this is the ground-truth answer key for Phase 5.
"""

import numpy as np
import pandas as pd

from ..common.grf import (
    generate_grf,
    lognormal_field_from_grf,
    nucleation_ic,
    nuisance_pattern,
    sample_uniform_range,
)
from ..common.metrics import transformed_area_fraction
from ..common.seeding import rng_for_sample
from .generate_dataset import solve_batched


def sample_phase4_inputs(config: dict, sample_index: int) -> dict:
    """
    Draw one pool sample's causal inputs from its dedicated rng stream.

    Inputs:
        config: dict, phase4 config.
        sample_index: int.

    Outputs:
        sample: dict with u0, mobility (float32 (nx, nx)), m, n_seeds, and
            metadata fields. The rng consumption order is fixed (m, n_seeds,
            IC, mobility texture) so Phase 5 intervention twins can replay
            selected draws.
    """
    rng = rng_for_sample(int(config["seed"]), sample_index)
    nx = int(config["nx"])
    domain = float(config["domain_size"])

    m = sample_uniform_range(rng, tuple(config["m_range"]))

    lo, hi = config["n_seeds_range"]
    n_seeds = int(rng.integers(int(lo), int(hi) + 1))

    u0 = nucleation_ic(
        nx, nx, domain,
        n_seeds=n_seeds,
        radius_range=tuple(config["seed_radius_range"]),
        interface_eps=float(config["eps"]),
        rng=rng,
    )

    mob_corr = sample_uniform_range(rng, tuple(config["mobility_corr_length_range"]))
    mob_decay = sample_uniform_range(rng, tuple(config["mobility_spectral_decay_range"]))
    grf_mob = generate_grf(nx, nx, domain, mob_corr, mob_decay, rng)
    texture = lognormal_field_from_grf(
        grf_mob, 1.0, float(config["mobility_log_std"]), tuple(config["mobility_clip"])
    )
    mobility = (m * texture).astype(np.float32)

    return {
        "u0": u0,
        "mobility": mobility,
        "m": m,
        "n_seeds": n_seeds,
        "mobility_corr_length": mob_corr,
        "mobility_spectral_decay": mob_decay,
    }


def generate_phase4_pool(config: dict) -> dict:
    """
    Generate the Phase 4 simulation pool.

    Inputs:
        config: dict, phase4 config.

    Outputs:
        pool: dict with u0 (N, nx, nx), mobility (N, nx, nx), u_final
            (N, nx, nx), snapshots {t: (N, nx, nx)}, severity (N,),
            metadata (pd.DataFrame with m, n_seeds, severity, master_seed).
    """
    from tqdm import tqdm

    n = int(config["pool_n"])
    nx = int(config["nx"])

    u0 = np.empty((n, nx, nx), dtype=np.float32)
    mobility = np.empty((n, nx, nx), dtype=np.float32)
    rows = []

    for i in tqdm(range(n), desc="Sampling pool inputs"):
        sample = sample_phase4_inputs(config, i)
        u0[i] = sample["u0"]
        mobility[i] = sample["mobility"]
        rows.append(
            {
                "sample_id": i,
                "master_seed": int(config["seed"]),
                "m": sample["m"],
                "n_seeds": sample["n_seeds"],
                "mobility_corr_length": sample["mobility_corr_length"],
                "mobility_spectral_decay": sample["mobility_spectral_decay"],
            }
        )

    eps = np.full(n, float(config["eps"]))
    solved = solve_batched(
        u0, mobility, eps, config,
        batch_size=int(config["solver_batch"]),
        desc="Solving pool (SBDF2)",
    )

    severity = transformed_area_fraction(solved["u_final"]).astype(np.float64)

    metadata = pd.DataFrame(rows)
    metadata["severity"] = severity

    return {
        "u0": u0,
        "mobility": mobility,
        "u_final": solved["u_final"],
        "snapshots": solved["snapshots"],
        "severity": severity,
        "metadata": metadata,
    }


def rank_standardize(values: np.ndarray) -> np.ndarray:
    """
    Map values to standard-normal scores by rank (Gaussian copula transform).

    Inputs:
        values: np.ndarray, shape (N,).

    Outputs:
        scores: np.ndarray, shape (N,), Phi^-1((rank + 0.5) / N).
    """
    from scipy.stats import norm

    order = np.argsort(np.argsort(values))
    return norm.ppf((order + 0.5) / len(values))


def assign_environments(pool_metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Partition the pool into disjoint environments and assign s_level per sample.

    Inputs:
        pool_metadata: pd.DataFrame with sample_id and severity.
        config: dict with environments {name: [rho, n_samples]} and seed.

    Outputs:
        answer_key: pd.DataFrame - pool metadata plus environment, rho_target,
            s_level, and realized Pearson/Spearman correlations per environment.
            This CSV IS the Phase 5 ground truth.
    """
    from scipy.stats import pearsonr, spearmanr

    env_spec = config["environments"]
    total_requested = sum(int(n) for _, n in env_spec.values())
    n_pool = len(pool_metadata)
    if total_requested > n_pool:
        raise ValueError(f"Environments request {total_requested} samples; pool has {n_pool}.")

    rng = np.random.default_rng(int(config["seed"]) + 555)
    permutation = rng.permutation(n_pool)

    answer_key = pool_metadata.copy()
    answer_key["environment"] = "unused"
    answer_key["rho_target"] = np.nan
    answer_key["s_level"] = np.nan

    cursor = 0
    realized = {}
    for env_name, (rho, n_env) in env_spec.items():
        rho = float(rho)
        n_env = int(n_env)
        idx = permutation[cursor:cursor + n_env]
        cursor += n_env

        severity_scores = rank_standardize(answer_key.loc[idx, "severity"].to_numpy())
        noise = rng.standard_normal(n_env)
        s_level = rho * severity_scores + np.sqrt(1.0 - rho**2) * noise

        answer_key.loc[idx, "environment"] = env_name
        answer_key.loc[idx, "rho_target"] = rho
        answer_key.loc[idx, "s_level"] = s_level

        sev = answer_key.loc[idx, "severity"].to_numpy()
        realized[env_name] = {
            "pearson": float(pearsonr(s_level, sev)[0]),
            "spearman": float(spearmanr(s_level, sev)[0]),
        }

    answer_key["rho_realized_pearson"] = answer_key["environment"].map(
        lambda e: realized.get(e, {}).get("pearson", np.nan)
    )
    answer_key["rho_realized_spearman"] = answer_key["environment"].map(
        lambda e: realized.get(e, {}).get("spearman", np.nan)
    )

    return answer_key


def build_nuisance_channel(
    s_levels: np.ndarray,
    config: dict,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    """
    Assemble the nuisance channel s(x) = s_level * P(x) + noise for many samples.

    Inputs:
        s_levels: np.ndarray, shape (N,).
        config: dict with nx, nuisance_n_waves, nuisance_noise_std.
        noise_rng: np.random.Generator for the additive pixel noise.

    Outputs:
        s: np.ndarray, shape (N, nx, nx), float32. Never passed to any solver.
    """
    nx = int(config["nx"])
    pattern = nuisance_pattern(nx, nx, n_waves=int(config["nuisance_n_waves"]))
    noise = noise_rng.normal(
        0.0, float(config["nuisance_noise_std"]), size=(len(s_levels), nx, nx)
    )
    s = s_levels[:, np.newaxis, np.newaxis] * pattern[np.newaxis] + noise
    return s.astype(np.float32)
