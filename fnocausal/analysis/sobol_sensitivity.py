"""Diagnostic 4: Sobol variance-based sensitivity (second line of evidence).

Two Sobol analyses over the same three factors (m, k, s_level):

    simulator:  sigma_sim(m, k) - s_level is passed but ignored by
                construction. Its Sobol index must be ~0. This validates the
                harness, not the model.
    model:      sigma_hat = biased severity head(u0(k), s(s_level)) - m is
                passed but invisible to the model, so its index must be ~0,
                while s_level's index should be LARGE.

The contrast S_s(simulator) ~ 0 vs S_s(model) >> 0 is the cleanest single
number for "the model uses a non-causal feature". All non-intervened
randomness (seed placement stream, mobility texture, s-channel noise) is held
fixed at a reference sample so the Sobol variance decomposes over exactly the
three named factors.
"""

import numpy as np
import torch
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from ..common.grf import generate_grf, lognormal_field_from_grf, nucleation_ic
from ..common.metrics import transformed_area_fraction
from ..common.normalization import normalize_array
from ..common.seeding import rng_for_sample
from ..sim.allen_cahn_spectral import solve_allen_cahn_imex
from ..sim.phase4_pool import build_nuisance_channel


def make_problem(config: dict) -> dict:
    """SALib problem definition over (m, k, s_level)."""
    return {
        "num_vars": 3,
        # np.array, not list: SALib 1.5.1 feeds names to pd.unique, which in
        # pandas >= 3.0 rejects plain lists.
        "names": np.array(["m", "k", "s_level"]),
        "bounds": [
            [float(config["m_range"][0]), float(config["m_range"][1])],
            [float(config["n_seeds_range"][0]), float(config["n_seeds_range"][1]) + 0.999],
            [-2.5, 2.5],
        ],
    }


def build_fields_for_rows(rows: np.ndarray, config: dict, reference_index: int = 0) -> dict:
    """
    Build (u0, mobility, s) fields for Saltelli sample rows with all
    non-intervened randomness fixed at a reference rng stream.

    Inputs:
        rows: np.ndarray, (N, 3) - columns m, k (continuous, floored), s_level.
        config: phase4 config.
        reference_index: int, pool sample whose rng stream supplies the fixed
            seed placements and mobility texture.

    Outputs:
        fields: dict with u0 (N, nx, nx), mobility (N, nx, nx), s (N, nx, nx),
            k_int (N,).
    """
    nx = int(config["nx"])
    domain = float(config["domain_size"])

    k_int = np.clip(rows[:, 1].astype(int), int(config["n_seeds_range"][0]),
                    int(config["n_seeds_range"][1]))

    # Fixed mobility texture from the reference stream (drawn once).
    rng_ref = rng_for_sample(int(config["seed"]), reference_index)
    _ = rng_ref.uniform(*config["m_range"])
    _ = rng_ref.integers(int(config["n_seeds_range"][0]), int(config["n_seeds_range"][1]) + 1)
    # Consume the IC placement draws at max k so the texture draw position is
    # fixed, then draw the texture.
    _ = nucleation_ic(nx, nx, domain, int(config["n_seeds_range"][1]),
                      tuple(config["seed_radius_range"]), float(config["eps"]), rng_ref)
    mob_corr = rng_ref.uniform(*config["mobility_corr_length_range"])
    mob_decay = rng_ref.uniform(*config["mobility_spectral_decay_range"])
    grf_mob = generate_grf(nx, nx, domain, mob_corr, mob_decay, rng_ref)
    texture = lognormal_field_from_grf(
        grf_mob, 1.0, float(config["mobility_log_std"]), tuple(config["mobility_clip"])
    )

    # u0 per unique k, with IDENTICAL placement stream (same spawn key).
    u0_by_k = {}
    for k in np.unique(k_int):
        rng_ic = rng_for_sample(int(config["seed"]) + 31337, 0)
        u0_by_k[int(k)] = nucleation_ic(
            nx, nx, domain, int(k),
            tuple(config["seed_radius_range"]), float(config["eps"]), rng_ic,
        )

    u0 = np.stack([u0_by_k[int(k)] for k in k_int])
    mobility = (rows[:, 0][:, np.newaxis, np.newaxis] * texture[np.newaxis]).astype(np.float32)

    noise_rng = np.random.default_rng(int(config["seed"]) + 424242)
    fixed_noise = noise_rng.normal(0.0, float(config["nuisance_noise_std"]), size=(nx, nx))
    from ..common.grf import nuisance_pattern
    pattern = nuisance_pattern(nx, nx, n_waves=int(config["nuisance_n_waves"]))
    s = (rows[:, 2][:, np.newaxis, np.newaxis] * pattern[np.newaxis]
         + fixed_noise[np.newaxis]).astype(np.float32)

    return {"u0": u0, "mobility": mobility, "s": s, "k_int": k_int}


def sobol_simulator(rows: np.ndarray, fields: dict, config: dict, batch_size: int = 128) -> np.ndarray:
    """
    Simulator severity for each Saltelli row (s_level ignored by construction).

    Outputs:
        sigma: np.ndarray, (N,).
    """
    from tqdm import tqdm

    n = rows.shape[0]
    eps = np.full(n, float(config["eps"]))
    backend = str(config.get("solver_backend", "numpy"))
    sigmas = []
    for start in tqdm(range(0, n, batch_size), desc="Sobol simulator solves"):
        end = min(start + batch_size, n)
        solved = solve_allen_cahn_imex(
            fields["u0"][start:end],
            eps[start:end],
            float(config["t_final"]),
            float(config["solver_dt"]),
            domain_size=float(config["domain_size"]),
            mobility=fields["mobility"][start:end],
            g=float(config["g"]),
            backend=backend,
        )
        sigmas.append(transformed_area_fraction(solved["u_final"]))
    return np.concatenate(sigmas)


def sobol_model(fields: dict, head, stats: dict, device, batch_size: int = 256) -> np.ndarray:
    """
    Biased severity-head predictions for each Saltelli row (m invisible).

    Outputs:
        sigma_hat: np.ndarray, (N,).
    """
    X = np.stack([fields["u0"], fields["s"]], axis=1)
    X_norm = normalize_array(X, stats["X_mean"], stats["X_std"])

    head.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X_norm[start:start + batch_size]).to(device)
            preds.append(head(xb).cpu().numpy()[:, 0])
    return np.concatenate(preds)


def analyze(problem: dict, outputs: np.ndarray) -> dict:
    """
    First-order and total Sobol indices.

    Outputs:
        dict name -> {"S1": float, "ST": float}.
    """
    result = sobol_analyze.analyze(problem, outputs, calc_second_order=False)
    return {
        name: {"S1": float(result["S1"][i]), "ST": float(result["ST"][i])}
        for i, name in enumerate(problem["names"])
    }
