"""Diagnostic 2: intervention-based latent probing.

The disentanglement test that is impossible with real photographs: because we
control the generator, we can produce matched twins that differ in exactly one
factor and watch which latent directions of the biased AE move.

    Intervention on S: same u0, sweep s_level over a grid (no re-simulation
        needed - s never touches the solver). The latent direction that
        responds is the nuisance subspace w_S.
    Intervention on C: same rng stream, force the nucleation seed count k
        (the causal content visible in u0), rebuild u0, keep s fixed. The
        responding direction is the causal subspace w_C.

Latents are not axis-aligned, so subspace directions (linear regression of z
on the intervened factor) are the primary object; per-dim displacement is
reported as secondary evidence.
"""

import numpy as np
import pandas as pd
import torch

from ..common.grf import nucleation_ic, sample_uniform_range
from ..common.normalization import normalize_array
from ..common.seeding import rng_for_sample
from ..sim.phase4_pool import build_nuisance_channel


def encode(ae, X: np.ndarray, stats: dict, device, batch_size: int = 256) -> np.ndarray:
    """
    Encode (u0, s) stacks with the biased AE.

    Inputs:
        ae: ConvAutoencoder.
        X: np.ndarray, (N, 2, nx, nx), unnormalized.
        stats: dict with X_mean/X_std.
        device: torch.device.

    Outputs:
        z: np.ndarray, (N, z_dim).
    """
    X_norm = normalize_array(X, stats["X_mean"], stats["X_std"])
    ae.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X_norm[start:start + batch_size]).to(device)
            outs.append(ae.encode(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def rebuild_u0_with_forced_k(config: dict, sample_index: int, k_forced: int) -> np.ndarray:
    """
    Regenerate one pool sample's u0 with the seed count forced to k_forced,
    replaying the same rng stream (m and the k draw are consumed identically,
    so seed placements are shared between twins up to the smaller k).

    Inputs:
        config: phase4 config.
        sample_index: int, pool sample id.
        k_forced: int.

    Outputs:
        u0: np.ndarray, (nx, nx) float32.
    """
    rng = rng_for_sample(int(config["seed"]), sample_index)
    nx = int(config["nx"])
    domain = float(config["domain_size"])

    _ = sample_uniform_range(rng, tuple(config["m_range"]))          # m (discarded)
    lo, hi = config["n_seeds_range"]
    _ = int(rng.integers(int(lo), int(hi) + 1))                       # original k (discarded)

    return nucleation_ic(
        nx, nx, domain,
        n_seeds=int(k_forced),
        radius_range=tuple(config["seed_radius_range"]),
        interface_eps=float(config["eps"]),
        rng=rng,
    )


def probe_interventions(
    ae,
    stats: dict,
    npz,
    answer_key: pd.DataFrame,
    config: dict,
    device,
    n_base: int = 150,
    s_grid: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0),
    k_grid: tuple = (2, 3, 4, 5, 6),
) -> dict:
    """
    Run both interventions and fit the S- and C-subspace directions.

    Inputs:
        ae, stats: biased AE and its normalizers.
        npz: pool arrays.
        answer_key: pd.DataFrame.
        config: phase4 config.
        device: torch.device.
        n_base: int, number of base samples (drawn from eval environments so
            they were never seen in AE training).
        s_grid: tuple, s_level intervention values.
        k_grid: tuple, forced seed counts.

    Outputs:
        result: dict with
            w_S, w_C: unit vectors (z_dim,) - fitted intervention directions,
            r2_S, r2_C: float, fit quality of z-response to each factor,
            per_dim: pd.DataFrame with per-dimension displacement stats,
            angle_deg: float, angle between w_S and w_C.
    """
    eval_ids = answer_key.loc[
        answer_key["environment"].str.startswith("eval_"), "sample_id"
    ].to_numpy()
    rng = np.random.default_rng(int(config["seed"]) + 111)
    base_ids = rng.choice(eval_ids, size=min(n_base, len(eval_ids)), replace=False)

    noise_rng = np.random.default_rng(int(config["seed"]) + 222)

    # --- Intervention on S: same u0, sweep s_level ---
    zs, s_values = [], []
    for s_level in s_grid:
        levels = np.full(len(base_ids), float(s_level))
        s_fields = build_nuisance_channel(levels, config, noise_rng)
        X = np.stack([npz["u0"][base_ids], s_fields], axis=1)
        zs.append(encode(ae, X, stats, device))
        s_values.append(levels)
    z_S = np.concatenate(zs, axis=0)
    s_values = np.concatenate(s_values)

    # --- Intervention on C: same stream, force k, keep s fixed at each
    # sample's observed value ---
    observed_s = answer_key.set_index("sample_id").loc[base_ids, "s_level"].to_numpy()
    s_fixed = build_nuisance_channel(observed_s, config, noise_rng)

    zc, k_values = [], []
    for k_forced in k_grid:
        u0_forced = np.stack([
            rebuild_u0_with_forced_k(config, int(i), int(k_forced)) for i in base_ids
        ])
        X = np.stack([u0_forced, s_fixed], axis=1)
        zc.append(encode(ae, X, stats, device))
        k_values.append(np.full(len(base_ids), float(k_forced)))
    z_C = np.concatenate(zc, axis=0)
    k_values = np.concatenate(k_values)

    def fit_direction(z, factor):
        """Least-squares direction of z response to the factor; returns unit
        vector and pooled R^2 of the projected response."""
        f_centered = factor - factor.mean()
        z_centered = z - z.mean(axis=0)
        w = z_centered.T @ f_centered / (f_centered @ f_centered)
        w_unit = w / (np.linalg.norm(w) + 1e-12)
        proj = z_centered @ w_unit
        ss_res = np.sum((proj - f_centered * (proj @ f_centered) / (f_centered @ f_centered)) ** 2)
        ss_tot = np.sum(proj**2) + 1e-30
        return w_unit, w, float(1.0 - ss_res / ss_tot)

    w_S_unit, w_S_raw, r2_S = fit_direction(z_S, s_values)
    w_C_unit, w_C_raw, r2_C = fit_direction(z_C, k_values)

    cos_angle = float(np.clip(np.abs(w_S_unit @ w_C_unit), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    # Per-dimension secondary evidence: |response slope| per latent dim.
    per_dim = pd.DataFrame(
        {
            "dim": np.arange(z_S.shape[1]),
            "abs_slope_S": np.abs(w_S_raw),
            "abs_slope_C": np.abs(w_C_raw),
        }
    )

    return {
        "w_S": w_S_unit,
        "w_C": w_C_unit,
        "r2_S": r2_S,
        "r2_C": r2_C,
        "per_dim": per_dim,
        "angle_deg": angle_deg,
        "base_ids": base_ids,
    }
