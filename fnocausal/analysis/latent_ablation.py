"""Diagnostic 5: latent ablation.

Remove the identified S- or C-subspace component from the biased AE's latent
and measure the consequences:

    ablate S: severity-probe accuracy should DROP on biased environments
              (the probe was leaning on the shortcut) but IMPROVE on the
              flipped environment (the shortcut was anti-informative there).
    ablate C: accuracy drops everywhere (the causal content is always useful).

Physical fidelity: decoding an S-ablated latent must leave the u0 channel
essentially unchanged (the nuisance is a separate channel by construction);
large u0 damage would mean the subspaces are entangled and the ablation is
removing physics along with the shortcut.
"""

import numpy as np
import pandas as pd
import torch

from ..common.metrics import relative_l2_numpy
from ..common.normalization import normalize_array


def ablate_direction(z: np.ndarray, w: np.ndarray, z_train_mean: np.ndarray) -> np.ndarray:
    """
    Mean-impute the component of z along the unit direction w.

    Inputs:
        z: np.ndarray, (N, d).
        w: np.ndarray, (d,), unit vector.
        z_train_mean: np.ndarray, (d,), training-set latent mean (the imputed
            component value comes from here, not from the eval batch).

    Outputs:
        z_ablated: np.ndarray, (N, d).
    """
    mean_component = float(z_train_mean @ w)
    return z - np.outer(z @ w - mean_component, w)


def probe_r2(z: np.ndarray, sigma: np.ndarray, coef: np.ndarray, z_ref_mean: np.ndarray,
             sigma_ref_mean: float) -> float:
    """
    R^2 of a fixed linear probe evaluated on (z, sigma).

    Inputs:
        z, sigma: eval data.
        coef: probe coefficients (fit elsewhere).
        z_ref_mean, sigma_ref_mean: centering constants from the probe's
            training environment (must not be re-estimated on eval data).

    Outputs:
        r2: float.
    """
    pred = (z - z_ref_mean) @ coef + sigma_ref_mean
    ss_res = np.sum((sigma - pred) ** 2)
    ss_tot = np.sum((sigma - sigma.mean()) ** 2) + 1e-30
    return float(1.0 - ss_res / ss_tot)


def ablation_table(
    z_by_env: dict,
    sigma_by_env: dict,
    coef: np.ndarray,
    w_S: np.ndarray,
    w_C: np.ndarray,
    z_train_mean: np.ndarray,
    sigma_train_mean: float,
) -> pd.DataFrame:
    """
    Probe R^2 per environment for intact / S-ablated / C-ablated latents.

    Inputs:
        z_by_env, sigma_by_env: eval environments.
        coef: severity-probe coefficients (fit on biased training envs).
        w_S, w_C: unit intervention directions.
        z_train_mean, sigma_train_mean: probe centering constants.

    Outputs:
        table: pd.DataFrame with environment x {intact, ablate_S, ablate_C} R^2.
    """
    rows = []
    for env, z in z_by_env.items():
        sigma = sigma_by_env[env]
        variants = {
            "intact": z,
            "ablate_S": ablate_direction(z, w_S, z_train_mean),
            "ablate_C": ablate_direction(z, w_C, z_train_mean),
        }
        for variant, z_v in variants.items():
            rows.append(
                {
                    "environment": env,
                    "variant": variant,
                    "probe_r2": probe_r2(z_v, sigma, coef, z_train_mean, sigma_train_mean),
                }
            )
    return pd.DataFrame(rows)


def decode_fidelity(
    ae,
    z: np.ndarray,
    z_ablated: np.ndarray,
    stats: dict,
    device,
    batch_size: int = 256,
) -> dict:
    """
    Compare decoded fields before/after ablation, per channel.

    Inputs:
        ae: ConvAutoencoder.
        z, z_ablated: (N, d) latents.
        stats: dict with X_mean/X_std (to unnormalize decoded fields).
        device: torch.device.

    Outputs:
        dict with u0_rel_change and s_rel_change (mean relative L2 between
        decoded channels before vs after ablation).
    """
    ae.eval()

    def decode_all(latents):
        outs = []
        with torch.no_grad():
            for start in range(0, latents.shape[0], batch_size):
                zb = torch.from_numpy(latents[start:start + batch_size].astype(np.float32)).to(device)
                outs.append(ae.decode(zb).cpu().numpy())
        fields = np.concatenate(outs, axis=0)
        return fields * stats["X_std"] + stats["X_mean"]

    before = decode_all(z)
    after = decode_all(z_ablated)

    u0_change = relative_l2_numpy(after[:, 0], before[:, 0]).mean()
    s_change = relative_l2_numpy(after[:, 1], before[:, 1]).mean()
    return {"u0_rel_change": float(u0_change), "s_rel_change": float(s_change)}
