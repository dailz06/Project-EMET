"""Diagnostic 1: shortcut collapse (the headline claim-2 result).

Evaluate the biased / control / oracle FNOs and the biased / control severity
heads on every evaluation environment (rho = 0.95, 0.5, 0, -0.95). Expected
answer-key pattern:

    biased models:  error rises monotonically as rho_eval departs from the
                    training correlation, worst at rho = -0.95 (flipped).
    control models: flat across environments (their s was never informative).
    oracle FNO:     flat and best (sees the true M).

Also settles Gate 4(iv): biased must beat control on eval_id (shortcut adopted).
"""

import numpy as np
import pandas as pd
import torch

from ..common.metrics import relative_l2_error_batch
from ..common.normalization import normalize_array


EVAL_ENVS = ("eval_id", "eval_rho05", "eval_broken", "eval_flipped")


def eval_fno_on_env(model, npz, answer_key, env, input_kind, stats, device, batch_size=64):
    """
    Mean relative L2 of a field surrogate on one environment.

    Inputs:
        model: torch.nn.Module (eval mode assumed handled here).
        npz: pool arrays (u0, mobility, s_fields, u_final).
        answer_key: pd.DataFrame.
        env: str, environment name.
        input_kind: "u0_s" or "u0_M".
        stats: dict with X_mean/X_std/y_mean/y_std (this model's normalizers).
        device: torch.device.

    Outputs:
        (mean_rel_l2, per_sample_errors): tuple(float, np.ndarray).
    """
    ids = answer_key.loc[answer_key["environment"] == env, "sample_id"].to_numpy()
    u0 = npz["u0"][ids]
    second = npz["s_fields"][ids] if input_kind == "u0_s" else npz["mobility"][ids]
    X = np.stack([u0, second], axis=1)
    y = npz["u_final"][ids][:, np.newaxis]

    X_norm = normalize_array(X, stats["X_mean"], stats["X_std"])

    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            xb = torch.from_numpy(X_norm[start:start + batch_size]).to(device)
            pred = model(xb).cpu().numpy() * stats["y_std"] + stats["y_mean"]
            err = relative_l2_error_batch(
                torch.from_numpy(pred), torch.from_numpy(y[start:start + batch_size]), eps=1e-6
            )
            errors.append(err.numpy())
    errors = np.concatenate(errors)
    return float(errors.mean()), errors


def eval_head_on_env(head, npz, answer_key, env, stats, device, batch_size=128):
    """
    Severity-head MAE on one environment (inputs are always (u0, s)).

    Outputs:
        (mae, per_sample_abs_errors): tuple(float, np.ndarray).
    """
    ids = answer_key.loc[answer_key["environment"] == env, "sample_id"].to_numpy()
    X = np.stack([npz["u0"][ids], npz["s_fields"][ids]], axis=1)
    sigma = answer_key.set_index("sample_id").loc[ids, "severity"].to_numpy(dtype=np.float32)

    X_norm = normalize_array(X, stats["X_mean"], stats["X_std"])

    head.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            xb = torch.from_numpy(X_norm[start:start + batch_size]).to(device)
            preds.append(head(xb).cpu().numpy()[:, 0])
    preds = np.concatenate(preds)
    abs_err = np.abs(preds - sigma)
    return float(abs_err.mean()), abs_err


def collapse_table(fno_results: dict, head_results: dict) -> pd.DataFrame:
    """
    Assemble the collapse table with effect sizes.

    Inputs:
        fno_results: dict role -> dict env -> mean rel-L2.
        head_results: dict role -> dict env -> MAE.

    Outputs:
        table: pd.DataFrame with one row per (model, env) plus a
            collapse_effect_size column: (err_flipped - err_id) / err_id.
    """
    rows = []
    for family, results, metric in (
        ("fno", fno_results, "rel_l2"),
        ("head", head_results, "sigma_mae"),
    ):
        for role, env_errors in results.items():
            effect = (env_errors["eval_flipped"] - env_errors["eval_id"]) / env_errors["eval_id"]
            for env in EVAL_ENVS:
                rows.append(
                    {
                        "model": f"{family}_{role}",
                        "metric": metric,
                        "environment": env,
                        "error": env_errors[env],
                        "collapse_effect_size": effect,
                    }
                )
    return pd.DataFrame(rows)
