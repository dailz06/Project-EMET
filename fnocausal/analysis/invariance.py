"""Diagnostic 3: invariance analysis (ICP / IRM style).

Theoretical backbone: a feature whose predictive relationship with the target
is STABLE across environments is (under the ICP assumptions) causal; one whose
relationship changes sign or magnitude with the environment is spurious.

Implementation on the biased AE latent z:
    1. Per-environment ridge probes z -> sigma for environments with different
       rho (train_rho95, train_rho80, control_train). Project each probe's
       coefficient vector onto the intervention directions w_S / w_C from
       latent_probing: the w_S component must track rho (unstable, sign-flips
       on flipped data), the w_C component must stay stable.
    2. An IRMv1-penalized linear probe trained across the environments: its
       w_S component should be driven toward zero relative to an ERM probe.
"""

import numpy as np
import pandas as pd
import torch


def ridge_probe(z: np.ndarray, sigma: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Closed-form ridge regression coefficients for z -> sigma.

    Inputs:
        z: np.ndarray, (N, d), centered internally.
        sigma: np.ndarray, (N,).
        alpha: float, L2 penalty.

    Outputs:
        coef: np.ndarray, (d,).
    """
    z_c = z - z.mean(axis=0)
    y_c = sigma - sigma.mean()
    gram = z_c.T @ z_c + alpha * np.eye(z.shape[1])
    return np.linalg.solve(gram, z_c.T @ y_c)


def per_environment_probes(
    z_by_env: dict,
    sigma_by_env: dict,
    w_S: np.ndarray,
    w_C: np.ndarray,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Fit one probe per environment and decompose coefficients onto w_S / w_C.

    Inputs:
        z_by_env: dict env -> (N_e, d) latents.
        sigma_by_env: dict env -> (N_e,) severities.
        w_S, w_C: unit intervention directions.
        alpha: ridge penalty.

    Outputs:
        table: pd.DataFrame with env, rho, coef_S, coef_C, r2 columns.
    """
    rows = []
    for env, z in z_by_env.items():
        sigma = sigma_by_env[env]
        coef = ridge_probe(z, sigma, alpha)

        z_c = z - z.mean(axis=0)
        pred = z_c @ coef
        y_c = sigma - sigma.mean()
        r2 = 1.0 - np.sum((y_c - pred) ** 2) / (np.sum(y_c**2) + 1e-30)

        rows.append(
            {
                "environment": env,
                "coef_S": float(coef @ w_S),
                "coef_C": float(coef @ w_C),
                "coef_norm": float(np.linalg.norm(coef)),
                "probe_r2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def univariate_score_probes(
    z_by_env: dict,
    sigma_by_env: dict,
    w_S: np.ndarray,
    w_C: np.ndarray,
) -> pd.DataFrame:
    """
    Univariate regressions of sigma on the S-score (z @ w_S) and C-score
    (z @ w_C) separately, per environment.

    Inputs:
        z_by_env, sigma_by_env: dict env -> arrays.
        w_S, w_C: unit intervention directions.

    Outputs:
        table: pd.DataFrame with env, beta_S, beta_C, r2_S, r2_C.

    Why univariate:
        ICP's invariance statement is about regressing on a candidate CAUSAL
        feature set alone: that coefficient is environment-invariant iff the
        set is causal. A multivariate probe over all of z has no such
        guarantee - when a correlated shortcut is available the fit
        legitimately shifts weight from the causal to the spurious direction
        (that reweighting is itself evidence of shortcut adoption, and is
        reported separately by per_environment_probes). The clean tests are:
        beta_C stable across environments; beta_S tracking rho.
    """
    rows = []
    for env, z in z_by_env.items():
        sigma = sigma_by_env[env]
        row = {"environment": env}
        for name, w in (("S", w_S), ("C", w_C)):
            score = z @ w
            score_c = score - score.mean()
            y_c = sigma - sigma.mean()
            beta = float(score_c @ y_c / (score_c @ score_c + 1e-30))
            resid = y_c - beta * score_c
            row[f"beta_{name}"] = beta
            row[f"r2_{name}"] = float(1.0 - (resid @ resid) / (y_c @ y_c + 1e-30))
        rows.append(row)
    return pd.DataFrame(rows)


def irm_probe(
    z_by_env: dict,
    sigma_by_env: dict,
    irm_lambda: float = 100.0,
    epochs: int = 5000,
    lr: float = 1e-2,
    seed: int = 0,
) -> np.ndarray:
    """
    Linear probe trained with the IRMv1 penalty across environments.

    Inputs:
        z_by_env: dict env -> (N_e, d) latents (jointly standardized inside).
        sigma_by_env: dict env -> (N_e,) severities (scaled by pooled std
            inside so the risk gradients are well-conditioned).
        irm_lambda: float, penalty weight (0 -> plain ERM).
        epochs, lr, seed: optimization controls (lr drops 10x halfway).

    Outputs:
        coef: np.ndarray, (d,), mapped back to unstandardized z and sigma
            units so it is directly comparable with ridge_probe coefficients
            and usable in probe_r2.

    IRMv1 (Arjovsky et al. 2019): minimize sum_e R_e(w * scale) +
    lambda * ||grad_scale R_e||^2 at scale = 1, with a shared linear w.
    """
    torch.manual_seed(seed)

    all_z = np.concatenate(list(z_by_env.values()), axis=0)
    all_sigma = np.concatenate(list(sigma_by_env.values()))
    mu, sd = all_z.mean(axis=0), all_z.std(axis=0) + 1e-8
    y_scale = float(all_sigma.std()) + 1e-12

    envs = []
    for env in z_by_env:
        z = torch.tensor((z_by_env[env] - mu) / sd, dtype=torch.float32)
        y = torch.tensor(sigma_by_env[env] / y_scale, dtype=torch.float32)
        envs.append((z, y - y.mean()))

    d = all_z.shape[1]
    w = torch.zeros(d, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 2, gamma=0.1)

    for _ in range(epochs):
        optimizer.zero_grad()
        total = torch.tensor(0.0)
        for z, y in envs:
            scale = torch.tensor(1.0, requires_grad=True)
            risk = torch.mean((z @ (w * scale) - y) ** 2)
            grad = torch.autograd.grad(risk, scale, create_graph=True)[0]
            total = total + risk + irm_lambda * grad**2
        total.backward()
        optimizer.step()
        scheduler.step()

    coef_std = w.detach().numpy()
    # Map back to unstandardized z and sigma units so projections onto
    # w_S/w_C and probe_r2 evaluations are comparable with the ridge probes.
    return coef_std / sd * y_scale
