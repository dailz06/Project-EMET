"""Error and physics metrics.

relative_l2_error_batch is ported from phase0_phase1_local_pipeline.py (L1444).
New physics metrics for Allen-Cahn verification and Phase 5 fidelity checks:
Ginzburg-Landau free energy, interface length, transformed area fraction
(the "severity" scalar), and the structure-factor coarsening length.

All physics metrics accept batched fields (..., nx, ny) and compute spatial
derivatives spectrally (periodic domain), consistent with the solvers.
"""

import numpy as np
import torch


def relative_l2_error_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Compute per-sample relative L2 error.

    Inputs:
        pred: torch.Tensor, shape (B, C, nx, ny), unnormalized.
        target: torch.Tensor, shape (B, C, nx, ny), unnormalized.
        eps: float.

    Outputs:
        rel_errors: torch.Tensor, shape (B,).
    """
    diff_norm = torch.linalg.vector_norm((pred - target).flatten(start_dim=1), dim=1)
    target_norm = torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1)

    return diff_norm / (target_norm + eps)


def relative_l2_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-sample relative L2 error for numpy arrays.

    Inputs:
        pred: np.ndarray, shape (B, ...) or (nx, ny).
        target: np.ndarray, same shape.
        eps: float.

    Outputs:
        rel_errors: np.ndarray, shape (B,) (scalar array if unbatched input).
    """
    if pred.ndim > 2:
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
    else:
        p = pred.reshape(1, -1)
        t = target.reshape(1, -1)

    diff = p - t
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(t, axis=1) + eps)


def _spectral_gradients(u: np.ndarray, domain_size: float) -> tuple:
    """
    Spectral gradients of batched periodic fields.

    Inputs:
        u: np.ndarray, shape (..., nx, ny).
        domain_size: float.

    Outputs:
        (ux, uy): np.ndarray pair, each shape (..., nx, ny).
    """
    nx, ny = u.shape[-2], u.shape[-1]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=domain_size / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=domain_size / ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")

    u_hat = np.fft.fft2(u, axes=(-2, -1))
    ux = np.fft.ifft2(1j * kxg * u_hat, axes=(-2, -1)).real
    uy = np.fft.ifft2(1j * kyg * u_hat, axes=(-2, -1)).real
    return ux, uy


def ginzburg_landau_energy(
    u: np.ndarray,
    eps_param: float,
    domain_size: float,
    g: float = 0.0,
) -> np.ndarray:
    """
    Ginzburg-Landau free energy E[u] = integral of eps^2/2 |grad u|^2 + F(u),
    with F(u) = (1 - u^2)^2 / 4 - g*u (tilted double well; g=0 is textbook AC).

    Inputs:
        u: np.ndarray, shape (..., nx, ny).
        eps_param: float, Allen-Cahn epsilon.
        domain_size: float.
        g: float, well tilt (bulk driving force).

    Outputs:
        energy: np.ndarray, shape (...,). Allen-Cahn is the L2 gradient flow of E
            (times mobility M >= 0), so E must be nonincreasing along trajectories.
    """
    ux, uy = _spectral_gradients(u, domain_size)
    grad_sq = ux**2 + uy**2
    bulk = 0.25 * (1.0 - u**2) ** 2 - g * u

    cell_area = (domain_size / u.shape[-2]) * (domain_size / u.shape[-1])
    density = 0.5 * eps_param**2 * grad_sq + bulk
    return density.sum(axis=(-2, -1)) * cell_area


def interface_length(u: np.ndarray, domain_size: float) -> np.ndarray:
    """
    Total interface length estimate: integral |grad u| dx / 2.

    Inputs:
        u: np.ndarray, shape (..., nx, ny), phases near +/-1.
        domain_size: float.

    Outputs:
        length: np.ndarray, shape (...,).

    Why /2:
        Across an equilibrium interface the profile jumps by Delta(u) = 2, so the
        line integral of |grad u| along the interface normal is ~2 per unit length.
    """
    ux, uy = _spectral_gradients(u, domain_size)
    grad_mag = np.sqrt(ux**2 + uy**2)
    cell_area = (domain_size / u.shape[-2]) * (domain_size / u.shape[-1])
    return grad_mag.sum(axis=(-2, -1)) * cell_area / 2.0


def transformed_area_fraction(u: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Fraction of the domain in the +1 (transformed/degraded) phase.

    Inputs:
        u: np.ndarray, shape (..., nx, ny).
        threshold: float, phase boundary value.

    Outputs:
        fraction: np.ndarray, shape (...,), in [0, 1]. This is the Phase 4/5
            severity scalar sigma.
    """
    return (u > threshold).mean(axis=(-2, -1))


def structure_factor_length(u: np.ndarray, domain_size: float, eps: float = 1e-12) -> np.ndarray:
    """
    Characteristic coarsening length from the first moment of the structure factor:
    l = 2*pi * sum(S(k)) / sum(k * S(k)), computed on the mean-subtracted field.

    Inputs:
        u: np.ndarray, shape (..., nx, ny).
        domain_size: float.
        eps: float.

    Outputs:
        length: np.ndarray, shape (...,). For nonconserved Allen-Cahn dynamics,
            l(t) ~ t^(1/2) during coarsening.
    """
    nx, ny = u.shape[-2], u.shape[-1]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=domain_size / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=domain_size / ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    k_rad = np.sqrt(kxg**2 + kyg**2)

    du = u - u.mean(axis=(-2, -1), keepdims=True)
    s_k = np.abs(np.fft.fft2(du, axes=(-2, -1))) ** 2
    s_k[..., 0, 0] = 0.0

    first_moment = (k_rad * s_k).sum(axis=(-2, -1))
    zeroth_moment = s_k.sum(axis=(-2, -1))
    return 2.0 * np.pi * zeroth_moment / (first_moment + eps)
