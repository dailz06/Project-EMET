"""Spectral resampling between grids.

Validator targets are produced at 128^2 and must be brought to the native 64^2
grid by Fourier truncation, never by slicing/striding - striding aliases the
above-Nyquist content of the fine solution into the coarse grid and would
contaminate the independence check.

Nyquist handling (even grid sizes): the coarse grid's Nyquist bin k = n/2
represents both +n/2 and -n/2. Upsampling splits its coefficient evenly across
the two fine-grid bins (the unique real, symmetric trigonometric interpolant);
downsampling sums the two fine bins back. With this convention
downsample(upsample(u)) == u exactly and both directions preserve realness.
"""

import numpy as np


def _spread_matrix(n_in: int, n_out: int) -> np.ndarray:
    """
    (n_out, n_in) matrix mapping coarse FFT bins to fine FFT bins (n_out >= n_in),
    with the coarse Nyquist coefficient split evenly across +/-n_in/2.
    """
    mat = np.zeros((n_out, n_in))
    half = n_in // 2
    for k in range(half):
        mat[k, k] = 1.0
    for k in range(1, half):
        mat[n_out - k, n_in - k] = 1.0
    # Coarse Nyquist bin (index half, representing +/-half).
    mat[half, half] = 0.5
    mat[n_out - half, half] = 0.5
    return mat


def _truncate_matrix(n_out: int, n_in: int) -> np.ndarray:
    """
    (n_out, n_in) matrix mapping fine FFT bins to coarse FFT bins (n_out <= n_in),
    with the coarse Nyquist bin collecting both fine +/-n_out/2 bins.
    """
    mat = np.zeros((n_out, n_in))
    half = n_out // 2
    for k in range(half):
        mat[k, k] = 1.0
    for k in range(1, half):
        mat[n_out - k, n_in - k] = 1.0
    mat[half, half] = 1.0
    mat[half, n_in - half] = 1.0
    return mat


def _apply_resample(u: np.ndarray, mat: np.ndarray, n_out: int) -> np.ndarray:
    u_hat = np.fft.fft2(u, axes=(-2, -1))
    resampled = np.einsum("ij,...jk,lk->...il", mat, u_hat, mat)
    scale = (n_out / u.shape[-1]) ** 2
    return np.fft.ifft2(resampled * scale, axes=(-2, -1)).real.astype(np.float32)


def fourier_downsample(u: np.ndarray, n_out: int) -> np.ndarray:
    """
    Downsample periodic fields by truncating the Fourier spectrum.

    Inputs:
        u: np.ndarray, shape (..., n_in, n_in).
        n_out: int, output grid points per side, <= n_in.

    Outputs:
        u_coarse: np.ndarray, shape (..., n_out, n_out), float32.
    """
    n_in = u.shape[-1]
    if u.shape[-2] != n_in:
        raise ValueError("fourier_downsample expects square fields.")
    if n_out > n_in:
        raise ValueError(f"n_out={n_out} exceeds input resolution {n_in}.")
    if n_out == n_in:
        return u.astype(np.float32)

    return _apply_resample(u, _truncate_matrix(n_out, n_in), n_out)


def fourier_upsample(u: np.ndarray, n_out: int) -> np.ndarray:
    """
    Upsample periodic fields by zero-padding the Fourier spectrum (exact
    trigonometric interpolation of band-limited fields).

    Inputs:
        u: np.ndarray, shape (..., n_in, n_in).
        n_out: int, output grid points per side, >= n_in.

    Outputs:
        u_fine: np.ndarray, shape (..., n_out, n_out), float32.

    Note:
        Used to lift native 64^2 inputs to the validator's 128^2 grid so the
        validator solves the SAME input fields; only the scheme/resolution of
        the solve differs. Pointwise-clipped fields (e.g. mobility) can slightly
        over/undershoot their clip bounds after interpolation - the caller
        re-clips mobility at zero if needed.
    """
    n_in = u.shape[-1]
    if u.shape[-2] != n_in:
        raise ValueError("fourier_upsample expects square fields.")
    if n_out < n_in:
        raise ValueError(f"n_out={n_out} is below input resolution {n_in}.")
    if n_out == n_in:
        return u.astype(np.float32)

    return _apply_resample(u, _spread_matrix(n_in, n_out), n_out)
