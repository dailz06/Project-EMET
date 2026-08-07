"""Gaussian random fields and initial-condition builders.

generate_grf and the lognormal/rescale transforms are ported from
phase0_phase1_local_pipeline.py (L412-503). New for Allen-Cahn: tanh ICs
(near-equilibrium two-phase fields) and nucleation ICs (seeded bumps in a
uniform background), plus the fixed nuisance pattern used in Phase 4.
"""

import numpy as np


def sample_uniform_range(rng: np.random.Generator, value_range: tuple) -> float:
    """
    Sample uniformly from a numeric range.

    Inputs:
        rng: np.random.Generator.
        value_range: tuple(float, float).

    Outputs:
        value: float.
    """
    return float(rng.uniform(value_range[0], value_range[1]))


def generate_grf(
    nx: int,
    ny: int,
    domain_size: float,
    corr_length: float,
    spectral_decay: float,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Generate a standardized 2D Gaussian random field using Fourier filtering.

    Inputs:
        nx: int, grid points in x.
        ny: int, grid points in y.
        domain_size: float, square domain side length.
        corr_length: float, correlation length in domain units.
        spectral_decay: float, high-frequency power spectral decay exponent.
        rng: np.random.Generator.
        eps: float, numerical stabilizer.

    Outputs:
        field: np.ndarray, shape (nx, ny), dtype float32, zero mean and unit std.

    Why GRFs:
        They provide a controllable, well-defined input distribution. Correlation
        length and spectral decay are physically interpretable and can be held out.
    """
    white_noise = rng.normal(loc=0.0, scale=1.0, size=(nx, ny))

    kx = np.fft.fftfreq(nx, d=domain_size / nx)
    ky = np.fft.fftfreq(ny, d=domain_size / ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_rad = 2.0 * np.pi * np.sqrt(kx_grid**2 + ky_grid**2)

    amplitude_filter = (1.0 + (corr_length * k_rad) ** 2) ** (-spectral_decay / 4.0)
    amplitude_filter[0, 0] = 0.0

    field = np.fft.ifft2(np.fft.fft2(white_noise) * amplitude_filter).real
    field = (field - field.mean()) / (field.std() + eps)

    return field.astype(np.float32)


def lognormal_field_from_grf(
    grf: np.ndarray,
    field_mean: float,
    field_log_std: float,
    field_clip: tuple,
) -> np.ndarray:
    """
    Convert a standardized GRF into a positive coefficient field (e.g. mobility M(x)).

    Inputs:
        grf: np.ndarray, shape (nx, ny), standardized field.
        field_mean: float, approximate mean value.
        field_log_std: float, log-space standard deviation.
        field_clip: tuple(float, float), min/max values.

    Outputs:
        field: np.ndarray, shape (nx, ny), dtype float32.
    """
    field = field_mean * np.exp(field_log_std * grf - 0.5 * field_log_std**2)
    field = np.clip(field, field_clip[0], field_clip[1])
    return field.astype(np.float32)


def tanh_ic_from_grf(grf: np.ndarray, interface_width: float) -> np.ndarray:
    """
    Convert a standardized GRF into a near-two-phase Allen-Cahn initial condition.

    Inputs:
        grf: np.ndarray, shape (nx, ny), standardized field.
        interface_width: float, tanh squash scale in grf std units. Smaller
            values push u0 closer to the +/-1 wells.

    Outputs:
        u0: np.ndarray, shape (nx, ny), dtype float32, values in (-1, 1).
    """
    return np.tanh(grf / interface_width).astype(np.float32)


def nucleation_ic(
    nx: int,
    ny: int,
    domain_size: float,
    n_seeds: int,
    radius_range: tuple,
    interface_eps: float,
    rng: np.random.Generator,
    background: float = -1.0,
) -> np.ndarray:
    """
    Build an initial condition of n_seeds tanh-profile circular nuclei (+1 phase)
    in a uniform background (-1 phase), with periodic minimum-image distances.

    Inputs:
        nx, ny: int, grid points.
        domain_size: float, square domain side length.
        n_seeds: int, number of nuclei.
        radius_range: tuple(float, float), nucleus radii in domain units.
        interface_eps: float, Allen-Cahn epsilon; interface profile is
            tanh((r - d)/(sqrt(2)*interface_eps)), the equilibrium 1D profile.
        rng: np.random.Generator.
        background: float, background phase value.

    Outputs:
        u0: np.ndarray, shape (nx, ny), dtype float32.
    """
    x = np.linspace(0.0, domain_size, nx, endpoint=False)
    y = np.linspace(0.0, domain_size, ny, endpoint=False)
    xg, yg = np.meshgrid(x, y, indexing="ij")

    u0 = np.full((nx, ny), background, dtype=np.float64)
    width = np.sqrt(2.0) * interface_eps

    for _ in range(n_seeds):
        xc = rng.uniform(0.0, domain_size)
        yc = rng.uniform(0.0, domain_size)
        radius = rng.uniform(radius_range[0], radius_range[1])

        dx = np.abs(xg - xc)
        dy = np.abs(yg - yc)
        dx = np.minimum(dx, domain_size - dx)
        dy = np.minimum(dy, domain_size - dy)
        dist = np.sqrt(dx**2 + dy**2)

        bump = np.tanh((radius - dist) / width)
        u0 = np.maximum(u0, bump)

    return u0.astype(np.float32)


def nuisance_pattern(nx: int, ny: int, n_waves: int = 3) -> np.ndarray:
    """
    Fixed deterministic low-frequency pattern P(x) for the Phase 4 nuisance channel.

    Inputs:
        nx, ny: int, grid points.
        n_waves: int, number of superposed plane waves.

    Outputs:
        pattern: np.ndarray, shape (nx, ny), dtype float32, zero mean, unit std.

    Note:
        Deterministic (hard-coded wave vectors/phases), so the same P(x) is used
        in every environment; only its amplitude s_level varies per sample. The
        nuisance channel never reaches any solver: s(x) = s_level * P(x) + noise
        is assembled in dataset code, and solver APIs take only (u0, M, eps, g).
    """
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    xg, yg = np.meshgrid(x, y, indexing="ij")

    wave_vectors = [(1, 2), (2, -1), (3, 1)][:n_waves]
    phases = [0.0, 1.1, 2.3][:n_waves]

    pattern = np.zeros((nx, ny), dtype=np.float64)
    for (kx, ky), phase in zip(wave_vectors, phases):
        pattern += np.sin(2.0 * np.pi * (kx * xg + ky * yg) + phase)

    pattern = (pattern - pattern.mean()) / pattern.std()
    return pattern.astype(np.float32)
