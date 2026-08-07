"""Primary Allen-Cahn solver: Fourier pseudo-spectral, semi-implicit SBDF2 (IMEX).

PDE (tilted double well, spatially varying mobility):

    du/dt = M(x) * (eps^2 lap(u) + u - u^3 + g)

Periodic square domain, batched over samples. The scheme splits the diffusion as

    M(x) eps^2 lap(u) = M_max eps^2 lap(u) + (M(x) - M_max) eps^2 lap(u)

and treats the first (constant-coefficient) part implicitly - a diagonal solve
in Fourier space - while the remainder and the reaction N(u) are explicit:

    N = (M(x) - M_max) eps^2 lap(u) + M(x) (u - u^3 + g)

Time stepping is second-order semi-implicit BDF (SBDF2), the standard scheme
for phase-field equations:

    (3 u^{n+1} - 4 u^n + u^{n-1}) / (2 dt) = M_max eps^2 lap(u^{n+1})
                                             + 2 N(u^n) - N(u^{n-1})

started with one semi-implicit Euler step. Since M(x) <= M_max the implicit
operator over-damps and the explicit term corrects, the standard stabilization
for variable coefficients. With M constant this reduces to the textbook
semi-implicit Allen-Cahn scheme. Second-order dt convergence and energy
monotonicity are checked empirically in verification (SBDF2 is not provably
gradient-stable, so the energy check is load-bearing, not decorative).
"""

import numpy as np


def make_k2(nx: int, ny: int, domain_size: float) -> np.ndarray:
    """
    Squared wavenumber magnitude grid for a periodic domain.

    Inputs:
        nx, ny: int, grid points.
        domain_size: float, square domain side length.

    Outputs:
        k2: np.ndarray, shape (nx, ny), k_x^2 + k_y^2.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=domain_size / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=domain_size / ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    return kxg**2 + kyg**2


def _as_batched(u0: np.ndarray) -> tuple:
    """Promote (nx, ny) input to (1, nx, ny); report whether it was batched."""
    if u0.ndim == 2:
        return u0[np.newaxis, ...], False
    if u0.ndim == 3:
        return u0, True
    raise ValueError(f"u0 must have 2 or 3 dims, got shape {u0.shape}.")


def _broadcast_param(value, batch: int) -> np.ndarray:
    """Broadcast a scalar or (B,) parameter to shape (B, 1, 1)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(batch, float(arr))
    if arr.shape != (batch,):
        raise ValueError(f"Per-sample parameter must be scalar or shape ({batch},), got {arr.shape}.")
    return arr[:, np.newaxis, np.newaxis]


def solve_allen_cahn_imex(
    u0: np.ndarray,
    eps_param,
    t_final: float,
    dt: float,
    domain_size: float = 1.0,
    mobility: np.ndarray = None,
    g: float = 0.0,
    snapshot_times: tuple = (),
    record_energy_every: int = 0,
    backend: str = "numpy",
) -> dict:
    """
    Solve the Allen-Cahn equation with the stabilized IMEX spectral scheme.

    Inputs:
        u0: np.ndarray, shape (B, nx, ny) or (nx, ny), initial condition.
        eps_param: float or np.ndarray (B,), Allen-Cahn epsilon per sample.
        t_final: float, integration horizon.
        dt: float, time step.
        domain_size: float, square domain side length (periodic).
        mobility: np.ndarray, shape (B, nx, ny) or (nx, ny), M(x) >= 0.
            None means M = 1 everywhere.
        g: float, well tilt (bulk driving force; g=0 is textbook AC).
        snapshot_times: tuple of floats, times at which to store the field.
            Each is rounded to the nearest step multiple.
        record_energy_every: int, if > 0, record the Ginzburg-Landau energy
            every that many steps (used by solver verification).
        backend: str, "numpy" (float64, the verified reference path) or
            "torch" (float32 on CUDA if available; ~100x faster for large
            batches, used for bulk dataset/pool/Sobol generation - torch-vs-
            numpy agreement is covered by a unit test). The torch path does
            not support record_energy_every.

    Outputs:
        result: dict with
            u_final: np.ndarray, (B, nx, ny) float32 (or (nx, ny) if unbatched input),
            snapshots: dict time -> np.ndarray (B, nx, ny) float32,
            energy_times: np.ndarray (n_rec,) (present if record_energy_every > 0),
            energies: np.ndarray (B, n_rec) (present if record_energy_every > 0).
    """
    from ..common.metrics import ginzburg_landau_energy

    u, was_batched = _as_batched(np.asarray(u0, dtype=np.float64))
    batch, nx, ny = u.shape

    if mobility is None:
        m_field = np.ones_like(u)
    else:
        m_field, _ = _as_batched(np.asarray(mobility, dtype=np.float64))
        if m_field.shape[0] == 1 and batch > 1:
            m_field = np.broadcast_to(m_field, u.shape).copy()
    if np.any(m_field < 0):
        raise ValueError("Mobility must be nonnegative.")

    eps_b = _broadcast_param(eps_param, batch)
    m_max = m_field.max(axis=(-2, -1), keepdims=True)

    k2 = make_k2(nx, ny, domain_size)
    implicit_rate = m_max * (eps_b**2) * k2
    euler_denominator = 1.0 + dt * implicit_rate
    sbdf2_denominator = 1.5 + dt * implicit_rate

    n_steps = int(round(t_final / dt))
    if not np.isclose(n_steps * dt, t_final, rtol=1e-6):
        raise ValueError(f"t_final={t_final} is not an integer multiple of dt={dt}.")

    snapshot_steps = {}
    for t_snap in snapshot_times:
        step = int(round(t_snap / dt))
        if not 0 < step <= n_steps:
            raise ValueError(f"Snapshot time {t_snap} outside (0, t_final].")
        snapshot_steps[step] = float(t_snap)

    snapshots = {}
    energy_times = []
    energies = []

    eps_flat = eps_b[:, 0, 0]

    if backend == "torch":
        if record_energy_every > 0:
            raise ValueError("record_energy_every requires the numpy backend.")
        u_final, snapshots = _imex_loop_torch(
            u, m_field, m_max, eps_b, k2, dt, n_steps, g, snapshot_steps,
            euler_denominator, sbdf2_denominator,
        )
        if not was_batched:
            u_final = u_final[0]
            snapshots = {t: s[0] for t, s in snapshots.items()}
        return {"u_final": u_final, "snapshots": snapshots}

    def explicit_term(field: np.ndarray) -> np.ndarray:
        lap = np.fft.ifft2(-k2 * np.fft.fft2(field, axes=(-2, -1)), axes=(-2, -1)).real
        return (m_field - m_max) * (eps_b**2) * lap + m_field * (field - field**3 + g)

    u_prev = None
    n_prev = None

    for step in range(1, n_steps + 1):
        n_curr = explicit_term(u)
        u_hat = np.fft.fft2(u, axes=(-2, -1))

        if u_prev is None:
            # Startup: one semi-implicit Euler step (first order, single step).
            u_hat_new = (u_hat + dt * np.fft.fft2(n_curr, axes=(-2, -1))) / euler_denominator
        else:
            u_prev_hat = np.fft.fft2(u_prev, axes=(-2, -1))
            rhs = (
                2.0 * u_hat
                - 0.5 * u_prev_hat
                + dt * np.fft.fft2(2.0 * n_curr - n_prev, axes=(-2, -1))
            )
            u_hat_new = rhs / sbdf2_denominator

        u_prev = u
        n_prev = n_curr
        u = np.fft.ifft2(u_hat_new, axes=(-2, -1)).real

        if step in snapshot_steps:
            snapshots[snapshot_steps[step]] = u.astype(np.float32).copy()

        if record_energy_every > 0 and step % record_energy_every == 0:
            energy_times.append(step * dt)
            energies.append(
                np.array([
                    ginzburg_landau_energy(u[b], float(eps_flat[b]), domain_size, g=g)
                    for b in range(batch)
                ])
            )

    u_final = u.astype(np.float32)
    if not was_batched:
        u_final = u_final[0]
        snapshots = {t: s[0] for t, s in snapshots.items()}

    result = {"u_final": u_final, "snapshots": snapshots}
    if record_energy_every > 0:
        result["energy_times"] = np.array(energy_times)
        result["energies"] = np.stack(energies, axis=1) if energies else np.empty((batch, 0))
    return result


def _imex_loop_torch(
    u, m_field, m_max, eps_b, k2, dt, n_steps, g, snapshot_steps,
    euler_denominator, sbdf2_denominator,
) -> tuple:
    """
    SBDF2 stepping on the GPU (float32). Same math as the numpy loop above.

    Outputs:
        (u_final, snapshots): np.float32 array (B, nx, ny) and dict t -> array.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u_t = torch.tensor(u, dtype=torch.float32, device=device)
    m_t = torch.tensor(m_field, dtype=torch.float32, device=device)
    m_max_t = torch.tensor(m_max, dtype=torch.float32, device=device)
    eps2_t = torch.tensor(eps_b**2, dtype=torch.float32, device=device)
    k2_t = torch.tensor(k2, dtype=torch.float32, device=device)
    euler_den = torch.tensor(euler_denominator, dtype=torch.float32, device=device)
    sbdf2_den = torch.tensor(sbdf2_denominator, dtype=torch.float32, device=device)

    def explicit_term(field):
        lap = torch.fft.ifft2(-k2_t * torch.fft.fft2(field)).real
        return (m_t - m_max_t) * eps2_t * lap + m_t * (field - field**3 + g)

    snapshots = {}
    u_prev = None
    n_prev = None

    for step in range(1, n_steps + 1):
        n_curr = explicit_term(u_t)
        u_hat = torch.fft.fft2(u_t)

        if u_prev is None:
            u_hat_new = (u_hat + dt * torch.fft.fft2(n_curr)) / euler_den
        else:
            rhs = (
                2.0 * u_hat
                - 0.5 * torch.fft.fft2(u_prev)
                + dt * torch.fft.fft2(2.0 * n_curr - n_prev)
            )
            u_hat_new = rhs / sbdf2_den

        u_prev = u_t
        n_prev = n_curr
        u_t = torch.fft.ifft2(u_hat_new).real

        if step in snapshot_steps:
            snapshots[snapshot_steps[step]] = u_t.cpu().numpy().astype(np.float32)

    return u_t.cpu().numpy().astype(np.float32), snapshots
