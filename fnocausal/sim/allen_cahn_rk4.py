"""Validator Allen-Cahn solver: explicit RK4, Fourier pseudo-spectral Laplacian.

Intentionally a different time integrator than the primary IMEX scheme
(generator != validator discipline). Meant to run at higher resolution and
smaller dt than the primary solver (default usage: 128^2 grid, dt/4);
validator targets are Fourier-downsampled to the native grid afterwards.

Being fully explicit, the scheme has a stability limit. The linear stiffness is
lam = M_max * (eps^2 * k_max^2 + reaction), and RK4's real-axis stability
interval is |lam*dt| < ~2.79; solve() raises if the requested dt violates it.
"""

import numpy as np

from .allen_cahn_spectral import _as_batched, _broadcast_param, make_k2

RK4_REAL_AXIS_LIMIT = 2.79


def solve_allen_cahn_rk4(
    u0: np.ndarray,
    eps_param,
    t_final: float,
    dt: float,
    domain_size: float = 1.0,
    mobility: np.ndarray = None,
    g: float = 0.0,
    snapshot_times: tuple = (),
    backend: str = "numpy",
) -> dict:
    """
    Solve the Allen-Cahn equation with explicit RK4 and spectral Laplacian.

    Inputs:
        u0: np.ndarray, shape (B, nx, ny) or (nx, ny), initial condition.
        eps_param: float or np.ndarray (B,), Allen-Cahn epsilon per sample.
        t_final: float, integration horizon.
        dt: float, time step (checked against the RK4 stability limit).
        domain_size: float, square domain side length (periodic).
        mobility: np.ndarray, M(x) >= 0; None means M = 1.
        g: float, well tilt.
        snapshot_times: tuple of floats, rounded to nearest step multiple.
        backend: str, "numpy" (float64 reference) or "torch" (float32 CUDA,
            for the bulk validator solves; agreement covered by a unit test).

    Outputs:
        result: dict with u_final and snapshots (same conventions as the IMEX solver).
    """
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
    k2 = make_k2(nx, ny, domain_size)

    # Stability check: worst-case linear rate over samples. The reaction term
    # contributes Lipschitz constant ~2 near the wells (d/du of u - u^3 at u=+-1).
    m_max = float(m_field.max())
    eps_max = float(np.max(eps_b))
    stiffness = m_max * (eps_max**2 * float(k2.max()) + 2.0)
    if stiffness * dt > RK4_REAL_AXIS_LIMIT:
        raise ValueError(
            f"RK4 unstable: stiffness*dt = {stiffness * dt:.2f} exceeds "
            f"{RK4_REAL_AXIS_LIMIT}. Reduce dt below {RK4_REAL_AXIS_LIMIT / stiffness:.2e}."
        )

    n_steps = int(round(t_final / dt))
    if not np.isclose(n_steps * dt, t_final, rtol=1e-6):
        raise ValueError(f"t_final={t_final} is not an integer multiple of dt={dt}.")

    snapshot_steps = {}
    for t_snap in snapshot_times:
        step = int(round(t_snap / dt))
        if not 0 < step <= n_steps:
            raise ValueError(f"Snapshot time {t_snap} outside (0, t_final].")
        snapshot_steps[step] = float(t_snap)

    if backend == "torch":
        u_final, snapshots = _rk4_loop_torch(
            u, m_field, eps_b, k2, dt, n_steps, g, snapshot_steps
        )
        if not was_batched:
            u_final = u_final[0]
            snapshots = {t: s[0] for t, s in snapshots.items()}
        return {"u_final": u_final, "snapshots": snapshots}

    def rhs(field: np.ndarray) -> np.ndarray:
        lap = np.fft.ifft2(-k2 * np.fft.fft2(field, axes=(-2, -1)), axes=(-2, -1)).real
        return m_field * ((eps_b**2) * lap + field - field**3 + g)

    snapshots = {}
    for step in range(1, n_steps + 1):
        k1 = rhs(u)
        k2_ = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2_)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2_ + 2.0 * k3 + k4)

        if step in snapshot_steps:
            snapshots[snapshot_steps[step]] = u.astype(np.float32).copy()

    u_final = u.astype(np.float32)
    if not was_batched:
        u_final = u_final[0]
        snapshots = {t: s[0] for t, s in snapshots.items()}

    return {"u_final": u_final, "snapshots": snapshots}


def _rk4_loop_torch(u, m_field, eps_b, k2, dt, n_steps, g, snapshot_steps) -> tuple:
    """
    RK4 stepping on the GPU (float32). Same math as the numpy loop above.

    Outputs:
        (u_final, snapshots): np.float32 array (B, nx, ny) and dict t -> array.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u_t = torch.tensor(u, dtype=torch.float32, device=device)
    m_t = torch.tensor(m_field, dtype=torch.float32, device=device)
    eps2_t = torch.tensor(eps_b**2, dtype=torch.float32, device=device)
    k2_t = torch.tensor(k2, dtype=torch.float32, device=device)

    def rhs(field):
        lap = torch.fft.ifft2(-k2_t * torch.fft.fft2(field)).real
        return m_t * (eps2_t * lap + field - field**3 + g)

    snapshots = {}
    for step in range(1, n_steps + 1):
        k1 = rhs(u_t)
        k2_stage = rhs(u_t + 0.5 * dt * k1)
        k3 = rhs(u_t + 0.5 * dt * k2_stage)
        k4 = rhs(u_t + dt * k3)
        u_t = u_t + (dt / 6.0) * (k1 + 2.0 * k2_stage + 2.0 * k3 + k4)

        if step in snapshot_steps:
            snapshots[snapshot_steps[step]] = u_t.cpu().numpy().astype(np.float32)

    return u_t.cpu().numpy().astype(np.float32), snapshots
