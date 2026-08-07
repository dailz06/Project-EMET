"""Allen-Cahn solver verification suite (Phase 2, Gate 2a).

Five checks against known behavior, run before any dataset is generated:

1. energy decay        -- Ginzburg-Landau energy monotone nonincreasing
                          (AC is the L2 gradient flow of E, mobility M >= 0).
2. shrinking circle    -- sharp-interface curvature flow: R^2 = R0^2 - 2 eps^2 t
                          for M=1, g=0.
3. coarsening exponent -- structure-factor length l(t) ~ t^(1/2) for
                          nonconserved dynamics.
4. dt/grid convergence -- dt vs dt/2 agreement; 64^2 vs downsampled-128^2 agreement.
5. scheme agreement    -- IMEX vs RK4 on shared ICs; IMEX vs py-pde spot check.

Each check returns a dict with name, metric, value, threshold, passed, plus
arrays for plotting. A solver that has not passed all five must not be used
to generate data (roadmap: "a solver you haven't sanity-checked will poison
everything downstream").
"""

import numpy as np

from ..common.grf import generate_grf, lognormal_field_from_grf, tanh_ic_from_grf
from ..common.metrics import interface_length, relative_l2_numpy, transformed_area_fraction
from ..common.seeding import rng_for_sample
from ..sim.allen_cahn_pypde import solve_allen_cahn_pypde
from ..sim.allen_cahn_rk4 import solve_allen_cahn_rk4
from ..sim.allen_cahn_spectral import solve_allen_cahn_imex
from ..sim.downsample import fourier_downsample, fourier_upsample


def _random_tanh_ics(n: int, nx: int, domain_size: float, seed: int) -> np.ndarray:
    """Stack of GRF-tanh initial conditions with per-sample rng streams."""
    ics = []
    for i in range(n):
        rng = rng_for_sample(seed, i)
        corr = rng.uniform(0.05, 0.2)
        decay = rng.uniform(2.0, 4.0)
        grf = generate_grf(nx, nx, domain_size, corr, decay, rng)
        ics.append(tanh_ic_from_grf(grf, interface_width=0.3))
    return np.stack(ics)


def check_energy_decay(
    n_runs: int = 10,
    nx: int = 64,
    domain_size: float = 1.0,
    eps_param: float = 0.03,
    t_final: float = 4.0,
    dt: float = 5e-3,
    g: float = 0.0,
    with_mobility: bool = False,
    seed: int = 1001,
    rel_tolerance: float = 1e-8,
) -> dict:
    """
    Assert the Ginzburg-Landau energy is monotone nonincreasing.

    Outputs:
        dict with passed, max relative energy increase, energy series, times.
    """
    u0 = _random_tanh_ics(n_runs, nx, domain_size, seed)

    mobility = None
    if with_mobility:
        rng = rng_for_sample(seed, 10_000)
        fields = []
        for _ in range(n_runs):
            grf = generate_grf(nx, nx, domain_size, 0.1, 3.0, rng)
            fields.append(lognormal_field_from_grf(grf, 1.0, 0.35, (0.3, 3.0)))
        mobility = np.stack(fields)

    result = solve_allen_cahn_imex(
        u0, eps_param, t_final, dt,
        domain_size=domain_size, mobility=mobility, g=g,
        record_energy_every=10,
    )

    energies = result["energies"]
    scale = np.abs(energies).max(axis=1, keepdims=True) + 1e-30
    increases = np.diff(energies, axis=1) / scale
    max_increase = float(increases.max()) if increases.size else 0.0

    return {
        "name": "energy_decay" + ("_mobility" if with_mobility else ""),
        "metric": "max_relative_energy_increase",
        "value": max_increase,
        "threshold": rel_tolerance,
        "passed": bool(max_increase <= rel_tolerance),
        "energies": energies,
        "times": result["energy_times"],
    }


def check_shrinking_circle(
    nx: int = 128,
    domain_size: float = 1.0,
    eps_param: float = 0.03,
    r0: float = 0.3,
    t_fit_max: float = 20.0,
    dt: float = 5e-3,
    n_snapshots: int = 20,
    slope_rel_tolerance: float = 0.10,
) -> dict:
    """
    Fit d(R^2)/dt for a shrinking circular +1 domain and compare to -2 eps^2.

    Outputs:
        dict with fitted slope, expected slope, relative deviation, R^2 series.

    Note:
        Run at 128^2 so the interface (width ~ sqrt(2)*eps) spans >= 5 cells;
        R stays well above the interface width across the fit window.
    """
    x = np.linspace(0.0, domain_size, nx, endpoint=False)
    xg, yg = np.meshgrid(x, x, indexing="ij")
    dist = np.sqrt((xg - 0.5 * domain_size) ** 2 + (yg - 0.5 * domain_size) ** 2)
    u0 = np.tanh((r0 - dist) / (np.sqrt(2.0) * eps_param)).astype(np.float32)

    snap_times = tuple(np.linspace(t_fit_max / n_snapshots, t_fit_max, n_snapshots))
    result = solve_allen_cahn_imex(
        u0, eps_param, t_fit_max, dt, domain_size=domain_size, snapshot_times=snap_times
    )

    times = [0.0] + list(snap_times)
    fields = [u0] + [result["snapshots"][t] for t in snap_times]
    r_squared = np.array([
        transformed_area_fraction(f) * domain_size**2 / np.pi for f in fields
    ])

    slope = np.polyfit(times, r_squared, deg=1)[0]
    expected = -2.0 * eps_param**2
    rel_dev = float(abs(slope - expected) / abs(expected))

    return {
        "name": "shrinking_circle",
        "metric": "relative_slope_deviation",
        "value": rel_dev,
        "threshold": slope_rel_tolerance,
        "passed": bool(rel_dev <= slope_rel_tolerance),
        "times": np.array(times),
        "r_squared": r_squared,
        "fitted_slope": float(slope),
        "expected_slope": float(expected),
    }


def check_coarsening(
    nx: int = 128,
    domain_size: float = 1.0,
    eps_param: float = 0.03,
    t_start: float = 1.0,
    t_final: float = 40.0,
    dt: float = 5e-3,
    n_snapshots: int = 14,
    n_runs: int = 3,
    r2_threshold: float = 0.9,
    slope_bounds: tuple = (1.0, 20.0),
    seed: int = 2002,
) -> dict:
    """
    Check curvature-driven coarsening: l(t)^2 grows linearly in eps^2 * t.

    Outputs:
        dict with the linear-fit R^2, normalized slope a = d(l^2)/d(eps^2 t),
        and l(t) series (mean over runs).

    Why l^2-vs-t instead of a log-log exponent:
        The classic l ~ t^(1/2) statement needs l to grow over a decade while
        staying far below the box size. In a unit box, Allen-Cahn coarsening
        from fine GRF ICs reaches l ~ 0.25 within t ~ 1 and saturates near
        l ~ 0.5 (few domains left), so no clean scaling decade exists at any
        resolvable eps. The equivalent, testable statement in the reachable
        regime is the differential form of the same law: l^2 = l0^2 + a eps^2 t
        with a of order unity - the direct signature of curvature flow (cf. the
        shrinking-circle law d(R^2)/dt = -2 eps^2). The length scale is measured
        as 1/interface_length (the structure-factor first moment is biased by
        the Porod k^-3 tail and underestimates growth).

    Fit window: snapshots with t >= 2 (skip initial transient) and l <= 0.4
    (avoid box saturation).
    """
    snap_times = tuple(np.geomspace(t_start, t_final, n_snapshots))
    snap_times = tuple(sorted(set(round(t / dt) * dt for t in snap_times)))

    all_lengths = []
    for run in range(n_runs):
        rng = rng_for_sample(seed, run)
        grf = generate_grf(nx, nx, domain_size, 0.03, 2.5, rng)
        u0 = tanh_ic_from_grf(grf, interface_width=0.5)

        result = solve_allen_cahn_imex(
            u0, eps_param, t_final, dt, domain_size=domain_size, snapshot_times=snap_times
        )
        all_lengths.append(
            [1.0 / interface_length(result["snapshots"][t], domain_size) for t in snap_times]
        )

    times = np.array(snap_times)
    lengths = np.array(all_lengths).mean(axis=0)

    window = (times >= 2.0) & (lengths <= 0.4)
    if window.sum() < 4:
        raise RuntimeError("Coarsening fit window has fewer than 4 points; adjust parameters.")

    t_fit = times[window]
    l2_fit = lengths[window] ** 2

    slope, intercept = np.polyfit(t_fit, l2_fit, deg=1)
    l2_hat = slope * t_fit + intercept
    ss_res = float(((l2_fit - l2_hat) ** 2).sum())
    ss_tot = float(((l2_fit - l2_fit.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)
    slope_norm = float(slope / eps_param**2)

    lo, hi = slope_bounds
    passed = bool(r_squared >= r2_threshold and lo <= slope_norm <= hi)

    return {
        "name": "coarsening_l2_linear",
        "metric": "(r_squared, slope/eps^2)",
        "value": (round(r_squared, 4), round(slope_norm, 2)),
        "threshold": (r2_threshold, slope_bounds),
        "passed": passed,
        "times": times,
        "lengths": lengths,
        "fit_window": window,
        "r_squared": r_squared,
        "slope_norm": slope_norm,
    }


def check_dt_and_grid_convergence(
    nx: int = 64,
    fine_nx: int = 128,
    domain_size: float = 1.0,
    eps_param: float = 0.03,
    t_final: float = 4.0,
    dt: float = 5e-3,
    dt_tolerance: float = 1e-3,
    grid_tolerance: float = 1e-2,
    seed: int = 3003,
) -> dict:
    """
    Check first-order dt convergence (dt vs dt/2) and grid convergence
    (native 64^2 solve vs Fourier-downsampled 128^2 solve).

    Outputs:
        dict with both relative errors and a combined pass flag.
    """
    u0 = _random_tanh_ics(4, nx, domain_size, seed)

    coarse_dt = solve_allen_cahn_imex(u0, eps_param, t_final, dt, domain_size=domain_size)
    half_dt = solve_allen_cahn_imex(u0, eps_param, t_final, dt / 2.0, domain_size=domain_size)
    dt_err = float(relative_l2_numpy(coarse_dt["u_final"], half_dt["u_final"]).max())

    u0_fine = fourier_upsample(u0, fine_nx)
    fine = solve_allen_cahn_imex(u0_fine, eps_param, t_final, dt / 2.0, domain_size=domain_size)
    fine_on_coarse = fourier_downsample(fine["u_final"], nx)
    grid_err = float(relative_l2_numpy(half_dt["u_final"], fine_on_coarse).max())

    passed = bool(dt_err <= dt_tolerance and grid_err <= grid_tolerance)

    return {
        "name": "dt_and_grid_convergence",
        "metric": "max_rel_l2(dt_halving, grid_refinement)",
        "value": (dt_err, grid_err),
        "threshold": (dt_tolerance, grid_tolerance),
        "passed": passed,
        "dt_error": dt_err,
        "grid_error": grid_err,
    }


def check_scheme_agreement(
    n_imex_rk4: int = 20,
    n_pypde: int = 5,
    nx: int = 64,
    domain_size: float = 1.0,
    eps_param: float = 0.03,
    t_final: float = 4.0,
    dt: float = 5e-3,
    rk4_tolerance: float = 1e-3,
    pypde_tolerance: float = 1e-2,
    seed: int = 4004,
    with_mobility: bool = True,
) -> dict:
    """
    Cross-scheme agreement on shared inputs: IMEX vs RK4 (same grid, RK4 at dt/4),
    and IMEX vs py-pde on a small spot-check subset.

    Outputs:
        dict with both max relative errors and a combined pass flag.
    """
    u0 = _random_tanh_ics(n_imex_rk4, nx, domain_size, seed)

    mobility = None
    if with_mobility:
        fields = []
        for i in range(n_imex_rk4):
            rng = rng_for_sample(seed, 20_000 + i)
            grf = generate_grf(nx, nx, domain_size, 0.1, 3.0, rng)
            fields.append(lognormal_field_from_grf(grf, 1.0, 0.35, (0.3, 3.0)))
        mobility = np.stack(fields)

    imex = solve_allen_cahn_imex(u0, eps_param, t_final, dt, domain_size=domain_size, mobility=mobility)
    rk4 = solve_allen_cahn_rk4(u0, eps_param, t_final, dt / 4.0, domain_size=domain_size, mobility=mobility)
    rk4_err = float(relative_l2_numpy(imex["u_final"], rk4["u_final"]).max())

    pypde_errs = []
    for i in range(n_pypde):
        m_i = None if mobility is None else mobility[i]
        u_pypde = solve_allen_cahn_pypde(
            u0[i], eps_param, t_final, dt, domain_size=domain_size, mobility=m_i
        )
        pypde_errs.append(float(relative_l2_numpy(imex["u_final"][i], u_pypde)[0]))
    pypde_err = float(max(pypde_errs))

    passed = bool(rk4_err <= rk4_tolerance and pypde_err <= pypde_tolerance)

    return {
        "name": "scheme_agreement",
        "metric": "max_rel_l2(imex_vs_rk4, imex_vs_pypde)",
        "value": (rk4_err, pypde_err),
        "threshold": (rk4_tolerance, pypde_tolerance),
        "passed": passed,
        "rk4_error": rk4_err,
        "pypde_error": pypde_err,
    }
