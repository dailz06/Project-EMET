"""Fast solver sanity tests (full verification lives in run_phase2_verify_solver.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fnocausal.common.grf import generate_grf, tanh_ic_from_grf
from fnocausal.common.metrics import ginzburg_landau_energy, relative_l2_numpy
from fnocausal.common.seeding import rng_for_sample
from fnocausal.sim.allen_cahn_rk4 import solve_allen_cahn_rk4
from fnocausal.sim.allen_cahn_spectral import solve_allen_cahn_imex
from fnocausal.sim.downsample import fourier_downsample, fourier_upsample


EPS = 0.03
DT = 5e-3


def _ic(n=2, nx=64, seed=0):
    ics = []
    for i in range(n):
        rng = rng_for_sample(seed, i)
        grf = generate_grf(nx, nx, 1.0, 0.1, 3.0, rng)
        ics.append(tanh_ic_from_grf(grf, 0.3))
    return np.stack(ics)


def test_imex_preserves_shape_and_wells():
    u0 = _ic()
    result = solve_allen_cahn_imex(u0, EPS, 0.5, DT)
    u = result["u_final"]
    assert u.shape == u0.shape
    # Reaction term pushes values toward the +/-1 wells; slight overshoot allowed.
    assert np.all(np.abs(u) < 1.2)


def test_imex_energy_decreases_short_run():
    u0 = _ic(n=1)
    result = solve_allen_cahn_imex(u0, EPS, 1.0, DT, record_energy_every=20)
    energies = result["energies"][0]
    assert np.all(np.diff(energies) <= 1e-10 * np.abs(energies[:-1]).max())


def test_imex_snapshots_align():
    u0 = _ic(n=1)
    result = solve_allen_cahn_imex(u0, EPS, 1.0, DT, snapshot_times=(0.5, 1.0))
    assert set(result["snapshots"].keys()) == {0.5, 1.0}
    assert np.array_equal(result["snapshots"][1.0], result["u_final"])


def test_imex_rk4_agree_short_run():
    u0 = _ic(n=2)
    imex = solve_allen_cahn_imex(u0, EPS, 0.5, DT)
    rk4 = solve_allen_cahn_rk4(u0, EPS, 0.5, DT / 4.0)
    err = relative_l2_numpy(imex["u_final"], rk4["u_final"]).max()
    assert err < 1e-3


def test_rk4_raises_on_unstable_dt():
    u0 = _ic(n=1, nx=128)
    with pytest.raises(ValueError, match="RK4 unstable"):
        solve_allen_cahn_rk4(u0, 0.05, 1.0, 0.1)


def test_fourier_resample_round_trip():
    u = _ic(n=1)[0]
    up = fourier_upsample(u, 128)
    down = fourier_downsample(up, 64)
    assert down.shape == (64, 64)
    assert relative_l2_numpy(u, down)[0] < 1e-6


def test_solver_deterministic():
    u0 = _ic(n=1)
    a = solve_allen_cahn_imex(u0, EPS, 0.5, DT)["u_final"]
    b = solve_allen_cahn_imex(u0, EPS, 0.5, DT)["u_final"]
    assert np.array_equal(a, b)


def test_torch_backends_match_numpy():
    u0 = _ic(n=2)
    m = np.full_like(u0, 1.3)

    imex_np = solve_allen_cahn_imex(u0, EPS, 0.5, DT, mobility=m, g=0.1)
    imex_th = solve_allen_cahn_imex(u0, EPS, 0.5, DT, mobility=m, g=0.1, backend="torch")
    assert relative_l2_numpy(imex_np["u_final"], imex_th["u_final"]).max() < 1e-4

    rk4_np = solve_allen_cahn_rk4(u0, EPS, 0.5, DT / 4.0, mobility=m, g=0.1)
    rk4_th = solve_allen_cahn_rk4(u0, EPS, 0.5, DT / 4.0, mobility=m, g=0.1, backend="torch")
    assert relative_l2_numpy(rk4_np["u_final"], rk4_th["u_final"]).max() < 1e-4


def test_mobility_scales_dynamics():
    # Doubling a constant mobility must equal doubling the horizon.
    u0 = _ic(n=1)
    m2 = np.full_like(u0, 2.0)
    fast = solve_allen_cahn_imex(u0, EPS, 1.0, DT, mobility=m2)["u_final"]
    slow_long = solve_allen_cahn_imex(u0, EPS, 2.0, DT)["u_final"]
    # Not exact (different dt-in-effective-time), so compare loosely.
    assert relative_l2_numpy(fast, slow_long)[0] < 5e-3
