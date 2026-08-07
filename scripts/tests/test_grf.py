"""GRF and IC-builder statistics tests."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fnocausal.common.grf import (
    generate_grf,
    lognormal_field_from_grf,
    nucleation_ic,
    nuisance_pattern,
    tanh_ic_from_grf,
)
from fnocausal.common.seeding import rng_for_sample


def test_grf_standardized():
    rng = np.random.default_rng(0)
    field = generate_grf(64, 64, 1.0, 0.1, 3.0, rng)
    assert field.shape == (64, 64)
    assert field.dtype == np.float32
    assert abs(field.mean()) < 1e-5
    assert abs(field.std() - 1.0) < 1e-3


def test_grf_reproducible_from_sample_rng():
    a = generate_grf(64, 64, 1.0, 0.1, 3.0, rng_for_sample(42, 7))
    b = generate_grf(64, 64, 1.0, 0.1, 3.0, rng_for_sample(42, 7))
    c = generate_grf(64, 64, 1.0, 0.1, 3.0, rng_for_sample(42, 8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_grf_correlation_length_orders_smoothness():
    rng_short = np.random.default_rng(1)
    rng_long = np.random.default_rng(1)
    short = generate_grf(64, 64, 1.0, 0.03, 3.0, rng_short)
    long = generate_grf(64, 64, 1.0, 0.25, 3.0, rng_long)
    # Longer correlation length -> smaller mean squared gradient.
    grad_short = np.mean(np.diff(short, axis=0) ** 2)
    grad_long = np.mean(np.diff(long, axis=0) ** 2)
    assert grad_long < grad_short


def test_lognormal_field_positive_and_clipped():
    rng = np.random.default_rng(2)
    grf = generate_grf(64, 64, 1.0, 0.1, 3.0, rng)
    field = lognormal_field_from_grf(grf, 1.0, 0.35, (0.3, 3.0))
    assert field.min() >= 0.3
    assert field.max() <= 3.0
    assert 0.5 < field.mean() < 1.5


def test_tanh_ic_bounded():
    rng = np.random.default_rng(3)
    grf = generate_grf(64, 64, 1.0, 0.1, 3.0, rng)
    u0 = tanh_ic_from_grf(grf, 0.3)
    # float32 rounding can make tanh saturate to exactly 1.0.
    assert np.all(np.abs(u0) <= 1.0)


def test_nucleation_ic_two_phase():
    rng = rng_for_sample(0, 0)
    u0 = nucleation_ic(64, 64, 1.0, n_seeds=4, radius_range=(0.05, 0.1),
                       interface_eps=0.03, rng=rng)
    assert u0.min() >= -1.0
    assert u0.max() <= 1.0
    frac = (u0 > 0).mean()
    assert 0.0 < frac < 0.5


def test_nuisance_pattern_deterministic_standardized():
    a = nuisance_pattern(64, 64)
    b = nuisance_pattern(64, 64)
    assert np.array_equal(a, b)
    assert abs(a.mean()) < 1e-6
    assert abs(a.std() - 1.0) < 1e-5
