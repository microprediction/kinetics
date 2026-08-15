"""Parity tests for the fastrace Rust kernel against the raceutil reference.

Skipped cleanly when the extension is not built (maturin develop --release
in rust/fastrace)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
fastrace = pytest.importorskip("fastrace")
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402
from scipy.special import log_ndtr  # noqa: E402

RNG = np.random.default_rng(29)


def _problem(n, k=2):
    mu = RNG.normal(0, 1.2, n)
    V = RNG.normal(0, 0.5 / np.sqrt(k), (n, k))
    D = RNG.uniform(0.4, 1.4, n)
    F, W = hermite_nodes(k)
    return mu, V, D, np.ascontiguousarray(F), np.ascontiguousarray(W)


def test_forward_parity_and_total():
    mu, V, D, F, W = _problem(60)
    p_rs, tot = fastrace.win_probabilities_factor(mu, V, D, F, W)
    p_py, tot_py = win_probabilities_factor(mu, V, D, F, W, return_total=True)
    assert np.abs(p_rs - p_py).max() < 1e-14
    assert abs(tot - tot_py) < 1e-12


def test_slope_parity():
    mu, V, D, F, W = _problem(40)
    _, sl_rs, _ = fastrace.forward_and_slopes(mu, V, D, F, W)
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T
    lo = M_all.min() - 8 * sd.max()
    hi = M_all.max() + 8 * sd.max()
    x = np.linspace(lo, hi, 1501)
    dx = x[1] - x[0]
    slope_py = np.zeros(len(mu))
    for c in range(len(F)):
        z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
        logS = log_ndtr(-z)
        f = np.exp(-0.5 * z**2) / (sd[:, None] * np.sqrt(2 * np.pi))
        rest = np.exp(np.clip(logS.sum(0)[None, :] - logS, -745, 0))
        slope_py += W[c] * np.sum(z * f / sd[:, None] * rest, axis=1) * dx
    assert np.abs(sl_rs - slope_py).max() < 1e-12


def test_rust_calibration_drop_in():
    from raceutil import abilities_from_probabilities_factor
    from rustcal import calibrate_rust
    mu, V, D, F, W = _problem(80)
    mu -= mu.mean()
    target = win_probabilities_factor(mu, V, D, F, W)
    mu_py, ip = abilities_from_probabilities_factor(target, V, D, F, W,
                                                    return_info=True)
    mu_rs, ir = calibrate_rust(target, V, D, F, W, return_info=True)
    assert ip["iterations"] == ir["iterations"]
    res = target > 1e-3
    assert np.abs(mu_py - mu_rs)[res].max() < 1e-10


def test_deep_tail_stability():
    mu = np.array([-6.0, 0.0, 6.0])
    V = np.zeros((3, 1)); D = np.ones(3)
    F = np.zeros((1, 1)); W = np.ones(1)
    p, tot = fastrace.win_probabilities_factor(mu, V, D, F, W)
    assert np.all(np.isfinite(p)) and p[2] > 0


def test_separated_kernel_converges_exponentially():
    mu, V, D, F, W = _problem(120)
    p_exact = win_probabilities_factor(mu, V, D, F, W)
    errs = []
    for rm, rs in ((16, 8), (32, 12), (48, 14)):
        p_sep, _ = fastrace.win_probabilities_factor_separated(
            mu, V, D, F, W, rm=rm, rs=rs)
        errs.append(np.abs(p_sep - p_exact).max())
    assert errs[1] < errs[0] / 10
    assert errs[2] < 1e-6
