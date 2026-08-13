"""Tests for experiments/raceutil.py: independent and factor-correlated transforms."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from raceutil import (  # noqa: E402
    abilities_from_probabilities,
    abilities_from_probabilities_factor,
    factor_model,
    hermite_nodes,
    qmc_nodes,
    win_probabilities,
    win_probabilities_factor,
)

RNG = np.random.default_rng(3)


def mc_win_probs(mu, C, scale, n, seed=9):
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(mu))
    for _ in range(n // 100_000):
        X = np.asarray(mu)[:, None] + scale * (L @ rng.standard_normal((len(mu), 100_000)))
        counts += np.bincount(np.argmin(X, axis=0), minlength=len(mu))
    return counts / counts.sum()


# ---- independent transform --------------------------------------------------


def test_independent_forward_matches_monte_carlo():
    mu = RNG.normal(0.0, 1.0, 12)
    p = win_probabilities(mu, 1.0)
    ref = mc_win_probs(mu, np.eye(12), 1.0, 2_000_000)
    assert np.abs(p - ref).max() < 3e-3


def test_independent_forward_sums_to_one_and_orders_correctly():
    mu = np.array([-1.0, 0.0, 1.0])
    p = win_probabilities(mu, 0.7)
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[0] > p[1] > p[2]  # min wins: lower ability stronger


def test_independent_inverse_roundtrip_skewed_targets():
    rates = RNG.lognormal(0.0, 1.5, 30)
    target = rates / rates.sum()
    mu = abilities_from_probabilities(target, 1.0)
    back = win_probabilities(mu, 1.0)
    assert np.abs(back - target).max() < 1e-6


def test_independent_inverse_rejects_zero_probabilities():
    with pytest.raises(ValueError):
        abilities_from_probabilities(np.array([0.5, 0.5, 0.0]))


# ---- factor model -------------------------------------------------------------


def test_factor_model_identity_invents_no_correlation():
    V, D = factor_model(np.eye(10), 2)
    off = V @ V.T + np.diag(D) - np.eye(10)
    assert np.abs(off).max() < 1e-6


def test_factor_model_equicorrelated_exact_at_k1():
    C = 0.6 * np.ones((8, 8)) + 0.4 * np.eye(8)
    V, D = factor_model(C, 1)
    assert np.abs(V @ V.T + np.diag(D) - C).max() < 1e-8


def test_factor_model_diagonal_always_exact():
    th = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    C = np.exp(-np.abs((th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi))
    V, D = factor_model(C, 3)
    assert np.abs(np.diag(V @ V.T + np.diag(D)) - 1.0).max() < 1e-9
    assert np.all(D > 0)


# ---- quadrature nodes ----------------------------------------------------------


def test_hermite_nodes_moments():
    for k in (1, 2, 3):
        F, W = hermite_nodes(k)
        assert abs(W.sum() - 1.0) < 5e-6
        assert np.abs(W @ F).max() < 5e-6
        assert np.abs(W @ (F**2) - 1.0).max() < 5e-6


def test_qmc_nodes_deterministic_and_normalized():
    F1, W1 = qmc_nodes(6, m=8, seed=4)
    F2, W2 = qmc_nodes(6, m=8, seed=4)
    assert np.array_equal(F1, F2)
    assert abs(W1.sum() - 1.0) < 1e-12
    assert F1.shape == (256, 6)


# ---- correlated transform -------------------------------------------------------


def test_factor_transform_identity_matches_independent():
    mu = RNG.normal(0.0, 0.8, 10)
    V, D = factor_model(np.eye(10), 2)
    F, W = hermite_nodes(2)
    p = win_probabilities_factor(mu, V, D, F, W)
    assert np.abs(p - win_probabilities(mu, 1.0)).max() < 1e-6


def test_factor_transform_known_model_matches_monte_carlo():
    n, k = 8, 2
    V = 0.6 * RNG.standard_normal((n, k))
    D = RNG.uniform(0.3, 0.9, n)
    mu = RNG.normal(0.0, 0.6, n)
    F, W = hermite_nodes(k)
    p = win_probabilities_factor(mu, V, D, F, W)
    ref = mc_win_probs(mu, V @ V.T + np.diag(D), 1.0, 2_000_000)
    assert np.abs(p - ref).max() < 3e-3


def test_factor_transform_deletion_ensemble_consistency():
    n = 7
    C = 0.5 * np.ones((n, n)) + 0.5 * np.eye(n)
    V, D = factor_model(C, 1)
    mu = RNG.normal(0.0, 0.5, n)
    F, W = hermite_nodes(1)
    p, q = win_probabilities_factor(mu, V, D, F, W, return_deletions=True)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.abs(np.diag(q)).max() == 0.0
    keep = np.setdiff1d(np.arange(n), [2])
    direct = win_probabilities_factor(mu, V, D, F, W, keep=keep)
    assert np.abs(direct - q[2][keep]).max() < 1e-10


def test_factor_inverse_roundtrip():
    n = 8
    C = 0.4 * np.ones((n, n)) + 0.6 * np.eye(n)
    V, D = factor_model(C, 1)
    F, W = hermite_nodes(1)
    mu_true = RNG.normal(0.0, 0.4, n)
    mu_true -= mu_true.mean()
    target = win_probabilities_factor(mu_true, V, D, F, W)
    mu_fit = abilities_from_probabilities_factor(target, V, D, F, W)
    back = win_probabilities_factor(mu_fit, V, D, F, W)
    assert np.abs(back - target).max() < 2e-3


def test_jacobian_vector_product_matches_finite_differences():
    """Referee-derived O(QNL) JVP formula (paper Prop. 4)."""
    from raceutil import jacobian_vector_product
    n = 10
    mu = RNG.normal(0, 0.6, n)
    V = RNG.normal(0, 0.4, (n, 2))
    D = RNG.uniform(0.6, 1.2, n)
    F, W = hermite_nodes(2)
    h = RNG.normal(0, 1, n); h -= h.mean()
    eps = 1e-5
    fd = (win_probabilities_factor(mu + eps * h, V, D, F, W)
          - win_probabilities_factor(mu - eps * h, V, D, F, W)) / (2 * eps)
    an = jacobian_vector_product(mu, V, D, F, W, h)
    assert np.abs(an - fd).max() < 1e-6
    assert abs(an.sum()) < 1e-12          # translation invariance: J^T 1 = 0 row sums
