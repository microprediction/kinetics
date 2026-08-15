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


def test_jvp_stable_on_referee_tail_case():
    """Third-review regression: N=8, heterogeneous D, GH(11) — the published
    density-derivative JVP returned NaN/inf here; the log-domain
    integration-by-parts form must stay finite and match finite differences."""
    from raceutil import jacobian_vector_product
    n = 8
    mu = np.linspace(-2.0, 2.0, n)
    D = np.logspace(-1, 0, n)
    V = np.random.default_rng(123).standard_normal((n, 2)) * 0.5
    F, W = hermite_nodes(2, Q=11)
    h = np.eye(n)[0] - 1.0 / n
    jv = jacobian_vector_product(mu, V, D, F, W, h)
    fd = (win_probabilities_factor(mu + 1e-5 * h, V, D, F, W)
          - win_probabilities_factor(mu - 1e-5 * h, V, D, F, W)) / 2e-5
    assert np.all(np.isfinite(jv))
    assert np.abs(jv - fd).max() < 1e-6


def test_forward_deep_tail_shares_positive():
    """log_ndtr survival: a runner 12 sd behind gets a genuine tiny positive
    share, not a floored zero, and parity with the independent transform holds."""
    mu = np.array([-6.0, 0.0, 6.0])
    p = win_probabilities_factor(mu, np.zeros((3, 1)), np.ones(3),
                                 np.zeros((1, 1)), np.ones(1))
    assert np.abs(p - win_probabilities(mu, 1.0)).max() < 1e-12
    assert p[2] > 0


def test_contrast_factor_fit_ignores_common_shock():
    """Third review: Sigma = tau^2 11^T + b b^T + D. Raw rank-1 fit spends its
    factor on the choice-irrelevant common component; the contrast-space fit
    captures b b^T. Compared in the quotient norm ||P (Sigma - fit) P||_F."""
    from raceutil import factor_model_contrast
    rng = np.random.default_rng(5)
    n = 12
    b = rng.normal(0, 1, n)
    Sig = 9.0 * np.ones((n, n)) + np.outer(b, b) + np.diag(rng.uniform(0.5, 1, n))
    P = np.eye(n) - np.ones((n, n)) / n
    V_raw, D_raw = factor_model(Sig, 1)
    V_con, D_con = factor_model_contrast(Sig, 1)
    err_raw = np.linalg.norm(P @ (Sig - V_raw @ V_raw.T - np.diag(D_raw)) @ P)
    err_con = np.linalg.norm(P @ (Sig - V_con @ V_con.T - np.diag(D_con)) @ P)
    assert err_con < 0.2 * err_raw
    assert np.abs(V_con.sum(axis=0)).max() < 1e-8     # centered loadings


def test_common_factor_shock_cannot_move_shares():
    """V -> V + 1 c^T leaves win probabilities unchanged (quotient-space fact
    behind the contrast fit)."""
    rng = np.random.default_rng(11)
    n = 7
    mu = rng.normal(0, 0.5, n)
    V = 0.4 * rng.standard_normal((n, 2))
    D = rng.uniform(0.5, 1.0, n)
    F, W = hermite_nodes(2)
    p0 = win_probabilities_factor(mu, V, D, F, W)
    p1 = win_probabilities_factor(mu, V + np.ones((n, 1)) * np.array([[0.9, -0.4]]),
                                  D, F, W)
    assert np.abs(p0 - p1).max() < 1e-12


def test_jacobian_symmetry_and_positivity():
    """Fourth review: <h, Jk> = <Jh, k> and h'Jh has the Laplacian sign for
    nonconstant h (min-wins J = -weighted Laplacian, so h'Jh < 0)."""
    from raceutil import jacobian_vector_product
    rng = np.random.default_rng(17)
    n = 9
    mu = rng.normal(0, 0.7, n)
    V = 0.5 * rng.standard_normal((n, 2))
    D = rng.uniform(0.4, 1.2, n)
    F, W = hermite_nodes(2)
    h = rng.normal(0, 1, n); h -= h.mean()
    k = rng.normal(0, 1, n); k -= k.mean()
    Jh = jacobian_vector_product(mu, V, D, F, W, h)
    Jk = jacobian_vector_product(mu, V, D, F, W, k)
    assert abs(h @ Jk - k @ Jh) < 1e-10          # symmetry
    assert h @ Jh < 0                             # -Laplacian quadratic form
    assert abs((h + 1.0) @ Jh - (h @ Jh + Jh.sum())) < 1e-12  # invariance sanity


def test_jvp_matches_normalized_map_both_conventions():
    """Corollaries 4b/4c: the JVP agrees with finite differences of the
    RETURNED normalized map in min-wins, and with the max-wins map under
    reflection (sign flip)."""
    from raceutil import jacobian_vector_product
    rng = np.random.default_rng(23)
    n = 8
    mu = rng.normal(0, 0.8, n)
    V = 0.4 * rng.standard_normal((n, 2))
    D = rng.uniform(0.5, 1.3, n)
    F, W = hermite_nodes(2)
    h = rng.normal(0, 1, n); h -= h.mean()
    eps = 1e-5
    jv = jacobian_vector_product(mu, V, D, F, W, h)
    fd_min = (win_probabilities_factor(mu + eps * h, V, D, F, W)
              - win_probabilities_factor(mu - eps * h, V, D, F, W)) / (2 * eps)
    assert np.abs(jv - fd_min).max() < 1e-6
    # max-wins: p_max(m) = p_min(-m); d/dm p_max[h] = -J_min(-m)[h]
    a = -mu
    fd_max = (win_probabilities_factor(-(mu + eps * h), V, D, F, W)
              - win_probabilities_factor(-(mu - eps * h), V, D, F, W)) / (2 * eps)
    jv_max = -jacobian_vector_product(a, V, D, F, W, h)
    assert np.abs(jv_max - fd_max).max() < 1e-6


def test_grid_jvp_exact_at_coarse_lattice_where_ibp_degrades():
    """Fifth review: the IBP/Laplacian JVP is the continuum derivative, not
    the derivative of the finite rectangle sum. The form="grid" variant IS
    the derivative of the frozen-grid map: it must match finite differences
    even on a coarse lattice where the IBP form visibly degrades."""
    from raceutil import jacobian_vector_product
    rng = np.random.default_rng(7)
    n = 9
    mu = rng.normal(0, 0.7, n)
    V = 0.5 * rng.standard_normal((n, 2))
    D = rng.uniform(0.4, 1.1, n)
    F, W = hermite_nodes(2)
    h = rng.normal(0, 1, n); h -= h.mean()
    eps = 1e-6
    L = 51
    # frozen-grid FD is approximated well by the adaptive map's FD here
    # because the envelope shift is O(eps); compare unnormalized forms via
    # the library map (normalized): both JVPs are of the unnormalized map,
    # and 1^T v = 0 keeps normalization corrections at quadrature level.
    fd = (win_probabilities_factor(mu + eps * h, V, D, F, W, points=L)
          - win_probabilities_factor(mu - eps * h, V, D, F, W, points=L)) / (2 * eps)
    jg = jacobian_vector_product(mu, V, D, F, W, h, points=L, form="grid")
    ji = jacobian_vector_product(mu, V, D, F, W, h, points=L, form="ibp")
    # residual vs the LIBRARY map includes normalization + envelope motion,
    # so "exact" here means: several times closer than the IBP form
    assert np.abs(jg - fd).max() < 1e-5
    assert np.abs(ji - fd).max() > 2 * np.abs(jg - fd).max()
    # and at production resolution the two coincide
    jg2 = jacobian_vector_product(mu, V, D, F, W, h, form="grid")
    ji2 = jacobian_vector_product(mu, V, D, F, W, h, form="ibp")
    assert np.abs(jg2 - ji2).max() < 1e-12


def test_projected_fit_certifies_heuristic():
    """Eighth review: the certified projected quotient fit changes the
    quotient residual by <1% vs the contrast heuristic on a boundary-style
    matrix, so the heuristic is not the binding constraint."""
    from raceutil import factor_model_contrast, factor_model_projected
    rng = np.random.default_rng(33)
    n = 30
    lam = np.arange(1, n + 1, dtype=float) ** -1.5
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    C = (Q * lam) @ Q.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    P = np.eye(n) - np.ones((n, n)) / n
    nrm = np.linalg.norm(P @ C @ P)
    rh = np.linalg.norm(P @ (C - (lambda VD: VD[0] @ VD[0].T + np.diag(VD[1]))(
        factor_model_contrast(C, 4))) @ P) / nrm
    rp = np.linalg.norm(P @ (C - (lambda VD: VD[0] @ VD[0].T + np.diag(VD[1]))(
        factor_model_projected(C, 4))) @ P) / nrm
    assert rp <= rh * 1.01
    assert abs(rp - rh) < 0.02
