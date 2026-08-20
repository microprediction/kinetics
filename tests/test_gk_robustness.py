"""Locks in experiment 40: replication, the dynamical-correlation ablation,
the rank bounds, and non-reversible index placement."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                      / "exp40_gk_robustness"))
import run_robustness as rb  # noqa: E402


def test_order_gain_holds_across_random_environments():
    """Every environment should gain a full order, not just the committed one."""
    rng = np.random.default_rng(40)
    gains = []
    for _ in range(6):
        m = int(rng.integers(4, 10))
        n = int(rng.integers(3, 9))
        L, pi, lam = rb.environment(rng, m, n, float(rng.uniform(0.3, 1.2)))
        lam_bar, K, _ = rb.kubo(L, pi, lam)
        s0, s1, _ = rb.orders_for(L, pi, lam, K, lam_bar, list(range(n)))
        assert 0.8 < s0 < 1.2, s0
        assert 1.75 < s1 < 2.2, s1
        gains.append(s1 - s0)
    assert min(gains) > 0.85


def test_equal_time_covariance_does_not_gain_the_order():
    """The correction needs the TIME INTEGRAL, not the static covariance.

    The static ablation is handed its best possible scalar timescale and still
    fails to reach second order.
    """
    L, pi, lam = rb.environment(np.random.default_rng(11), 6, 5, 0.8)
    lam_bar, K, lam_t = rb.kubo(L, pi, lam)
    n = len(lam)
    A = list(range(n))
    C = np.array([[pi @ (lam_t[j] * lam_t[k]) for k in range(n)] for j in range(n)])
    best = min((rb.orders_for(L, pi, lam, K, lam_bar, A, Kalt=tau * C)[2][-1], tau)
               for tau in np.logspace(-2, 1, 60))
    _, order_static, _ = rb.orders_for(L, pi, lam, K, lam_bar, A,
                                       Kalt=best[1] * C)
    _, order_true, err_true = rb.orders_for(L, pi, lam, K, lam_bar, A)
    assert order_static < 1.4, order_static
    assert order_true > 1.8, order_true
    assert best[0] > 3 * err_true[-1]


def test_rank_bounds():
    """rank(K) = r for rank-r loadings, capped by the environment's dimension."""
    rng = np.random.default_rng(3)
    m, n = 6, 10
    L, pi, _ = rb.environment(rng, m, n, 0.5)
    for r in range(1, m):
        B = rng.normal(0, 0.6, (n, r))
        z = rng.normal(0, 1.0, (r, m))
        z = z - (z @ pi)[:, None]
        d = B @ z
        _, K, _ = rb.kubo(L, pi, d - d.min() + 0.5)
        sv = np.linalg.svd(K, compute_uv=False)
        assert int((sv > 1e-10 * sv.max()).sum()) == r
    # generic loadings are capped by m-1, not by N
    _, Kfull, _ = rb.kubo(L, pi, np.exp(rng.normal(0, 0.8, (n, m))))
    sv = np.linalg.svd(Kfull, compute_uv=False)
    assert int((sv > 1e-10 * sv.max()).sum()) == min(n, m - 1)


def test_transposing_K_breaks_the_correction():
    """K is asymmetric off reversibility, so the index order in the theorem
    is a real claim and not a convention."""
    L, pi, lam = rb.environment(np.random.default_rng(2026), 6, 5, 0.7)
    lam_bar, K, _ = rb.kubo(L, pi, lam)
    A = list(range(len(lam)))
    assert np.abs(K - K.T).max() / np.abs(K).max() > 1e-3
    _, right, _ = rb.orders_for(L, pi, lam, K, lam_bar, A)
    _, wrong, _ = rb.orders_for(L, pi, lam, K, lam_bar, A, Kalt=K.T)
    assert right > 1.8
    assert wrong < right - 0.2


def test_general_start_restores_second_order():
    """From a non-stationary start the stationary formula loses an order, and
    the general form recovers it; the extra term vanishes under pi."""
    L, pi, lam = rb.environment(np.random.default_rng(99), 6, 4, 0.6)
    lam_bar, K, lam_t = rb.kubo(L, pi, lam)
    n, m = len(lam), len(pi)
    A = list(range(n))
    Pi = np.outer(np.ones(m), pi)
    dev = lambda g: np.linalg.solve(Pi - L, g - pi @ g)
    Lb = lam_bar.sum()
    soft = lam_bar / Lb
    m_const = -(K.sum(axis=0) - soft * K.sum()) / Lb
    LamT = lam_t.sum(0)

    def orders(mu0):
        extra = np.array([-(mu0 @ dev(soft[i] * LamT - lam_t[i])) for i in range(n)])
        e_s, e_g = [], []
        for eps in rb.EPS_GRID:
            u = np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)
            p0 = mu0 @ u
            e_s.append(np.abs(p0 - (soft + eps * m_const)).max())
            e_g.append(np.abs(p0 - (soft + eps * (m_const + extra))).max())
        return rb.order(e_s), rb.order(e_g), np.abs(extra).max()

    s_stat, g_stat, extra_pi = orders(pi)
    assert extra_pi < 1e-14                      # vanishes in equilibrium
    assert g_stat > 1.8

    s_pt, g_pt, extra_pt = orders(np.eye(m)[0])
    assert extra_pt > 1e-3                       # genuinely present off equilibrium
    assert s_pt < 1.3, s_pt                      # stationary formula loses the order
    assert g_pt > 1.8, g_pt                      # general formula restores it
