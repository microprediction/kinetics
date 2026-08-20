"""Locks in experiment 39: the Green-Kubo correction predicts the runner-up kernel."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                      / "exp39_runner_up_bridge"))
import run_runner_up_bridge as rb  # noqa: E402


def _setup():
    L, pi, lam = rb.build_environment()
    lam_bar, K = rb.green_kubo(L, pi, lam)
    return L, pi, lam, lam_bar, K


def test_runner_up_identity_is_exact():
    """q_j^(-i) = p_j + p_i M_ij holds for every eps, not asymptotically."""
    L, pi, lam, _, _ = _setup()
    A = list(range(len(lam)))
    for eps in (0.5, 0.05):
        p, M, q = rb.exact_runner_up(L, pi, lam, A, eps)
        for i in A:
            for j in A:
                if i != j:
                    assert abs(q[i, j] - (p[j] + p[i] * M[i, j])) < 1e-12


def test_runner_up_kernel_is_stochastic():
    L, pi, lam, _, _ = _setup()
    A = list(range(len(lam)))
    _, M, _ = rb.exact_runner_up(L, pi, lam, A, 0.1)
    assert (M >= -1e-14).all()
    assert np.allclose(M.sum(axis=1), 1.0, atol=1e-12)


def test_green_kubo_gains_an_order_on_the_kernel():
    """Luce predicts M to O(eps); the correction predicts it to O(eps^2)."""
    L, pi, lam, lam_bar, K = _setup()
    n = len(lam)
    A = list(range(n))
    off = ~np.eye(n, dtype=bool)
    Lbar = lam_bar.sum()
    M_luce = np.array([[lam_bar[j] / (Lbar - lam_bar[i]) if i != j else 0.0
                        for j in range(n)] for i in range(n)])
    eps_grid = np.array([0.08, 0.02, 0.005])
    e_luce, e_gk = [], []
    for eps in eps_grid:
        _, M, _ = rb.exact_runner_up(L, pi, lam, A, eps)
        p_th = rb.theory_shares(lam_bar, K, A, eps)
        M_gk = np.zeros((n, n))
        for i in A:
            rest = [a for a in A if a != i]
            q_th = rb.theory_shares(lam_bar, K, rest, eps)
            for pos, j in enumerate(rest):
                M_gk[i, j] = (q_th[pos] - p_th[j]) / p_th[i]
        e_luce.append(np.abs(M - M_luce)[off].max())
        e_gk.append(np.abs(M - M_gk)[off].max())
    s_luce = np.polyfit(np.log(eps_grid), np.log(e_luce), 1)[0]
    s_gk = np.polyfit(np.log(eps_grid), np.log(e_gk), 1)[0]
    assert 0.8 < s_luce < 1.2, s_luce
    assert 1.8 < s_gk < 2.2, s_gk
    assert e_gk[-1] < e_luce[-1] / 50


def test_leading_order_kernel_is_luce():
    """At leading order the runner-up is drawn from the race among the rest."""
    L, pi, lam, lam_bar, _ = _setup()
    n = len(lam)
    A = list(range(n))
    _, M, _ = rb.exact_runner_up(L, pi, lam, A, 1e-4)
    Lbar = lam_bar.sum()
    for i in A:
        for j in A:
            if i != j:
                assert abs(M[i, j] - lam_bar[j] / (Lbar - lam_bar[i])) < 1e-5
