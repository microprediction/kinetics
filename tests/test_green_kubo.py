"""Tests for the exp07 Green-Kubo machinery: theorem invariants as a fast suite."""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "exp07_green_kubo"))
from run_green_kubo import (  # noqa: E402
    exact_win_probs,
    gk_prediction,
    green_kubo_matrix,
    make_chain,
)

RNG = np.random.default_rng(11)


def small_world(seed=11, n_channels=6):
    rng = np.random.default_rng(seed)
    L, pi = make_chain(rng)
    lam = rng.lognormal(0.0, 0.8, size=(n_channels, L.shape[0]))
    return L, pi, lam


def test_invariant_law_is_invariant():
    L, pi = make_chain(RNG)
    assert np.abs(pi @ L).max() < 1e-12
    assert abs(pi.sum() - 1.0) < 1e-12


def test_win_probs_sum_to_one_at_every_eps():
    L, pi, lam = small_world()
    A = np.arange(len(lam))
    for eps in (1e-3, 0.1, 1.0, 10.0):
        p = exact_win_probs(L, pi, lam, A, eps)
        assert abs(p.sum() - 1.0) < 1e-10
        assert np.all(p > 0)


def test_green_kubo_matrix_matches_time_integrated_covariance():
    """K from the deviation-integral solve == brute-force time quadrature."""
    L, pi, lam = small_world()
    K = green_kubo_matrix(L, pi, lam)
    ltil = lam - (lam @ pi)[:, None]
    ts = np.linspace(0.0, 60.0, 6001)
    cov = np.array([
        np.einsum("jy,y,yz,kz->jk", ltil, pi, expm(L * t), ltil) for t in ts
    ])
    K_brute = np.trapezoid(cov, ts, axis=0)
    # tolerance is the trapezoid rule's, not the solve's
    assert np.abs(K - K_brute).max() < 1e-4


def test_softmax_limit_first_order():
    L, pi, lam = small_world()
    A = np.arange(len(lam))
    p0, _ = gk_prediction(pi, lam, green_kubo_matrix(L, pi, lam), A)
    errs = [np.abs(exact_win_probs(L, pi, lam, A, eps) - p0).max()
            for eps in (0.02, 0.01, 0.005)]
    assert errs[1] / errs[0] < 0.6  # roughly halves with eps
    assert errs[2] / errs[1] < 0.6


def test_green_kubo_correction_is_second_order():
    L, pi, lam = small_world()
    K = green_kubo_matrix(L, pi, lam)
    A = np.arange(len(lam))
    p0, c1 = gk_prediction(pi, lam, K, A)
    errs = [np.abs(exact_win_probs(L, pi, lam, A, eps) - p0 - eps * c1).max()
            for eps in (0.02, 0.01, 0.005)]
    assert errs[1] / errs[0] < 0.35  # roughly quarters with eps halved
    assert errs[2] / errs[1] < 0.35


def test_one_pair_corrects_all_subsets():
    L, pi, lam = small_world()
    K = green_kubo_matrix(L, pi, lam)
    eps = 0.05
    rng = np.random.default_rng(2)
    for _ in range(10):
        A = np.sort(rng.choice(len(lam), size=int(rng.integers(2, len(lam))),
                               replace=False))
        p = exact_win_probs(L, pi, lam, A, eps)
        p0, c1 = gk_prediction(pi, lam, K, A)
        assert np.abs(p - p0 - eps * c1).max() < np.abs(p - p0).max() + 1e-15


def test_common_mode_gives_luce_exactly_at_all_eps():
    """lambda_i = a_i c(y) is a common time change: Luce holds EXACTLY, not
    just asymptotically -- the proportional-hazards normal form."""
    L, pi, _ = small_world()
    rng = np.random.default_rng(4)
    a = rng.lognormal(0.0, 1.0, 5)
    c = rng.lognormal(0.0, 0.8, L.shape[0])
    lam = np.outer(a, c)
    A = np.arange(5)
    luce = a / a.sum()
    for eps in (0.01, 0.3, 3.0):
        p = exact_win_probs(L, pi, lam, A, eps)
        assert np.abs(p - luce).max() < 1e-12
    K = green_kubo_matrix(L, pi, lam)
    _, c1 = gk_prediction(pi, lam, K, A)
    assert np.abs(c1).max() < 1e-12


def test_low_rank_loadings_give_low_rank_K():
    L, pi, _ = small_world()
    rng = np.random.default_rng(6)
    for r in (1, 2):
        # positive BY CONSTRUCTION: clipping would break the low-rank structure
        B = rng.uniform(-0.4, 0.4, size=(7, r)) / r
        z = rng.uniform(-1.0, 1.0, size=(r, L.shape[0]))
        lam = 1.0 + B @ z
        assert np.all(lam > 0)
        sv = np.linalg.svd(green_kubo_matrix(L, pi, lam), compute_uv=False)
        assert int(np.sum(sv > 1e-9 * sv[0])) <= r


def test_gk_coefficients_sum_to_zero():
    """Corrections preserve normalization: sum_i c1_i = 0 on every subset."""
    L, pi, lam = small_world()
    K = green_kubo_matrix(L, pi, lam)
    rng = np.random.default_rng(8)
    for _ in range(5):
        A = np.sort(rng.choice(len(lam), size=4, replace=False))
        _, c1 = gk_prediction(pi, lam, K, A)
        assert abs(c1.sum()) < 1e-12
