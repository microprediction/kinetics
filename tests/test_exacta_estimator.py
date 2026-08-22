"""Locks in experiment 44: the exacta-board estimator recovers the identified
part of eps*K, in the right gauges, at the right orders."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                      / "exp44_exacta_estimator"))
import run_exacta_estimator as ex  # noqa: E402


def test_linear_model_recovery_is_second_order():
    """On boards generated exactly from the first-order model, the gauge-
    projected cosine to truth approaches 1 at rate eps^2."""
    rng = np.random.default_rng(5)
    N = 6
    lb = rng.uniform(0.5, 2.0, N)
    lb /= lb.sum()
    K = rng.normal(0, 1.0, (N, N))
    full = list(range(N))
    D_full, _ = ex.design_rows(lb, N, full)
    defects = []
    for eps in (0.02, 0.01, 0.005):
        k = eps * K.ravel()
        p = lb + D_full @ k
        board = np.zeros((N, N))
        for i in full:
            rest = [a for a in full if a != i]
            D_rest, c_rest = ex.design_rows(lb, N, rest)
            board[i, rest] = (c_rest + D_rest @ k) - p[rest]
        board = p[:, None] * (board / board.sum(1, keepdims=True))
        k_hat, P2, rank = ex.fit_from_board(board, N)
        assert rank == N * N - 2 * N
        k_ref = P2 @ k
        cos = k_hat @ k_ref / np.linalg.norm(k_hat) / np.linalg.norm(k_ref)
        defects.append((1 - cos) / eps**2)
    assert max(defects) < 20
    assert max(defects) / min(defects) < 1.2   # constant => eps^2 scaling


def test_markov_model_recovery_is_first_order():
    """On exact boards from the chain model, the error of the identified part
    is O(eps) once the reference uses the board's time unit (rates sum to 1)."""
    rng = np.random.default_rng(44)
    N, m = 6, 7
    L, pi, lam = ex.environment(rng, m, N)
    lam_bar, K = ex.kubo(L, pi, lam)
    ratios = []
    for eps in (0.04, 0.02, 0.01):
        board = ex.exact_board(L, pi, lam, eps)
        k_hat, P2, rank = ex.fit_from_board(board, N)
        assert rank == N * N - 2 * N
        k_ref = P2 @ (eps * K.ravel() / lam_bar.sum())
        rel = np.linalg.norm(k_hat - k_ref) / np.linalg.norm(k_ref)
        ratios.append(rel / eps)
    assert max(ratios) < 3.0
    assert max(ratios) / min(ratios) < 1.3     # constant => first order


def test_common_mode_board_returns_zero():
    """Proportional hazards have no identified content and the estimator says
    so at machine precision."""
    rng = np.random.default_rng(7)
    N, m = 6, 7
    L, pi, _ = ex.environment(rng, m, N)
    a = rng.uniform(0.5, 2.0, N)
    c = np.exp(rng.normal(0, 0.7, m))
    lam = np.outer(a, c)
    _, K = ex.kubo(L, pi, lam)
    for eps in (0.04, 0.01):
        board = ex.exact_board(L, pi, lam, eps)
        k_hat, _, _ = ex.fit_from_board(board, N)
        assert np.linalg.norm(k_hat) < 1e-10 * eps * np.linalg.norm(K)
