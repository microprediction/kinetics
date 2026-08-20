"""Locks in experiment 41: identifiability of K and what data recovers it."""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                      / "exp41_estimate_K"))
import run_estimate_K as ek  # noqa: E402


def _design(lam_bar, N, eps=0.05):
    subs = [list(c) for r in range(2, N + 1) for c in itertools.combinations(range(N), r)]
    return np.vstack([ek.design_rows(lam_bar, N, A, eps)[0] for A in subs])


def test_two_null_families_are_exactly_null():
    rng = np.random.default_rng(1)
    for N in (3, 4, 5):
        lam_bar = rng.uniform(0.5, 2.0, N)
        T = _design(lam_bar, N)
        for _ in range(5):
            Z = (np.outer(rng.normal(size=N), lam_bar)
                 + rng.normal() * np.diag(lam_bar))
            assert np.linalg.norm(T @ Z.ravel()) / np.linalg.norm(Z) < 1e-12


def test_identified_dimension_is_N2_minus_N_minus_1():
    rng = np.random.default_rng(2)
    for N in (3, 4, 5, 6):
        lam_bar = rng.uniform(0.5, 2.0, N)
        assert np.linalg.matrix_rank(_design(lam_bar, N), tol=1e-9) == N * N - N - 1


def test_runner_up_matches_every_intervention():
    """Full-set ranked data identifies as much as all blocked-subset experiments,
    while winner-only data identifies almost nothing."""
    rng = np.random.default_rng(3)
    for N in (4, 5):
        lam_bar = rng.uniform(0.5, 2.0, N)
        eps = 0.05
        full = list(range(N))
        D_full, c_full = ek.design_rows(lam_bar, N, full, eps)
        rows = []
        for i in full:
            rest = [a for a in full if a != i]
            D_rest, _ = ek.design_rows(lam_bar, N, rest, eps)
            for pos, j in enumerate(rest):
                rows.append((D_rest[pos] - D_full[full.index(j)]) / c_full[i])
        rank_ranked = np.linalg.matrix_rank(np.vstack([D_full, np.array(rows)]), tol=1e-9)
        rank_winner = np.linalg.matrix_rank(D_full, tol=1e-9)
        assert rank_ranked == N * N - N - 1
        assert rank_winner == N - 1
