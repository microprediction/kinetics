"""Tests for exp10: geometric hitting splits and the substitution resolvent."""

import sys
from pathlib import Path

import numpy as np

for sub in ("exp01_narrow_escape", "exp08_ranked_escape", "exp10_green_function_M"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / sub))
import run_green_function_M as gm  # noqa: E402


def small(monkey=(12, 48)):
    old = gm.NR, gm.NTH
    gm.NR, gm.NTH = monkey
    try:
        return gm.sparse_disk_generator()
    finally:
        gm.NR, gm.NTH = old


def test_disjoint_windows_are_disjoint():
    rng = np.random.default_rng(0)
    centers, halfw = gm.make_disjoint_windows(rng)
    assert len(centers) == gm.N_W
    for i in range(gm.N_W):
        j = (i + 1) % gm.N_W
        gap = (centers[j] - centers[i]) % (2 * np.pi)
        assert gap > halfw[i] + halfw[j]
    assert halfw.min() >= 0.02


def test_sparse_generator_invariants():
    Lg, pi, r_cent, theta_cent = small()
    scale = np.abs(Lg.diagonal()).max()
    assert np.abs(np.asarray(Lg.sum(axis=1)).ravel()).max() < 1e-9 * scale
    assert np.abs(pi @ Lg).max() < 1e-9 * scale
    assert abs(pi.sum() - 1.0) < 1e-12


def test_hitting_splits_sum_to_one_and_respect_size():
    """Splits are a probability vector, and a wider window catches more."""
    gm.NR, gm.NTH = 12, 48
    try:
        Lg, pi, r_cent, theta_cent = gm.sparse_disk_generator()
        centers = np.array([0.5, 2.5, 4.5])
        halfw = np.array([0.15, 0.3, 0.6])
        base = (gm.NR - 1) * gm.NTH
        cells = []
        for c, h in zip(centers, halfw):
            d = np.abs((theta_cent - c + np.pi) % (2 * np.pi) - np.pi)
            cells.append(base + np.nonzero(d <= h)[0])
        start = np.zeros(Lg.shape[0]); start[: gm.NTH] = pi[: gm.NTH]
        splits = gm.hitting_splits(Lg, start, cells)
        assert abs(splits.sum() - 1.0) < 1e-8
        assert np.all(splits > 0)
        assert splits[0] < splits[1] < splits[2]
    finally:
        gm.NR, gm.NTH = 60, 176


def test_hitting_split_symmetric_windows_are_equal():
    """Two identical windows placed symmetrically must split 50/50."""
    gm.NR, gm.NTH = 12, 48
    try:
        Lg, pi, r_cent, theta_cent = gm.sparse_disk_generator()
        base = (gm.NR - 1) * gm.NTH
        cells = []
        for c in (1.0, 1.0 + np.pi):
            d = np.abs((theta_cent - c + np.pi) % (2 * np.pi) - np.pi)
            cells.append(base + np.nonzero(d <= 0.3)[0])
        start = np.zeros(Lg.shape[0]); start[: gm.NTH] = pi[: gm.NTH]
        splits = gm.hitting_splits(Lg, start, cells)
        assert abs(splits[0] - splits[1]) < 1e-8
    finally:
        gm.NR, gm.NTH = 60, 176
