"""Tests for exp09's finite-volume disk generator and its Green-Kubo layer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "exp01_narrow_escape"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "exp07_green_kubo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "exp09_gk_narrow_escape"))
import run_gk_narrow_escape as gk9  # noqa: E402
from run_green_kubo import exact_win_probs, gk_prediction, green_kubo_matrix  # noqa: E402


def small_grid(monkey_nr=10, monkey_nth=24):
    old = gk9.NR, gk9.NTH
    gk9.NR, gk9.NTH = monkey_nr, monkey_nth
    try:
        return gk9.build_disk_generator()
    finally:
        gk9.NR, gk9.NTH = old


def test_grid_generator_invariants():
    Lg, pi, r_cent, theta_cent = small_grid()
    assert np.abs(Lg.sum(axis=1)).max() < 1e-10          # conservation
    assert np.abs(pi @ Lg).max() < 1e-10                 # uniform law invariant
    db = pi[:, None] * Lg - (pi[:, None] * Lg).T
    assert np.abs(db).max() < 1e-10                      # reversibility
    assert abs(pi.sum() - 1.0) < 1e-12
    # off-diagonal rates nonnegative
    off = Lg - np.diag(np.diag(Lg))
    assert off.min() >= 0


def test_grid_pi_is_uniform_on_the_disk():
    _, pi, r_cent, _ = small_grid()
    # cell probability proportional to cell area: constant within each ring
    pis = pi.reshape(len(r_cent), -1)
    assert np.abs(pis - pis[:, :1]).max() < 1e-15        # angular uniformity
    ratio = pis[:, 0] / r_cent                            # radial: pi ~ r dr dtheta
    assert np.abs(ratio - ratio[0]).max() < 1e-15


def test_gk_layer_on_grid_softmax_and_correction_orders():
    rng = np.random.default_rng(0)
    Lg, pi, r_cent, theta_cent = small_grid()
    # three bands
    n = len(pi)
    rr = np.repeat(r_cent, len(theta_cent))
    tt = np.tile(theta_cent, len(r_cent))
    g = np.zeros((3, n))
    for i, c in enumerate((0.5, 2.5, 4.5)):
        d = np.abs((tt - c + np.pi) % (2 * np.pi) - np.pi)
        g[i] = ((rr > 0.7) & (d <= 0.35 + 0.1 * i)).astype(float)
    K = green_kubo_matrix(Lg, pi, g)
    assert np.abs(K - K.T).max() < 1e-10
    A = np.arange(3)
    p0, c1 = gk_prediction(pi, g, K, A)
    assert abs(c1.sum()) < 1e-12
    assert abs(p0.sum() - 1.0) < 1e-12
    errs_s, errs_g = [], []
    for kappa in (0.05, 0.025, 0.0125):
        p = exact_win_probs(Lg, pi, g, A, kappa)
        assert abs(p.sum() - 1.0) < 1e-9
        errs_s.append(np.abs(p - p0).max())
        errs_g.append(np.abs(p - p0 - kappa * c1).max())
    assert errs_s[1] / errs_s[0] < 0.6                    # first order
    assert errs_g[1] / errs_g[0] < 0.35                   # second order
    assert errs_g[2] / errs_g[1] < 0.35


def test_softmax_limit_is_band_area_share():
    """As kappa -> 0 the win probability is the band's share of total band area."""
    Lg, pi, r_cent, theta_cent = small_grid()
    n = len(pi)
    rr = np.repeat(r_cent, len(theta_cent))
    tt = np.tile(theta_cent, len(r_cent))
    g = np.zeros((2, n))
    for i, (c, h) in enumerate(((1.0, 0.3), (4.0, 0.6))):
        d = np.abs((tt - c + np.pi) % (2 * np.pi) - np.pi)
        g[i] = ((rr > 0.7) & (d <= h)).astype(float)
    areas = g @ pi
    p = exact_win_probs(Lg, pi, g, np.arange(2), 1e-4)
    assert np.abs(p - areas / areas.sum()).max() < 1e-3
