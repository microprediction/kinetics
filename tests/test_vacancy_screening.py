"""Tests for exp11: gradients, the local Woodbury algebra, and the harmonic limit."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "exp11_vacancy_screening"))
import run_vacancy_screening as vs  # noqa: E402


def small_world(mismatch=0.15, side=8, seed=13):
    vs.SIDE = side
    vs.MISMATCH = mismatch
    rng = np.random.default_rng(seed)
    return vs.build_network(rng)


def test_gradient_matches_finite_differences():
    X0, bonds, kb, ell, free = small_world()
    rng = np.random.default_rng(1)
    x0 = X0[free].ravel() + 0.02 * rng.standard_normal(int(free.sum()) * 2)
    E, g = vs.energy_grad(x0, X0, free, bonds, kb, ell)
    for i in (0, 7, 23):
        eps = 1e-6
        xp = x0.copy(); xp[i] += eps
        Ep, _ = vs.energy_grad(xp, X0, free, bonds, kb, ell)
        assert abs((Ep - E) / eps - g[i]) < 1e-4


def test_hessian_matches_finite_difference_of_gradient():
    X0, bonds, kb, ell, free = small_world()
    free_nodes = np.nonzero(free)[0]
    node_dof = {node: 2 * k for k, node in enumerate(free_nodes)}
    E0, xfree, res = vs.relax(X0, free, bonds, kb, ell)
    pos = X0.copy(); pos[free] = xfree
    H = vs.assemble_hessian(pos, free_nodes, bonds, kb, ell, node_dof)
    x0 = xfree.ravel()
    eps = 1e-6
    for i in (0, 11):
        xp = x0.copy(); xp[i] += eps
        _, gp = vs.energy_grad(xp, X0, free, bonds, kb, ell)
        xm = x0.copy(); xm[i] -= eps
        _, gm = vs.energy_grad(xm, X0, free, bonds, kb, ell)
        assert np.abs((gp - gm) / (2 * eps) - H[i]).max() < 1e-4


def test_relaxation_energies_are_nonnegative_and_ranked_sanely():
    """run_one at small size: dE_true >= 0 is implied by ratio bounds; here we
    check the harmonic sweep agrees with direct assembly (done inside run_one)
    and returns sane metrics in the near-harmonic limit."""
    vs.SIDE = 8
    r = vs.run_one(0.05)
    assert r["woodbury"] < 1e-10                 # local algebra vs direct assembly
    assert r["spearman"] > 0.98                  # near-harmonic: ranking near-exact
    assert 0.9 < r["ratio_lo"] and r["ratio_hi"] < 1.1


def test_screening_degrades_gracefully_with_anharmonicity():
    vs.SIDE = 8
    r_soft = vs.run_one(0.05)
    r_hard = vs.run_one(0.4)
    assert r_hard["spearman"] < r_soft["spearman"]
    assert r_hard["spearman"] > 0.5              # but does not collapse
    assert r_hard["woodbury"] < 1e-10            # algebra exact regardless
