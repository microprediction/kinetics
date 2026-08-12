"""Tests for exp12: MATPOWER parsing, the MLODF identity, islanding, AC solver."""

import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1] / "experiments" / "exp12_grid_contingency"
sys.path.insert(0, str(EXP))
import run_grid_contingency as gc  # noqa: E402


def grid30():
    return gc.Grid(EXP / "data" / "case30.m")


def test_case30_parses_to_known_shape():
    g = grid30()
    assert g.n == 30 and g.m == 41
    assert np.all(g.rate > 0)                          # fully rated system
    assert g.slack == 0


def test_mlodf_matches_direct_resolve():
    g = grid30()
    f0, PTDF = gc.dc_setup(g)
    rng = np.random.default_rng(3)
    for _ in range(10):
        S = list(rng.choice(g.m, size=2, replace=False))
        pred = gc.mlodf_flows(f0, PTDF, S)
        if pred is None:
            continue
        assert np.abs(pred - gc.dc_direct(g, S)).max() < 1e-12


def test_islanding_detection_matches_graph_connectivity():
    g = grid30()
    f0, PTDF = gc.dc_setup(g)
    rng = np.random.default_rng(5)
    for _ in range(40):
        S = list(rng.choice(g.m, size=2, replace=False))
        # graph connectivity without branches S
        keep = np.setdiff1d(np.arange(g.m), S)
        adj = {i: set() for i in range(g.n)}
        for k in keep:
            adj[g.f[k]].add(g.t[k]); adj[g.t[k]].add(g.f[k])
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        connected = len(seen) == g.n
        assert (gc.mlodf_flows(f0, PTDF, S) is not None) == connected


def test_dc_flows_conserve_at_every_bus():
    g = grid30()
    f0, _ = gc.dc_setup(g)
    net = np.zeros(g.n)
    np.add.at(net, g.f, -f0)
    np.add.at(net, g.t, f0)
    inj = g.Pg - g.Pd
    resid = inj + net
    resid[g.slack] = 0.0                               # slack absorbs imbalance
    assert np.abs(resid).max() < 1e-10


def test_ac_newton_converges_with_tiny_residual():
    g = grid30()
    Y = gc.build_ybus(g)
    V, ok = gc.newton_pf(g, Y)
    assert ok
    S = V * np.conj(Y @ V)
    non_slack = np.setdiff1d(np.arange(g.n), [g.slack])
    assert np.abs((g.Pg - g.Pd - S.real)[non_slack]).max() < 1e-8
    pq = np.setdiff1d(np.arange(g.n), np.concatenate([[g.slack], g.pv]))
    assert np.abs((-g.Qd - S.imag)[pq]).max() < 1e-8
    assert 0.9 < np.abs(V).min() and np.abs(V).max() < 1.15
