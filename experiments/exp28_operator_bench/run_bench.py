"""Experiment 28: operator microbenchmarks for the Gaussian perturbed-max
layer (Experiment I of the companion paper).

Grid: N in {10, 100, 1000, 10000} x k in {0, 1, 2, 4, 8, 15}.
Factor nodes: product Gauss-Hermite (order 15, pruned) for k <= 4;
fixed scrambled Sobol for k in {8, 15}. Lattice L = 257 (training mode)
and L = 501 for N = 10000.

Reference per problem: the same operator at high resolution (GH order 21
pruned / L = 2001 for k <= 4; 4x Sobol nodes / L = 1001 for k > 4).

Baselines: hard-argmax Monte Carlo at M in {1e4, 1e5, 1e6} draws, and the
temperature-softmax surrogate (Collier et al.) at M = 1000 with
tau in {0.1, 1.0}.

Reported: max abs error, max log-odds error over shares >= 1e-3/N... no:
over shares >= 1e-4, potential error |G_hat - G_ref|, and wall time.

Run:  python experiments/exp28_operator_bench/run_bench.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import log_ndtr
from scipy.stats import norm, qmc

HERE = Path(__file__).resolve().parent
SEED = 28


def gh_nodes(k, order=15, prune=1e-7):
    if k == 0:
        return np.zeros((1, 1)), np.ones(1)
    x1, w1 = hermegauss(order)
    w1 = w1 / w1.sum()
    grids = np.array(np.meshgrid(*([x1] * k))).reshape(k, -1).T
    ws = np.array(np.meshgrid(*([w1] * k))).reshape(k, -1).T.prod(axis=1)
    keep = ws > prune * ws.max()
    return grids[keep], ws[keep] / ws[keep].sum()


def sobol_nodes(k, m, seed=0):
    F = norm.ppf(qmc.Sobol(k, scramble=True, seed=seed).random_base2(m))
    return F, np.full(len(F), 1.0 / len(F))


def operator(mu, V, D, F, W, L):
    """Forward p and potential G, chunked over factor nodes. Max-wins."""
    N = len(mu)
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T
    lo = M_all.min() - 8 * sd.max()
    hi = M_all.max() + 8 * sd.max()
    x = np.linspace(lo, hi, L)
    dx = x[1] - x[0]
    p = np.zeros(N)
    G = 0.0
    chunk = max(1, int(5e6 / (N * L)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        logF = log_ndtr(z)
        g = np.exp(-0.5 * z * z) / (sd[None, :, None] * np.sqrt(2 * np.pi))
        field = logF.sum(axis=1)
        rest = np.exp(np.clip(field[:, None, :] - logF, -745, 0))
        p += (Wc[:, None] * ((g * rest).sum(axis=2) * dx)).sum(axis=0)
        H = np.exp(np.clip(field, -745, 0))
        intH = (H.sum(axis=1) - 0.5 * (H[:, 0] + H[:, -1])) * dx
        G += float(Wc @ (x[-1] - intH))
    return p / p.sum(), G


def mc_hard(mu, V, D, M, rng):
    N = len(mu)
    k = V.shape[1]
    counts = np.zeros(N)
    Gacc = 0.0
    chunk = max(1, int(2e7 / N))
    done = 0
    while done < M:
        m = min(chunk, M - done)
        f = rng.standard_normal((m, k))
        U = mu[None, :] + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((m, N))
        counts += np.bincount(np.argmax(U, axis=1), minlength=N)
        Gacc += U.max(axis=1).sum()
        done += m
    return counts / M, Gacc / M


def mc_softmax(mu, V, D, M, tau, rng):
    N = len(mu)
    k = V.shape[1]
    acc = np.zeros(N)
    chunk = max(1, int(2e7 / N))
    done = 0
    while done < M:
        m = min(chunk, M - done)
        f = rng.standard_normal((m, k))
        U = (mu[None, :] + f @ V.T
             + np.sqrt(D)[None, :] * rng.standard_normal((m, N))) / tau
        U -= U.max(axis=1, keepdims=True)
        E = np.exp(U)
        acc += (E / E.sum(axis=1, keepdims=True)).sum(axis=0)
        done += m
    return acc / M


def main():
    rng = np.random.default_rng(SEED)
    rows = ["N,k,method,setting,seconds,max_abs_err,max_logodds_err,G_err"]
    configs = []
    for n in (10, 100, 1000):
        for k in (0, 1, 2, 4, 8, 15):
            configs.append((n, k))
    configs.append((10000, 2))

    for n, k in configs:
        mu = rng.normal(0, 1.5, n)
        V = (rng.normal(0, 0.5 / np.sqrt(max(k, 1)), (n, k))
             if k > 0 else np.zeros((n, 1)))
        D = rng.uniform(0.5, 1.5, n)
        kk = max(k, 1)
        if k <= 4:
            F, W = gh_nodes(k if k > 0 else 0)
            Fr, Wr = gh_nodes(k if k > 0 else 0, order=21, prune=1e-9)
        else:
            m = 10 if n >= 1000 else 12
            F, W = sobol_nodes(kk, m, seed=1)
            Fr, Wr = sobol_nodes(kk, m + 2, seed=2)
        L = 501 if n >= 10000 else 257
        # reference
        p_ref, G_ref = operator(mu, V, D, Fr, Wr, 2001 if k <= 4 else 1001)
        res = p_ref >= 1e-4

        def score(p, G, name, setting, dt):
            ae = float(np.abs(p - p_ref).max())
            le = float(np.abs(np.log(np.maximum(p[res], 1e-300))
                              - np.log(p_ref[res])).max()) if res.any() else np.nan
            ge = abs(G - G_ref) if G is not None else np.nan
            print(f"N={n:>5} k={k:>2} {name:>12} {setting:>10} "
                  f"{dt:8.3f}s abs {ae:.1e} lo {le:.1e} G {ge if ge==ge else float('nan'):.1e}")
            rows.append(f"{n},{k},{name},{setting},{dt:.4f},{ae:.3e},"
                        f"{le:.3e},{ge:.3e}")

        t0 = time.perf_counter()
        p_op, G_op = operator(mu, V, D, F, W, L)
        score(p_op, G_op, "operator", f"Q={len(F)},L={L}",
              time.perf_counter() - t0)

        for M in (10**4, 10**5, 10**6):
            if n >= 10000 and M >= 10**6:
                continue
            t0 = time.perf_counter()
            p_mc, G_mc = mc_hard(mu, V, D, M, np.random.default_rng(5))
            score(p_mc, G_mc, "mc_hard", f"M={M}", time.perf_counter() - t0)

        for tau in (0.1, 1.0):
            t0 = time.perf_counter()
            p_sm = mc_softmax(mu, V, D, 1000, tau, np.random.default_rng(6))
            score(p_sm, None, "mc_softmax", f"tau={tau}",
                  time.perf_counter() - t0)

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
