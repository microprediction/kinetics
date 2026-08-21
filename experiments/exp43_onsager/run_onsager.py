"""Experiment 43: Onsager reciprocity for the race, and the irreversibility
witness.

K is a transport-like matrix built from time correlations, so Onsager's
question applies: is it symmetric under microscopic reversibility? Yes, and
the asymmetry is partially observable, which turns orders of arrival into a
witness of hidden broken detailed balance.

  (a) RECIPROCITY. If the driver satisfies detailed balance then K = K^T,
      because the semigroup is self-adjoint in ell^2(pi) and the correlation
      function is symmetric under exchange of the two channels.
  (b) INVARIANTS. The observable class shifts antisym(K) by the
      (N-1)-dimensional family (d lam^T - lam d^T)/2, so the gauge-invariant
      functionals of the asymmetry number N(N-1)/2 - (N-1) = (N-1)(N-2)/2.
      All vanish for reversible drivers. None exist for N = 2, so two
      channels can never witness hidden nonequilibrium and three can, the
      arrival-order analogue of needing a cycle to carry a current.
  (c) WITNESS. On a driven chain the single N = 3 invariant is zero at zero
      drive and grows monotonically with the driving strength, and it is
      class-invariant, hence computable from any member of the observable
      class.

Run:  python experiments/exp43_onsager/run_onsager.py   (~10 s)
Outputs: results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def kubo(L, pi, lam):
    m = len(pi)
    Pi = np.outer(np.ones(m), pi)
    lam_bar = lam @ pi
    lam_t = lam - lam_bar[:, None]
    dev = lambda g: np.linalg.solve(Pi - L, g - pi @ g)
    n = len(lam)
    K = np.array([[pi @ (lam_t[j] * dev(lam_t[k])) for k in range(n)]
                  for j in range(n)])
    return lam_bar, K


def reversible_generator(rng, m):
    w = rng.uniform(0.2, 1.5, (m, m))
    w = (w + w.T) / 2
    np.fill_diagonal(w, 0.0)
    mu = rng.uniform(0.5, 2.0, m)
    L = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a != b:
                L[a, b] = w[a, b] * np.sqrt(mu[b] / mu[a])
    np.fill_diagonal(L, -L.sum(1))
    return L, mu / mu.sum()


def driven_generator(rng, m, drive):
    L0, _ = reversible_generator(rng, m)
    for a in range(m):
        L0[a, (a + 1) % m] += drive
    np.fill_diagonal(L0, 0)
    np.fill_diagonal(L0, -L0.sum(1))
    w, V = np.linalg.eig(L0.T)
    pi = np.real(V[:, np.argmin(np.abs(w))])
    return L0, pi / pi.sum()


def invariants(K, lam_bar):
    """Values of the gauge-invariant functionals of antisym(K)."""
    N = len(lam_bar)
    A = (K - K.T) / 2
    pairs = [(j, k) for j in range(N) for k in range(j + 1, N)]
    avec = np.array([A[j, k] for j, k in pairs])
    G = np.zeros((len(pairs), N))
    for row, (j, k) in enumerate(pairs):
        for a in range(N):
            d = np.eye(N)[a]
            G[row, a] = (d[j] * lam_bar[k] - d[k] * lam_bar[j]) / 2
    U, s, _ = np.linalg.svd(G, full_matrices=True)
    rank_g = int((s > 1e-12 * max(s.max(), 1e-300)).sum())
    basis = U[:, rank_g:]
    return basis.T @ avec, rank_g, len(pairs) - rank_g


def main() -> None:
    rows = []
    rng = np.random.default_rng(21)
    m = 6

    # (a) reciprocity, and counts (b)
    for N in (2, 3, 4, 5):
        lam = rng.uniform(0.4, 2.0, (N, m))
        L, pi = reversible_generator(np.random.default_rng(N), m)
        db = np.abs(pi[:, None] * L - (pi[:, None] * L).T).max()
        lb, K = kubo(L, pi, lam)
        asym = np.abs(K - K.T).max() / np.abs(K).max()
        vals, rank_g, n_inv = invariants(K, lb)
        rows += [(f"N{N}_detailed_balance_defect", f"{db:.2e}"),
                 (f"N{N}_K_asymmetry_reversible", f"{asym:.2e}"),
                 (f"N{N}_gauge_dim_antisym", str(rank_g)),
                 (f"N{N}_invariants", str(n_inv)),
                 (f"N{N}_invariants_reversible_max",
                  f"{np.abs(vals).max() if n_inv else 0:.2e}")]

    # (c) the N=3 witness against driving strength, and class invariance
    N = 3
    lam = rng.uniform(0.4, 2.0, (N, m))
    for drive in (0.0, 0.1, 0.3, 1.0, 3.0):
        L, pi = driven_generator(np.random.default_rng(55), m, drive)
        lb, K = kubo(L, pi, lam)
        vals, _, _ = invariants(K, lb)
        rows.append((f"witness_drive_{drive}", f"{vals[0]:+.6f}"))

    L, pi = driven_generator(np.random.default_rng(55), m, 1.0)
    lb, K = kubo(L, pi, lam)
    v0, _, _ = invariants(K, lb)
    rng2 = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        Kp = K + np.outer(rng2.normal(size=N), lb) + np.diag(rng2.normal(size=N))
        v1, _, _ = invariants(Kp, lb)
        worst = max(worst, np.abs(v1 - v0).max())
    rows.append(("witness_class_invariance", f"{worst:.2e}"))

    with open(HERE / "results.csv", "w") as fh:
        fh.write("quantity,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")
    for k, v in rows:
        print(f"{k:36s} {v}")


if __name__ == "__main__":
    main()
