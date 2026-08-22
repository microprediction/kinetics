"""Experiment 42: counting the driver's modes from the observable class.

Resolves the rank anomaly found while probing Peter's structural-diagnostic
suggestion, and turns it into an exact estimator.

The observer recovers K only modulo the family K + d lam^T + t diag(lam).
Three facts, the first two of which explain the anomaly:

  (a) FAT SET. {K + d lam^T} is an N-dimensional subfamily consisting
      entirely of matrices of rank <= r+1, since d lam^T is rank one. An
      alternating low-rank fit at rank r+1 therefore has an N-dimensional
      solution set and finds it easily.
  (b) THIN SLICE. Rank <= r requires d in col(K), an r-dimensional slice of
      codimension N-r. The optimizer never lands on it, which is why the
      first machine-zero residual appears at rank r+1 and not at rank r,
      despite the true K sitting in the search space.
  (c) ALGEBRAIC ESTIMATOR. No optimization is needed. For any member
      M = K + d lam^T of the t-free family, the combinations
      lam_k M_j - lam_j M_k cancel d exactly and lie in col(K), so the span
      of all pairwise combinations has dimension exactly r, generically.

Part B (rate gauge): verifies Proposition "Rate gauge": replacing
(lam, K) by (lam + eps*eta, K - diag(eta)) cancels exactly in every subset's
first-order share, and the nuisance-rate invisible family {d lam^T + diag(v)}
has dimension exactly 2N, equal to the measured null space of the
all-experiments design.

Part C (mode counting in the 2N class): when the rates are estimated too,
the class gains an arbitrary diagonal and the Part A read-off is
contaminated. A submatrix whose row and column sets are disjoint contains no
diagonal entry, and on such blocks the class shifts K by rank one at most.
The largest diagonal-avoiding block of a generic member has rank
min(r+1, floor(N/2)), so its rank minus one recovers r whenever N >= 2(r+1)
and otherwise reports the detection floor floor(N/2) - 1.

Pipeline: one alternating fit at rank r+1 recovers t exactly (the diag
direction is not in the fat set, so it must be fit away), then the pairwise
combinations of the resulting M deliver r as a numerical rank with
machine-zero trailing singular values.

Run:  python experiments/exp42_mode_count/run_mode_count.py   (~20 s)
Outputs: results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def environment(rng, m_states, n_channels, r):
    Q = rng.uniform(0.2, 1.5, (m_states, m_states))
    np.fill_diagonal(Q, 0.0)
    L = Q - np.diag(Q.sum(1))
    w, V = np.linalg.eig(L.T)
    pi = np.real(V[:, np.argmin(np.abs(w))])
    pi = pi / pi.sum()
    B = rng.normal(0, 0.6, (n_channels, r))
    z = rng.normal(0, 1.0, (r, m_states))
    z = z - (z @ pi)[:, None]
    d = B @ z
    lam = d - d.min() + 0.5
    return L, pi, lam


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


def fat_solve(K_obs, lam_bar, rho, iters=600):
    """Alternating rank-rho fit within the class; returns (t_hat, M)."""
    N = len(lam_bar)
    A = np.column_stack([np.kron(np.eye(N)[a], lam_bar) for a in range(N)]
                        + [np.diag(lam_bar).ravel()])
    x = np.zeros(N + 1)
    for _ in range(iters):
        M = K_obs + (A @ x).reshape(N, N)
        U, s, Vt = np.linalg.svd(M)
        X = (U[:, :rho] * s[:rho]) @ Vt[:rho]
        x, *_ = np.linalg.lstsq(A, (X - K_obs).ravel(), rcond=None)
    return x[N], K_obs + (A @ x).reshape(N, N)


def mode_count(M, lam_bar, tol=1e-8):
    """Rank of span{lam_k M_j - lam_j M_k}: d cancels, col(K) remains."""
    N = len(lam_bar)
    cols = [lam_bar[k] * M[:, j] - lam_bar[j] * M[:, k]
            for j in range(N) for k in range(j + 1, N)]
    sv = np.linalg.svd(np.column_stack(cols), compute_uv=False)
    return int((sv > tol * sv.max()).sum()), sv


def main() -> None:
    rng = np.random.default_rng(77)
    rows = []
    m, N = 10, 8
    for r in (1, 2, 3, 4, 5):
        L, pi, lam = environment(rng, m, N, r)
        lam_bar, K = kubo(L, pi, lam)
        t_true = 0.7
        K_obs = (K + np.outer(rng.normal(size=N), lam_bar)
                 + t_true * np.diag(lam_bar))

        # geometry: the whole t-free family has rank exactly r+1
        fam_ranks = set()
        for _ in range(100):
            sv = np.linalg.svd(K + np.outer(rng.normal(0, 1, N), lam_bar),
                               compute_uv=False)
            fam_ranks.add(int((sv > 1e-10 * sv.max()).sum()))

        t_hat, M = fat_solve(K_obs, lam_bar, r + 1)
        r_hat, sv = mode_count(M, lam_bar)
        rows += [(f"r{r}_family_ranks", " ".join(map(str, sorted(fam_ranks)))),
                 (f"r{r}_t_recovery_err", f"{abs(t_hat + t_true):.2e}"),
                 (f"r{r}_estimate", str(r_hat)),
                 (f"r{r}_sv_gap", f"{sv[r] / sv[r - 1]:.2e}" if r < len(sv)
                  else "na")]
        print(f"r={r}: family ranks {sorted(fam_ranks)}, "
              f"t err {abs(t_hat + t_true):.1e}, r_hat={r_hat}, "
              f"sv[r]/sv[r-1]={sv[r]/sv[r-1]:.1e}")

    # ---- Part C: mode counting in the 2N (nuisance-rates) class -----------
    from itertools import combinations

    m2, N2 = 12, 10
    h = N2 // 2
    for r in (1, 2, 3, 4, 5, 6):
        rng_c = np.random.default_rng(100 + r)
        L, pi, lam = environment(rng_c, m2, N2, r)
        lam_bar, K = kubo(L, pi, lam)
        M = (K + np.outer(rng_c.normal(size=N2), lam_bar)
             + np.diag(rng_c.normal(size=N2)))
        best = 0
        for R in combinations(range(N2), h):
            C = [c for c in range(N2) if c not in R]
            sv = np.linalg.svd(M[np.ix_(R, C)], compute_uv=False)
            best = max(best, int((sv > 1e-9 * sv.max()).sum()))
        rows.append((f"C_r{r}_rhat_2Nclass", str(best - 1)))
        print(f"Part C r={r}: r_hat = {best - 1} (cap {h - 1})")

    # ---- Part B: the rate gauge -------------------------------------------
    import itertools

    def design_rows_local(lb, n, A):
        Lb = lb[A].sum()
        c = lb[A] / Lb
        rws = []
        for pos in range(len(A)):
            g = np.zeros((n, n))
            for j in A:
                g[j, A[pos]] += 1.0
            for j in A:
                for k in A:
                    g[j, k] -= c[pos]
            rws.append(-(1.0 / Lb) * g.ravel())
        return np.array(rws)

    def jac_c(lb, n, A):
        S = lb[A].sum()
        J = np.zeros((len(A), n))
        for pos, j in enumerate(A):
            for l in A:
                J[pos, l] = (1.0 * (l == j)) / S - lb[j] / S**2
        return J

    for n in (5, 6, 7):
        rg = np.random.default_rng(n)
        lb = rg.uniform(0.5, 2.0, n)
        lb /= lb.sum()
        D_full = design_rows_local(lb, n, list(range(n)))
        allsub = [list(c) for k in range(2, n + 1)
                  for c in itertools.combinations(range(n), k)]
        Dm = np.vstack([design_rows_local(lb, n, A) - jac_c(lb, n, A) @ D_full
                        for A in allsub])
        eta = rg.normal(size=n)
        worst = max(np.abs(design_rows_local(lb, n, A) @ np.diag(eta).ravel()
                           + jac_c(lb, n, A) @ eta).max() for A in allsub)
        fam = ([np.outer(np.eye(n)[a], lb).ravel() for a in range(n)]
               + [np.diag(np.eye(n)[a]).ravel() for a in range(n)])
        F = np.array(fam).T
        null_dim = n * n - np.linalg.matrix_rank(Dm, tol=1e-9)
        rows += [(f"gauge_N{n}_cancellation", f"{worst:.2e}"),
                 (f"gauge_N{n}_null_dim", str(null_dim)),
                 (f"gauge_N{n}_family_dim", str(np.linalg.matrix_rank(F))),
                 (f"gauge_N{n}_family_annihilated",
                  f"{np.linalg.norm(Dm @ F)/np.linalg.norm(F):.2e}")]

    with open(HERE / "results.csv", "w") as fh:
        fh.write("quantity,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")


if __name__ == "__main__":
    main()
