"""Experiment 41: what choice data reveals about K, and how to estimate it.

The paper computes K from a known generator. A practitioner does not have the
generator. They have races. This experiment asks what K can be recovered from
observed outcomes alone, and supplies an estimator.

Part A. IDENTIFIABILITY. The correction is linear in K, so stacking it over
every availability set gives a linear map T from K to observable deviations.
Its null space is exactly what no amount of choice data can reveal. Two
families are null, each verifiable in one line from the correction

    B(A,i) = sum_{j in A} K_ji - c_i sum_{j,k in A} K_jk :

  (i)  K -> K + d lam_bar^T for any vector d. Then the first sum gains
       lam_bar_i D_A and the second gains D_A Lbar_A, and since
       c_i = lam_bar_i / Lbar_A the two cancel.
  (ii) K -> K + t diag(lam_bar). The first sum gains t lam_bar_i, the second
       gains t Lbar_A, and they cancel the same way.

Together these span N+1 dimensions, so at most N^2 - N - 1 combinations of K
are identified. Part A checks the null space is no bigger than that.

Part B. AN ESTIMATOR. Given observed shares on a collection of availability
sets, each estimated from finitely many races, fit K by least squares on the
identified subspace. Report how well the fitted K predicts HELD-OUT subsets
that the fit never saw, which is the question a practitioner actually has.

Part D. WHAT RANKED DATA BUYS. Blocked-subset experiments are interventions
and are often impossible. The runner-up costs nothing extra to observe wherever
a finishing order is recorded. Compare the rank of three designs: every
blocked-subset experiment, winner-only on the full set, and winner plus
runner-up on the full set.

Part C. WHAT IT COSTS. Sweep the number of races per observed set and report
the held-out error, so the data requirement is explicit rather than implied.

Run:  python experiments/exp41_estimate_K/run_estimate_K.py   (~40 s)
Outputs: results.csv, figures/estimator.png
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def environment(rng, m_states, n_channels, dispersion=0.7):
    Q = rng.uniform(0.2, 1.5, (m_states, m_states))
    np.fill_diagonal(Q, 0.0)
    L = Q - np.diag(Q.sum(1))
    w, V = np.linalg.eig(L.T)
    pi = np.real(V[:, np.argmin(np.abs(w))])
    pi = pi / pi.sum()
    lam = np.exp(rng.normal(0.0, dispersion, (n_channels, m_states)))
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


def exact_shares(L, pi, lam, A, eps):
    u = np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)
    return pi @ u


def design_rows(lam_bar, N, A, eps):
    """Rows of the linear map from vec(K) to the correction on subset A."""
    Lb = lam_bar[A].sum()
    c = lam_bar[A] / Lb
    rows = []
    for pos in range(len(A)):
        g = np.zeros((N, N))
        for j in A:
            g[j, A[pos]] += 1.0
        for j in A:
            for k in A:
                g[j, k] -= c[pos]
        rows.append(-(eps / Lb) * g.ravel())
    return np.array(rows), c


def null_basis(lam_bar, N):
    B = [np.outer(np.eye(N)[a], lam_bar).ravel() for a in range(N)]
    B.append(np.diag(lam_bar).ravel())
    Q, _ = np.linalg.qr(np.array(B).T)
    return Q                                     # columns span the null space


def main() -> None:
    rows = []
    rng = np.random.default_rng(41)
    N, m = 5, 6
    L, pi, lam = environment(rng, m, N)
    lam_bar, K_true = kubo(L, pi, lam)
    eps = 0.05
    all_subsets = [list(c) for r in range(2, N + 1)
                   for c in itertools.combinations(range(N), r)]

    # ---- Part A: identifiability ------------------------------------------
    T = np.vstack([design_rows(lam_bar, N, A, eps)[0] for A in all_subsets])
    rank = np.linalg.matrix_rank(T, tol=1e-9)
    Q = null_basis(lam_bar, N)
    worst_null = 0.0
    for _ in range(20):
        d = rng.normal(size=N)
        t = rng.normal()
        Z = np.outer(d, lam_bar) + t * np.diag(lam_bar)
        worst_null = max(worst_null,
                         np.linalg.norm(T @ Z.ravel()) / np.linalg.norm(Z))
    rows += [("N", str(N)),
             ("identified_dim", str(rank)),
             ("predicted_identified_dim", str(N * N - N - 1)),
             ("nullity", str(N * N - rank)),
             ("null_family_dim", str(np.linalg.matrix_rank(Q.T))),
             ("worst_null_residual", f"{worst_null:.3e}")]

    # ---- Part B: estimator on noisy data ----------------------------------
    # observed sets: all pairs and triples; held out: everything larger
    fit_sets = [A for A in all_subsets if len(A) <= 3]
    test_sets = [A for A in all_subsets if len(A) > 3]

    def fit_K(n_races, seed):
        r = np.random.default_rng(seed)
        rows_d, rhs = [], []
        for A in fit_sets:
            p = exact_shares(L, pi, lam, A, eps)
            counts = r.multinomial(n_races, p) / n_races      # observed shares
            D, c = design_rows(lam_bar, N, A, eps)
            rows_d.append(D)
            rhs.append(counts - c)                            # deviation from softmax
        Dm = np.vstack(rows_d)
        y = np.concatenate(rhs)
        # least squares restricted to the identified subspace
        P_id = np.eye(N * N) - Q @ Q.T
        Dm_id = Dm @ P_id
        sol, *_ = np.linalg.lstsq(Dm_id, y, rcond=None)
        return (P_id @ sol).reshape(N, N)

    def heldout_error(Kh):
        errs = []
        for A in test_sets:
            p = exact_shares(L, pi, lam, A, eps)
            D, c = design_rows(lam_bar, N, A, eps)
            pred = c + D @ Kh.ravel()
            errs.append(np.abs(p - pred).max())
        return float(np.max(errs))

    # oracle: the true K, and the softmax-only baseline
    err_oracle = heldout_error(K_true)
    err_softmax = heldout_error(np.zeros((N, N)))
    rows += [("heldout_err_softmax_only", f"{err_softmax:.3e}"),
             ("heldout_err_true_K", f"{err_oracle:.3e}")]

    # ---- Part C: data requirement -----------------------------------------
    race_counts = [10_000, 100_000, 1_000_000, 10_000_000]
    med_errs = []
    for n_races in race_counts:
        e = [heldout_error(fit_K(n_races, 1000 + s)) for s in range(5)]
        med_errs.append(float(np.median(e)))
        rows.append((f"heldout_err_{n_races}_races", f"{med_errs[-1]:.3e}"))

    # how close does the fitted K get to the true K, modulo the null space?
    Kh = fit_K(10_000_000, 7)
    P_id = np.eye(N * N) - Q @ Q.T
    diff_id = P_id @ (Kh - K_true).ravel()
    rows.append(("identified_part_recovery_err", f"{np.linalg.norm(diff_id):.3e}"))
    rows.append(("identified_part_norm", f"{np.linalg.norm(P_id @ K_true.ravel()):.3e}"))

    # ---- Part D: what ranked data buys ------------------------------------
    full = list(range(N))
    D_full, c_full = design_rows(lam_bar, N, full, eps)
    rank_full = np.linalg.matrix_rank(D_full, tol=1e-9)
    rows_M = []
    for i in full:
        rest = [a for a in full if a != i]
        D_rest, _ = design_rows(lam_bar, N, rest, eps)
        for pos, j in enumerate(rest):
            # M_ij = (q_j^(-i) - p_j) / p_i, linear in K through both subsets
            rows_M.append((D_rest[pos] - D_full[full.index(j)]) / c_full[i])
    T_ranked = np.vstack([D_full, np.array(rows_M)])
    rank_ranked = np.linalg.matrix_rank(T_ranked, tol=1e-9)
    rows += [("rank_all_blocked_experiments", str(rank)),
             ("rank_full_set_winner_only", str(rank_full)),
             ("rank_full_set_with_runner_up", str(rank_ranked))]

    with open(HERE / "results.csv", "w") as fh:
        fh.write("quantity,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")
    for k, v in rows:
        print(f"{k:32s} {v}")

    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.loglog(race_counts, med_errs, "o-", label="estimated $K$ (held-out sets)")
    ax.axhline(err_softmax, ls="--", c="grey", label="softmax only")
    ax.axhline(err_oracle, ls=":", c="k", label="true $K$")
    ax.set_xlabel("races observed per availability set")
    ax.set_ylabel("worst held-out error")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "estimator.png", dpi=150)


if __name__ == "__main__":
    main()
