"""Experiment 39: the Green-Kubo correction predicts the runner-up kernel.

This closes the bridge between Paper 1 (softmax from fast mixing) and Paper 2
(the runner-up principle). Paper 2 shows the runner-up kernel

    M_ij = P(runner-up = j | winner = i)

is exactly the object that identifies single-deletion counterfactuals, and that
winner-only data leaves it completely unconstrained. Paper 1 computes the choice
law from the physics. The question here is whether Paper 1 also predicts M.

Exact identity. Removing channel i changes the winner only in those realizations
where i won, and there the runner-up inherits, so

    q_j^(-i) = p_j + p_i M_ij                                          (exact)

for every eps, with q^(-i) the deletion counterfactual. Applying Theorem 1 to
both the full set A and the reduced set A \\ {i} therefore predicts M.

Leading order gives the Luce (IIA) runner-up kernel

    M_ij = lbar_j / (Lbar_A - lbar_i) + O(eps),

and adding the Green-Kubo term should gain one order.

The joint law is computed exactly by a two-stage resolvent. Conditioning on the
first firing time and state,

    P(win = i, runner-up = j) = pi . (Lambda_A - L/eps)^{-1} [lambda_i * u_j^{A\\i}]

which is exact on a finite chain, so no simulation error enters.

Run:  python experiments/exp39_runner_up_bridge/run_runner_up_bridge.py
Outputs: results.csv, figures/runner_up_orders.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def build_environment(seed: int = 7, m_states: int = 6, n_channels: int = 5):
    """A non-reversible ergodic chain and strictly positive channel intensities."""
    rng = np.random.default_rng(seed)
    Q = rng.uniform(0.2, 1.5, (m_states, m_states))
    np.fill_diagonal(Q, 0.0)
    L = Q - np.diag(Q.sum(1))
    w, V = np.linalg.eig(L.T)
    pi = np.real(V[:, np.argmin(np.abs(w))])
    pi = pi / pi.sum()
    assert pi.min() > 0 and np.allclose(pi @ L, 0, atol=1e-12)
    lam = rng.uniform(0.3, 2.0, (n_channels, m_states))
    return L, pi, lam


def green_kubo(L, pi, lam):
    """K_jk = int_0^inf Cov_pi(lambda_j(Y_0), lambda_k(Y_t)) dt."""
    m = len(pi)
    Pi = np.outer(np.ones(m), pi)
    lam_bar = lam @ pi
    lam_t = lam - lam_bar[:, None]

    def deviation(g):
        return np.linalg.solve(Pi - L, g - pi @ g)

    n = len(lam)
    K = np.array([[pi @ (lam_t[j] * deviation(lam_t[k])) for k in range(n)]
                  for j in range(n)])
    return lam_bar, K


def win_functions(L, lam, A, eps):
    """u_i^A(y), columns ordered as A."""
    return np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)


def theory_shares(lam_bar, K, A, eps):
    """Theorem 1: softmax plus the Green-Kubo correction."""
    lb = lam_bar[A]
    Lb = lb.sum()
    soft = lb / Lb
    KA = K[np.ix_(A, A)]
    return soft - (eps / Lb) * (KA.sum(axis=0) - soft * KA.sum())


def exact_runner_up(L, pi, lam, A, eps):
    """M_ij and the exact deletion counterfactuals, by two-stage resolvent."""
    n = len(lam)
    u = win_functions(L, lam, A, eps)
    p = pi @ u
    resolvent = -np.linalg.inv(L / eps - np.diag(lam[A].sum(0)))
    M = np.zeros((n, n))
    q = np.zeros((n, n))
    for i in A:
        rest = [a for a in A if a != i]
        u_rest = win_functions(L, lam, rest, eps)
        q_i = pi @ u_rest
        for pos, j in enumerate(rest):
            M[i, j] = (pi @ (resolvent @ (lam[i] * u_rest[:, pos]))) / p[i]
            q[i, j] = q_i[pos]
    return p, M, q


def main() -> None:
    L, pi, lam = build_environment()
    n = len(lam)
    lam_bar, K = green_kubo(L, pi, lam)
    A = list(range(n))
    off = ~np.eye(n, dtype=bool)

    Lbar_A = lam_bar.sum()
    M_luce = np.array([[lam_bar[j] / (Lbar_A - lam_bar[i]) if i != j else 0.0
                        for j in range(n)] for i in range(n)])

    eps_grid = np.array([0.08, 0.04, 0.02, 0.01, 0.005])
    err_luce, err_gk, identity_err, rowsum_err = [], [], [], []

    for eps in eps_grid:
        p, M, q = exact_runner_up(L, pi, lam, A, eps)

        # the exact runner-up identity q_j^(-i) = p_j + p_i M_ij
        identity_err.append(max(abs(q[i, j] - (p[j] + p[i] * M[i, j]))
                                for i in range(n) for j in range(n) if i != j))
        rowsum_err.append(max(abs(M[i].sum() - 1.0) for i in range(n)))

        p_th = theory_shares(lam_bar, K, A, eps)
        M_gk = np.zeros((n, n))
        for i in A:
            rest = [a for a in A if a != i]
            q_th = theory_shares(lam_bar, K, rest, eps)
            for pos, j in enumerate(rest):
                M_gk[i, j] = (q_th[pos] - p_th[j]) / p_th[i]

        err_luce.append(np.abs(M - M_luce)[off].max())
        err_gk.append(np.abs(M - M_gk)[off].max())

    slope = lambda e: float(np.polyfit(np.log(eps_grid), np.log(e), 1)[0])
    s_luce, s_gk = slope(err_luce), slope(err_gk)

    rows = [("identity_max_err", f"{max(identity_err):.3e}"),
            ("rowsum_max_err", f"{max(rowsum_err):.3e}"),
            ("luce_slope", f"{s_luce:.4f}"),
            ("gk_slope", f"{s_gk:.4f}"),
            ("luce_err_at_eps0.005", f"{err_luce[-1]:.3e}"),
            ("gk_err_at_eps0.005", f"{err_gk[-1]:.3e}")]
    with open(HERE / "results.csv", "w") as fh:
        fh.write("quantity,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")
    for k, v in rows:
        print(f"{k:24s} {v}")

    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.loglog(eps_grid, err_luce, "o-", label=f"Luce (IIA), order {s_luce:.2f}")
    ax.loglog(eps_grid, err_gk, "s-", label=f"Green--Kubo, order {s_gk:.2f}")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"max error in $M_{ij}$")
    ax.set_title("Predicting the runner-up kernel")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "runner_up_orders.png", dpi=150)


if __name__ == "__main__":
    main()
