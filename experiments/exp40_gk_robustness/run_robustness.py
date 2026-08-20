"""Experiment 40: robustness of the Green-Kubo race expansion.

Experiment 7 established the expansion on one chain. This one answers the
questions a referee asks next, and commits the independent check that was
previously only recorded in a commit message.

Part A. REPLICATION. Twenty random environments, varying the number of hidden
states, the number of channels, and the dispersion of the intensities. Report
the distribution of measured convergence orders rather than a single instance.

Part B. THE CORRELATION MUST BE DYNAMICAL. The paper claims K is a Green-Kubo
object, meaning the time integral matters and equal-time covariance is not
enough. Test it: replace K by the equal-time covariance scaled by any single
timescale and check that the order gain disappears. The best possible scalar
timescale is fitted, so the ablation is given every advantage.

Part C. RANK BOUND. Rank-r loadings give rank(K) <= r, but there is a second
cap: deviations live in the mean-zero subspace of the environment, so
rank(K) <= min(N, m-1) whatever the loadings. Verify both.

Part D. NON-REVERSIBLE INDEX PLACEMENT. K is asymmetric off reversibility, so
K_ji and K_ij differ and the theorem's index order is testable. Check the
correction against brute-force time integration of the covariance, and confirm
that swapping the index order breaks the order gain.

Run:  python experiments/exp40_gk_robustness/run_robustness.py   (~30 s)
Outputs: results.csv, figures/replication.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
EPS_GRID = np.array([0.08, 0.04, 0.02, 0.01, 0.005])


def environment(rng, m_states, n_channels, dispersion):
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
    return lam_bar, K, lam_t


def exact_shares(L, pi, lam, A, eps):
    u = np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)
    return pi @ u


def predicted(lam_bar, Kmat, A, eps):
    lb = lam_bar[A]
    Lb = lb.sum()
    soft = lb / Lb
    KA = Kmat[np.ix_(A, A)]
    return soft, soft - (eps / Lb) * (KA.sum(axis=0) - soft * KA.sum())


def order(errs):
    return float(np.polyfit(np.log(EPS_GRID), np.log(errs), 1)[0])


def orders_for(L, pi, lam, K, lam_bar, A, Kalt=None):
    e_soft, e_gk = [], []
    for eps in EPS_GRID:
        p = exact_shares(L, pi, lam, A, eps)
        s, g = predicted(lam_bar, K if Kalt is None else Kalt, A, eps)
        e_soft.append(np.abs(p - s).max())
        e_gk.append(np.abs(p - g).max())
    return order(e_soft), order(e_gk), e_gk


def main() -> None:
    rows = []

    # ---- Part A: replication over 20 environments --------------------------
    soft_orders, gk_orders = [], []
    rng_master = np.random.default_rng(40)
    for trial in range(20):
        m = int(rng_master.integers(4, 10))
        n = int(rng_master.integers(3, 9))
        disp = float(rng_master.uniform(0.3, 1.2))
        L, pi, lam = environment(rng_master, m, n, disp)
        lam_bar, K, _ = kubo(L, pi, lam)
        A = list(range(n))
        s0, s1, _ = orders_for(L, pi, lam, K, lam_bar, A)
        soft_orders.append(s0)
        gk_orders.append(s1)
    soft_orders = np.array(soft_orders)
    gk_orders = np.array(gk_orders)
    gain = gk_orders - soft_orders
    rows += [("replication_trials", "20"),
             ("softmax_order_median", f"{np.median(soft_orders):.4f}"),
             ("softmax_order_min", f"{soft_orders.min():.4f}"),
             ("softmax_order_max", f"{soft_orders.max():.4f}"),
             ("gk_order_median", f"{np.median(gk_orders):.4f}"),
             ("gk_order_min", f"{gk_orders.min():.4f}"),
             ("gk_order_max", f"{gk_orders.max():.4f}"),
             ("order_gain_min", f"{gain.min():.4f}"),
             ("order_gain_median", f"{np.median(gain):.4f}")]

    # ---- Part B: the correlation must be dynamical -------------------------
    L, pi, lam = environment(np.random.default_rng(11), 6, 5, 0.8)
    lam_bar, K, lam_t = kubo(L, pi, lam)
    A = list(range(len(lam)))
    C_static = np.array([[pi @ (lam_t[j] * lam_t[k]) for k in range(len(lam))]
                         for j in range(len(lam))])
    # give the ablation its best shot: fit the single timescale that minimises
    # the error at the smallest eps
    taus = np.logspace(-2, 1, 200)
    best_tau, best_err = None, np.inf
    for tau in taus:
        _, _, e = orders_for(L, pi, lam, K, lam_bar, A, Kalt=tau * C_static)
        if e[-1] < best_err:
            best_err, best_tau = e[-1], tau
    s_true, g_true, e_true = orders_for(L, pi, lam, K, lam_bar, A)
    s_ab, g_ab, e_ab = orders_for(L, pi, lam, K, lam_bar, A,
                                  Kalt=best_tau * C_static)
    rows += [("ablation_best_tau", f"{best_tau:.4f}"),
             ("ablation_static_order", f"{g_ab:.4f}"),
             ("true_gk_order", f"{g_true:.4f}"),
             ("ablation_err_ratio_at_min_eps", f"{e_ab[-1] / e_true[-1]:.1f}")]

    # ---- Part C: rank bounds ----------------------------------------------
    rng = np.random.default_rng(3)
    m, n = 6, 10
    L, pi, _ = environment(rng, m, n, 0.5)
    ranks = []
    for r in range(1, 6):
        B = rng.normal(0, 0.6, (n, r))
        z = rng.normal(0, 1.0, (r, m))
        z = z - (z @ pi)[:, None]
        dev_part = B @ z
        # keep intensities strictly positive without disturbing the rank of
        # the deviation, by lifting with a constant
        lam_lr = dev_part - dev_part.min() + 0.5
        assert lam_lr.min() > 0
        _, Kr, _ = kubo(L, pi, lam_lr)
        sv = np.linalg.svd(Kr, compute_uv=False)
        ranks.append(int((sv > 1e-10 * max(sv.max(), 1e-300)).sum()))
    # generic (full-dispersion) intensities: rank capped by m-1, not by N
    _, Kfull, _ = kubo(L, pi, np.exp(rng.normal(0, 0.8, (n, m))))
    sv_full = np.linalg.svd(Kfull, compute_uv=False)
    generic_rank = int((sv_full > 1e-10 * sv_full.max()).sum())
    rows += [("lowrank_ranks_r1_to_r5", " ".join(map(str, ranks))),
             ("generic_rank_N10_m6", str(generic_rank)),
             ("generic_rank_bound_min_N_mminus1", str(min(n, m - 1)))]

    # ---- Part D: non-reversible index placement ---------------------------
    L, pi, lam = environment(np.random.default_rng(2026), 6, 5, 0.7)
    lam_bar, K, lam_t = kubo(L, pi, lam)
    A = list(range(len(lam)))
    asym = np.abs(K - K.T).max() / np.abs(K).max()

    def brute(j, k, T=60.0, n_steps=20000):
        dt = T / n_steps
        step = expm(L * dt)
        P = np.eye(len(pi))
        vals = np.empty(n_steps)
        for a in range(n_steps):
            vals[a] = pi @ (lam_t[j] * (P @ lam_t[k]))
            P = P @ step
        return float(np.trapezoid(vals, dx=dt))

    bf_err = max(abs(K[j, k] - brute(j, k)) for j, k in [(0, 2), (1, 3), (4, 0)])
    _, g_right, _ = orders_for(L, pi, lam, K, lam_bar, A)
    _, g_wrong, _ = orders_for(L, pi, lam, K, lam_bar, A, Kalt=K.T)
    rows += [("K_relative_asymmetry", f"{asym:.4f}"),
             ("K_vs_brute_force_max_err", f"{bf_err:.3e}"),
             ("order_correct_index", f"{g_right:.4f}"),
             ("order_transposed_index", f"{g_wrong:.4f}")]

    with open(HERE / "results.csv", "w") as fh:
        fh.write("quantity,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")
    for k, v in rows:
        print(f"{k:34s} {v}")

    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(soft_orders, gk_orders, s=28)
    ax.axhline(2, ls="--", c="grey", lw=0.8)
    ax.axvline(1, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("measured order, softmax law")
    ax.set_ylabel("measured order, with Green--Kubo term")
    ax.set_title("Twenty random environments")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "replication.png", dpi=150)


if __name__ == "__main__":
    main()
