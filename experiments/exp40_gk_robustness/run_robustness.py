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

Part G. PROPORTIONAL HAZARDS UNDER STRESS. Proposition 1 claims lambda_i =
a_i c(y) gives Luce exactly, pointwise in the start state, for any c >= 0 and
any eps. Stress it: let c vanish on most states, span nine orders of magnitude,
and push eps to 100.

Part F. EXHAUSTIVE SUBSETS. Experiment 7 samples 41 availability sets, none of
size two. Sweep ALL subsets of size >= 2 of the committed exp07 chain instead,
so the quoted maxima are true maxima, and fit the order on every one of them.

Part E. THE START LAW. The theorem is stated from stationarity. From any other
start the leading softmax term is unchanged, but the first-order correction
picks up an extra term, so the stationary formula is only first-order accurate
there. Verify the general form restores second order, and that its extra term
vanishes under the invariant law.

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

    # ---- Part E: general start law ----------------------------------------
    L, pi, lam = environment(np.random.default_rng(99), 6, 4, 0.6)
    lam_bar, K, lam_t = kubo(L, pi, lam)
    n = len(lam)
    A = list(range(n))
    m_states = len(pi)
    Pi = np.outer(np.ones(m_states), pi)
    dev = lambda g: np.linalg.solve(Pi - L, g - pi @ g)
    Lb = lam_bar.sum()
    soft = lam_bar / Lb
    m_const = -(K.sum(axis=0) - soft * K.sum()) / Lb
    LamT = lam_t.sum(0)
    starts = [("point_mass", np.eye(m_states)[0]), ("stationary", pi)]
    for label, mu0 in starts:
        extra = np.array([-(mu0 @ dev(soft[i] * LamT - lam_t[i]))
                          for i in range(n)])
        e_stat, e_gen = [], []
        for eps in EPS_GRID:
            u = np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)
            p0 = mu0 @ u
            e_stat.append(np.abs(p0 - (soft + eps * m_const)).max())
            e_gen.append(np.abs(p0 - (soft + eps * (m_const + extra))).max())
        rows += [(f"start_{label}_stationary_form_order", f"{order(e_stat):.4f}"),
                 (f"start_{label}_general_form_order", f"{order(e_gen):.4f}"),
                 (f"start_{label}_extra_term_max", f"{np.abs(extra).max():.3e}")]

    # ---- Part F: exhaustive subset sweep on the exp07 instance -------------
    import itertools
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / "exp07_green_kubo"))
    import run_green_kubo as gk7  # noqa: E402

    rng7 = np.random.default_rng(gk7.SEED)
    L7, pi7 = gk7.make_chain(rng7)
    lam7 = rng7.lognormal(0.0, 0.8, size=(gk7.N, gk7.M))
    lam_bar7, K7, _ = kubo(L7, pi7, lam7)
    asym7 = np.abs(K7 - K7.T).max()
    rows += [("exp07_K_asymmetry_abs", f"{asym7:.4f}"),
             ("exp07_K_max_abs", f"{np.abs(K7).max():.4f}"),
             ("exp07_K_relative_asymmetry", f"{asym7 / np.abs(K7).max():.4f}")]

    idx = list(range(gk7.N))
    subsets = [list(c) for r in range(2, gk7.N + 1)
               for c in itertools.combinations(idx, r)]
    eps_ref = 0.05
    worst_soft = worst_gk = 0.0
    n_worse = 0
    ord_gk, ord_soft = [], []
    for A7 in subsets:
        p7 = exact_shares(L7, pi7, lam7, A7, eps_ref)
        s7, g7 = predicted(lam_bar7, K7, A7, eps_ref)
        es, eg = np.abs(p7 - s7).max(), np.abs(p7 - g7).max()
        worst_soft = max(worst_soft, es)
        worst_gk = max(worst_gk, eg)
        n_worse += eg > es
        so, go, _ = orders_for(L7, pi7, lam7, K7, lam_bar7, A7)
        ord_soft.append(so)
        ord_gk.append(go)
    rows += [("subsets_swept", str(len(subsets))),
             ("subset_worst_softmax_err", f"{worst_soft:.3e}"),
             ("subset_worst_gk_err", f"{worst_gk:.3e}"),
             ("subsets_where_gk_worse", str(int(n_worse))),
             ("subset_gk_order_min", f"{min(ord_gk):.4f}"),
             ("subset_gk_order_max", f"{max(ord_gk):.4f}"),
             ("subset_softmax_order_min", f"{min(ord_soft):.4f}"),
             ("subset_softmax_order_max", f"{max(ord_soft):.4f}")]

    # ---- Part G: proportional hazards under stress -------------------------
    rngc = np.random.default_rng(5150)
    Lc, pic, _ = environment(rngc, 6, 5, 0.5)
    a = rngc.uniform(0.5, 2.0, 5)
    worst = 0.0
    cases = {
        "c_positive": rngc.uniform(0.4, 3.0, 6),
        "c_zero_on_four_of_six": np.array([0.0, 0.0, 0.0, 0.0, 1.3, 0.7]),
        "c_nine_decades": np.array([1e-9, 1e-5, 1e-2, 1.0, 1e3, 1e9]),
    }
    for name, cvec in cases.items():
        lam_c = a[:, None] * cvec[None, :]
        for eps in (1e-3, 0.3, 3.0, 100.0):
            A2 = [0, 1, 3]
            u = np.linalg.solve(Lc / eps - np.diag(lam_c[A2].sum(0)), -lam_c[A2].T)
            luce = a[A2] / a[A2].sum()
            worst = max(worst, np.abs(u - luce[None, :]).max())   # pointwise in y
    rows += [("prop_hazards_worst_pointwise_dev", f"{worst:.3e}")]

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
