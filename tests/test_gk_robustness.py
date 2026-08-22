"""Locks in experiment 40: replication, the dynamical-correlation ablation,
the rank bounds, and non-reversible index placement."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                      / "exp40_gk_robustness"))
import run_robustness as rb  # noqa: E402


def test_order_gain_holds_across_random_environments():
    """Every environment should gain a full order, not just the committed one."""
    rng = np.random.default_rng(40)
    gains = []
    for _ in range(6):
        m = int(rng.integers(4, 10))
        n = int(rng.integers(3, 9))
        L, pi, lam = rb.environment(rng, m, n, float(rng.uniform(0.3, 1.2)))
        lam_bar, K, _ = rb.kubo(L, pi, lam)
        s0, s1, _ = rb.orders_for(L, pi, lam, K, lam_bar, list(range(n)))
        assert 0.8 < s0 < 1.2, s0
        assert 1.75 < s1 < 2.2, s1
        gains.append(s1 - s0)
    assert min(gains) > 0.85


def test_equal_time_covariance_does_not_gain_the_order():
    """The correction needs the TIME INTEGRAL, not the static covariance.

    The static ablation is handed its best possible scalar timescale and still
    fails to reach second order.
    """
    L, pi, lam = rb.environment(np.random.default_rng(11), 6, 5, 0.8)
    lam_bar, K, lam_t = rb.kubo(L, pi, lam)
    n = len(lam)
    A = list(range(n))
    C = np.array([[pi @ (lam_t[j] * lam_t[k]) for k in range(n)] for j in range(n)])
    best = min((rb.orders_for(L, pi, lam, K, lam_bar, A, Kalt=tau * C)[2][-1], tau)
               for tau in np.logspace(-2, 1, 60))
    _, order_static, _ = rb.orders_for(L, pi, lam, K, lam_bar, A,
                                       Kalt=best[1] * C)
    _, order_true, err_true = rb.orders_for(L, pi, lam, K, lam_bar, A)
    assert order_static < 1.4, order_static
    assert order_true > 1.8, order_true
    assert best[0] > 3 * err_true[-1]


def test_rank_bounds():
    """rank(K) = r for rank-r loadings, capped by the environment's dimension."""
    rng = np.random.default_rng(3)
    m, n = 6, 10
    L, pi, _ = rb.environment(rng, m, n, 0.5)
    for r in range(1, m):
        B = rng.normal(0, 0.6, (n, r))
        z = rng.normal(0, 1.0, (r, m))
        z = z - (z @ pi)[:, None]
        d = B @ z
        _, K, _ = rb.kubo(L, pi, d - d.min() + 0.5)
        sv = np.linalg.svd(K, compute_uv=False)
        assert int((sv > 1e-10 * sv.max()).sum()) == r
    # generic loadings are capped by m-1, not by N
    _, Kfull, _ = rb.kubo(L, pi, np.exp(rng.normal(0, 0.8, (n, m))))
    sv = np.linalg.svd(Kfull, compute_uv=False)
    assert int((sv > 1e-10 * sv.max()).sum()) == min(n, m - 1)


def test_transposing_K_breaks_the_correction():
    """K is asymmetric off reversibility, so the index order in the theorem
    is a real claim and not a convention."""
    L, pi, lam = rb.environment(np.random.default_rng(2026), 6, 5, 0.7)
    lam_bar, K, _ = rb.kubo(L, pi, lam)
    A = list(range(len(lam)))
    assert np.abs(K - K.T).max() / np.abs(K).max() > 1e-3
    _, right, _ = rb.orders_for(L, pi, lam, K, lam_bar, A)
    _, wrong, _ = rb.orders_for(L, pi, lam, K, lam_bar, A, Kalt=K.T)
    assert right > 1.8
    assert wrong < right - 0.2


def test_general_start_restores_second_order():
    """From a non-stationary start the stationary formula loses an order, and
    the general form recovers it; the extra term vanishes under pi."""
    L, pi, lam = rb.environment(np.random.default_rng(99), 6, 4, 0.6)
    lam_bar, K, lam_t = rb.kubo(L, pi, lam)
    n, m = len(lam), len(pi)
    A = list(range(n))
    Pi = np.outer(np.ones(m), pi)
    dev = lambda g: np.linalg.solve(Pi - L, g - pi @ g)
    Lb = lam_bar.sum()
    soft = lam_bar / Lb
    m_const = -(K.sum(axis=0) - soft * K.sum()) / Lb
    LamT = lam_t.sum(0)

    def orders(mu0):
        extra = np.array([-(mu0 @ dev(soft[i] * LamT - lam_t[i])) for i in range(n)])
        e_s, e_g = [], []
        for eps in rb.EPS_GRID:
            u = np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)
            p0 = mu0 @ u
            e_s.append(np.abs(p0 - (soft + eps * m_const)).max())
            e_g.append(np.abs(p0 - (soft + eps * (m_const + extra))).max())
        return rb.order(e_s), rb.order(e_g), np.abs(extra).max()

    s_stat, g_stat, extra_pi = orders(pi)
    assert extra_pi < 1e-14                      # vanishes in equilibrium
    assert g_stat > 1.8

    s_pt, g_pt, extra_pt = orders(np.eye(m)[0])
    assert extra_pt > 1e-3                       # genuinely present off equilibrium
    assert s_pt < 1.3, s_pt                      # stationary formula loses the order
    assert g_pt > 1.8, g_pt                      # general formula restores it


def test_proportional_hazards_is_exact_pointwise_under_stress():
    """Prop 1 needs no positivity of c and no bound on eps, and holds for
    every start state, not only in equilibrium."""
    rng = np.random.default_rng(5150)
    L, pi, _ = rb.environment(rng, 6, 5, 0.5)
    a = rng.uniform(0.5, 2.0, 5)
    A = [0, 1, 3]
    luce = a[A] / a[A].sum()
    for cvec in (rng.uniform(0.4, 3.0, 6),
                 np.array([0.0, 0.0, 0.0, 0.0, 1.3, 0.7]),
                 np.array([1e-9, 1e-5, 1e-2, 1.0, 1e3, 1e9])):
        lam = a[:, None] * cvec[None, :]
        for eps in (1e-3, 0.3, 3.0, 100.0):
            u = np.linalg.solve(L / eps - np.diag(lam[A].sum(0)), -lam[A].T)
            assert np.abs(u - luce[None, :]).max() < 1e-12


def test_mode_count_estimator():
    """exp42: the driver's mode count is recovered exactly, and the fat set
    explains the rank anomaly (whole t-free family has rank r+1)."""
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                          / "exp42_mode_count"))
    mc = importlib.import_module("run_mode_count")
    rng = np.random.default_rng(5)
    for r in (1, 3):
        L, pi, lam = mc.environment(rng, 9, 7, r)
        lam_bar, K = mc.kubo(L, pi, lam)
        K_obs = K + np.outer(rng.normal(size=7), lam_bar) + 0.4 * np.diag(lam_bar)
        t_hat, M = mc.fat_solve(K_obs, lam_bar, r + 1)
        assert abs(t_hat + 0.4) < 1e-6
        r_hat, sv = mc.mode_count(M, lam_bar)
        assert r_hat == r
        # the fat set: random members of the t-free family have rank r+1
        svf = np.linalg.svd(K + np.outer(rng.normal(size=7), lam_bar),
                            compute_uv=False)
        assert int((svf > 1e-10 * svf.max()).sum()) == r + 1


def test_remainder_bound_dominates():
    """The explicit constant eps^2 ||Lam u2||/Lam_min bounds the true error."""
    rng = np.random.default_rng(404)
    L, pi, _ = rb.environment(rng, 7, 4, 0.0)
    lam = rng.uniform(0.4, 2.0, (4, 7))
    m = 7
    Pi = np.outer(np.ones(m), pi)
    dev = lambda g: np.linalg.solve(Pi - L, g - pi @ g)
    lb = lam @ pi; Lb = lb.sum(); c = lb / Lb
    lt = lam - lb[:, None]; Lam = lam.sum(0)
    u1t = np.column_stack([-dev(c[i]*(Lam-Lb) - lt[i]) for i in range(4)])
    m1 = -np.array([pi @ (Lam*u1t[:, i]) for i in range(4)]) / Lb
    u1 = u1t + m1
    u2t = np.column_stack([-dev(Lam*u1[:, i]) for i in range(4)])
    m2 = -np.array([pi @ (Lam*u2t[:, i]) for i in range(4)]) / Lb
    u2 = u2t + m2
    C = np.abs(Lam[:, None]*u2).max() / Lam.min()
    for eps in (0.2, 0.05, 0.01):
        u = np.linalg.solve(L/eps - np.diag(Lam), -lam.T)
        U = c[None, :] + eps*u1 + eps**2*u2
        # defect is exactly -eps^2 Lam u2
        defect = (L/eps) @ U - Lam[:, None]*U + lam.T
        assert np.abs(defect + eps**2 * (Lam[:, None] * u2)).max() < 1e-9
        assert np.abs(u - U).max() <= C * eps**2


def test_rate_gauge_is_exact_and_complete():
    """(lam, K) -> (lam + eps*eta, K - diag(eta)) cancels exactly on every
    subset, and {d lam^T + diag(v)} is the entire nuisance-rate null space."""
    import itertools
    rng = np.random.default_rng(6)
    n = 6
    lb = rng.uniform(0.5, 2.0, n); lb /= lb.sum()

    def drows(A):
        Lb = lb[A].sum(); c = lb[A]/Lb
        out = []
        for pos in range(len(A)):
            g = np.zeros((n, n))
            for j in A: g[j, A[pos]] += 1.0
            for j in A:
                for k in A: g[j, k] -= c[pos]
            out.append(-(1.0/Lb)*g.ravel())
        return np.array(out)

    def jc(A):
        S = lb[A].sum()
        J = np.zeros((len(A), n))
        for pos, j in enumerate(A):
            for l in A:
                J[pos, l] = (1.0*(l == j))/S - lb[j]/S**2
        return J

    D_full = drows(list(range(n)))
    allsub = [list(c) for k in range(2, n+1)
              for c in itertools.combinations(range(n), k)]
    eta = rng.normal(size=n)
    for A in allsub:
        assert np.abs(drows(A) @ np.diag(eta).ravel() + jc(A) @ eta).max() < 1e-12
    Dm = np.vstack([drows(A) - jc(A) @ D_full for A in allsub])
    F = np.array([np.outer(np.eye(n)[a], lb).ravel() for a in range(n)]
                 + [np.diag(np.eye(n)[a]).ravel() for a in range(n)]).T
    assert n*n - np.linalg.matrix_rank(Dm, tol=1e-9) == 2*n
    assert np.linalg.matrix_rank(F) == 2*n
    assert np.linalg.norm(Dm @ F) < 1e-12


def test_onsager_reciprocity_and_witness():
    """Reversible driver => K symmetric; the (N-1)(N-2)/2 gauge-invariant
    asymmetry functionals vanish under reversibility, are class-invariant,
    and the N=3 witness responds to driving."""
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                          / "exp43_onsager"))
    on = importlib.import_module("run_onsager")
    rng = np.random.default_rng(21)
    m = 6
    for N in (3, 4):
        lam = rng.uniform(0.4, 2.0, (N, m))
        L, pi = on.reversible_generator(np.random.default_rng(N), m)
        lb, K = on.kubo(L, pi, lam)
        assert np.abs(K - K.T).max() / np.abs(K).max() < 1e-12
        vals, rank_g, n_inv = on.invariants(K, lb)
        assert rank_g == N - 1
        assert n_inv == (N - 1) * (N - 2) // 2
        assert np.abs(vals).max() < 1e-14
    N = 3
    lam = rng.uniform(0.4, 2.0, (N, m))
    L, pi = on.driven_generator(np.random.default_rng(55), m, 1.0)
    lb, K = on.kubo(L, pi, lam)
    v_driven, _, _ = on.invariants(K, lb)
    assert np.abs(v_driven).max() > 1e-4
    rng2 = np.random.default_rng(0)
    for _ in range(5):
        Kp = K + np.outer(rng2.normal(size=N), lb) + np.diag(rng2.normal(size=N))
        v1, _, _ = on.invariants(Kp, lb)
        assert np.abs(v1 - v_driven).max() < 1e-12


def test_K_is_the_clock_fluctuation_covariance():
    """The compensators Theta_i(t) = int lam_i(Y_s) ds satisfy a joint CLT
    whose covariance is the symmetrized K: Cov(Theta_j, Theta_k)/t ->
    eps*(K_jk + K_kj). Checked against the exact finite-t double integral,
    non-reversible chain, at two eps values to confirm the eps scaling."""
    rng = np.random.default_rng(3)
    m, n = 6, 3
    L, pi, lam = rb.environment(rng, m, n, 0.7)
    lam_bar, K, lam_t = rb.kubo(L, pi, lam)
    evals, R = np.linalg.eig(L)
    Rinv = np.linalg.inv(R)

    def J(kappa, t):
        # int_0^t (t - s) e^{kappa s} ds
        if abs(kappa) < 1e-13:
            return t * t / 2
        return (np.exp(kappa * t) - 1) / kappa**2 - t / kappa

    t = 100.0
    for eps in (0.05, 0.01):
        C = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                ajk = ((pi * lam_t[j]) @ R) * (Rinv @ lam_t[k])
                akj = ((pi * lam_t[k]) @ R) * (Rinv @ lam_t[j])
                C[j, k] = np.real(sum((ajk[l] + akj[l]) * J(evals[l] / eps, t)
                                      for l in range(m)))
        target = eps * (K + K.T)
        rel = np.abs(C / t - target).max() / np.abs(target).max()
        assert rel < 25 * eps / t, rel   # finite-t tail is O(eps^2/t)


def test_shared_clock_is_one_null_direction():
    """Common-mode rates give K exactly proportional to lbar lbar^T, the
    d = gamma*lbar member of the invisible family; every subset's first-order
    bracket annihilates it."""
    import itertools

    rng = np.random.default_rng(9)
    m, n = 6, 5
    L, pi, _ = rb.environment(rng, m, n, 0.7)
    a = rng.uniform(0.5, 2.0, n)
    c = np.exp(rng.normal(0, 0.7, m))
    lam = np.outer(a, c)
    lam_bar, K, _ = rb.kubo(L, pi, lam)
    P = np.outer(lam_bar, lam_bar)
    coef = (K * P).sum() / (P * P).sum()
    assert np.linalg.norm(K - coef * P) < 1e-12 * np.linalg.norm(K)
    worst = 0.0
    for r in range(2, n + 1):
        for A in itertools.combinations(range(n), r):
            A = list(A)
            cA = lam_bar[A] / lam_bar[A].sum()
            for pos, i in enumerate(A):
                B = (sum(P[j, i] for j in A)
                     - cA[pos] * sum(P[j, k] for j in A for k in A))
                worst = max(worst, abs(B))
    assert worst < 1e-12


def test_mode_count_in_the_2N_gauge_class():
    """With rates estimated too, the class gains an arbitrary diagonal.
    Diagonal-avoiding blocks still count the modes: the largest block with
    disjoint rows and columns has rank min(r+1, floor(N/2)), so rank minus
    one recovers r below the detection floor and reports the floor above."""
    import itertools

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                          / "exp42_mode_count"))
    import run_mode_count as mc

    m, N = 12, 10
    h = N // 2
    for r in (1, 3, 4, 6):
        rng = np.random.default_rng(100 + r)
        L, pi, lam = mc.environment(rng, m, N, r)
        lam_bar, K = mc.kubo(L, pi, lam)
        M = (K + np.outer(rng.normal(size=N), lam_bar)
             + np.diag(rng.normal(size=N)))
        best = 0
        for R in itertools.combinations(range(N), h):
            C = [c for c in range(N) if c not in R]
            sv = np.linalg.svd(M[np.ix_(R, C)], compute_uv=False)
            best = max(best, int((sv > 1e-9 * sv.max()).sum()))
        assert best - 1 == min(r, h - 1)
