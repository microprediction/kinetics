"""Experiment 44: estimating K from the exacta board alone.

Closes the estimation gap. Experiment 41 fit K from blocked-subset
interventions with known rates. Here nothing is known and nothing is blocked.
The input is the exacta board of the full set, the N(N-1) joint probabilities
of (winner, runner-up), and the rates lam_bar are a nuisance eliminated by
substitution.

The chain of exact facts behind the estimator:

  (i)   The bridge q_j^(-i) = p_j + p_i M_ij (Experiment 39, exact at every
        eps) converts the board into every leave-one-out share without
        performing a single intervention.
  (ii)  Winner shares give lam_bar = p - D_full k + O(eps^2), where
        k = vec(eps K), so the nuisance is eliminated by linearizing each
        subset's softmax at p. The effective design is
        D_eff^(i) = D_rest - Jc D_full.
  (iii) After elimination the identified class is K modulo the 2N-dimensional
        family {d lam_bar^T + diag(v)} of the rate-gauge proposition, and the
        design rank is exactly N^2 - 2N, the saturation value. The fit is
        least squares on the orthocomplement of that family.
  (iv)  The board carries no time unit: shares are invariant under
        (lam, K) -> (s lam, s K), a global rescaling of the clock. The
        estimator therefore returns eps*K in the units where sum(lam_bar)=1,
        and comparisons must normalize the reference the same way.

An earlier version of this pipeline appeared to be missing a first-order term
(direction cosine ~0.85 to truth, independent of eps). It was not missing a
term; it was missing two gauges. It was being compared against a target containing the diag(v) gauge
component, which no choice data determines. In the gauge-invariant projection
the recovery is exact to the order the expansion carries.

Parts:
  A. Exact linear-model boards: residual of the effective design is
     O(eps^2), rank is N^2 - 2N, gauge-projected cosine -> 1 as eps^2.
  B. Exact boards from the Markov model (resolvent, no sampling): recovery
     error of the identified part is O(eps), the order the first-order
     model permits.
  C. Common-mode degenerate case: the identified part of K is zero and the
     estimator returns zero, at every eps.
  D. Sampled boards: multinomial exacta counts, error vs number of races.

Run:  python experiments/exp44_exacta_estimator/run_exacta_estimator.py  (~30 s)
Outputs: results.csv
"""

from __future__ import annotations

from pathlib import Path

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


def design_rows(lb, N, A):
    """Rows of the map vec(eps*K) -> first-order share deviation on A."""
    Lb = lb[A].sum()
    c = lb[A] / Lb
    rows = []
    for pos in range(len(A)):
        g = np.zeros((N, N))
        for j in A:
            g[j, A[pos]] += 1.0
        for j in A:
            for k in A:
                g[j, k] -= c[pos]
        rows.append(-(1.0 / Lb) * g.ravel())
    return np.array(rows), c


def exact_board(L, pi, lam, eps):
    """Joint exacta probabilities P[i, j] = P(first = i, second = j), from the
    bridge q_j^(-i) = p_j + p_i M_ij run in reverse. Exact at every eps."""
    N = len(lam)
    full = list(range(N))
    p = exact_shares(L, pi, lam, full, eps)
    P = np.zeros((N, N))
    for i in full:
        rest = [a for a in full if a != i]
        q = exact_shares(L, pi, lam, rest, eps)
        P[i, rest] = q - p[rest]                 # p_i * M_ij, by the bridge
    return p[:, None] * (P / P.sum(1, keepdims=True))


def gauge_projector(lb, N):
    """Orthoprojector onto the complement of {d lb^T + diag(v)}, dim 2N."""
    fam = ([np.outer(np.eye(N)[a], lb).ravel() for a in range(N)]
           + [np.diag(np.eye(N)[a]).ravel() for a in range(N)])
    Q, _ = np.linalg.qr(np.array(fam).T)
    return np.eye(N * N) - Q @ Q.T


def fit_from_board(board, N):
    """Estimate the gauge-projected vec(eps*K) from an exacta board."""
    p = board.sum(1)
    M = board / np.maximum(p[:, None], 1e-300)
    full = list(range(N))
    D_full, _ = design_rows(p, N, full)
    rows_d, rhs = [], []
    for i in full:
        rest = [a for a in full if a != i]
        q = p + p[i] * M[i]
        D_rest, c = design_rows(p, N, rest)
        Jc = np.zeros((len(rest), N))
        for pos, j in enumerate(rest):
            Jc[pos, j] = 1.0 / (1 - p[i])
            Jc[pos, i] = p[j] / (1 - p[i]) ** 2
        rows_d.append(D_rest - Jc @ D_full)
        rhs.append(q[rest] - c)
    Dm = np.vstack(rows_d)
    y = np.concatenate(rhs)
    P2 = gauge_projector(p, N)
    sol, *_ = np.linalg.lstsq(Dm @ P2, y, rcond=None)
    return P2 @ sol, P2, np.linalg.matrix_rank(Dm, tol=1e-11)


def main() -> None:
    rows = []
    N, m = 6, 7
    full = list(range(N))

    # ---- Part A: exact linear-model boards --------------------------------
    rng = np.random.default_rng(5)
    lb_t = rng.uniform(0.5, 2.0, N)
    lb_t /= lb_t.sum()
    K_t = rng.normal(0, 1.0, (N, N))
    D_full_t, _ = design_rows(lb_t, N, full)
    for eps in (0.02, 0.01, 0.005):
        k = eps * K_t.ravel()
        p = lb_t + D_full_t @ k
        board = np.zeros((N, N))
        for i in full:
            rest = [a for a in full if a != i]
            D_rest_t, c_rest_t = design_rows(lb_t, N, rest)
            q = c_rest_t + D_rest_t @ k
            board[i, rest] = q - p[rest]
        board = p[:, None] * (board / board.sum(1, keepdims=True))
        k_hat, P2, rank = fit_from_board(board, N)
        k_ref = P2 @ k
        cos = k_hat @ k_ref / np.linalg.norm(k_hat) / np.linalg.norm(k_ref)
        rows += [(f"A_eps{eps}_rank", str(rank)),
                 (f"A_eps{eps}_cosine", f"{cos:.6f}"),
                 (f"A_eps{eps}_one_minus_cos_over_eps2",
                  f"{(1 - cos) / eps**2:.3f}")]

    # ---- Part B: exact boards from the Markov model -----------------------
    rng = np.random.default_rng(44)
    L, pi, lam = environment(rng, m, N)
    lam_bar, K = kubo(L, pi, lam)
    for eps in (0.04, 0.02, 0.01):
        board = exact_board(L, pi, lam, eps)
        k_hat, P2, rank = fit_from_board(board, N)
        k_ref = P2 @ (eps * K.ravel() / lam_bar.sum())   # time unit: sum(lb)=1
        rel = np.linalg.norm(k_hat - k_ref) / np.linalg.norm(k_ref)
        rows += [(f"B_eps{eps}_rank", str(rank)),
                 (f"B_eps{eps}_rel_err", f"{rel:.4f}"),
                 (f"B_eps{eps}_rel_err_over_eps", f"{rel / eps:.3f}")]

    # ---- Part C: common-mode degenerate case ------------------------------
    rng = np.random.default_rng(7)
    L2, pi2, _ = environment(rng, m, N)
    a = rng.uniform(0.5, 2.0, N)
    c = np.exp(rng.normal(0, 0.7, m))
    lam2 = np.outer(a, c)
    lb2, K2 = kubo(L2, pi2, lam2)
    for eps in (0.04, 0.01):
        board = exact_board(L2, pi2, lam2, eps)
        k_hat, P2, _ = fit_from_board(board, N)
        scale = eps * np.linalg.norm(K2)
        rows.append((f"C_eps{eps}_khat_over_scale",
                     f"{np.linalg.norm(k_hat) / scale:.2e}"))

    # ---- Part D: sampled boards -------------------------------------------
    eps = 0.02
    board = exact_board(L, pi, lam, eps)
    cells = board.ravel()
    k_ref = None
    for n_races in (10_000, 100_000, 1_000_000, 10_000_000):
        errs = []
        for s in range(5):
            r = np.random.default_rng(1000 + s)
            counts = r.multinomial(n_races, cells / cells.sum())
            b_hat = counts.reshape(N, N) / n_races
            k_hat, P2, _ = fit_from_board(b_hat, N)
            if k_ref is None:
                k_ref = (gauge_projector(board.sum(1), N)
                         @ (eps * K.ravel() / lam_bar.sum()))
            errs.append(np.linalg.norm(k_hat - k_ref) / np.linalg.norm(k_ref))
        rows.append((f"D_{n_races}_races_med_rel_err",
                     f"{float(np.median(errs)):.3f}"))

    with open(HERE / "results.csv", "w") as fh:
        fh.write("quantity,value\n")
        for kk, v in rows:
            fh.write(f"{kk},{v}\n")
    for kk, v in rows:
        print(f"{kk:36s} {v}")


if __name__ == "__main__":
    main()
