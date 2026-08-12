"""Experiment 7 (program Q7, the flagship): softmax as homogenized kinetics, and the
Green-Kubo correction -- verified exactly on a finite-state chain.

Model. A hidden environment Y_t is a CTMC on m states with generator L / eps
(fast mixing as eps -> 0), invariant law pi. Channel i fires with intensity
lambda_i(Y_t) >= 0; the first channel to fire wins. For an available set A, the win
probability from state y solves the killed-resolvent equation

    u_i^A = (Lambda_A - L/eps)^{-1} lambda_i,        Lambda_A = diag(sum_{j in A} lambda_j)

and p_i^A = pi . u_i^A from a stationary start. On a finite chain everything is an
exact linear solve, so the two claims of the theorem can be tested to machine
precision:

  (1) LEADING ORDER (softmax):   p_i^A -> lbar_i / Lbar_A,  lbar_i = pi . lambda_i,
      with error O(eps).
  (2) GREEN-KUBO CORRECTION: with K_jk = int_0^inf Cov_pi(lambda_j(Y_0),
      lambda_k(Y_t)) dt  (computed via the deviation integral
      D = (Pi - L)^{-1}(I - Pi), K_jk = pi . (ltil_j * (D ltil_k)) ),

      p_i^A = lbar_i/Lbar_A - (eps/Lbar_A) [ sum_{j in A} K_ji
              - (lbar_i/Lbar_A) sum_{j,k in A} K_jk ] + O(eps^2).

  (3) UNIFORMITY: the same (lbar, K) predicts EVERY blocked subset A by restricting
      the sums -- one global pair answers the whole deletion ensemble.
  (4) COMMON MODE: lambda_i(y) = a_i c(y) makes the correction vanish identically.
  (5) LOW RANK: lambda deviations spanned by r environmental modes make K rank r.

A Gillespie simulation (real event-driven kinetics, hidden state included) checks the
resolvent arithmetic at one moderate eps.

Run:  python experiments/exp07_green_kubo/run_green_kubo.py
Outputs: results.csv, figures/convergence.png, figures/subsets.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
M = 6          # hidden states
N = 10         # channels
SEED = 5


def make_chain(rng):
    """Random irreducible generator L (rows sum to zero) and its invariant law."""
    Q = rng.exponential(1.0, size=(M, M))
    np.fill_diagonal(Q, 0.0)
    L = Q - np.diag(Q.sum(axis=1))
    w, V = np.linalg.eig(L.T)
    pi = np.real(V[:, np.argmin(np.abs(w))])
    pi = np.abs(pi) / np.abs(pi).sum()
    return L, pi


def green_kubo_matrix(L, pi, lam):
    """K_jk = int_0^inf Cov_pi(lambda_j(Y_0), lambda_k(Y_t)) dt, exactly.

    Uses the deviation integral D = (Pi - L)^{-1} (I - Pi) = int (e^{tL} - Pi) dt:
    K_jk = sum_y pi_y ltil_j(y) (D ltil_k)(y).
    """
    m = L.shape[0]
    Pi = np.outer(np.ones(m), pi)
    D = np.linalg.solve(Pi - L, np.eye(m) - Pi)
    ltil = lam - (lam @ pi)[:, None]              # centered intensities (N, m)
    return np.einsum("jy,y,yk->jk", ltil, pi, D @ ltil.T)


def exact_win_probs(L, pi, lam, A, eps):
    """p_i^A by direct resolvent solve, for i in A."""
    LamA = np.diag(lam[A].sum(axis=0))
    R = LamA - L / eps
    U = np.linalg.solve(R, lam[A].T)              # (M, |A|)
    return pi @ U


def gk_prediction(pi, lam, K, A):
    """Softmax leading order and the Green-Kubo coefficient, for i in A."""
    lbar = lam @ pi
    Lbar = lbar[A].sum()
    p0 = lbar[A] / Lbar
    K_Ai = K[np.ix_(A, A)].sum(axis=0)            # sum_{j in A} K_ji, i in A
    K_AA = K[np.ix_(A, A)].sum()
    c1 = -(K_Ai - p0 * K_AA) / Lbar
    return p0, c1


def gillespie(L, pi, lam, A, eps, n_traj, rng):
    """Event-driven simulation with the hidden state; returns win frequencies."""
    m = L.shape[0]
    jump_rates = -np.diag(L) / eps
    P_jump = L / eps
    np.fill_diagonal(P_jump, 0.0)
    P_jump = P_jump / P_jump.sum(axis=1, keepdims=True)
    counts = np.zeros(len(A), dtype=int)
    lamA = lam[A]
    for _ in range(n_traj):
        y = rng.choice(m, p=pi)
        while True:
            total_fire = lamA[:, y].sum()
            total = jump_rates[y] + total_fire
            if rng.random() < total_fire / total:
                counts[rng.choice(len(A), p=lamA[:, y] / total_fire)] += 1
                break
            y = rng.choice(m, p=P_jump[y])
    return counts / counts.sum()


def main():
    rng = np.random.default_rng(SEED)
    L, pi = make_chain(rng)
    lam = rng.lognormal(0.0, 0.8, size=(N, M))    # channel intensities per state
    K = green_kubo_matrix(L, pi, lam)
    A_full = np.arange(N)
    rows = ["part,quantity,value"]

    # ---- (1)+(2): convergence orders on the full set ---------------------------
    eps_grid = np.logspace(-3, 0, 13)
    e0, e1 = [], []
    p0, c1 = gk_prediction(pi, lam, K, A_full)
    for eps in eps_grid:
        p = exact_win_probs(L, pi, lam, A_full, eps)
        e0.append(np.abs(p - p0).max())
        e1.append(np.abs(p - p0 - eps * c1).max())
    s0 = np.polyfit(np.log(eps_grid[:6]), np.log(e0[:6]), 1)[0]
    s1 = np.polyfit(np.log(eps_grid[:6]), np.log(e1[:6]), 1)[0]
    print(f"convergence to softmax:      slope {s0:.3f}  (theory: 1)")
    print(f"after Green-Kubo correction: slope {s1:.3f}  (theory: 2)")
    rows += [f"1,softmax_slope,{s0:.4f}", f"2,gk_slope,{s1:.4f}"]

    # ---- (3): one (lbar, K) answers every blocked subset ------------------------
    eps = 0.05
    subs_err0, subs_err1 = [], []
    subsets = [A_full] + [np.sort(rng.choice(N, size=s, replace=False))
                          for s in rng.integers(3, N, size=40)]
    for A in subsets:
        p = exact_win_probs(L, pi, lam, A, eps)
        q0, qc = gk_prediction(pi, lam, K, A)
        subs_err0.append(np.abs(p - q0).max())
        subs_err1.append(np.abs(p - q0 - eps * qc).max())
    print(f"\n{len(subsets)} blocked subsets at eps={eps}: "
          f"max |exact - softmax| = {max(subs_err0):.2e}, "
          f"max |exact - GK|      = {max(subs_err1):.2e}")
    rows += [f"3,max_softmax_err,{max(subs_err0):.3e}",
             f"3,max_gk_err,{max(subs_err1):.3e}"]

    # ---- (4): common-mode cancellation ------------------------------------------
    a = rng.lognormal(0.0, 1.0, N)
    c = rng.lognormal(0.0, 0.8, M)
    lam_cm = np.outer(a, c)
    K_cm = green_kubo_matrix(L, pi, lam_cm)
    A = np.arange(N)
    _, c1_cm = gk_prediction(pi, lam_cm, K_cm, A)
    p_cm = exact_win_probs(L, pi, lam_cm, A, 0.3)
    lbar_cm = lam_cm @ pi
    print(f"\ncommon mode lambda_i = a_i c(y): |GK coefficient| = "
          f"{np.abs(c1_cm).max():.2e}; |exact - softmax| at eps=0.3 = "
          f"{np.abs(p_cm - lbar_cm[A] / lbar_cm[A].sum()).max():.2e}")
    rows += [f"4,common_mode_gk_coeff,{np.abs(c1_cm).max():.3e}"]

    # ---- (5): low-rank environmental loadings -----------------------------------
    r = 2
    # positive by construction (clipping would break the low-rank structure)
    B = rng.uniform(-0.4, 0.4, size=(N, r)) / r
    z = rng.uniform(-1.0, 1.0, size=(r, M))
    lam_lr = 1.0 + B @ z
    K_lr = green_kubo_matrix(L, pi, lam_lr)
    sv = np.linalg.svd(K_lr, compute_uv=False)
    print(f"low-rank loadings (r={r}): K singular values {sv[:4].round(6)} "
          f"-> rank {int(np.sum(sv > 1e-10 * sv[0]))}")
    rows += [f"5,K_rank,{int(np.sum(sv > 1e-10 * sv[0]))}"]

    # ---- Gillespie check ----------------------------------------------------------
    t0 = time.perf_counter()
    A = np.sort(rng.choice(N, size=6, replace=False))
    freq = gillespie(L, pi, lam, A, 0.3, 40_000, rng)
    p_exact = exact_win_probs(L, pi, lam, A, 0.3)
    mc_noise = np.sqrt(p_exact.max() / 40_000)
    print(f"\nGillespie (40k trajectories, eps=0.3, {time.perf_counter()-t0:.0f}s): "
          f"max |sim - resolvent| = {np.abs(freq - p_exact).max():.2e} "
          f"(MC noise ~{mc_noise:.0e})")
    rows += [f"6,gillespie_err,{np.abs(freq - p_exact).max():.3e}"]

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---- figures --------------------------------------------------------------------
    fig_dir = HERE / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(eps_grid, e0, "o-", color="#2a1a12", label=f"|exact − softmax|  (slope {s0:.2f})")
    ax.loglog(eps_grid, e1, "s-", color="#c2410c",
              label=f"|exact − softmax − ε·GK|  (slope {s1:.2f})")
    ax.set_xlabel("time-scale separation ε")
    ax.set_ylabel("max abs win-probability error")
    ax.set_title("Softmax is the homogenized choice law;\n"
                 "the Green–Kubo term is the exact first correction", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(fig_dir / "convergence.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(5.2, 4.2))
    ax2.plot(subs_err0, subs_err1, ".", color="#c2410c")
    lim = max(subs_err0)
    ax2.plot([0, lim], [0, lim], ":", color="#9a9a9a", label="no improvement")
    ax2.set_xlabel("softmax-only error per subset")
    ax2.set_ylabel("error after Green–Kubo correction")
    ax2.set_title(f"One (λ̄, K) pair corrects all {len(subsets)} blocked subsets "
                  f"(ε = {eps})", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout(); fig2.savefig(fig_dir / "subsets.png", dpi=150)
    print("\nwrote results.csv, figures/convergence.png, figures/subsets.png")


if __name__ == "__main__":
    main()
