"""Experiment 6 (program Q6): a FAST transform for correlated fields.

Experiment 3 established that a correlated race predicts real first-passage
counterfactuals ~9x better than independent races -- but priced the race by Monte
Carlo. This experiment builds and validates a fast deterministic transform:

  Sigma ~= V V^T + diag(D)     (k factors + idiosyncratic variance, fit by
                                iterated principal-factor analysis)

  p_i = E_f [ int f_i(x|f) * S_field(x|f) / S_i(x|f) dx ],   f ~ N(0, I_k)

Conditionally on the k factors the competitors are independent, so the
multiplicative cavity (field product, divide one out) applies at every quadrature
node, O(N L) per node. The factor expectation uses product Gauss-Hermite for small
k and scrambled-Sobol QMC for larger k. The two leave-one-out identities compose:
the Gaussian/Schur side compresses the coupling into factors; the field product
prices the race. The single-deletion ENSEMBLE survives: one conditional field pass
yields P(j wins | i removed) for all (i, j), and the transform is smooth and
deterministic in mu -- which is what the inverse (fixed-point) transform needs,
and what plain Monte Carlo pricing lacks.

Relation to the thurstone package: this is `multiray` (ability = mu + Z.v per
condition) with the ray coordinate promoted to a latent Gaussian and integrated
out; the package's core transform is the independent limit (V = 0).

Findings encoded below:
  * Given the factor model, the quadrature is exact to reference-MC noise.
  * The pipeline error is DOMINATED by the factor-approximation residual
    ||Sigma_hat - Sigma||_offdiag, not by the quadrature. Smooth kernels converge
    fast in k; the exponential kernel's kink at d = 0 gives 1/m^2 eigenvalue decay
    and needs k ~ 8-12 (QMC) for ~1e-3 accuracy.
  * Two traps documented: naive eigen-truncation invents off-diagonal correlation
    (catastrophic near C = I; use factor analysis), and the squared-exponential of
    GEODESIC distance is not PSD on a circle (use the chordal version).

Run:  python experiments/exp06_fast_correlated_transform/run_fast_transform.py
Outputs: results.csv, figures/error_vs_rank.png, printed summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp01_narrow_escape"))
from raceutil import (abilities_from_probabilities_factor, factor_model,  # noqa: E402
                      hermite_nodes, qmc_nodes, win_probabilities,
                      win_probabilities_factor)
from run_narrow_escape import N_W, make_windows, simulate, tv  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 42


def exp_kernel(centers: np.ndarray, ell: float) -> np.ndarray:
    d = np.abs((centers[:, None] - centers[None, :] + np.pi) % (2 * np.pi) - np.pi)
    return np.exp(-d / ell)


def chordal_se_kernel(centers: np.ndarray, ell: float) -> np.ndarray:
    return np.exp(-(1.0 - np.cos(centers[:, None] - centers[None, :])) / ell**2)


def mc_win_probs(mu, C, n_draws, rng, chunk=500_000):
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))
    counts = np.zeros(len(mu))
    done = 0
    while done < n_draws:
        n = min(chunk, n_draws - done)
        X = mu[:, None] + L @ rng.standard_normal((len(mu), n))
        counts += np.bincount(np.argmin(X, axis=0), minlength=len(mu))
        done += n
    return counts / counts.sum()


def nodes_for(k: int):
    """GH product rule while affordable, scrambled-Sobol QMC beyond."""
    if k <= 4:
        return hermite_nodes(k, {1: 21, 2: 15, 3: 11, 4: 9}[k])
    return qmc_nodes(k, m=13)


def offdiag_err(C, V, D):
    R = V @ V.T + np.diag(D) - C
    return np.abs(R - np.diag(np.diag(R))).max()


def main() -> None:
    rng = np.random.default_rng(SEED)
    centers, halfw = make_windows(rng)
    mu_test = np.random.default_rng(1).normal(0.0, 0.8, size=N_W)
    rows = ["part,case,k,offdiag_resid,max_abs_err,seconds"]

    # ---- Part A: quadrature is exact given the factor model --------------------
    print("Part A: exactness given the model (reference: 8M-draw MC, noise ~2e-4)")
    Vk = 0.6 * np.random.default_rng(5).standard_normal((N_W, 2))
    Dk = np.random.default_rng(5).uniform(0.3, 0.9, N_W)
    Ck = Vk @ Vk.T + np.diag(Dk)
    F, W = nodes_for(2)
    p = win_probabilities_factor(mu_test, Vk, Dk, F, W)
    e = np.abs(p - mc_win_probs(mu_test, Ck, 8_000_000, np.random.default_rng(9))).max()
    print(f"  known 2-factor model:  err {e:.1e}")
    rows.append(f"A,known_factors,2,0,{e:.2e},")

    rho = 0.6
    Ceq = rho * np.ones((N_W, N_W)) + (1 - rho) * np.eye(N_W)
    V, D = factor_model(Ceq, 1)
    F, W = nodes_for(1)
    p = win_probabilities_factor(mu_test, V, D, F, W)
    e = np.abs(p - mc_win_probs(mu_test, Ceq, 8_000_000, np.random.default_rng(9))).max()
    print(f"  equicorrelated (k=1 exact): err {e:.1e}")
    rows.append(f"A,equicorrelated,1,0,{e:.2e},")

    V, D = factor_model(np.eye(N_W), 2)
    F, W = nodes_for(2)
    e = np.abs(win_probabilities_factor(mu_test, V, D, F, W)
               - win_probabilities(mu_test, 1.0)).max()
    print(f"  C = I vs independent transform: err {e:.1e} "
          "(naive eigen-truncation gives 1e-1 here)")
    rows.append(f"A,identity,2,0,{e:.2e},")

    # ---- Part B: error tracks the factor residual across rank ------------------
    print("\nPart B: error vs rank (error is factor-model error, not quadrature)")
    cases = {"exp ell=1.6": exp_kernel(centers, 1.6),
             "chordal-SE ell=1.6": chordal_se_kernel(centers, 1.6)}
    KS = [1, 3, 5, 8, 12]
    errs = {}
    for name, C in cases.items():
        ref = mc_win_probs(mu_test, C, 8_000_000, np.random.default_rng(9))
        errs[name] = []
        for k in KS:
            V, D = factor_model(C, k)
            F, W = nodes_for(k)
            t0 = time.perf_counter()
            p = win_probabilities_factor(mu_test, V, D, F, W)
            dt = time.perf_counter() - t0
            e, r = np.abs(p - ref).max(), offdiag_err(C, V, D)
            errs[name].append(e)
            rows.append(f"B,{name},{k},{r:.4f},{e:.2e},{dt:.2f}")
        print(f"  {name}: " + "  ".join(f"k={k}: {e:.1e}"
                                        for k, e in zip(KS, errs[name])))

    # ---- Part C: the deletion ensemble from one pass ---------------------------
    print("\nPart C: single-deletion ensemble, one pass vs per-deletion recompute")
    C = exp_kernel(centers, 1.6)
    V, D = factor_model(C, 3)
    F, W = nodes_for(3)
    t0 = time.perf_counter()
    _, q_ens = win_probabilities_factor(mu_test, V, D, F, W, return_deletions=True)
    t_one = time.perf_counter() - t0
    t0 = time.perf_counter()
    worst = 0.0
    for i in range(N_W):
        keep = np.setdiff1d(np.arange(N_W), [i])
        q_i = win_probabilities_factor(mu_test, V, D, F, W, keep=keep)
        worst = max(worst, np.abs(q_i - np.delete(q_ens[i], i)).max())
    t_sep = time.perf_counter() - t0
    print(f"  one pass {t_one:.2f}s vs {N_W} separate transforms {t_sep:.2f}s; "
          f"max discrepancy {worst:.1e}")
    rows.append(f"C,deletion_ensemble,3,,{worst:.2e},{t_one:.2f}")

    # ---- Part D: exp03 end-to-end, Monte Carlo replaced ------------------------
    print("\nPart D: exp03 counterfactual, fully deterministic (exp kernel, k=8)")
    base = simulate(centers, halfw, np.ones(N_W, bool), 30_000, rng, "open")
    p_hat = (base + 0.5) / (base.sum() + 0.5 * N_W)
    blocked = np.argsort(base)[-3:]
    keep = np.setdiff1d(np.arange(N_W), blocked)
    mask = np.ones(N_W, bool); mask[blocked] = False
    cf = simulate(centers, halfw, mask, 60_000, rng, "test")
    q_true = cf[keep] / cf.sum()

    V, D = factor_model(C, 8)
    F_fit, W_fit = qmc_nodes(8, m=11)
    F_fin, W_fin = qmc_nodes(8, m=13)
    t0 = time.perf_counter()
    mu_fit = abilities_from_probabilities_factor(p_hat, V, D, F_fit, W_fit)
    q_pred = win_probabilities_factor(mu_fit, V, D, F_fin, W_fin, keep=keep)
    t_fit = time.perf_counter() - t0
    e_corr = tv(q_pred, q_true)
    e_iia = tv(p_hat[keep] / p_hat[keep].sum(), q_true)
    print(f"  TV fast correlated: {e_corr:.4f}   Harville/IIA: {e_iia:.4f}   "
          f"(exp03's Monte Carlo pipeline: 0.0094)")
    print(f"  fit + predict {t_fit:.0f}s, no Monte Carlo anywhere")
    rows.append(f"D,exp03_replication,8,,{e_corr:.4f},{t_fit:.0f}")

    # ---- Part E: scaling ---------------------------------------------------------
    print("\nPart E: N = 400 timing (equally spaced, exp kernel ell=1.6, k=8)")
    N = 400
    cN = 2 * np.pi * np.arange(N) / N
    CN = exp_kernel(cN, 1.6)
    muN = np.random.default_rng(2).normal(0.0, 0.8, size=N)
    VN, DN = factor_model(CN, 8)
    FN, WN = qmc_nodes(8, m=12)
    t0 = time.perf_counter()
    pN = win_probabilities_factor(muN, VN, DN, FN, WN)
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    pMC = mc_win_probs(muN, CN, 400_000, np.random.default_rng(4))
    t_mc = time.perf_counter() - t0
    print(f"  fast {t_fast:.1f}s vs MC(400k) {t_mc:.1f}s; "
          f"max |fast - MC| = {np.abs(pN - pMC).max():.1e} "
          f"(MC noise ~{np.sqrt(pN.max() / 400_000):.0e}); "
          "cost O(nodes * N * L), smooth and deterministic in mu")
    rows.append(f"E,N400,8,,{np.abs(pN - pMC).max():.2e},{t_fast:.1f}")

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---- figure -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (name, e), c in zip(errs.items(), ("#c2410c", "#2a1a12")):
        ax.semilogy(KS, e, "o-", color=c, label=name)
    ax.axhline(2e-4, color="#9a9a9a", ls=":", label="reference-MC noise")
    ax.set_xticks(KS)
    ax.set_xlabel("factor rank k")
    ax.set_ylabel("max abs win-probability error")
    ax.set_title("Fast correlated transform: error vs rank\n"
                 "(GH nodes for k <= 4, scrambled-Sobol QMC beyond)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "error_vs_rank.png", dpi=150)
    print("\nwrote results.csv, figures/error_vs_rank.png")


if __name__ == "__main__":
    main()
