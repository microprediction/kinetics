"""Experiment 3 (program Q6): does a CORRELATED race capture the geometry that
independent races miss?

Experiment 1 showed that real diffusive kinetics violates IIA geometrically -- a
walker turned away from a blocked window is absorbed nearby -- and that neither the
exponential (Harville) nor the independent Thurstone surrogate captures the effect.
The conjecture (Q6) is that the missing object is correlation between competitors.

Model. Windows race with performances X = mu + L eps, where the noise correlation is
geometry-informed:  corr(i, j) = exp(-d_ij / ell)  with d_ij the angular distance
between window centers. In a correlated race, deleting a competitor hands its wins
disproportionately to its correlated partners -- exactly the neighbor-inheritance
the physics displays. At ell -> 0 the model reduces to the independent Thurstone race.

Protocol (all ground truths are re-simulated Brownian dynamics, same geometry and
seed as experiment 1):
  1. Fit abilities mu(ell) to the open-geometry win frequencies for each ell.
  2. CALIBRATE ell on one intervention (block windows ranked 4th-5th busiest).
  3. TEST on a different intervention (block the 3 busiest, as in experiment 1),
     comparing Harville, independent Thurstone, and correlated race at ell*.

Win probabilities under correlation are computed by Monte Carlo with common random
numbers (fixed noise panel), which makes the fitting objective deterministic.

Run:  python experiments/exp03_correlated_race/run_correlated_race.py
Outputs: results.csv, figures/tv_vs_ell.png, figures/test_redistribution.png
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
from run_narrow_escape import N_W, make_windows, simulate, tv  # noqa: E402

HERE = Path(__file__).resolve().parent

R_MC = 400_000        # common-random-number noise panel
ELL_GRID = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
SEED = 42             # same geometry as experiment 1


def corr_matrix(centers: np.ndarray, ell: float) -> np.ndarray:
    d = np.abs((centers[:, None] - centers[None, :] + np.pi) % (2 * np.pi) - np.pi)
    C = np.exp(-d / ell)
    return C + 1e-9 * np.eye(len(centers))


def mc_win_probs(mu: np.ndarray, chol: np.ndarray, Z: np.ndarray,
                 keep: np.ndarray | None = None) -> np.ndarray:
    """P(win) by argmin over the (optionally restricted) correlated field."""
    X = mu[:, None] + chol @ Z
    if keep is not None:
        X = X[keep]
    w = np.argmin(X, axis=0)
    return np.bincount(w, minlength=X.shape[0]) / X.shape[1]


def fit_abilities(p_hat: np.ndarray, chol: np.ndarray, Z: np.ndarray,
                  n_iter: int = 800) -> np.ndarray:
    """Damped fixed point, as in raceutil but under correlated MC forward.

    Under strong correlation the win probabilities become hyper-sensitive to
    ability gaps (the effective noise between competitors i, j has variance
    2(1 - rho_ij) sigma^2), so the step is scaled by the smallest effective
    pairwise noise scale to keep the iteration stable.
    """
    C = chol @ chol.T
    rho_max = np.max(C - np.diag(np.diag(C)))
    step = 0.5 * np.sqrt(max(2.0 * (1.0 - rho_max), 1e-4))
    floor = 1.0 / (2 * Z.shape[1])
    logp = np.log(p_hat)
    mu = -(logp - logp.mean()) / 2.0
    for _ in range(n_iter):
        model = np.maximum(mc_win_probs(mu, chol, Z), floor)
        resid = np.clip(np.log(model) - logp, -4.0, 4.0)
        mu = mu + step * resid
        mu -= mu.mean()
    return mu


def main() -> None:
    rng = np.random.default_rng(SEED)
    centers, halfw = make_windows(rng)
    all_open = np.ones(N_W, dtype=bool)

    print("simulating ground truths (open, calibration block, test block):")
    base_counts = simulate(centers, halfw, all_open, 30_000, rng, "open")
    p_open = base_counts / base_counts.sum()
    p_hat = (base_counts + 0.5) / (base_counts.sum() + 0.5 * N_W)

    order = np.argsort(p_open)
    cal_block = order[-5:-3]          # 4th and 5th busiest
    test_block = order[-3:]           # the 3 busiest (as in experiment 1)

    def blocked_truth(block, n_walk, label):
        mask = all_open.copy(); mask[block] = False
        counts = simulate(centers, halfw, mask, n_walk, rng, label)
        keep = np.setdiff1d(np.arange(N_W), block)
        return keep, counts[keep] / counts.sum()

    cal_keep, q_cal = blocked_truth(cal_block, 40_000, "calibration")
    test_keep, q_test = blocked_truth(test_block, 60_000, "test")

    Z = np.random.default_rng(SEED + 1).standard_normal((N_W, R_MC))

    # --- sweep ell: fit on open data, score on both interventions --------------
    rows = ["ell,tv_fit,tv_calibration,tv_test"]
    tv_cal, tv_test = [], []
    t0 = time.perf_counter()
    for ell in ELL_GRID:
        chol = np.linalg.cholesky(corr_matrix(centers, ell))
        mu = fit_abilities(p_hat, chol, Z)
        e_fit = tv(mc_win_probs(mu, chol, Z), p_hat)   # 0 unless the fit failed
        e_cal = tv(mc_win_probs(mu, chol, Z, cal_keep), q_cal)
        e_test = tv(mc_win_probs(mu, chol, Z, test_keep), q_test)
        tv_cal.append(e_cal); tv_test.append(e_test)
        rows.append(f"{ell},{e_fit:.6f},{e_cal:.6f},{e_test:.6f}")
        print(f"ell={ell:>5}: TV fit {e_fit:.4f}   calibration {e_cal:.4f}   "
              f"test {e_test:.4f}")
    print(f"sweep took {time.perf_counter() - t0:.1f}s")

    # --- baselines on the test intervention ------------------------------------
    q_harville = p_hat[test_keep] / p_hat[test_keep].sum()
    e_harville = tv(q_harville, q_test)
    e_indep = tv_test[0]                      # ell = 0.01 ~ independent
    i_star = int(np.argmin(tv_cal))
    ell_star, e_star = ELL_GRID[i_star], tv_test[i_star]
    print(f"\ntest intervention (block 3 busiest):")
    print(f"  Harville / IIA:            TV {e_harville:.4f}")
    print(f"  independent Thurstone:     TV {e_indep:.4f}")
    print(f"  correlated, ell*={ell_star} (chosen on calibration): TV {e_star:.4f}")
    rows.append(f"harville,,{e_harville:.6f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # --- figures ----------------------------------------------------------------
    fig_dir = HERE / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogx(ELL_GRID, tv_cal, "o-", color="#9a9a9a", label="calibration intervention")
    ax.semilogx(ELL_GRID, tv_test, "s-", color="#c2410c", label="test intervention")
    ax.axhline(e_harville, color="#2a1a12", ls=":", label="Harville / IIA (test)")
    ax.axvline(ell_star, color="#c2410c", ls="--", lw=1, alpha=0.6,
               label=f"ell* = {ell_star} (from calibration)")
    ax.set_xlabel("correlation length ell (rad)")
    ax.set_ylabel("TV error of blocked-geometry prediction")
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "tv_vs_ell.png", dpi=150)

    chol_star = np.linalg.cholesky(corr_matrix(centers, ell_star))
    mu_star = fit_abilities(p_hat, chol_star, Z)
    q_corr = mc_win_probs(mu_star, chol_star, Z, test_keep)
    chol_ind = np.linalg.cholesky(corr_matrix(centers, ELL_GRID[0]))
    q_ind = mc_win_probs(fit_abilities(p_hat, chol_ind, Z), chol_ind, Z, test_keep)

    fig2, ax2 = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(test_keep)); w = 0.21
    ax2.bar(xs - 1.5 * w, q_test, w, label="ground truth (re-simulated)", color="#2a1a12")
    ax2.bar(xs - 0.5 * w, q_harville, w, label="Harville / IIA", color="#9a9a9a")
    ax2.bar(xs + 0.5 * w, q_ind, w, label="independent Thurstone", color="#e8a87c")
    ax2.bar(xs + 1.5 * w, q_corr, w, label=f"correlated race (ell*={ell_star})", color="#c2410c")
    near = np.array([np.min(np.abs((centers[k] - centers[test_block] + np.pi)
                                   % (2 * np.pi) - np.pi)) for k in test_keep])
    ax2.set_xticks(xs, [f"w{k}" + ("*" if near[j] < 0.6 else "")
                        for j, k in enumerate(test_keep)], fontsize=8)
    ax2.set_ylabel("win probability after blocking")
    ax2.set_title("Test intervention: blocking the 3 busiest windows "
                  "(* = within 0.6 rad of a blocked window)", fontsize=10)
    ax2.legend(fontsize=8.5)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "test_redistribution.png", dpi=150)

    print("wrote results.csv, figures/tv_vs_ell.png, figures/test_redistribution.png")


if __name__ == "__main__":
    main()
