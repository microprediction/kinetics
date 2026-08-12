"""Experiment 5 (Q6 x Q2): does the correlated race also fix the temperature-transfer
residual?

Experiment 4 left both independent surrogates with the same residual transfer error
(TV ~0.034 at kT 1.0 -> 0.55, ~4x sampling noise), attributed to angular mixing:
a particle engaged with one channel often crosses at a neighbor, correlating adjacent
channels. Experiment 3 showed a geometry-informed correlated race repairs exactly this
kind of error for blocked-window counterfactuals. Here the two are combined on the
Kramers barrier-crossing system:

  performances  X = mu * (kT1/kT2) + L eps,   corr(i,j) = exp(-d_ij / ell),

with d_ij the angular distance between channel centers. Abilities mu are fit at kT1
under each ell; the correlation length is CALIBRATED on the kT2 = 0.7 ground truth and
TESTED on the deeper kT2 = 0.55 extrapolation. ell -> 0 recovers experiment 4's
independent Thurstone transfer.

Run:  python experiments/exp05_correlated_transfer/run_correlated_transfer.py
Outputs: results.csv, figures/tv_vs_ell_transfer.png
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp03_correlated_race"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp04_kramers_transfer"))
from run_correlated_race import fit_abilities, mc_win_probs  # noqa: E402
from run_kramers_transfer import KT1, N_CH, SECTORS, make_barriers, simulate, tv  # noqa: E402

HERE = Path(__file__).resolve().parent

KT_CAL, KT_TEST = 0.7, 0.55
N_FIT, N_TRUTH = 25_000, 40_000
R_MC = 400_000
ELL_GRID = [0.01, 0.1, 0.2, 0.4, 0.8, 1.6]
SEED = 3          # same quenched barriers as experiment 4


def corr_matrix(ell: float) -> np.ndarray:
    d = np.abs((SECTORS[:, None] - SECTORS[None, :] + np.pi) % (2 * np.pi) - np.pi)
    return np.exp(-d / ell) + 1e-9 * np.eye(N_CH)


def main() -> None:
    rng = np.random.default_rng(SEED)
    E = make_barriers(rng)
    counts1 = simulate(E, KT1, N_FIT, rng, "fit")
    p1 = (counts1 + 0.5) / (counts1.sum() + 0.5 * N_CH)
    q_cal = simulate(E, KT_CAL, N_TRUTH, rng, f"truth kT={KT_CAL}")
    q_cal = q_cal / q_cal.sum()
    q_test = simulate(E, KT_TEST, N_TRUTH, rng, f"truth kT={KT_TEST}")
    q_test = q_test / q_test.sum()

    # independent baselines (as in experiment 4)
    q_h_cal = p1 ** (KT1 / KT_CAL); q_h_cal /= q_h_cal.sum()
    q_h_test = p1 ** (KT1 / KT_TEST); q_h_test /= q_h_test.sum()

    Z = np.random.default_rng(SEED + 1).standard_normal((N_CH, R_MC))
    rows = ["ell,tv_fit,tv_cal_kt0.7,tv_test_kt0.55"]
    tv_cal, tv_test = [], []
    t0 = time.perf_counter()
    for ell in ELL_GRID:
        chol = np.linalg.cholesky(corr_matrix(ell))
        mu = fit_abilities(p1, chol, Z)
        e_fit = tv(mc_win_probs(mu, chol, Z), p1)
        e_cal = tv(mc_win_probs(mu * (KT1 / KT_CAL), chol, Z), q_cal)
        e_test = tv(mc_win_probs(mu * (KT1 / KT_TEST), chol, Z), q_test)
        tv_cal.append(e_cal); tv_test.append(e_test)
        rows.append(f"{ell},{e_fit:.6f},{e_cal:.6f},{e_test:.6f}")
        print(f"ell={ell:>5}: TV fit {e_fit:.4f}   kT={KT_CAL} {e_cal:.4f}   "
              f"kT={KT_TEST} {e_test:.4f}")
    print(f"sweep took {time.perf_counter() - t0:.1f}s")

    i_star = int(np.argmin(tv_cal))
    ell_star = ELL_GRID[i_star]
    e_h = tv(q_h_test, q_test)
    print(f"\ndeep transfer kT {KT1} -> {KT_TEST}:")
    print(f"  Harville/Arrhenius:              TV {e_h:.4f}")
    print(f"  independent Thurstone (ell~0):   TV {tv_test[0]:.4f}")
    print(f"  correlated, ell*={ell_star} (chosen at kT={KT_CAL}): TV {tv_test[i_star]:.4f}")
    rows.append(f"harville,,,{e_h:.6f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogx(ELL_GRID, tv_cal, "o-", color="#9a9a9a",
                label=f"calibration (kT={KT_CAL})")
    ax.semilogx(ELL_GRID, tv_test, "s-", color="#c2410c",
                label=f"test (kT={KT_TEST})")
    ax.axhline(e_h, color="#2a1a12", ls=":", label="Harville/Arrhenius (test)")
    ax.axvline(ell_star, color="#c2410c", ls="--", lw=1, alpha=0.6,
               label=f"ell* = {ell_star} (from calibration)")
    ax.set_xlabel("correlation length ell (rad)")
    ax.set_ylabel("TV error of transferred prediction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "tv_vs_ell_transfer.png", dpi=150)
    print("wrote results.csv, figures/tv_vs_ell_transfer.png")


if __name__ == "__main__":
    main()
