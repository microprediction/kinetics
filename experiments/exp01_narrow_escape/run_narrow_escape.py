"""Experiment 1 (program Q2): kinetic surrogates on a real first-passage simulation.

Physics. A Brownian particle diffuses in the unit disk (overdamped Langevin dynamics,
no potential) and is absorbed at one of N windows on the boundary -- the classical
narrow-escape geometry. Each simulated trajectory is a race: the windows compete for
the particle, and we record only the winner, as a kinetic simulation would.

Counterfactual. Block the B busiest windows (make them reflecting) and predict the
new win distribution among the survivors. Ground truth comes from re-simulating the
blocked geometry. Diffusion makes this genuinely non-IIA: a walker turned away from a
blocked window is most likely absorbed *nearby*, so neighboring windows gain more
than proportional renormalization predicts.

Surrogates fit to winner-only data from the OPEN geometry:
  * Harville / exponential (the KMC assumption): blocked probability is
    redistributed proportionally (IIA).
  * Thurstone (independent Gaussian race via the fast ability transform): invert
    frequencies to latent abilities, then forward-price the surviving field.

Honest question: the Thurstone race is independent too -- it bends IIA through
ability gaps, not geometry. Does that help on real diffusive kinetics, or does
capturing the neighbor effect require a correlated race (program Q6)?

Run:  python experiments/exp01_narrow_escape/run_narrow_escape.py
Outputs: results.csv, figures/redistribution.png, figures/error_vs_R.png
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
from raceutil import abilities_from_probabilities, win_probabilities  # noqa: E402

HERE = Path(__file__).resolve().parent

# --- geometry: 16 windows, one tight cluster plus scattered singletons ---------
N_W = 16
B_BLOCK = 3
DT = 5e-5                     # diffusion step variance 2*DT (D = 1)
MAX_STEPS = 600_000
R_BASE = 30_000               # trajectories, open geometry
R_TRUTH = 60_000              # trajectories, blocked geometry (ground truth)
R_GRID = [500, 2000, 8000, 30_000]
SIGMA = 1.0                   # Thurstone race scale (untuned default)
SEED = 42


def make_windows(rng: np.random.Generator):
    """Angular centers and half-widths; a cluster around 0 plus scattered windows."""
    cluster = np.linspace(-0.45, 0.45, 6)                       # 6 windows in ~52 deg
    scattered = rng.uniform(0.9, 2 * np.pi - 0.9, size=N_W - 6)
    centers = np.sort(np.concatenate([cluster % (2 * np.pi), scattered]))
    halfw = rng.lognormal(np.log(0.030), 0.5, size=N_W)         # varied widths
    return centers, np.clip(halfw, 0.012, 0.09)


def simulate(centers, halfw, open_mask, n_walkers, rng, label=""):
    """Vectorized Brownian dynamics; returns winner counts per window."""
    pos = np.zeros((n_walkers, 2))
    counts = np.zeros(N_W, dtype=int)
    step_sd = np.sqrt(2.0 * DT)
    open_centers = centers[open_mask]
    open_halfw = halfw[open_mask]
    open_ids = np.nonzero(open_mask)[0]
    t0 = time.perf_counter()
    for step in range(MAX_STEPS):
        if len(pos) == 0:
            break
        pos += step_sd * rng.standard_normal(pos.shape)
        r = np.hypot(pos[:, 0], pos[:, 1])
        out = r >= 1.0
        if np.any(out):
            theta = np.arctan2(pos[out, 1], pos[out, 0]) % (2 * np.pi)
            d = np.abs((theta[:, None] - open_centers[None, :] + np.pi) % (2 * np.pi) - np.pi)
            hit = d <= open_halfw[None, :]
            absorbed = hit.any(axis=1)
            if np.any(absorbed):
                widx = open_ids[np.argmax(hit[absorbed], axis=1)]
                counts += np.bincount(widx, minlength=N_W)
            # reflect the rest back inside
            keep_out = ~absorbed
            idx_out = np.nonzero(out)[0]
            refl = idx_out[keep_out]
            pos[refl] *= ((2.0 - r[refl]) / r[refl])[:, None]
            pos = np.delete(pos, idx_out[absorbed], axis=0)
    n_lost = len(pos)
    print(f"  simulate[{label}]: {counts.sum()} absorbed, {n_lost} lost, "
          f"{time.perf_counter() - t0:.1f}s")
    return counts


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def main() -> None:
    rng = np.random.default_rng(SEED)
    centers, halfw = make_windows(rng)
    all_open = np.ones(N_W, dtype=bool)

    print("base simulation (open geometry):")
    base_counts = simulate(centers, halfw, all_open, R_BASE, rng, "open")
    p_open = base_counts / base_counts.sum()

    blocked = np.argsort(p_open)[-B_BLOCK:]
    keep = np.setdiff1d(np.arange(N_W), blocked)
    open_mask = all_open.copy()
    open_mask[blocked] = False
    print(f"blocking busiest windows {sorted(blocked.tolist())}; ground-truth re-simulation:")
    cf_counts = simulate(centers, halfw, open_mask, R_TRUTH, rng, "blocked")
    q_true = cf_counts[keep] / cf_counts.sum()

    # how non-IIA is the physics? (using the full base run as best estimate of p)
    q_iia_best = p_open[keep] / p_open[keep].sum()
    print(f"\nnon-IIA magnitude of the physics: TV(truth, proportional) = "
          f"{tv(q_true, q_iia_best):.4f}")

    # --- surrogates at increasing amounts of race data -----------------------
    rows = ["R,tv_harville,tv_thurstone"]
    errs_h, errs_t = [], []
    for R in R_GRID:
        sub = rng.multinomial(R, p_open)          # subsample the base races
        p_hat = (sub + 0.5) / (R + 0.5 * N_W)
        q_harville = p_hat[keep] / p_hat[keep].sum()
        mu_hat = abilities_from_probabilities(p_hat, SIGMA)
        q_thur = win_probabilities(mu_hat[keep], SIGMA)
        e_h, e_t = tv(q_harville, q_true), tv(q_thur, q_true)
        errs_h.append(e_h); errs_t.append(e_t)
        rows.append(f"{R},{e_h:.6f},{e_t:.6f}")
        print(f"R={R:>6}: TV Harville {e_h:.4f}   TV Thurstone {e_t:.4f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # --- figure 1: the redistribution, window by window -----------------------
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(keep))
    w = 0.27
    mu_full = abilities_from_probabilities(
        (base_counts + 0.5) / (base_counts.sum() + 0.5 * N_W), SIGMA)
    q_thur_full = win_probabilities(mu_full[keep], SIGMA)
    ax.bar(xs - w, q_true, w, label="ground truth (re-simulated)", color="#2a1a12")
    ax.bar(xs, q_iia_best, w, label="Harville / IIA renormalization", color="#9a9a9a")
    ax.bar(xs + w, q_thur_full, w, label="Thurstone surrogate", color="#c2410c")
    near = np.array([np.min(np.abs((centers[k] - centers[blocked] + np.pi)
                                   % (2 * np.pi) - np.pi)) for k in keep])
    labels = [f"w{k}" + ("*" if near[j] < 0.6 else "") for j, k in enumerate(keep)]
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylabel("win probability after blocking")
    ax.set_title(f"Redistribution after blocking the {B_BLOCK} busiest windows "
                 "(* = within 0.6 rad of a blocked window)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "redistribution.png", dpi=150)

    # --- figure 2: error vs data ----------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.loglog(R_GRID, errs_h, "o-", color="#2a1a12", label="Harville (IIA)")
    ax2.loglog(R_GRID, errs_t, "s-", color="#c2410c", label="Thurstone")
    ax2.axhline(tv(q_true, q_iia_best), color="#9a9a9a", ls=":",
                label="non-IIA magnitude of the physics")
    ax2.set_xlabel("observed first events R")
    ax2.set_ylabel("TV error of blocked-geometry prediction")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(HERE / "figures" / "error_vs_R.png", dpi=150)

    print("\nwrote results.csv, figures/redistribution.png, figures/error_vs_R.png")


if __name__ == "__main__":
    main()
