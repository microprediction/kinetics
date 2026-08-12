"""Experiment 9 (Q7 x Q3): the Green-Kubo theorem on CONTINUOUS physics.

Experiment 7 verified the softmax-homogenization theorem and its Green-Kubo
correction exactly on an abstract finite-state chain. This experiment repeats the
verification where the hidden state is real physics: reflected Brownian motion in
the unit disk, with N "reaction windows" -- boundary bands in which channel i fires
at rate kappa (Robin-style partial reactivity). The winner is the first channel to
fire; kappa plays the role of the time-scale separation (slow reaction = fast
mixing), via the rescaling t -> kappa t.

Three layers, so every error source is separated:

  A. REAL SIMULATION: Brownian dynamics with per-step firing (50k trajectories per
     kappa). Contains MC noise + time-discretization error.
  B. EXACT ON THE DISCRETIZED GENERATOR: finite-volume Neumann Laplacian on a polar
     grid (invariant law = cell areas, verified), killed-resolvent solve per kappa.
     Contains spatial-discretization error relative to A, but is EXACT for the
     discrete model -- so B vs (softmax + GK) tests the THEOREM cleanly.
  C. THEORY: p_i(kappa) ~= p0_i + kappa * c1_i with p0 the band-area softmax and c1
     the Green-Kubo coefficient from ONE deviation-integral solve (formulas imported
     unchanged from experiment 7 -- the same code that passed the chain tests).

Care checklist (per the project's standing rule that theorems are checked
numerically): pi L = 0 and detailed balance for the grid generator; win probability
normalization at every kappa; K symmetry (reversible generator); GK coefficients sum
to zero; convergence ORDERS fitted, not just errors.

Run:  python experiments/exp09_gk_narrow_escape/run_gk_narrow_escape.py
Outputs: results.csv, figures/gk_continuous.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp01_narrow_escape"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp07_green_kubo"))
from run_green_kubo import exact_win_probs, gk_prediction, green_kubo_matrix  # noqa: E402
from run_narrow_escape import make_windows  # noqa: E402

HERE = Path(__file__).resolve().parent

NR, NTH = 40, 88          # polar finite-volume grid (n = 3520 cells)
BAND = 0.12               # radial depth of the reaction bands
KAPPAS = [2.0, 5.0, 10.0, 20.0, 40.0]
DT = 2e-4
N_TRAJ = 50_000
MAX_STEPS = 3_000_000
SEED = 42                 # same window geometry as experiments 1/3/8


def build_disk_generator():
    """Finite-volume generator of reflected BM on the unit disk (polar grid).

    Conductance-based construction gives detailed balance w.r.t. cell areas, so the
    invariant law is uniform on the disk (as it must be) by construction -- and we
    verify rather than assume it.
    """
    dr = 1.0 / NR
    dth = 2.0 * np.pi / NTH
    r_cent = (np.arange(NR) + 0.5) * dr
    n = NR * NTH
    idx = lambda m, j: m * NTH + (j % NTH)
    area = np.repeat(r_cent * dr * dth, NTH)
    Lg = np.zeros((n, n))
    for m in range(NR):
        for j in range(NTH):
            i = idx(m, j)
            # radial neighbor outward (reflecting at r = 1: no face)
            if m + 1 < NR:
                cond = ((m + 1) * dr * dth) / dr          # face length / distance
                k = idx(m + 1, j)
                Lg[i, k] += cond / area[i]
                Lg[k, i] += cond / area[k]
            # angular neighbor
            cond = dr / (r_cent[m] * dth)
            k = idx(m, j + 1)
            Lg[i, k] += cond / area[i]
            Lg[k, i] += cond / area[k]
    np.fill_diagonal(Lg, Lg.diagonal() - Lg.sum(axis=1))
    pi = area / area.sum()
    # --- verify, don't assume (tolerances scale with the operator norm) --------
    scale = np.abs(np.diag(Lg)).max()
    assert np.abs(Lg.sum(axis=1)).max() < 1e-12 * scale, "rows must sum to zero"
    assert np.abs(pi @ Lg).max() < 1e-12 * scale, "uniform law must be invariant"
    db = pi[:, None] * Lg - (pi[:, None] * Lg).T
    assert np.abs(db).max() < 1e-12 * scale, "detailed balance (reversibility)"
    theta_cent = (np.arange(NTH) + 0.5) * dth
    return Lg, pi, r_cent, theta_cent


def band_indicators(centers, halfw, r_cent, theta_cent):
    """g_i on the grid: FRACTIONAL coverage of cell area by window i's band.

    Sharp 0/1 indicators mis-size windows narrower than a grid cell (the
    narrowest here is half a cell wide), which showed up as a 5e-2 systematic
    gap between the simulation and the discretized model. Fractional coverage
    makes every band's grid area exact.
    """
    dr = 1.0 / NR
    dth = 2.0 * np.pi / NTH
    g = np.zeros((len(centers), NR * NTH))
    rr = np.repeat(r_cent, NTH)          # cell centers
    tt = np.tile(theta_cent, NR)
    r_lo, r_hi = rr - dr / 2.0, rr + dr / 2.0
    # area-weighted radial overlap with [1 - BAND, 1]
    ov_lo = np.maximum(r_lo, 1.0 - BAND)
    ov_hi = np.minimum(r_hi, 1.0)
    frac_r = np.clip(ov_hi**2 - ov_lo**2, 0.0, None) / (r_hi**2 - r_lo**2)
    for i, (c, h) in enumerate(zip(centers, halfw)):
        delta = (tt - c + np.pi) % (2.0 * np.pi) - np.pi   # cell-center offset
        ov = np.minimum(delta + dth / 2.0, h) - np.maximum(delta - dth / 2.0, -h)
        frac_th = np.clip(ov, 0.0, dth) / dth
        g[i] = frac_r * frac_th
    return g


def simulate_firing(centers, halfw, kappa, n_traj, rng):
    """Brownian motion in the disk; in band i, fire at rate kappa. First fire wins."""
    pos = np.zeros((n_traj, 2))
    winner = np.full(n_traj, -1, dtype=int)
    active = np.arange(n_traj)
    step_sd = np.sqrt(2.0 * DT)
    p_fire = 1.0 - np.exp(-kappa * DT)
    t0 = time.perf_counter()
    for _ in range(MAX_STEPS):
        if len(active) == 0:
            break
        pos[active] += step_sd * rng.standard_normal((len(active), 2))
        x, y = pos[active, 0], pos[active, 1]
        r = np.hypot(x, y)
        refl = r >= 1.0
        if np.any(refl):
            rr = r[refl]
            pos[active[refl]] *= ((2.0 - rr) / rr)[:, None]
            r[refl] = 2.0 - rr
        in_band = r > 1.0 - BAND
        if np.any(in_band):
            cand = active[in_band]
            theta = np.arctan2(pos[cand, 1], pos[cand, 0]) % (2 * np.pi)
            d = np.abs((theta[:, None] - centers[None, :] + np.pi)
                       % (2 * np.pi) - np.pi)
            win = d <= halfw[None, :]
            # windows can OVERLAP: each covering window fires independently
            # (intensities add), matching the model lambda_i = kappa g_i.
            # Simultaneous same-step multi-fires (O((kappa dt)^2)) go to the
            # lowest index -- a negligible, documented bias.
            fire_mat = win & (rng.random(win.shape) < p_fire)
            fired = fire_mat.any(axis=1)
            if np.any(fired):
                winner[cand[fired]] = np.argmax(fire_mat[fired], axis=1)
        active = active[winner[active] < 0]
    done = winner >= 0
    print(f"  kappa={kappa}: {done.sum()}/{n_traj} fired, "
          f"{time.perf_counter() - t0:.0f}s")
    counts = np.bincount(winner[done], minlength=len(centers))
    return counts / counts.sum()


def main():
    rng = np.random.default_rng(SEED)
    centers, halfw = make_windows(rng)
    N = len(centers)
    A = np.arange(N)

    print("building finite-volume generator and Green-Kubo prediction:")
    t0 = time.perf_counter()
    Lg, pi, r_cent, theta_cent = build_disk_generator()
    g = band_indicators(centers, halfw, r_cent, theta_cent)
    K = green_kubo_matrix(Lg, pi, g)
    assert np.abs(K - K.T).max() < 1e-10, "K symmetric for reversible generator"
    p0, c1 = gk_prediction(pi, g, K, A)
    assert abs(c1.sum()) < 1e-12, "GK coefficients preserve normalization"
    print(f"  n={len(pi)} cells, one deviation solve, {time.perf_counter()-t0:.0f}s")

    rows = ["kappa,layer,max_abs_err"]
    err_soft, err_gk, sim_pts = [], [], {}
    for kappa in KAPPAS:
        p_exact = exact_win_probs(Lg, pi, g, A, kappa)   # eps = kappa by rescaling
        assert abs(p_exact.sum() - 1.0) < 1e-9
        e_s = np.abs(p_exact - p0).max()
        e_g = np.abs(p_exact - p0 - kappa * c1).max()
        err_soft.append(e_s)
        err_gk.append(e_g)
        rows += [f"{kappa},exact_vs_softmax,{e_s:.3e}",
                 f"{kappa},exact_vs_gk,{e_g:.3e}"]

    s0 = np.polyfit(np.log(KAPPAS[:3]), np.log(err_soft[:3]), 1)[0]
    s1 = np.polyfit(np.log(KAPPAS[:3]), np.log(err_gk[:3]), 1)[0]
    print(f"\ndiscretized-exact vs theory: softmax slope {s0:.2f} (theory 1), "
          f"GK slope {s1:.2f} (theory 2)")
    rows += [f",softmax_slope,{s0:.3f}", f",gk_slope,{s1:.3f}"]

    print("\nreal Brownian simulation (layer A):")
    for kappa in (5.0, 20.0):
        p_sim = simulate_firing(centers, halfw, kappa, N_TRAJ, rng)
        sim_pts[kappa] = p_sim
        p_exact = exact_win_probs(Lg, pi, g, A, kappa)
        e = np.abs(p_sim - p_exact).max()
        noise = np.sqrt(p_exact.max() / N_TRAJ)
        print(f"  kappa={kappa}: max |sim - discretized exact| = {e:.2e} "
              f"(MC noise ~{noise:.0e})")
        rows.append(f"{kappa},sim_vs_exact,{e:.3e}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.loglog(KAPPAS, err_soft, "o-", color="#2a1a12",
              label=f"|exact − softmax|  (slope {s0:.2f})")
    ax.loglog(KAPPAS, err_gk, "s-", color="#c2410c",
              label=f"|exact − softmax − κ·GK|  (slope {s1:.2f})")
    ax.set_xlabel("reactivity κ (time-scale separation)")
    ax.set_ylabel("max abs win-probability error")
    ax.set_title("Theorem check on the discretized disk", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    top = np.argsort(p0)[-4:]
    colors = plt.cm.copper(np.linspace(0.3, 0.9, 4))
    kk = np.linspace(0, max(KAPPAS), 50)
    for c, i in zip(colors, top):
        ax.plot(kk, p0[i] + kk * c1[i], "-", color=c, lw=1.5)
        exacts = [exact_win_probs(Lg, pi, g, A, k)[i] for k in KAPPAS]
        ax.plot(KAPPAS, exacts, "o", color=c, ms=4, label=f"window {i}")
        for kappa, p_sim in sim_pts.items():
            ax.plot([kappa], [p_sim[i]], "x", color=c, ms=8, mew=2)
    ax.set_xlabel("reactivity κ")
    ax.set_ylabel("win probability")
    ax.set_title("Lines: softmax + κ·GK.  Dots: exact.  X: real Brownian sim",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "gk_continuous.png", dpi=150)
    print("\nwrote results.csv, figures/gk_continuous.png")


if __name__ == "__main__":
    main()
