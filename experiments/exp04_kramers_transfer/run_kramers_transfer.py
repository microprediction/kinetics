"""Experiment 4 (program Q2, chemistry angle): temperature transfer of kinetic
surrogates on a real barrier-crossing simulation.

Physics. Overdamped Langevin dynamics in a 2D well enclosed by a barrier ring of
angularly varying height: U(r, theta) = A(theta) * exp(-(r - r0)^2 / 2w^2), with
A(theta) interpolating N per-channel barrier heights E_i (drawn once, quenched
disorder). A particle starting at the origin escapes by thermally crossing the ring;
the escape channel (angular sector) is the winner of a race between N activated
pathways. This is a minimal model of competing reaction channels.

Question. Fit each surrogate to escape-channel frequencies observed at temperature
kT1, then predict the channel distribution at a colder temperature kT2 (ground truth:
re-simulation). Temperature transfer is THE canonical counterfactual of chemical
kinetics, and the two surrogates extrapolate differently:

  * Harville/Arrhenius (KMC assumption): channels are independent exponential clocks,
    rates k_i ~ p_i; cooling sharpens rates as  p_i^(kT1/kT2), renormalized.
  * Thurstone/Arrhenius: performance = barrier + thermal noise, X_i = mu_i + kT * eps;
    cooling rescales abilities as  mu_i * (kT1/kT2)  with unit noise. The race is
    still independent -- what changes vs Harville is the noise law (Gaussian vs
    Gumbel-like), i.e. HOW win probability responds to ability gaps as they widen.

Baseline: "no transfer" (use the kT1 frequencies unchanged) shows the effect size.

Run:  python experiments/exp04_kramers_transfer/run_kramers_transfer.py
Outputs: results.csv, figures/transfer.png
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

N_CH = 12               # escape channels (angular sectors)
R0, W = 0.6, 0.10       # barrier ring radius and width
R_ABS = 0.95            # absorption radius (beyond the ring)
KT1 = 1.0               # fit temperature (E_i in units of KT1)
KT2_LIST = (0.7, 0.55)  # test temperatures: moderate and deep extrapolation
DT = 2e-4
MAX_STEPS = 2_000_000
N_FIT = 25_000          # trajectories at KT1
N_TEST = 40_000         # trajectories at KT2 (ground truth)
SEED = 3


def make_barriers(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(2.0, 3.5, size=N_CH)          # quenched disorder, in kT1


SECTORS = 2.0 * np.pi * np.arange(N_CH) / N_CH


def angular_profile(E: np.ndarray):
    """Periodic linear interpolation of barrier heights; returns A(theta), A'(theta)."""
    grid = np.concatenate([SECTORS, [2 * np.pi]])
    vals = np.concatenate([E, [E[0]]])

    def A(theta):
        return np.interp(theta % (2 * np.pi), grid, vals)

    def dA(theta):
        th = theta % (2 * np.pi)
        idx = np.minimum((th // (2 * np.pi / N_CH)).astype(int), N_CH - 1)
        return (vals[idx + 1] - vals[idx]) / (2 * np.pi / N_CH)

    return A, dA


def simulate(E: np.ndarray, kT: float, n_walkers: int, rng: np.random.Generator,
             label: str = "") -> np.ndarray:
    """Langevin dynamics until absorption at r >= R_ABS; returns channel counts."""
    A, dA = angular_profile(E)
    pos = np.zeros((n_walkers, 2))
    counts = np.zeros(N_CH, dtype=int)
    noise_sd = np.sqrt(2.0 * kT * DT)
    t0 = time.perf_counter()
    for _ in range(MAX_STEPS):
        if len(pos) == 0:
            break
        x, y = pos[:, 0], pos[:, 1]
        r = np.maximum(np.hypot(x, y), 1e-9)
        th = np.arctan2(y, x)
        B = np.exp(-0.5 * ((r - R0) / W) ** 2)
        dUdr = A(th) * (-(r - R0) / W**2) * B
        dUdth = dA(th) * B
        fx = -(dUdr * x / r - dUdth * y / r**2)
        fy = -(dUdr * y / r + dUdth * x / r**2)
        pos[:, 0] += fx * DT + noise_sd * rng.standard_normal(len(pos))
        pos[:, 1] += fy * DT + noise_sd * rng.standard_normal(len(pos))
        out = np.hypot(pos[:, 0], pos[:, 1]) >= R_ABS
        if np.any(out):
            th_out = np.arctan2(pos[out, 1], pos[out, 0]) % (2 * np.pi)
            ch = np.round(th_out / (2 * np.pi / N_CH)).astype(int) % N_CH
            counts += np.bincount(ch, minlength=N_CH)
            pos = pos[~out]
    print(f"  simulate[{label}] kT={kT}: {counts.sum()} escaped, {len(pos)} lost, "
          f"{time.perf_counter() - t0:.1f}s")
    return counts


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def main() -> None:
    rng = np.random.default_rng(SEED)
    E = make_barriers(rng)
    print(f"barrier heights (kT1 units): {E.round(2)}")

    counts1 = simulate(E, KT1, N_FIT, rng, "fit")
    p1 = (counts1 + 0.5) / (counts1.sum() + 0.5 * N_CH)
    mu1 = abilities_from_probabilities(p1, sigma=1.0)        # abilities in kT1 units

    rows = ["kT2,method,tv"]
    (HERE / "figures").mkdir(exist_ok=True)
    for KT2 in KT2_LIST:
        counts2 = simulate(E, KT2, N_TEST, rng, f"truth kT={KT2}")
        q_true = counts2 / counts2.sum()

        # --- transfer predictions ----------------------------------------------
        q_none = p1 / p1.sum()                               # no transfer
        q_harville = p1 ** (KT1 / KT2)
        q_harville /= q_harville.sum()                       # Arrhenius rate scaling
        q_thur = win_probabilities(mu1 * (KT1 / KT2), sigma=1.0)  # cool: gaps widen

        e_none, e_h, e_t = (tv(q_none, q_true), tv(q_harville, q_true),
                            tv(q_thur, q_true))
        print(f"\ntransfer kT {KT1} -> {KT2}, TV vs re-simulated truth:")
        print(f"  no transfer:          {e_none:.4f}")
        print(f"  Harville/Arrhenius:   {e_h:.4f}")
        print(f"  Thurstone/Arrhenius:  {e_t:.4f}")
        rows += [f"{KT2},none,{e_none:.6f}", f"{KT2},harville,{e_h:.6f}",
                 f"{KT2},thurstone,{e_t:.6f}"]

        # --- figure -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4.2))
        xs = np.arange(N_CH); w = 0.21
        ax.bar(xs - 1.5 * w, q_true, w, label=f"ground truth at kT={KT2}",
               color="#2a1a12")
        ax.bar(xs - 0.5 * w, q_none, w, label=f"no transfer (kT={KT1} freqs)",
               color="#d8d3ce")
        ax.bar(xs + 0.5 * w, q_harville, w, label="Harville/Arrhenius",
               color="#9a9a9a")
        ax.bar(xs + 1.5 * w, q_thur, w, label="Thurstone/Arrhenius", color="#c2410c")
        ax.set_xticks(xs, [f"ch{c}\nE={E[c]:.1f}" for c in range(N_CH)], fontsize=7.5)
        ax.set_ylabel(f"escape-channel probability at kT={KT2}")
        ax.set_title(f"Temperature transfer kT {KT1} -> {KT2}: "
                     f"TV none {e_none:.3f}, Harville {e_h:.3f}, "
                     f"Thurstone {e_t:.3f}", fontsize=10)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(HERE / "figures" / f"transfer_kt{KT2}.png", dpi=150)

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv and transfer figures")


if __name__ == "__main__":
    main()
