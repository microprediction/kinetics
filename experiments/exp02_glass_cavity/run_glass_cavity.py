"""Experiment 2 (program Q1/Q4): the rank-one cavity on a disordered elastic network.

Physics. A 2D random spring network -- nodes on a jittered triangular-ish lattice,
springs with log-normally disordered stiffnesses -- is a standard harmonic model of an
amorphous solid. Its Hessian H (here the weighted graph Laplacian plus a weak pinning
term) has inverse G = H^{-1}, the static susceptibility: G_jj is the linear response of
node j to a unit force on itself (local compliance).

Deletion questions, all answered from ONE inverse:
  * Pin site i (remove its degree of freedom). The softening it removes,
        drop_i = Tr G - Tr G^(i) = (sum_j G_ji^2) / G_ii,
    is an O(n) scalar per site after one solve -- the "soft spot" map of the glass.
  * Pin a PAIR {i, j}: a 2x2 Schur downdate gives Tr G^(ij) directly. The
    nonadditivity  I_ij = drop_i + drop_j - (Tr G - Tr G^(ij))  is the elastic
    interaction between the two pinning defects, obtainable for ALL pairs without
    re-solving anything.

The experiment verifies both against direct re-inversion, times the full single-site
sweep both ways, and shows the defect-defect interaction decaying with distance.

Run:  python experiments/exp02_glass_cavity/run_glass_cavity.py
Outputs: figures/soft_spots.png, figures/pair_interaction.png, printed summary.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SIDE = 28            # SIDE x SIDE nodes
PIN = 0.05           # substrate coupling: sets a finite screening length ~ sqrt(k/PIN)
SEED = 7


def build_network(rng: np.random.Generator):
    """Jittered square lattice with nearest + diagonal springs, disordered stiffness."""
    n = SIDE * SIDE
    ix, iy = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    xy = np.column_stack([ix.ravel() + 0.3 * rng.standard_normal(n),
                          iy.ravel() + 0.3 * rng.standard_normal(n)])
    idx = lambda i, j: i * SIDE + j
    bonds = []
    for i in range(SIDE):
        for j in range(SIDE):
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < SIDE and 0 <= b < SIDE:
                    bonds.append((idx(i, j), idx(a, b)))
    bonds = np.array(bonds)
    k = rng.lognormal(0.0, 0.9, size=len(bonds))     # strong stiffness disorder
    H = np.zeros((n, n))
    H[bonds[:, 0], bonds[:, 1]] -= k
    H[bonds[:, 1], bonds[:, 0]] -= k
    np.add.at(np.ravel(H), (bonds[:, 0] * (n + 1)), k)
    np.add.at(np.ravel(H), (bonds[:, 1] * (n + 1)), k)
    H += PIN * np.eye(n)
    return H, xy


def main() -> None:
    rng = np.random.default_rng(SEED)
    H, xy = build_network(rng)
    n = H.shape[0]

    # ---- one global solve ----------------------------------------------------
    t0 = time.perf_counter()
    G = np.linalg.inv(H)
    t_solve = time.perf_counter() - t0
    trG = np.trace(G)

    # ---- all n single-site pinning responses from the one inverse -------------
    t0 = time.perf_counter()
    drop = np.einsum("ji,ji->i", G, G) / np.diag(G)     # drop_i = ||G_{.i}||^2 / G_ii
    t_sweep = time.perf_counter() - t0

    # ---- verify + time the naive route on a sample of sites -------------------
    sample = rng.choice(n, size=12, replace=False)
    t0 = time.perf_counter()
    worst = 0.0
    for i in sample:
        keep = np.arange(n) != i
        direct = trG - np.trace(np.linalg.inv(H[np.ix_(keep, keep)]))
        worst = max(worst, abs(direct - drop[i]))
    t_naive_sample = time.perf_counter() - t0
    t_naive_full = t_naive_sample * n / len(sample)
    print(f"n = {n} sites")
    print(f"single-site sweep: cavity {t_solve + t_sweep:.3f}s (one solve + downdates) "
          f"vs naive ~{t_naive_full:.1f}s ({n} re-inversions)   "
          f"speedup ~{t_naive_full / (t_solve + t_sweep):.0f}x")
    print(f"max |cavity - direct| over {len(sample)} checked sites: {worst:.3e}")

    # ---- pair pinning: 2x2 downdates, interaction vs distance -----------------
    pairs = rng.choice(n, size=(4000, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    t0 = time.perf_counter()
    i, j = pairs[:, 0], pairs[:, 1]
    Gii, Gjj, Gij = np.diag(G)[i], np.diag(G)[j], G[i, j]
    det = Gii * Gjj - Gij**2
    ci = np.einsum("ji,ji->i", G[:, i], G[:, i])        # ||G_{.i}||^2
    cj = np.einsum("ji,ji->i", G[:, j], G[:, j])
    cij = np.einsum("ji,ji->i", G[:, i], G[:, j])       # <G_{.i}, G_{.j}>
    drop_pair = (Gjj * ci - 2.0 * Gij * cij + Gii * cj) / det
    t_pairs = time.perf_counter() - t0
    interaction = drop[i] + drop[j] - drop_pair
    dist = np.hypot(*(xy[i] - xy[j]).T)

    # verify a few pairs directly
    worst_pair = 0.0
    for row, (a, b) in enumerate(pairs[:6]):
        keep = np.setdiff1d(np.arange(n), [a, b])
        direct = trG - np.trace(np.linalg.inv(H[np.ix_(keep, keep)]))
        worst_pair = max(worst_pair, abs(direct - drop_pair[row]))
    print(f"{len(pairs)} pair deletions priced in {t_pairs * 1e3:.0f}ms "
          f"(2x2 downdates); max |cavity - direct| on checked pairs: {worst_pair:.3e}")

    # ---- figures ---------------------------------------------------------------
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.log10(drop), s=26, cmap="inferno")
    ax.set_title("Soft-spot map: log10 softening removed by pinning each site\n"
                 "(all sites from ONE inverse)", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(fig_dir / "soft_spots.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    ax2.semilogy(dist, np.abs(interaction), ".", ms=2, alpha=0.35, color="#c2410c")
    bins = np.linspace(0, dist.max(), 24)
    mids = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(np.abs(interaction)[(dist >= a) & (dist < b)])
           for a, b in zip(bins[:-1], bins[1:])]
    ax2.semilogy(mids, med, "k-", lw=2, label="median")
    ax2.set_xlabel("distance between pinned sites")
    ax2.set_ylabel("|pair interaction|  $I_{ij}$")
    ax2.set_title("Defect-defect interaction from 2x2 downdates\n"
                  f"({len(pairs)} pairs, no re-solves)", fontsize=10)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(fig_dir / "pair_interaction.png", dpi=150)

    print("wrote figures/soft_spots.png, figures/pair_interaction.png")


if __name__ == "__main__":
    main()
