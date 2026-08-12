"""Experiment 10 (Q3 x Q7): predicting the substitution kernel from pure geometry.

Experiment 8 showed the runner-up kernel M -- measured from trajectory data -- feeds
a Markov substitution resolvent that answers blocked-set counterfactuals at the
noise floor. This experiment asks whether M (and the winner distribution p itself)
can be PREDICTED with no trajectory data at all: from Green-function hitting splits
of the same finite-volume disk generator validated in experiment 9.

Objects computed from the generator alone (sparse solves; no simulation):
  p_geo[j]   = P(first window hit from the center is j)
  M_geo[i,j] = P(next distinct window hit is j | just hit window i)
               (start: invariant law restricted to window i's boundary ring;
                sensitivity to that choice is reported)

Both feed the substitution resolvent  q_A = p_A + p_B (I - M_BB)^{-1} M_BA.

Models compared on random blocked sets against held-out encounter sequences:
  Harville (winner-only renormalization)          [trajectory p, no structure]
  empirical M resolvent                            [trajectory p and M]
  hybrid: trajectory p + GEOMETRIC M               [trajectory p, geometric structure]
  pure geometry: p_geo + M_geo                     [NO trajectory data at all]

Care notes. Windows are DISJOINT BY CONSTRUCTION (experiment 9 found the seed-42
geometry has overlapping pairs, which poisons encounter semantics). Encounters on
the grid are entries into a window's outermost ring of cells, the closest analogue
of boundary contact; ring depth is a documented discretization parameter.

Run:  python experiments/exp10_green_function_M/run_green_function_M.py
Outputs: results.csv, figures/M_scatter.png, figures/models.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp01_narrow_escape"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp08_ranked_escape"))
from run_narrow_escape import tv  # noqa: E402
from run_ranked_escape import first_in, simulate_sequences  # noqa: E402

HERE = Path(__file__).resolve().parent

N_W = 16
NR, NTH = 60, 176
N_TRAJ = 24_000
N_BLOCKS = 12
SEED = 7


def make_disjoint_windows(rng):
    """16 windows with guaranteed disjointness: min gap, half-widths clipped."""
    while True:
        centers = np.sort(rng.uniform(0.0, 2 * np.pi, size=N_W))
        gaps = np.diff(np.concatenate([centers, [centers[0] + 2 * np.pi]]))
        if gaps.min() > 0.10:
            break
    halfw = np.clip(rng.lognormal(np.log(0.035), 0.4, size=N_W), 0.02, 0.09)
    left_gap = np.roll(gaps, 1)
    halfw = np.minimum(halfw, 0.45 * np.minimum(gaps, left_gap))
    # verify, don't assume
    for i in range(N_W):
        j = (i + 1) % N_W
        gap = (centers[j] - centers[i]) % (2 * np.pi)
        assert gap > halfw[i] + halfw[j], "windows must be disjoint"
    return centers, halfw


def sparse_disk_generator():
    """Sparse finite-volume Neumann generator (same construction as exp09)."""
    dr = 1.0 / NR
    dth = 2.0 * np.pi / NTH
    r_cent = (np.arange(NR) + 0.5) * dr
    n = NR * NTH
    idx = lambda m, j: m * NTH + (j % NTH)
    area = np.repeat(r_cent * dr * dth, NTH)
    Lg = lil_matrix((n, n))
    for m in range(NR):
        for j in range(NTH):
            i = idx(m, j)
            if m + 1 < NR:
                cond = ((m + 1) * dr * dth) / dr
                k = idx(m + 1, j)
                Lg[i, k] += cond / area[i]
                Lg[k, i] += cond / area[k]
            cond = dr / (r_cent[m] * dth)
            k = idx(m, j + 1)
            Lg[i, k] += cond / area[i]
            Lg[k, i] += cond / area[k]
    Lg = csr_matrix(Lg)
    diag = -np.asarray(Lg.sum(axis=1)).ravel()
    Lg = Lg + csr_matrix((diag, (np.arange(n), np.arange(n))), shape=(n, n))
    pi = area / area.sum()
    assert np.abs(np.asarray(Lg.sum(axis=1)).ravel()).max() < 1e-9 * np.abs(diag).max()
    assert np.abs(pi @ Lg).max() < 1e-9 * np.abs(diag).max()
    theta_cent = (np.arange(NTH) + 0.5) * dth
    return Lg, pi, r_cent, theta_cent


def window_ring_cells(centers, halfw, theta_cent):
    """Outermost-ring cell indices per window (encounter = entry into the ring)."""
    base = (NR - 1) * NTH
    cells = []
    for c, h in zip(centers, halfw):
        d = np.abs((theta_cent - c + np.pi) % (2 * np.pi) - np.pi)
        cells.append(base + np.nonzero(d <= h)[0])
    return cells


def hitting_splits(Lg, start_weights, target_cells):
    """P(first entry into union of targets is target k), from a start distribution.

    Solve on free cells: Q h_k = -L[free, cells_k] 1; splits from a start on free
    cells are start . h_k. Rows over k sum to 1 (verified by caller).
    """
    n = Lg.shape[0]
    absorbing = np.concatenate(target_cells)
    free = np.setdiff1d(np.arange(n), absorbing)
    Q = Lg[np.ix_(free, free)]
    lu = splu(Q.tocsc())
    pos = {c: i for i, c in enumerate(free)}
    w = start_weights[free]
    w = w / w.sum()
    splits = np.zeros(len(target_cells))
    for k, cells in enumerate(target_cells):
        rhs = -np.asarray(Lg[np.ix_(free, cells)].sum(axis=1)).ravel()
        splits[k] = w @ lu.solve(rhs)
    return splits


def main():
    rng = np.random.default_rng(SEED)
    centers, halfw = make_disjoint_windows(rng)
    print(f"disjoint geometry: halfw in [{halfw.min():.3f}, {halfw.max():.3f}]")

    # ---- geometric predictions (no trajectories) --------------------------------
    t0 = time.perf_counter()
    Lg, pi, r_cent, theta_cent = sparse_disk_generator()
    rings = window_ring_cells(centers, halfw, theta_cent)
    n = Lg.shape[0]

    center_start = np.zeros(n)
    center_start[:NTH] = pi[:NTH]                      # innermost ring ~ the center
    p_geo = hitting_splits(Lg, center_start, rings)
    assert abs(p_geo.sum() - 1.0) < 1e-8
    p_geo /= p_geo.sum()

    # exact ENTRY PROFILE: probability of first arriving at window i via cell e
    # (start = center). No start-profile assumption: the first-transition kernel
    # is a two-leg hitting problem, computed exactly on the grid.
    cell_groups = [np.array([c]) for ring in rings for c in ring]
    owner = np.concatenate([[i] * len(ring) for i, ring in enumerate(rings)])
    phi = hitting_splits(Lg, center_start, cell_groups)
    assert abs(phi.sum() - 1.0) < 1e-8

    M_geo = np.zeros((N_W, N_W))          # uniform-ring start (the naive choice)
    M_prof = np.zeros((N_W, N_W))         # exact entry-profile start
    for i in range(N_W):
        others = [rings[j] for j in range(N_W) if j != i]
        start = np.zeros(n); start[rings[i]] = pi[rings[i]]
        row = hitting_splits(Lg, start, others)
        M_geo[i, np.arange(N_W) != i] = row / row.sum()
        prof = np.zeros(n)
        prof[np.concatenate(cell_groups)[owner == i]] = phi[owner == i]
        row2 = hitting_splits(Lg, prof, others)
        M_prof[i, np.arange(N_W) != i] = row2 / row2.sum()
    sens = np.abs(M_geo - M_prof).max()
    print(f"geometric p and M from sparse solves: {time.perf_counter()-t0:.0f}s; "
          f"uniform-ring vs exact-entry-profile M: max diff {sens:.3f}")

    # ---- trajectory data ----------------------------------------------------------
    print("reflecting simulation (encounter sequences):")
    seqs = simulate_sequences(centers, halfw, N_TRAJ, rng)
    train, test = seqs[: N_TRAJ // 2], seqs[N_TRAJ // 2 :]

    def freqs_from(S, keep):
        mask = np.zeros(N_W, dtype=bool); mask[keep] = True
        w = first_in(S, mask)
        ok = w >= 0
        c = np.bincount(w[ok], minlength=N_W)[keep]
        return c / c.sum()

    all_idx = np.arange(N_W)
    p_hat = freqs_from(train, all_idx)
    def emp_kernel(cols):
        M = np.zeros((N_W, N_W))
        for c in cols:
            a, b = train[:, c], train[:, c + 1]
            ok = (a >= 0) & (b >= 0)
            np.add.at(M, (a[ok], b[ok]), 1.0)
        return M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)

    M_first = emp_kernel([0])                          # first transitions only
    M_emp = emp_kernel(range(train.shape[1] - 1))      # stationary encounter chain
    # the two differ (max 0.21), mostly on adjacent entries: the encounter chain
    # is not the first-transition kernel. Counterfactuals want the latter.

    e_p = np.abs(p_geo - p_hat).max()
    e_M = np.abs(M_geo - M_first).max()
    e_Mp = np.abs(M_prof - M_first).max()
    print(f"|p_geo - p_hat|_max = {e_p:.4f} (train noise ~{np.sqrt(p_hat.max()/12000):.3f})")
    print(f"|M_geo - M_first|_max  = {e_M:.4f}   (uniform ring start)")
    print(f"|M_prof - M_first|_max = {e_Mp:.4f}   (exact entry profile)")

    # ---- blocked-set evaluation ----------------------------------------------------
    def resolvent(p, M, B, keep):
        MBB = M[np.ix_(B, B)]
        q = p[keep] + p[B] @ np.linalg.solve(np.eye(len(B)) - MBB, M[np.ix_(B, keep)])
        return q / q.sum()

    models = {
        "Harville (trajectory p)": lambda B, keep: p_hat[keep] / p_hat[keep].sum(),
        "empirical M (first transitions)": lambda B, keep: resolvent(p_hat, M_first, B, keep),
        "empirical M (all pairs)": lambda B, keep: resolvent(p_hat, M_emp, B, keep),
        "hybrid: traj. p + M_geo (ring start)": lambda B, keep: resolvent(p_hat, M_geo, B, keep),
        "hybrid: traj. p + M_geo (entry profile)": lambda B, keep: resolvent(p_hat, M_prof, B, keep),
        "pure geometry (p_geo + M_prof)": lambda B, keep: resolvent(p_geo, M_prof, B, keep),
    }
    results = {name: {s: [] for s in (1, 2, 3)} for name in models}
    brng = np.random.default_rng(1)
    rows = ["block_size,model,mean_tv,max_tv"]
    for size in (1, 2, 3):
        for _ in range(N_BLOCKS):
            B = np.sort(brng.choice(N_W, size=size, replace=False))
            keep = np.setdiff1d(all_idx, B)
            truth = freqs_from(test, keep)
            for name, fn in models.items():
                results[name][size].append(tv(fn(B, keep), truth))
    print(f"\nTV vs held-out sequences ({N_BLOCKS} random blocks per size):")
    print(f"{'model':>36} {'singles':>9} {'pairs':>9} {'triples':>9}")
    for name in models:
        means = [np.mean(results[name][s]) for s in (1, 2, 3)]
        print(f"{name:>36} {means[0]:>9.4f} {means[1]:>9.4f} {means[2]:>9.4f}")
        for s, m in zip((1, 2, 3), means):
            rows.append(f"{s},{name},{m:.6f},{max(results[name][s]):.6f}")
    rows += [f",p_geo_vs_p_hat,{e_p:.6f}", f",M_geo_vs_M_first,{e_M:.6f}",
             f",M_prof_vs_M_first,{e_Mp:.6f}", f",ring_vs_profile,{sens:.6f}"]
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---- figures ----------------------------------------------------------------------
    fig_dir = HERE / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    off = ~np.eye(N_W, dtype=bool)
    ax.plot(M_first[off], M_geo[off], ".", ms=5, color="#9a9a9a", alpha=0.6,
            label="uniform ring start")
    ax.plot(M_first[off], M_prof[off], ".", ms=5, color="#c2410c", alpha=0.7,
            label="exact entry profile")
    ax.legend(fontsize=8.5)
    lim = max(M_first.max(), M_geo.max(), M_prof.max())
    ax.plot([0, lim], [0, lim], ":", color="#9a9a9a")
    ax.set_xlabel("empirical M, first transitions (12k trajectories)")
    ax.set_ylabel("geometric M (Green-function solves)")
    ax.set_title("The substitution kernel, predicted from geometry", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "M_scatter.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    xs = np.arange(3); wd = 0.14
    colors = ["#9a9a9a", "#2a1a12", "#6b5d52", "#8a6a52", "#c2410c", "#e8a87c"]
    for j, (name, c) in enumerate(zip(models, colors)):
        means = [np.mean(results[name][s]) for s in (1, 2, 3)]
        ax2.bar(xs + (j - 2.5) * wd, means, wd, label=name, color=c)
    ax2.set_xticks(xs, ["singletons", "pairs", "triples"])
    ax2.set_ylabel("mean TV vs held-out sequences")
    ax2.set_title("Counterfactuals from geometry alone", fontsize=10)
    ax2.legend(fontsize=7.5)
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout(); fig2.savefig(fig_dir / "models.png", dpi=150)
    print("\nwrote results.csv, figures/M_scatter.png, figures/models.png")


if __name__ == "__main__":
    main()
