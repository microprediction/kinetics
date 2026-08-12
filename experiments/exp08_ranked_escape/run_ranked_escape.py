"""Experiment 8 (program Q3): ranked narrow escape -- the runner-up principle on
real first-passage physics.

Physics. The same disk-and-windows geometry as experiments 1 and 3, but the boundary
is ALL-REFLECTING and we passively record each trajectory's sequence of distinct
window encounters. The point is the truncation identity: for any absorbing set A, the
counterfactual winner is exactly the first element of the sequence lying in A --
absorption merely truncates the reflecting path at its first A-encounter. One
trajectory therefore answers EVERY blocked-set counterfactual, and a held-out batch
of sequences is exact ground truth for all of them at once. (A direct absorption
re-simulation cross-checks the identity for one block set.)

Question. How much substitution structure does each DATA REGIME buy? The runner-up
principle says winner-only data identifies nothing about redistribution; the
runner-up identity determines every singleton scratch; deeper prefixes determine
deeper deletions. Models compared, by the data they consume:

  winner-only:      Harville / IIA renormalization of the winner frequencies
                    independent Thurstone (fast ability transform, sigma = 1)
  winner+runner-up: empirical runner-up kernel M with the exact singleton identity
                    q_j = p_j + p_i M_ij, composed for multi-blocks by the Markov
                    substitution resolvent  q_A = p_A + p_B (I - M_BB)^{-1} M_BA
  top-k prefixes:   direct evaluation of training sequences (k = 4)

Evaluated on random singleton, pair, and triple block sets against held-out-sequence
ground truth.

Run:  python experiments/exp08_ranked_escape/run_ranked_escape.py
Outputs: results.csv, figures/regimes.png
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
from raceutil import abilities_from_probabilities, win_probabilities  # noqa: E402
from run_narrow_escape import N_W, make_windows, simulate, tv  # noqa: E402

HERE = Path(__file__).resolve().parent

DT = 5e-5
MAX_STEPS = 3_000_000
K_SEQ = 8                 # distinct windows recorded per trajectory
N_TRAJ = 24_000           # half train, half held-out truth
N_BLOCKS = 12             # random block sets per size
SEED = 42                 # same geometry as experiments 1 and 3


def simulate_sequences(centers, halfw, n_walkers, rng):
    """Reflecting Brownian motion; per walker, the first K_SEQ distinct windows hit."""
    pos = np.zeros((n_walkers, 2))
    seqs = np.full((n_walkers, K_SEQ), -1, dtype=np.int16)
    counts = np.zeros(n_walkers, dtype=np.int8)
    seen = np.zeros((n_walkers, N_W), dtype=bool)
    active = np.arange(n_walkers)
    step_sd = np.sqrt(2.0 * DT)
    t0 = time.perf_counter()
    for _ in range(MAX_STEPS):
        if len(active) == 0:
            break
        pos[active] += step_sd * rng.standard_normal((len(active), 2))
        r = np.hypot(pos[active, 0], pos[active, 1])
        out = r >= 1.0
        if np.any(out):
            hit_idx = active[out]
            theta = np.arctan2(pos[hit_idx, 1], pos[hit_idx, 0]) % (2 * np.pi)
            d = np.abs((theta[:, None] - centers[None, :] + np.pi) % (2 * np.pi) - np.pi)
            win = d <= halfw[None, :]
            in_window = win.any(axis=1)
            widx = np.argmax(win, axis=1)
            # record first-time encounters (passively; path is unchanged)
            rec = in_window & ~seen[hit_idx, widx]
            if np.any(rec):
                w_rec = hit_idx[rec]
                seqs[w_rec, counts[w_rec]] = widx[rec]
                seen[w_rec, widx[rec]] = True
                counts[w_rec] += 1
            # reflect everyone who left
            rr = r[out]
            pos[hit_idx] *= ((2.0 - rr) / rr)[:, None]
            active = active[counts[active] < K_SEQ]
    n_full = int(np.sum(counts >= K_SEQ))
    print(f"  sequences: {n_full}/{n_walkers} walkers reached {K_SEQ} distinct windows, "
          f"{time.perf_counter() - t0:.0f}s")
    return seqs


def first_in(seqs, allowed_mask):
    """Winner per trajectory for available set: first sequence entry in the set."""
    n = len(seqs)
    winners = np.full(n, -1, dtype=int)
    undecided = np.ones(n, dtype=bool)
    for c in range(seqs.shape[1]):
        col = seqs[:, c]
        ok = undecided & (col >= 0) & allowed_mask[np.clip(col, 0, None)]
        winners[ok] = col[ok]
        undecided &= ~ok
    return winners  # -1 where the recorded prefix never met the set


def freqs_from(seqs, keep):
    mask = np.zeros(N_W, dtype=bool)
    mask[keep] = True
    w = first_in(seqs, mask)
    ok = w >= 0
    counts = np.bincount(w[ok], minlength=N_W)[keep]
    return counts / counts.sum(), 1.0 - ok.mean()


def main():
    rng = np.random.default_rng(SEED)
    centers, halfw = make_windows(rng)

    print("reflecting simulation (encounter sequences):")
    seqs = simulate_sequences(centers, halfw, N_TRAJ, rng)
    train, test = seqs[: N_TRAJ // 2], seqs[N_TRAJ // 2 :]

    # cross-check the truncation identity against a real absorbing simulation
    all_idx = np.arange(N_W)
    check_block = np.array([3, 6])
    keep = np.setdiff1d(all_idx, check_block)
    mask = np.ones(N_W, bool); mask[check_block] = False
    q_abs = simulate(centers, halfw, mask, 30_000, rng, "identity check")
    q_abs = q_abs[keep] / q_abs.sum()
    q_seq, _ = freqs_from(seqs, keep)
    print(f"  truncation identity: TV(sequence eval, absorbing sim) = "
          f"{tv(q_seq, q_abs):.4f} (sampling noise ~0.006)")

    # --- data regimes from the TRAIN half --------------------------------------
    p_hat, _ = freqs_from(train, all_idx)          # winner-only frequencies
    p_hat = np.maximum(p_hat, 1e-9)
    mu_hat = abilities_from_probabilities(p_hat, 1.0)

    # runner-up kernel from (winner, runner-up) pairs only
    M = np.zeros((N_W, N_W))
    w1, w2 = train[:, 0], train[:, 1]
    ok = (w1 >= 0) & (w2 >= 0)
    np.add.at(M, (w1[ok], w2[ok]), 1.0)
    M = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)

    def model_harville(B, keep):
        q = p_hat[keep]
        return q / q.sum()

    def model_thurstone(B, keep):
        return win_probabilities(mu_hat[keep], 1.0)

    def model_markov(B, keep):
        MBB = M[np.ix_(B, B)]
        MBA = M[np.ix_(B, keep)]
        q = p_hat[keep] + p_hat[B] @ np.linalg.solve(np.eye(len(B)) - MBB, MBA)
        return q / q.sum()

    def model_topk(B, keep):
        q, _ = freqs_from(train[:, :4], keep)      # top-4 prefixes only
        return q

    models = {"Harville (winner only)": model_harville,
              "indep. Thurstone (winner only)": model_thurstone,
              "Markov M (winner+runner-up)": model_markov,
              "top-4 prefixes": model_topk}

    # --- evaluation over random block sets --------------------------------------
    rows = ["block_size,model,mean_tv,max_tv"]
    results = {name: {s: [] for s in (1, 2, 3)} for name in models}
    brng = np.random.default_rng(1)
    for size in (1, 2, 3):
        blocks = [np.sort(brng.choice(N_W, size=size, replace=False))
                  for _ in range(N_BLOCKS)]
        for B in blocks:
            keep = np.setdiff1d(all_idx, B)
            truth, dropped = freqs_from(test, keep)
            for name, fn in models.items():
                results[name][size].append(tv(fn(B, keep), truth))
    print(f"\nTV vs held-out-sequence truth ({N_BLOCKS} random blocks per size):")
    print(f"{'model':>34} {'singles':>9} {'pairs':>9} {'triples':>9}")
    for name in models:
        means = [np.mean(results[name][s]) for s in (1, 2, 3)]
        print(f"{name:>34} {means[0]:>9.4f} {means[1]:>9.4f} {means[2]:>9.4f}")
        for s, m in zip((1, 2, 3), means):
            rows.append(f"{s},{name},{m:.6f},{max(results[name][s]):.6f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # --- figure -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(3)
    wd = 0.19
    colors = ["#9a9a9a", "#e8a87c", "#c2410c", "#2a1a12"]
    for j, (name, c) in enumerate(zip(models, colors)):
        means = [np.mean(results[name][s]) for s in (1, 2, 3)]
        ax.bar(xs + (j - 1.5) * wd, means, wd, label=name, color=c)
    ax.set_xticks(xs, ["singletons", "pairs", "triples"])
    ax.set_ylabel("mean TV vs held-out-sequence truth")
    ax.set_title("The runner-up principle on real first-passage physics:\n"
                 "what each data regime buys", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "regimes.png", dpi=150)
    print("\nwrote results.csv, figures/regimes.png")


if __name__ == "__main__":
    main()
