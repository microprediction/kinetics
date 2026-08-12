"""Experiment 11 (program Q1/Q4, Paper 3): harmonic cavity screening for RELAXED
vacancies in a pre-stressed vector elastic network.

The review's prescription, implemented in full:
  * VECTOR elasticity (2D displacements), not a scalar Laplacian;
  * TRUE vacancies: remove a site and all incident bonds (a local Woodbury update
    plus a leave-2-out downdate), not pinning;
  * ANHARMONIC ground truth: full nonlinear energy minimization per vacancy
    (geometric nonlinearity from finite rotations of pre-stressed bonds);
  * SCREENING metrics: rank correlation, top-K recall at a relax-budget, and the
    saved relaxation calls -- not machine-precision self-agreement.

Physics. Nodes on a jittered triangular patch, boundary clamped; harmonic bond law
V_b = (k_b/2)(r - ell_b)^2 with frustrated natural lengths ell_b (quenched mismatch),
relaxed to a pre-stressed equilibrium x*. Removing site v leaves unbalanced forces
g_v on its neighbors; the network re-relaxes, releasing energy dE(v) >= 0.

Harmonic prediction, ALL vacancies from ONE global inverse G = H^{-1}:
  dE_harm(v) = (1/2) g^T (H_v)^{-1} g, with H_v reached from G by a 2x2 leave-out
  (the site's own dofs) and a push-through Woodbury on the neighbor block:
      val = g^T (I - S C)^{-1} S g,
  S = downdated neighbor-block of G, C = sum of removed bond Hessian blocks.
  (The push-through form needs no C^{-1}, so degenerate bond states are safe.)
  Verified against direct dense re-assembly on sample vacancies.

Run:  python experiments/exp11_vacancy_screening/run_vacancy_screening.py
Outputs: results.csv, figures/screening.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SIDE = 22           # triangular patch of ~SIDE^2 nodes
MISMATCH = 0.08     # natural-length frustration (quenched disorder)
SEED = 13
TOP_K = 20
BUDGET = 2 * TOP_K  # relax-budget for screen-then-relax


def build_network(rng):
    """Jittered triangular lattice; outer layer clamped; frustrated bonds."""
    pts = []
    for row in range(SIDE):
        for col in range(SIDE):
            x = col + 0.5 * (row % 2) + 0.06 * rng.standard_normal()
            y = row * np.sqrt(3) / 2 + 0.06 * rng.standard_normal()
            pts.append((x, y))
    X = np.array(pts)
    n = len(X)
    # bonds: pairs closer than 1.35 lattice units
    bonds = []
    for i in range(n):
        d = np.hypot(*(X[i] - X[i + 1 :]).T)
        for j in np.nonzero(d < 1.35)[0]:
            bonds.append((i, i + 1 + j))
    bonds = np.array(bonds)
    kb = rng.lognormal(0.0, 0.3, size=len(bonds))
    d0 = np.hypot(*(X[bonds[:, 0]] - X[bonds[:, 1]]).T)
    ell = d0 * (1.0 + MISMATCH * rng.standard_normal(len(bonds)))
    # clamp the outer layer (removes rigid modes; sample held at its edges)
    row = np.repeat(np.arange(SIDE), SIDE)
    col = np.tile(np.arange(SIDE), SIDE)
    clamped = (row == 0) | (row == SIDE - 1) | (col == 0) | (col == SIDE - 1)
    return X, bonds, kb, ell, ~clamped  # free-node mask


def energy_grad(xflat, X0, free, bonds, kb, ell):
    """Total energy and gradient w.r.t. free-node coordinates."""
    pos = X0.copy()
    pos[free] = xflat.reshape(-1, 2)
    dvec = pos[bonds[:, 0]] - pos[bonds[:, 1]]
    r = np.hypot(dvec[:, 0], dvec[:, 1])
    E = 0.5 * np.sum(kb * (r - ell) ** 2)
    fb = (kb * (r - ell) / r)[:, None] * dvec        # bond force on endpoint 0
    grad = np.zeros_like(pos)
    np.add.at(grad, bonds[:, 0], fb)
    np.add.at(grad, bonds[:, 1], -fb)
    return E, grad[free].ravel()


def relax(X0, free, bonds, kb, ell, x_start=None):
    x0 = (X0[free] if x_start is None else x_start).ravel()
    res = minimize(energy_grad, x0, args=(X0, free, bonds, kb, ell),
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": 4000, "ftol": 1e-15, "gtol": 1e-10})
    return res.fun, res.x.reshape(-1, 2), res


def bond_hessian_blocks(pos, bonds, kb, ell):
    """Per-bond 2x2 block: k t t^T + (k (r-ell)/r)(I - t t^T)."""
    dvec = pos[bonds[:, 0]] - pos[bonds[:, 1]]
    r = np.hypot(dvec[:, 0], dvec[:, 1])
    t = dvec / r[:, None]
    tt = t[:, :, None] * t[:, None, :]
    eye = np.eye(2)[None, :, :]
    return kb[:, None, None] * tt + (kb * (r - ell) / r)[:, None, None] * (eye - tt)


def assemble_hessian(pos, free_nodes, bonds, kb, ell, node_dof):
    """Dense Hessian on free dofs."""
    nf = 2 * len(free_nodes)
    H = np.zeros((nf, nf))
    B = bond_hessian_blocks(pos, bonds, kb, ell)
    for (i, j), Bb in zip(bonds, B):
        di, dj = node_dof.get(i), node_dof.get(j)
        if di is not None:
            H[di : di + 2, di : di + 2] += Bb
        if dj is not None:
            H[dj : dj + 2, dj : dj + 2] += Bb
        if di is not None and dj is not None:
            H[di : di + 2, dj : dj + 2] -= Bb
            H[dj : dj + 2, di : di + 2] -= Bb
    return H


def run_one(mismatch, make_figure=False):
    global MISMATCH
    MISMATCH = mismatch
    rng = np.random.default_rng(SEED)
    X0, bonds, kb, ell, free = build_network(rng)
    free_nodes = np.nonzero(free)[0]
    node_dof = {node: 2 * k for k, node in enumerate(free_nodes)}
    print(f"network: {len(X0)} nodes ({len(free_nodes)} free), {len(bonds)} bonds")

    # ---- pristine relaxed state --------------------------------------------------
    E0, xfree, res = relax(X0, free, bonds, kb, ell)
    assert res.success or np.abs(res.jac).max() < 1e-7, "pristine relaxation failed"
    pos = X0.copy(); pos[free] = xfree
    print(f"pristine relaxed: E0 = {E0:.4f}, |grad|_max = {np.abs(res.jac).max():.1e}")

    H = assemble_hessian(pos, free_nodes, bonds, kb, ell, node_dof)
    np.linalg.cholesky(H)                              # PD at a strict minimum
    t0 = time.perf_counter()
    G = np.linalg.inv(H)
    t_inv = time.perf_counter() - t0

    # candidates: free sites all of whose neighbors are also free
    nbrs = {v: [] for v in range(len(X0))}
    for bidx, (i, j) in enumerate(bonds):
        nbrs[i].append((bidx, j)); nbrs[j].append((bidx, i))
    cands = [v for v in free_nodes
             if all(nb in node_dof for _, nb in nbrs[v])]
    print(f"{len(cands)} vacancy candidates")

    # ---- harmonic screening: every vacancy from the ONE inverse -------------------
    B_all = bond_hessian_blocks(pos, bonds, kb, ell)
    dvec = pos[bonds[:, 0]] - pos[bonds[:, 1]]
    r = np.hypot(dvec[:, 0], dvec[:, 1])
    fb = (kb * (r - ell) / r)[:, None] * dvec          # force of bond on endpoint 0

    t0 = time.perf_counter()
    dE_harm = np.zeros(len(cands))
    for c, v in enumerate(cands):
        items = nbrs[v]
        z = len(items)
        nb_dofs = np.concatenate([[node_dof[nb], node_dof[nb] + 1] for _, nb in items])
        v_dofs = np.array([node_dof[v], node_dof[v] + 1])
        # residual force on neighbors from deleting v's bonds (pristine grad is 0)
        g = np.zeros(2 * z)
        C = np.zeros((2 * z, 2 * z))
        for a, (bidx, nb) in enumerate(items):
            i, j = bonds[bidx]
            sgn = 1.0 if j == nb else -1.0             # force on the neighbor end
            g[2 * a : 2 * a + 2] = -sgn * fb[bidx]     # remaining-network gradient
            C[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] = B_all[bidx]
        # leave-2-out downdate of G at v's dofs, on the neighbor block only
        Gnn = G[np.ix_(nb_dofs, nb_dofs)]
        Gnv = G[np.ix_(nb_dofs, v_dofs)]
        Gvv = G[np.ix_(v_dofs, v_dofs)]
        S = Gnn - Gnv @ np.linalg.solve(Gvv, Gnv.T)
        # push-through Woodbury for the bond removal (no C^{-1} needed)
        Sg = np.linalg.solve(np.eye(2 * z) - S @ C, S @ g)
        dE_harm[c] = 0.5 * g @ Sg
    t_harm = time.perf_counter() - t0
    print(f"harmonic sweep: all {len(cands)} vacancies in {t_harm*1e3:.0f} ms "
          f"(+ one {H.shape[0]}x{H.shape[0]} inverse, {t_inv:.2f}s)")

    # ---- exactness check of the local algebra (theorem hygiene) -------------------
    worst = 0.0
    for v in [cands[k] for k in (0, len(cands) // 2, len(cands) - 1)]:
        keep_bonds = np.array([not (v in b) for b in bonds])
        keep_nodes = free_nodes[free_nodes != v]
        nd = {node: 2 * k for k, node in enumerate(keep_nodes)}
        Hv = assemble_hessian(pos, keep_nodes, bonds[keep_bonds],
                              kb[keep_bonds], ell[keep_bonds], nd)
        gfull = np.zeros(2 * len(keep_nodes))
        for bidx, nb in nbrs[v]:
            i, j = bonds[bidx]
            sgn = 1.0 if j == nb else -1.0
            gfull[nd[nb] : nd[nb] + 2] = -sgn * fb[bidx]
        direct = 0.5 * gfull @ np.linalg.solve(Hv, gfull)
        worst = max(worst, abs(direct - dE_harm[cands.index(v)]))
    print(f"local Woodbury vs direct re-assembly: max |diff| = {worst:.2e}")

    # ---- anharmonic ground truth ----------------------------------------------------
    print("nonlinear relaxations (ground truth):")
    t0 = time.perf_counter()
    dE_true = np.zeros(len(cands))
    for c, v in enumerate(cands):
        keep_bonds = np.array([not (v in b) for b in bonds])
        free_v = free.copy(); free_v[v] = False
        # energy of the damaged network at the pristine configuration
        dvec_k = pos[bonds[keep_bonds][:, 0]] - pos[bonds[keep_bonds][:, 1]]
        rk = np.hypot(dvec_k[:, 0], dvec_k[:, 1])
        E_frozen = 0.5 * np.sum(kb[keep_bonds] * (rk - ell[keep_bonds]) ** 2)
        E_min, _, r2 = relax(X0, free_v, bonds[keep_bonds], kb[keep_bonds],
                             ell[keep_bonds], x_start=pos[free_v])
        dE_true[c] = E_frozen - E_min
        if (c + 1) % 100 == 0:
            print(f"  {c+1}/{len(cands)} done, {time.perf_counter()-t0:.0f}s")
    t_true = time.perf_counter() - t0
    print(f"  total {t_true:.0f}s ({t_true/len(cands)*1e3:.0f} ms per relaxation)")

    # ---- screening metrics -----------------------------------------------------------
    rho = spearmanr(dE_harm, dE_true).statistic
    order_true = np.argsort(-dE_true)
    order_harm = np.argsort(-dE_harm)
    topk_true = set(order_true[:TOP_K].tolist())
    recall_budget = len(topk_true & set(order_harm[:BUDGET].tolist())) / TOP_K
    recall_topk = len(topk_true & set(order_harm[:TOP_K].tolist())) / TOP_K
    ratio = dE_harm / dE_true
    print(f"\nscreening: Spearman rho = {rho:.4f}")
    print(f"top-{TOP_K} recall @ budget {BUDGET}: {recall_budget:.2f} "
          f"(@ budget {TOP_K}: {recall_topk:.2f})")
    print(f"harmonic/true energy ratio: median {np.median(ratio):.3f}, "
          f"5-95% [{np.quantile(ratio, 0.05):.3f}, {np.quantile(ratio, 0.95):.3f}]")
    speed = t_true / (t_true * BUDGET / len(cands) + t_inv + t_harm)
    print(f"screen-then-relax cost: {speed:.1f}x cheaper than relax-everything")

    metrics = dict(mismatch=mismatch, spearman=rho, recall_budget=recall_budget,
                   recall_topk=recall_topk, ratio_lo=float(np.quantile(ratio, 0.05)),
                   ratio_hi=float(np.quantile(ratio, 0.95)),
                   woodbury=worst, sweep_ms=t_harm * 1e3, relax_s=t_true,
                   speedup=speed)

    # ---- figure --------------------------------------------------------------------------
    if not make_figure:
        return metrics
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.loglog(dE_true, dE_harm, ".", ms=5, color="#c2410c", alpha=0.6)
    lim = [min(dE_true.min(), dE_harm.min()) * 0.8,
           max(dE_true.max(), dE_harm.max()) * 1.2]
    ax.plot(lim, lim, ":", color="#9a9a9a")
    hits = np.array([c in topk_true for c in range(len(cands))])
    ax.loglog(dE_true[hits], dE_harm[hits], "o", ms=7, mfc="none", mec="#2a1a12",
              label=f"true top {TOP_K}")
    ax.set_xlabel("relaxed (anharmonic) vacancy energy release")
    ax.set_ylabel("harmonic cavity prediction")
    ax.set_title(f"Harmonic screening of {len(cands)} vacancies: "
                 f"Spearman {rho:.3f},\n"
                 f"top-{TOP_K} recall {recall_budget:.0%} at a {BUDGET}-relaxation "
                 f"budget", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "screening.png", dpi=150)
    return metrics


def main():
    all_rows = ["mismatch,spearman,recall_budget,recall_topk,ratio_q05,ratio_q95,"
                "woodbury,sweep_ms,relax_s,speedup"]
    sweep = []
    for mis in (0.1, 0.2, 0.3, 0.4):
        print(f"\n===== mismatch {mis} =====")
        r = run_one(mis, make_figure=(mis == 0.3))     # headline figure at 0.3
        sweep.append(r)
        all_rows.append(f"{mis},{r['spearman']:.4f},{r['recall_budget']:.3f},"
                        f"{r['recall_topk']:.3f},{r['ratio_lo']:.3f},{r['ratio_hi']:.3f},"
                        f"{r['woodbury']:.2e},{r['sweep_ms']:.0f},{r['relax_s']:.0f},"
                        f"{r['speedup']:.1f}")
    (HERE / "results.csv").write_text("\n".join(all_rows) + "\n")

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ms = [r["mismatch"] for r in sweep]
    ax.plot(ms, [r["spearman"] for r in sweep], "o-", color="#2a1a12",
            label="Spearman rank correlation")
    ax.plot(ms, [r["recall_budget"] for r in sweep], "s-", color="#c2410c",
            label=f"top-{TOP_K} recall @ {BUDGET}-relaxation budget")
    ax.set_xlabel("bond-length mismatch (anharmonicity / frustration)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("When does harmonic cavity screening break?", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "breakdown.png", dpi=150)
    print("\nwrote results.csv, figures/screening.png, figures/breakdown.png")


if __name__ == "__main__":
    main()
