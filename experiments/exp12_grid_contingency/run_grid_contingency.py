"""Experiment 12: the deletion ensemble on a REAL engineering problem --
N-1 / N-2 transmission contingency screening (NERC TPL-mandated) on IEEE test
systems, against full AC power flow ground truth.

Why this is the direct connection. Grid operators must verify, continuously, that
the network survives every credible outage. The industry's own screening tool --
line outage distribution factors (LODF), generalized to multiple outages (MLODF) --
IS the leave-k-out Schur/Woodbury identity on one inverse of the DC susceptance
matrix: for an outage set S,

    f' = f0 + PTDF[:, S] (I - PTDF[S, S])^{-1} f0[S],

with islanding detected as singularity of the k x k block. This experiment runs the
whole workflow on real data (Alsac & Stott 1974, the classic 30-bus security-analysis
system with true line ratings; IEEE 118-bus for scale), and asks the exp11 question:
how well does the cheap linear (DC) deletion ensemble SCREEN the expensive nonlinear
(full AC Newton power flow) truth?

Layers, mirroring exp11:
  * exactness: MLODF vs direct DC re-solve (machine precision -- the identity);
  * screening: DC severity rank vs AC severity rank over ALL feasible N-2 pairs
    (the 30-bus system is small enough to run full AC truth on every pair);
  * blind spots reported honestly: islanding (detected exactly), AC non-convergence
    (voltage collapse -- invisible to DC), reactive/voltage limits.

Data: MATPOWER case30.m / case118.m (public), parsed directly; no power-systems
libraries. AC solver is a standard polar Newton-Raphson written here and validated
by residual checks and DC/AC base-case consistency.

Run:  python experiments/exp12_grid_contingency/run_grid_contingency.py
Outputs: results.csv, figures/screening.png
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# MATPOWER case parsing
# ---------------------------------------------------------------------------


def parse_case(path):
    text = Path(path).read_text()

    def block(name):
        m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
        rows = []
        for line in m.group(1).strip().splitlines():
            line = line.split("%")[0].strip().rstrip(";")
            if line:
                rows.append([float(x) for x in line.split()])
        return np.array(rows)

    base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text).group(1))
    return base, block("bus"), block("gen"), block("branch")


class Grid:
    def __init__(self, path):
        self.base, bus, gen, br = parse_case(path)
        br = br[br[:, 10] != 0]                       # in-service branches
        self.bus_ids = bus[:, 0].astype(int)
        self.n = len(bus)
        idx = {b: i for i, b in enumerate(self.bus_ids)}
        self.f = np.array([idx[int(b)] for b in br[:, 0]])
        self.t = np.array([idx[int(b)] for b in br[:, 1]])
        self.r, self.x, self.b_line = br[:, 2], br[:, 3], br[:, 4]
        self.rate = br[:, 5]                          # MVA; 0 = unrated
        self.tap = np.where(br[:, 8] == 0.0, 1.0, br[:, 8])
        self.shift = np.deg2rad(br[:, 9])
        self.m = len(br)
        self.Pd, self.Qd = bus[:, 2] / self.base, bus[:, 3] / self.base
        self.Gs, self.Bs = bus[:, 4] / self.base, bus[:, 5] / self.base
        self.slack = int(np.nonzero(bus[:, 1] == 3)[0][0])
        self.pv = np.nonzero(bus[:, 1] == 2)[0]
        gid = np.array([idx[int(g)] for g in gen[:, 0]])
        on = gen[:, 7] > 0
        self.Pg = np.zeros(self.n)
        np.add.at(self.Pg, gid[on], gen[on, 1] / self.base)
        self.Vg = np.ones(self.n)
        self.Vg[gid[on]] = gen[on, 5]


# ---------------------------------------------------------------------------
# DC layer: one inverse, then the whole N-1 / N-2 ensemble by MLODF
# ---------------------------------------------------------------------------


def dc_setup(g: Grid):
    """Reduced susceptance inverse, PTDF, base flows (per-unit, lossless)."""
    w = 1.0 / (g.x * g.tap)                           # branch susceptance
    A = np.zeros((g.m, g.n))
    A[np.arange(g.m), g.f] = 1.0
    A[np.arange(g.m), g.t] = -1.0
    B = A.T @ (w[:, None] * A)
    keep = np.setdiff1d(np.arange(g.n), [g.slack])
    Binv = np.linalg.inv(B[np.ix_(keep, keep)])
    P = g.Pg - g.Pd
    theta = np.zeros(g.n)
    theta[keep] = Binv @ P[keep]
    f0 = w * (A @ theta)
    H = (w[:, None] * A[:, keep]) @ Binv              # PTDF to bus injections
    PTDF_br = H @ A[:, keep].T                        # branch-to-branch
    return f0, PTDF_br


def mlodf_flows(f0, PTDF_br, S):
    """Post-outage flows for outage set S via the leave-k-out resolvent.
    Returns None if the outage islands the system (singular k x k block)."""
    E = np.eye(len(S)) - PTDF_br[np.ix_(S, S)]
    if abs(np.linalg.det(E)) < 1e-8:
        return None
    corr = PTDF_br[:, S] @ np.linalg.solve(E, f0[S])
    f = f0 + corr
    f[S] = 0.0
    return f


def dc_direct(g: Grid, S):
    """Direct DC re-solve with branches S removed (for exactness checks)."""
    keep_br = np.setdiff1d(np.arange(g.m), S)
    w = (1.0 / (g.x * g.tap))[keep_br]
    A = np.zeros((len(keep_br), g.n))
    A[np.arange(len(keep_br)), g.f[keep_br]] = 1.0
    A[np.arange(len(keep_br)), g.t[keep_br]] = -1.0
    B = A.T @ (w[:, None] * A)
    keep = np.setdiff1d(np.arange(g.n), [g.slack])
    P = (g.Pg - g.Pd)[keep]
    theta = np.zeros(g.n)
    theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], P)
    full = np.zeros(g.m)
    full[keep_br] = w * (A @ theta)
    return full


# ---------------------------------------------------------------------------
# AC layer: standard polar Newton-Raphson (ground truth "anharmonicity")
# ---------------------------------------------------------------------------


def build_ybus(g: Grid, drop=()):
    Y = np.zeros((g.n, g.n), dtype=complex)
    for k in range(g.m):
        if k in drop:
            continue
        ys = 1.0 / (g.r[k] + 1j * g.x[k])
        tap = g.tap[k] * np.exp(1j * g.shift[k])
        i, j = g.f[k], g.t[k]
        Y[i, i] += (ys + 1j * g.b_line[k] / 2) / (g.tap[k] ** 2)
        Y[j, j] += ys + 1j * g.b_line[k] / 2
        Y[i, j] += -ys / np.conj(tap)
        Y[j, i] += -ys / tap
    Y[np.arange(g.n), np.arange(g.n)] += g.Gs + 1j * g.Bs
    return Y


def newton_pf(g: Grid, Y, V0=None, tol=1e-9, itmax=30):
    """Polar NR; PV buses hold V and P, slack holds V and angle. Q-limits ignored
    (stated); returns (V complex, converged)."""
    Vm = np.where(g.Vg > 0, g.Vg, 1.0).astype(float).copy()
    Va = np.zeros(g.n)
    if V0 is not None:
        Vm, Va = np.abs(V0).copy(), np.angle(V0).copy()
    pq = np.setdiff1d(np.arange(g.n), np.concatenate([[g.slack], g.pv]))
    pvpq = np.concatenate([g.pv, pq])
    Psp = g.Pg - g.Pd
    Qsp = -g.Qd
    for _ in range(itmax):
        V = Vm * np.exp(1j * Va)
        S = V * np.conj(Y @ V)
        dP = Psp[pvpq] - S.real[pvpq]
        dQ = Qsp[pq] - S.imag[pq]
        mis = np.concatenate([dP, dQ])
        if np.abs(mis).max() < tol:
            return V, True
        # Jacobian (dense; fine at this scale). Standard complex-derivative forms:
        # dS/dVa = j diag(V) conj(diag(I) - Y diag(V));  I = Y V
        # dS/d|V| = diag(Vnorm conj(I)) + diag(V) conj(Y) diag(conj(Vnorm))
        dS_dVa = 1j * np.diag(V) @ (np.diag(np.conj(Y @ V)) - np.conj(Y) @ np.diag(np.conj(V)))
        dS_dVm = np.diag(V / Vm * np.conj(Y @ V)) + np.diag(V) @ np.conj(Y) @ np.diag(np.conj(V) / Vm)
        J = np.block([
            [dS_dVa.real[np.ix_(pvpq, pvpq)], dS_dVm.real[np.ix_(pvpq, pq)]],
            [dS_dVa.imag[np.ix_(pq, pvpq)], dS_dVm.imag[np.ix_(pq, pq)]],
        ])
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return Vm * np.exp(1j * Va), False
        Va[pvpq] += dx[: len(pvpq)]
        Vm[pq] += dx[len(pvpq):]
        if Vm.min() < 0.3:                            # collapsing
            return Vm * np.exp(1j * Va), False
    return Vm * np.exp(1j * Va), False


def branch_mva(g: Grid, V, drop=()):
    """|S| at the from end of each branch, per unit."""
    out = np.zeros(g.m)
    for k in range(g.m):
        if k in drop:
            continue
        ys = 1.0 / (g.r[k] + 1j * g.x[k])
        tap = g.tap[k] * np.exp(1j * g.shift[k])
        i, j = g.f[k], g.t[k]
        Vi, Vj = V[i] / tap, V[j]
        Iij = ys * (Vi - Vj) + 1j * g.b_line[k] / 2 * Vi
        out[k] = abs((V[i] / tap) * np.conj(Iij))
    return out


def severity(flows_pu, rate_pu):
    with np.errstate(divide="ignore"):
        load = np.abs(flows_pu) / rate_pu
    return float(np.nanmax(np.where(rate_pu > 0, load, np.nan)))


def main():
    rows = ["case,quantity,value"]

    # ================= case30: full study with AC ground truth ==================
    g = Grid(HERE / "data" / "case30.m")
    print(f"case30: {g.n} buses, {g.m} branches, all lines rated")
    rate_pu = g.rate / g.base
    f0, PTDF = dc_setup(g)

    # exactness of the deletion ensemble (the identity itself)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(30):
        S = list(rng.choice(g.m, size=2, replace=False))
        pred = mlodf_flows(f0, PTDF, S)
        if pred is None:
            continue
        worst = max(worst, np.abs(pred - dc_direct(g, S)).max())
    print(f"  MLODF vs direct DC re-solve (30 random pairs): max |diff| = {worst:.2e}")
    rows.append(f"case30,mlodf_exactness,{worst:.3e}")

    # AC base case, validated by residual + DC consistency
    # (screening severity below uses the industry-standard base-Q augmentation:
    # DC predicts active power; reactive flow is held at its base-case value,
    # S_est = sqrt(P_dc^2 + Q_base^2). Pure-MW screening is reactive-blind and
    # fails against MVA ratings -- measured and reported.)
    Y = build_ybus(g)
    V, ok = newton_pf(g, Y)
    assert ok, "base-case AC power flow failed"
    S_res = V * np.conj(Y @ V)
    pq = np.setdiff1d(np.arange(g.n), np.concatenate([[g.slack], g.pv]))
    assert np.abs((g.Pg - g.Pd - S_res.real)[np.setdiff1d(np.arange(g.n), [g.slack])]).max() < 1e-8
    mva0 = branch_mva(g, V)
    Q0 = np.sqrt(np.maximum(mva0**2 - f0**2, 0.0))     # base reactive component
    corr0 = np.corrcoef(np.abs(f0), mva0)[0, 1]
    print(f"  AC base case converged; DC vs AC base-flow correlation {corr0:.3f}; "
          f"peak base loading {severity(mva0, rate_pu):.2f}")
    rows.append(f"case30,dc_ac_base_corr,{corr0:.4f}")

    # full N-2 ensemble: DC screen and AC truth for EVERY pair
    pairs = [(a, b) for a in range(g.m) for b in range(a + 1, g.m)]
    t0 = time.perf_counter()
    sev_dc, sev_dc_p, feasible = {}, {}, []
    for S in pairs:
        fpred = mlodf_flows(f0, PTDF, list(S))
        if fpred is None:
            continue
        feasible.append(S)
        sev_dc_p[S] = severity(fpred, rate_pu)         # reactive-blind (reported)
        s_est = np.sqrt(fpred**2 + Q0**2)
        s_est[list(S)] = 0.0
        sev_dc[S] = severity(s_est, rate_pu)           # base-Q augmented (standard)
    t_dc = time.perf_counter() - t0
    print(f"  DC ensemble: {len(pairs)} pairs screened in {t_dc*1e3:.0f} ms "
          f"({len(pairs) - len(feasible)} islanding pairs detected exactly)")
    rows.append(f"case30,dc_ms_all_pairs,{t_dc*1e3:.0f}")
    rows.append(f"case30,islanding_pairs,{len(pairs) - len(feasible)}")

    t0 = time.perf_counter()
    sev_ac, collapsed = {}, []
    for S in feasible:
        Yk = build_ybus(g, drop=set(S))
        Vk, okk = newton_pf(g, Yk, V0=V)
        if not okk:
            collapsed.append(S)
            continue
        sev_ac[S] = severity(branch_mva(g, Vk, drop=set(S)), rate_pu)
    t_ac = time.perf_counter() - t0
    print(f"  AC truth: {len(sev_ac)} converged in {t_ac:.0f}s; "
          f"{len(collapsed)} non-converged (voltage collapse -- DC-invisible)")
    rows.append(f"case30,ac_nonconverged,{len(collapsed)}")

    common = [S for S in feasible if S in sev_ac]
    d = np.array([sev_dc[S] for S in common])
    dp = np.array([sev_dc_p[S] for S in common])
    a = np.array([sev_ac[S] for S in common])
    rho = spearmanr(d, a).statistic
    K, BUD = 20, 40
    top_true = set(np.argsort(-a)[:K].tolist())
    recall = len(top_true & set(np.argsort(-d)[:BUD].tolist())) / K
    recall_p = len(top_true & set(np.argsort(-dp)[:BUD].tolist())) / K
    print(f"  screening over {len(common)} N-2 pairs (top-{K} recall @ budget {BUD}):")
    print(f"    DC + base-Q (standard practice): {recall:.2f}   [Spearman {rho:.3f}; "
          f"rank ties dominate: one base-overloaded line caps most pairs]")
    print(f"    DC active-power only:            {recall_p:.2f}   [reactive-blind]")
    print(f"    speed ratio AC/DC = {t_ac / t_dc:.0f}x")
    rows += [f"case30,recall_baseQ,{recall:.3f}", f"case30,recall_p_only,{recall_p:.3f}",
             f"case30,spearman,{rho:.4f}", f"case30,ac_over_dc_time,{t_ac / t_dc:.1f}"]

    # ================= case118: scale of the ensemble ============================
    g2 = Grid(HERE / "data" / "case118.m")
    f02, PTDF2 = dc_setup(g2)
    pairs2 = [(x, y) for x in range(g2.m) for y in range(x + 1, g2.m)]
    t0 = time.perf_counter()
    island2 = 0
    for S in pairs2:
        if mlodf_flows(f02, PTDF2, list(S)) is None:
            island2 += 1
    t2 = time.perf_counter() - t0
    worst2 = 0.0
    for _ in range(10):
        S = list(rng.choice(g2.m, size=2, replace=False))
        pred = mlodf_flows(f02, PTDF2, S)
        if pred is not None:
            worst2 = max(worst2, np.abs(pred - dc_direct(g2, S)).max())
    print(f"case118: {g2.n} buses, {g2.m} branches; all {len(pairs2)} N-2 pairs "
          f"from one inverse in {t2:.1f}s ({island2} islanding); "
          f"exactness {worst2:.1e}")
    rows += [f"case118,n2_pairs,{len(pairs2)}", f"case118,seconds,{t2:.2f}",
             f"case118,islanding,{island2}", f"case118,exactness,{worst2:.2e}"]

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(a, d, ".", ms=4, color="#c2410c", alpha=0.5)
    hits = np.array([i in top_true for i in range(len(common))])
    ax.plot(a[hits], d[hits], "o", ms=7, mfc="none", mec="#2a1a12",
            label=f"true AC top {K}")
    lim = [min(a.min(), d.min()) * 0.95, max(a.max(), d.max()) * 1.05]
    ax.plot(lim, lim, ":", color="#9a9a9a")
    ax.set_xlabel("AC post-contingency peak loading (truth)")
    ax.set_ylabel("DC-ensemble predicted peak loading")
    ax.set_title(f"IEEE 30-bus, all {len(common)} feasible N-2 outages:\n"
                 f"Spearman {rho:.3f}, top-{K} recall {recall:.0%} at budget {BUD}",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "screening.png", dpi=150)
    print("\nwrote results.csv, figures/screening.png")


if __name__ == "__main__":
    main()
