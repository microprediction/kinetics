"""Rust-accelerated calibration: the raceutil damped Jacobi quasi-Newton
inversion with the per-iteration forward+slope pass delegated to the
fastrace kernel (rust/fastrace; machine-precision parity with raceutil,
tested). The iteration logic is byte-for-byte the same as
raceutil.abilities_from_probabilities_factor; only the O(QNL) passes move
to Rust. Requires `maturin develop --release` in rust/fastrace.
"""

from __future__ import annotations

import numpy as np

import fastrace
from raceutil import abilities_from_probabilities_factor  # warm start reuse

_PFLOOR = 1e-300


def calibrate_rust(p, V, D, F, W, n_iter: int = 50, tol: float = 1e-6,
                   return_info: bool = False):
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    p = p / p.sum()
    logp = np.log(p)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    sd = np.sqrt(D)
    N = len(p)
    F = np.ascontiguousarray(F, dtype=float)
    W = np.ascontiguousarray(W, dtype=float)
    floor = max(1e-9, 1e-4 / N)
    ident = p > floor
    if F.shape[1] >= 1 and np.any(V != 0.0):
        sd_tot = np.sqrt(D + np.sum(V**2, axis=1))
        mu = abilities_from_probabilities_factor(
            p, np.zeros((N, 1)), sd_tot**2, np.zeros((1, 1)), np.ones(1),
            n_iter=n_iter, tol=tol)
    else:
        mu = (logp - logp.mean()) / 2.0
    step_cap = 1.0 * np.sqrt(D + np.sum(V**2, axis=1))
    prev_res = np.inf
    damp = 1.0
    res = np.inf
    it = 0
    for it in range(n_iter):
        phat_n, slope, total = fastrace.forward_and_slopes(mu, V, D, F, W)
        phat = np.maximum(phat_n, _PFLOOR)
        resid = np.log(phat) - logp
        res = np.abs(resid[ident]).max() if np.any(ident) else np.abs(resid).max()
        if res < tol:
            break
        if res > prev_res * 1.2:
            damp = max(0.25, damp * 0.5)
        prev_res = res
        dlogp = (slope / total) / phat        # negative for min-wins
        dlogp = np.minimum(dlogp, -1e-3 / (sd + 1e-9))
        delta = np.clip(damp * resid / dlogp, -step_cap, step_cap)
        mu = mu - delta
        mu -= mu.mean()
    if return_info:
        return mu, {"iterations": it + 1, "residual": float(res),
                    "converged": bool(res < tol)}
    return mu
