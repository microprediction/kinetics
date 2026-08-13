"""Shared race-transform utilities for the experiments.

Gaussian Thurstone race, argmin convention: performance X_i = mu_i + sigma * eps_i,
lowest wins. Forward transform (abilities -> win probabilities) uses the multiplicative
cavity identity: field survival built once, each competitor's rest-field by division.
Inverse transform (win probabilities -> abilities) is a damped fixed point on log-probs.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_TINY = 1e-300
_PFLOOR = 1e-15


def _lattice(mu: np.ndarray, sigma: float, points: int = 3001) -> np.ndarray:
    """Lattice adapted to the field: covers every competitor's density."""
    lo = float(np.min(mu)) - 8.0 * sigma
    hi = float(np.max(mu)) + 8.0 * sigma
    return np.linspace(lo, hi, points)


def win_probabilities(mu: np.ndarray, sigma: float = 1.0,
                      x: np.ndarray | None = None) -> np.ndarray:
    """P(X_i = min_j X_j) for X_i = mu_i + sigma*eps, eps ~ N(0,1) iid."""
    mu = np.asarray(mu, dtype=float)
    if x is None:
        x = _lattice(mu, sigma)
    z = (x[None, :] - mu[:, None]) / sigma
    S = 1.0 - ndtr(z)
    f = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))
    dx = x[1] - x[0]
    log_S_field = np.sum(np.log(np.maximum(S, _TINY)), axis=0)
    log_rest = log_S_field[None, :] - np.log(np.maximum(S, _TINY))
    rest = np.exp(np.clip(log_rest, -745.0, 0.0))
    p = np.sum(f * rest, axis=1) * dx
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("race integration failed; widen the lattice")
    return p / total  # remove lattice quadrature error


def abilities_from_probabilities(p: np.ndarray, sigma: float = 1.0,
                                 n_iter: int = 500, step: float = 0.5,
                                 tol: float = 1e-9) -> np.ndarray:
    """Invert the race: find mu (mean zero) with win_probabilities(mu) = p.

    Damped fixed point: win probability is decreasing in mu (argmin race), so raise
    the ability of overpriced competitors and lower it for underpriced ones. Residuals
    are clipped so a vanishing model probability cannot destabilize the iteration.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    p = p / p.sum()
    logp = np.log(p)
    mu = -sigma * (logp - logp.mean()) / 2.0  # conservative warm start
    for _ in range(n_iter):
        model = np.maximum(win_probabilities(mu, sigma), _PFLOOR)
        resid = np.clip(np.log(model) - logp, -4.0, 4.0)
        mu = mu + step * sigma * resid
        mu -= mu.mean()
        if np.abs(resid).max() < tol:
            break
    return mu


# ---------------------------------------------------------------------------
# Fast transform for CORRELATED fields (program Q6).
#
# Decompose the covariance as Sigma ~= V V^T + D (k factors + idiosyncratic
# diagonal; eigen-truncation leaves a nonnegative diagonal residual and matches
# the diagonal exactly). Conditionally on the k factors the competitors are
# independent, so the multiplicative cavity applies at every quadrature node:
#
#   p_i = E_f [ integral f_i(x|f) * S_field(x|f) / S_i(x|f) dx ],
#
# a k-dimensional Gauss-Hermite quadrature wrapped around the O(N) independent
# transform. The two leave-one-out identities compose: the Gaussian/Schur side
# compresses the coupling into factors; the field product prices the race.
# ---------------------------------------------------------------------------


def factor_model(C: np.ndarray, k: int, n_iter: int = 200,
                 tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Fit C ~= V V^T + diag(D) by iterated principal-factor analysis.

    Unlike naive eigen-truncation (which invents off-diagonal correlation --
    catastrophically so for C near identity), the iteration fits the
    off-diagonals: eigendecompose C - diag(D), re-estimate D from the exact
    diagonal, repeat. V has k columns; D is the idiosyncratic variance.
    """
    C = np.asarray(C, dtype=float)
    D = np.full(len(C), 0.5 * float(np.mean(np.diag(C))))
    V = np.zeros((len(C), k))
    for _ in range(n_iter):
        lam, U = np.linalg.eigh(C - np.diag(D))
        idx = np.argsort(lam)[::-1][:k]
        V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        D_new = np.clip(np.diag(C) - np.sum(V**2, axis=1), 1e-3, None)
        if np.abs(D_new - D).max() < tol:
            D = D_new
            break
        D = D_new
    return V, D


def hermite_nodes(k: int, Q: int = 15, prune: float = 1e-7):
    """Product Gauss-Hermite rule for E over N(0, I_k); returns (nodes, weights)."""
    x, w = np.polynomial.hermite_e.hermegauss(Q)
    w = w / np.sqrt(2.0 * np.pi)
    if k == 1:
        return x[:, None], w
    grids = np.meshgrid(*([x] * k), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for d in range(k):
        W *= w[np.searchsorted(x, F[:, d])]
    keep = W > prune * W.max()
    return F[keep], W[keep]


def win_probabilities_factor(mu: np.ndarray, V: np.ndarray, D: np.ndarray,
                             F: np.ndarray, W: np.ndarray,
                             keep: np.ndarray | None = None,
                             points: int = 1501,
                             return_deletions: bool = False):
    """Win probabilities for X = mu + V f + sqrt(D) eps, argmin wins.

    With return_deletions=True also returns the FULL single-deletion ensemble
    q[i, j] = P(j wins | i removed) from the same conditional field pass --
    the multiplicative cavity, conditionally: divide S_field by S_i (and S_j).
    """
    mu = np.asarray(mu, dtype=float)
    if keep is not None:
        mu, V, D = mu[keep], V[keep], D[keep]
    N = len(mu)
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T                      # (nodes, N) cond. means
    lo = M_all.min() - 8.0 * sd.max()
    hi = M_all.max() + 8.0 * sd.max()
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]

    p = np.zeros(N)
    q = np.zeros((N, N)) if return_deletions else None
    chunk = max(1, int(5e6 / (N * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]                          # (nc, N)
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        S = np.maximum(1.0 - ndtr(z), _TINY)            # (nc, N, L)
        f = np.exp(-0.5 * z**2) / (sd[None, :, None] * np.sqrt(2.0 * np.pi))
        logS = np.log(S)
        logSfield = logS.sum(axis=1)                    # (nc, L)
        rest = np.exp(np.clip(logSfield[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f * rest, axis=2) * dx)       # (nc, N) -> (N,)
        if return_deletions:
            for i in range(N):
                # divide the deleted competitor's survival back out
                rest_i = np.exp(np.clip(
                    logSfield[:, None, :] - logS - logS[:, i:i + 1, :],
                    -745.0, 0.0))
                contrib = np.sum(f * rest_i, axis=2) * dx
                contrib[:, i] = 0.0
                q[i] += Wc @ contrib
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("factor race integration failed")
    if return_deletions:
        q = q / q.sum(axis=1, keepdims=True)
        return p / total, q
    return p / total


def abilities_from_probabilities_factor(p: np.ndarray, V: np.ndarray,
                                        D: np.ndarray, F: np.ndarray,
                                        W: np.ndarray, n_iter: int = 50,
                                        tol: float = 1e-6) -> np.ndarray:
    """Inverse transform under the factor model, by coordinate-wise Newton.

    Design synthesis (credit where due): the coordinate-Newton-against-a-frozen-
    field structure is the ORIGINAL fast-ability-transform inversion (winning /
    thurstone); the independent-inverse warm start and the observation that the
    choice-space Jacobian is intrinsically well-conditioned are from the
    allocation package (allocation/_thurstone/calibrate.py and
    experiments/preconditioner.py). This version adds: k-factor quadrature,
    analytic per-coordinate slopes dp_i/dmu_i = sum_w W_w int (z f(z)/sd_i)
    rest_i dx computed in the same chunked lattice pass as p_hat, and a
    tail-aware tolerance (convergence is not held hostage by runners whose
    target probability is unidentifiably small). Typical cost: ~10 forward-pass
    equivalents, versus hundreds for the damped Picard iteration it replaces.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    p = p / p.sum()
    logp = np.log(p)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    sd = np.sqrt(D)
    N = len(p)
    # tail-aware convergence: runners below the floor are matched best-effort
    floor = max(1e-9, 1e-4 / N)
    ident = p > floor
    # warm start: exact INDEPENDENT inversion (allocation's design), using each
    # runner's total sd, via this same Newton with a single zero factor node
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
    for _ in range(n_iter):
        M_all = mu[None, :] + F @ V.T
        lo = M_all.min() - 8.0 * sd.max()
        hi = M_all.max() + 8.0 * sd.max()
        x = np.linspace(lo, hi, 1501)
        dx = x[1] - x[0]
        phat = np.zeros(N)
        slope = np.zeros(N)
        chunk = max(1, int(5e6 / (N * len(x))))
        for a in range(0, len(F), chunk):
            M = M_all[a:a + chunk]
            Wc = W[a:a + chunk]
            z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
            S = np.maximum(1.0 - ndtr(z), _TINY)
            f = np.exp(-0.5 * z**2) / (sd[None, :, None] * np.sqrt(2.0 * np.pi))
            logS = np.log(S)
            logSfield = logS.sum(axis=1)
            rest = np.exp(np.clip(logSfield[:, None, :] - logS, -745.0, 0.0))
            phat += Wc @ (np.sum(f * rest, axis=2) * dx)
            slope += Wc @ (np.sum(z * f / sd[None, :, None] * rest, axis=2) * dx)
        phat = np.maximum(phat / phat.sum(), _PFLOOR)
        resid = np.log(phat) - logp
        res = np.abs(resid[ident]).max() if np.any(ident) else np.abs(resid).max()
        if res < tol:
            break
        if res > prev_res * 1.2:
            damp = max(0.25, damp * 0.5)     # simple safeguard
        prev_res = res
        dlogp = slope / phat                  # negative for min-wins
        dlogp = np.minimum(dlogp, -1e-3 / (sd + 1e-9))
        delta = np.clip(damp * resid / dlogp, -step_cap, step_cap)
        mu = mu - delta                      # Newton: mu <- mu - resid / dlogp
        mu -= mu.mean()
    return mu


def qmc_nodes(k: int, m: int = 13, seed: int = 0):
    """Scrambled-Sobol nodes for E over N(0, I_k): 2^m equal-weight points.

    Deterministic given the seed; error decays ~n^-1 on smooth integrands vs
    n^-1/2 for plain Monte Carlo. Use for factor ranks beyond the reach of
    product Gauss-Hermite (k >~ 4).
    """
    from scipy.stats import norm, qmc

    F = norm.ppf(qmc.Sobol(k, scramble=True, seed=seed).random_base2(m))
    return F, np.full(len(F), 1.0 / len(F))
