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
