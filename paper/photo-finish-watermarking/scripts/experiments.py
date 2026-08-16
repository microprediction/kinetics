#!/usr/bin/env python3
"""Synthetic verification experiments for Photo-Finish Watermarking.

All experiments are deterministic given SEED and use only synthetic probability
vectors.  They verify algebraic identities and illustrate the Gaussian key-
exposure construction; they are not language-model benchmarks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType fonts in vector figures; avoid Type 3 fonts in journal PDFs.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import null_space
from scipy.special import log_ndtr, softmax
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SEED = 20260816
RNG = np.random.default_rng(SEED)


def gh_rule(k: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Product Gauss-Hermite rule for N(0,I_k)."""
    if k == 0:
        return np.zeros((1, 0)), np.ones(1)
    x, w = hermgauss(order)
    x = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    grids = np.meshgrid(*([x] * k), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=1)
    wgrids = np.meshgrid(*([w] * k), indexing="ij")
    weights = np.prod(np.stack(wgrids, axis=0), axis=0).ravel()
    return nodes, weights


def trapz_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.trapezoid(y, x=x, axis=-1)


def gaussian_choice_forward(
    mu: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    *,
    gh_order: int = 17,
    lattice: int = 1201,
    return_jacobian: bool = False,
    energy_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Winner probabilities for low-rank-plus-diagonal Gaussian utilities.

    Parameters
    ----------
    mu : (N,)
    V : (N,k)
    D : (N,), strictly positive idiosyncratic variances
    energy_vectors : (N,r), optional columns h for h' J h

    Returns
    -------
    p, J, energies, pre_normalization_defect
    """
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    D = np.asarray(D, dtype=float)
    n = mu.size
    if V.ndim == 1:
        V = V[:, None]
    k = V.shape[1]
    assert V.shape[0] == n and D.shape == (n,) and np.all(D > 0)

    nodes, weights = gh_rule(k, gh_order)
    means_all = mu[None, :] + nodes @ V.T if k else np.broadcast_to(mu, (1, n))
    sd = np.sqrt(D)
    lo = float(np.min(means_all) - 8.5 * np.max(sd))
    hi = float(np.max(means_all) + 8.5 * np.max(sd))
    x = np.linspace(lo, hi, lattice)

    p = np.zeros(n)
    J = np.zeros((n, n)) if return_jacobian else None
    if energy_vectors is not None:
        H = np.asarray(energy_vectors, dtype=float)
        if H.ndim == 1:
            H = H[:, None]
        assert H.shape[0] == n
        energies = np.zeros(H.shape[1])
    else:
        H = None
        energies = None

    log_norm_const = 0.5 * np.log(2.0 * np.pi)

    for node_weight, m in zip(weights, means_all):
        z = (x[None, :] - m[:, None]) / sd[:, None]
        logF = log_ndtr(z)
        logg = -0.5 * z * z - np.log(sd)[:, None] - log_norm_const
        log_field = np.sum(logF, axis=0)
        log_r = logg + log_field[None, :] - logF
        r = np.exp(log_r)
        p += node_weight * trapz_rows(r, x)

        if return_jacobian:
            for i in range(n):
                for j in range(i + 1, n):
                    log_w = logg[i] + logg[j] + log_field - logF[i] - logF[j]
                    wij = node_weight * float(np.trapezoid(np.exp(log_w), x=x))
                    J[i, i] += wij
                    J[j, j] += wij
                    J[i, j] -= wij
                    J[j, i] -= wij

        if H is not None:
            # Continuum photo-finish quadratic form, evaluated in O(NL) per h.
            hazards = np.exp(logg - logF)
            Lambda = np.sum(hazards, axis=0)
            for col in range(H.shape[1]):
                h = H[:, col]
                A = h @ hazards
                integrand = np.sum(h[:, None] * r * (h[:, None] * Lambda[None, :] - A[None, :]), axis=0)
                energies[col] += node_weight * float(np.trapezoid(integrand, x=x))

    total = float(np.sum(p))
    defect = abs(1.0 - total)
    p = p / total
    return p, J, energies, defect


def calibrate_mu(
    target: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    *,
    gh_order: int = 17,
    lattice: int = 1201,
    tol: float = 2e-10,
    max_iter: int = 35,
) -> tuple[np.ndarray, dict]:
    """Damped Newton calibration on the mean-zero quotient."""
    target = np.asarray(target, dtype=float)
    target = target / target.sum()
    n = target.size
    B = null_space(np.ones((1, n)))  # N x (N-1), orthonormal
    # A log-share warm start, scaled to the marginal utility standard deviation.
    scale = float(np.sqrt(np.mean(D + np.sum(V * V, axis=1))))
    mu = scale * (np.log(target) - np.mean(np.log(target)))
    eta = B.T @ mu

    history: list[float] = []
    for it in range(max_iter):
        mu = B @ eta
        p, J, _, defect = gaussian_choice_forward(
            mu, V, D, gh_order=gh_order, lattice=lattice, return_jacobian=True
        )
        log_resid = np.log(p) - np.log(target)
        err = float(np.max(np.abs(log_resid)))
        history.append(err)
        if err < tol:
            return mu, {"iterations": it, "max_log_error": err, "defect": defect, "history": history}

        # Solve the exact reduced Jacobian system for probability residuals.
        rhs = B.T @ (p - target)
        Hred = B.T @ J @ B
        step = np.linalg.solve(Hred, rhs)
        # Cap extreme Newton moves in the utility metric.
        step_norm = float(np.linalg.norm(step))
        if step_norm > 3.0 * scale:
            step *= (3.0 * scale / step_norm)

        base_obj = float(np.linalg.norm(p - target))
        accepted = False
        damping = 1.0
        for _ in range(14):
            trial_eta = eta - damping * step
            trial_mu = B @ trial_eta
            trial_p, _, _, _ = gaussian_choice_forward(
                trial_mu, V, D, gh_order=gh_order, lattice=lattice, return_jacobian=False
            )
            trial_obj = float(np.linalg.norm(trial_p - target))
            if trial_obj < base_obj:
                eta = trial_eta
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            # Fallback along the projected log-residual, a robust Jacobi-like step.
            eta = eta - 0.1 * (B.T @ log_resid)

    mu = B @ eta
    p, _, _, defect = gaussian_choice_forward(mu, V, D, gh_order=gh_order, lattice=lattice)
    err = float(np.max(np.abs(np.log(p) - np.log(target))))
    return mu, {"iterations": max_iter, "max_log_error": err, "defect": defect, "history": history}


def synthid_exact_identity_experiment() -> dict:
    n = 100
    raw = np.exp(-0.06 * np.arange(n))
    raw *= np.exp(0.15 * RNG.normal(size=n))
    p = raw / raw.sum()
    g = RNG.integers(0, 2, size=n).astype(float)
    a = float(p @ g)
    J = np.diag(p) - np.outer(p, p)
    q = p * (1.0 + g - a)
    q_lap = p + J @ g
    energy = float(g @ J @ g)
    tv = 0.5 * float(np.sum(np.abs(q - p)))
    chi2 = float(np.sum((q - p) ** 2 / p))
    if 1e-14 < a < 1 - 1e-14:
        beta = math.log((2.0 - a) / (1.0 - a))
        q_tilt = softmax(np.log(p) + beta * g)
        tilt_err = float(np.max(np.abs(q - q_tilt)))
    else:
        beta = float("nan")
        tilt_err = float("nan")

    # Random-key averages.
    draws = 100_000
    theta = 0.5
    G = RNG.binomial(1, theta, size=(draws, n)).astype(float)
    a_draw = G @ p
    # g'Jg = sum p_i g_i - (p'g)^2 for binary g.
    e_draw = a_draw - a_draw * a_draw
    expected_formula = theta * (1 - theta) * (1 - float(p @ p))

    result = {
        "N": n,
        "a": a,
        "laplacian_max_error": float(np.max(np.abs(q - q_lap))),
        "mass_error": abs(float(q.sum()) - 1.0),
        "min_q": float(q.min()),
        "energy": energy,
        "tv": tv,
        "chi2": chi2,
        "energy_tv_error": abs(energy - tv),
        "energy_chi2_error": abs(energy - chi2),
        "tilt_beta": beta,
        "tilt_max_error": tilt_err,
        "expected_energy_mc": float(e_draw.mean()),
        "expected_energy_formula": expected_formula,
        "expected_energy_mc_se": float(e_draw.std(ddof=1) / np.sqrt(draws)),
    }
    return result


def synthid_layers_experiment() -> dict:
    n = 120
    # A moderately concentrated Zipf-like next-token distribution.
    p0 = 1.0 / (np.arange(1, n + 1) ** 1.05)
    p0 /= p0.sum()
    paths = 20_000
    layers = 30
    theta = 0.5
    P = np.broadcast_to(p0, (paths, n)).copy()
    mean_collision = [float(np.mean(np.sum(P * P, axis=1)))]
    mean_signal = []
    predicted_signal = []
    for _ in range(layers):
        G = RNG.binomial(1, theta, size=(paths, n)).astype(float)
        a = np.sum(P * G, axis=1)
        signal = a * (1.0 - a)
        mean_signal.append(float(np.mean(signal)))
        predicted_signal.append(float(theta * (1 - theta) * (1.0 - mean_collision[-1])))
        P *= 1.0 + G - a[:, None]
        # Floating-point renormalization only; the algebraic map already sums to one.
        P /= P.sum(axis=1, keepdims=True)
        mean_collision.append(float(np.mean(np.sum(P * P, axis=1))))

    layer_idx = np.arange(1, layers + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(layer_idx, mean_signal, marker="o", markersize=3, linewidth=1.4, label="Monte Carlo mean cut energy")
    ax.plot(layer_idx, predicted_signal, linestyle="--", linewidth=1.5, label=r"$\frac{1}{4}(1-\mathbb{E}C_\ell)$ prediction")
    ax.set_xlabel("Tournament layer")
    ax.set_ylabel("Expected score drift")
    ax.set_title("Diminishing signal in an iterated binary tournament")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "synthid_layer_signal.pdf", bbox_inches="tight")
    fig.savefig(FIG / "synthid_layer_signal.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(np.arange(0, layers + 1), mean_collision, marker="o", markersize=3, linewidth=1.4)
    ax.set_xlabel("Tournament layers applied")
    ax.set_ylabel(r"Expected collision probability $\mathbb{E}\sum_i p_i^2$")
    ax.set_title("Each layer concentrates the conditional token distribution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "synthid_layer_collision.pdf", bbox_inches="tight")
    fig.savefig(FIG / "synthid_layer_collision.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "N": n,
        "paths": paths,
        "layers": layers,
        "initial_collision": mean_collision[0],
        "final_collision": mean_collision[-1],
        "first_signal": mean_signal[0],
        "last_signal": mean_signal[-1],
        "max_prediction_error": float(np.max(np.abs(np.array(mean_signal) - np.array(predicted_signal)))),
        "mean_collision": mean_collision,
        "mean_signal": mean_signal,
        "predicted_signal": predicted_signal,
    }


def gaussian_exposure_experiment() -> dict:
    n = 10
    # Target distribution with no very small cells, to isolate geometry rather than tails.
    logits = np.array([1.00, 0.72, 0.45, 0.27, 0.05, -0.08, -0.22, -0.38, -0.57, -0.80])
    target = softmax(logits)
    # One factor with a non-monotone loading pattern, centered to remove the common-factor gauge.
    v = np.array([-1.15, -0.80, -0.35, 0.20, 0.72, 1.05, 0.60, 0.05, -0.45, 0.13])
    v = v - v.mean()
    V = (0.62 * v)[:, None]
    D = np.array([0.78, 0.83, 0.88, 0.82, 0.76, 0.91, 0.86, 0.80, 0.89, 0.84])

    mu, cal = calibrate_mu(target, V, D, gh_order=19, lattice=1401, tol=1e-9)
    p_check, _, _, defect = gaussian_choice_forward(mu, V, D, gh_order=23, lattice=1801)
    validation_log_err = float(np.max(np.abs(np.log(p_check) - np.log(target))))

    rho_grid = np.linspace(0.0, 1.0, 11)
    outer_nodes, outer_weights = gh_rule(1, 23)
    metrics = []
    for rho in rho_grid:
        qbar = np.zeros(n)
        mi = 0.0
        collision = 0.0
        linear_drift = 0.0
        stein_rhs = 0.0
        residual_V = np.sqrt(max(0.0, 1.0 - rho)) * V
        for node, weight in zip(outer_nodes, outer_weights):
            r = float(node[0])
            shifted_mu = mu + np.sqrt(rho) * V[:, 0] * r
            q, _, energies, _ = gaussian_choice_forward(
                shifted_mu,
                residual_V,
                D,
                gh_order=19,
                lattice=1201,
                energy_vectors=V,
            )
            qbar += weight * q
            mi += weight * float(np.sum(q * (np.log(q) - np.log(target))))
            collision += weight * float(q @ q)
            linear_drift += weight * r * float(V[:, 0] @ q)
            stein_rhs += weight * float(energies[0]) * np.sqrt(rho)
        metrics.append(
            {
                "rho": float(rho),
                "mi": float(mi),
                "posterior_collision": float(collision),
                "linear_score_drift": float(linear_drift),
                "stein_rhs": float(stein_rhs),
                "marginal_max_error": float(np.max(np.abs(qbar - target))),
            }
        )

    rho = np.array([m["rho"] for m in metrics])
    mi = np.array([m["mi"] for m in metrics])
    collision = np.array([m["posterior_collision"] for m in metrics])
    drift = np.array([m["linear_score_drift"] for m in metrics])
    stein = np.array([m["stein_rhs"] for m in metrics])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(rho, mi, marker="o", linewidth=1.5, label=r"$I(R;Y)$")
    ax.plot(rho, collision - float(target @ target), marker="s", linewidth=1.5, label="excess posterior collision")
    ax.set_xlabel(r"Key exposure $\rho$")
    ax.set_ylabel("Information / concentration")
    ax.set_title("More exposed Gaussian factor means more evidence and less diversity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gaussian_key_exposure.pdf", bbox_inches="tight")
    fig.savefig(FIG / "gaussian_key_exposure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(rho, drift, marker="o", linewidth=1.5, label=r"$\mathbb{E}[R v_Y]$")
    ax.plot(rho, stein, linestyle="--", linewidth=1.6, label=r"$\sqrt{\rho}\,\mathbb{E}[v^\top J_R v]$")
    ax.set_xlabel(r"Key exposure $\rho$")
    ax.set_ylabel("Expected linear detector score")
    ax.set_title("Stein identity for the model-free Gaussian detector")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gaussian_stein_score.pdf", bbox_inches="tight")
    fig.savefig(FIG / "gaussian_stein_score.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "N": n,
        "target": target.tolist(),
        "V": V[:, 0].tolist(),
        "D": D.tolist(),
        "mu": mu.tolist(),
        "calibration": cal,
        "validation_log_error": validation_log_err,
        "validation_mass_defect": defect,
        "base_collision": float(target @ target),
        "metrics": metrics,
        "max_nondistortion_error": float(max(m["marginal_max_error"] for m in metrics)),
        "max_stein_abs_error": float(max(abs(m["linear_score_drift"] - m["stein_rhs"]) for m in metrics)),
        "mi_monotonicity_min_increment": float(np.min(np.diff(mi))),
        "collision_monotonicity_min_increment": float(np.min(np.diff(collision))),
    }


def exact_null_detector_experiment() -> dict:
    # Conditional on any fixed token sequence, the standardized Gaussian score is exactly N(0,1).
    T = 240
    k = 3
    n = 80
    V = RNG.normal(size=(n, k))
    V -= V.mean(axis=0, keepdims=True)
    tokens = RNG.integers(0, n, size=T)
    denom = float(np.sqrt(np.sum(np.sum(V[tokens] ** 2, axis=1))))
    reps = 100_000
    # Generate in chunks to keep memory modest.
    zvals = np.empty(reps)
    chunk = 5000
    coeff = V[tokens]
    for start in range(0, reps, chunk):
        end = min(reps, start + chunk)
        R = RNG.normal(size=(end - start, T, k))
        scores = np.einsum("rtk,tk->r", R, coeff)
        zvals[start:end] = scores / denom
    threshold = norm.ppf(0.99)
    result = {
        "T": T,
        "k": k,
        "repetitions": reps,
        "mean": float(zvals.mean()),
        "sd": float(zvals.std(ddof=1)),
        "empirical_upper_1pct": float(np.mean(zvals > threshold)),
        "threshold": float(threshold),
    }
    return result


def main() -> None:
    results = {
        "seed": SEED,
        "synthid_identities": synthid_exact_identity_experiment(),
        "synthid_layers": synthid_layers_experiment(),
        "gaussian_exposure": gaussian_exposure_experiment(),
        "exact_null_detector": exact_null_detector_experiment(),
    }
    with (DATA / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({
        "synthid_identities": results["synthid_identities"],
        "synthid_layers_summary": {k: results["synthid_layers"][k] for k in ["initial_collision", "final_collision", "first_signal", "last_signal", "max_prediction_error"]},
        "gaussian_summary": {k: results["gaussian_exposure"][k] for k in ["validation_log_error", "max_nondistortion_error", "max_stein_abs_error", "mi_monotonicity_min_increment", "collision_monotonicity_min_increment"]},
        "null_detector": results["exact_null_detector"],
    }, indent=2))


if __name__ == "__main__":
    main()
