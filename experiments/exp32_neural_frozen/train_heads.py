"""Experiment 32: frozen-feature linear heads on CIFAR-10 (Experiment IV
of the companion paper).

Head mu = W h + b over 512-d frozen ResNet-18 features, N = 10 classes.
Methods, all trained with Adam at matched settings, measured on wall
clock:

  softmax   : cross-entropy (baseline; the Gumbel member)
  fy_mc1    : Gaussian Fenchel-Young, 1-sample perturbed argmax gradient
              (Berthet-style: grad ~ e_argmax(mu + s z) - e_y)
  fy_mc16   : same with 16-sample average
  fy_exact  : deterministic shared-field evaluator, grad = p - e_y
  nll_exact : exact Gaussian winner likelihood, grad = -J e_y / p_y
              (boundary-aware hard-negative mining; convex by Prop. 2)

iid Gaussian geometry (k = 0, D_i = s^2), s swept. The exact operators
are batched NumPy: per-sample 1-D lattice over the shared product of 10
conditional CDFs; L = 129 suffices at N = 10 (checked against L = 513).

Metrics: test accuracy, test NLL (under the model each method trains),
test ECE (15 bins), plus PROBE metrics for every head under the SAME
exact-Gaussian evaluation, so model classes are compared fairly.

Run:  python experiments/exp32_neural_frozen/train_heads.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, softmax as sp_softmax

HERE = Path(__file__).resolve().parent
L = 129
RNG = np.random.default_rng(32)


def gauss_forward(mu, s):
    """Batched iid-Gaussian argmax winner probabilities. mu: (B, N)."""
    B, N = mu.shape
    lo = mu.min(axis=1) - 8 * s
    hi = mu.max(axis=1) + 8 * s
    x = lo[:, None] + (hi - lo)[:, None] * (np.arange(L) / (L - 1))[None, :]
    dx = (hi - lo) / (L - 1)                                # (B,)
    z = (x[:, None, :] - mu[:, :, None]) / s                # (B,N,L)
    logF = log_ndtr(z)
    g = np.exp(-0.5 * z * z) / (s * np.sqrt(2 * np.pi))
    field = logF.sum(axis=1)                                # (B,L)
    rest = np.exp(np.clip(field[:, None, :] - logF, -745, 0))
    p = (g * rest).sum(axis=2) * dx[:, None]
    return np.maximum(p, 1e-300) / p.sum(axis=1, keepdims=True)


def gauss_w_row(mu, s, y):
    """w_{y j} rows for label indices y. mu: (B, N), y: (B,)."""
    B, N = mu.shape
    lo = mu.min(axis=1) - 8 * s
    hi = mu.max(axis=1) + 8 * s
    x = lo[:, None] + (hi - lo)[:, None] * (np.arange(L) / (L - 1))[None, :]
    dx = (hi - lo) / (L - 1)
    z = (x[:, None, :] - mu[:, :, None]) / s
    logF = log_ndtr(z)
    g = np.exp(-0.5 * z * z) / (s * np.sqrt(2 * np.pi))
    field = logF.sum(axis=1)
    gy = g[np.arange(B), y]                                 # (B,L)
    logFy = logF[np.arange(B), y]
    rest2 = np.exp(np.clip(field[:, None, :] - logF - logFy[:, None, :],
                           -745, 0))
    w = (gy[:, None, :] * g * rest2).sum(axis=2) * dx[:, None]   # (B,N)
    w[np.arange(B), y] = 0.0
    return w


def ece(probs, y, bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def train(method, Xtr, ytr, Xte, yte, s=1.0, epochs=12, bs=512, lr=1e-3):
    B, dim = Xtr.shape
    N = 10
    rng = np.random.default_rng(7)
    W = np.zeros((dim, N)); b = np.zeros(N)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t0 = time.perf_counter()
    step = 0
    curve = []
    for ep in range(epochs):
        perm = rng.permutation(B)
        for a in range(0, B, bs):
            idx = perm[a:a + bs]
            h = Xtr[idx]; y = ytr[idx]
            mu = h @ W + b
            nb = len(idx)
            if method == "softmax":
                p = sp_softmax(mu, axis=1)
                gmu = p.copy(); gmu[np.arange(nb), y] -= 1.0
            elif method == "fy_exact":
                p = gauss_forward(mu, s)
                gmu = p.copy(); gmu[np.arange(nb), y] -= 1.0
            elif method == "nll_exact":
                p = gauss_forward(mu, s)
                w = gauss_w_row(mu, s, y)
                py = p[np.arange(nb), y]
                gmu = w / py[:, None]
                gmu[np.arange(nb), y] = -w.sum(axis=1) / py
            elif method.startswith("fy_mc"):
                M = int(method[5:])
                gmu = np.zeros_like(mu)
                for _ in range(M):
                    zpert = rng.standard_normal(mu.shape)
                    am = np.argmax(mu + s * zpert, axis=1)
                    gmu[np.arange(nb), am] += 1.0 / M
                gmu[np.arange(nb), y] -= 1.0
            gmu /= nb
            gW = h.T @ gmu; gb = gmu.sum(axis=0)
            step += 1
            for P, g_, m_, v_ in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m_ *= beta1; m_ += (1 - beta1) * g_
                v_ *= beta2; v_ += (1 - beta2) * g_ * g_
                mh = m_ / (1 - beta1**step); vh = v_ / (1 - beta2**step)
                P -= lr * mh / (np.sqrt(vh) + eps)
        # end-of-epoch eval (excluded from wall time via marker)
        t_train = time.perf_counter() - t0
        mu_te = Xte @ W + b
        acc = float((mu_te.argmax(axis=1) == yte).mean())
        curve.append((ep, t_train, acc))
    # final probe under the exact Gaussian evaluation and softmax eval
    mu_te = Xte @ W + b
    p_gauss = gauss_forward(mu_te, s)
    p_soft = sp_softmax(mu_te, axis=1)
    nll_g = float(-np.log(p_gauss[np.arange(len(yte)), yte]).mean())
    nll_s = float(-np.log(np.maximum(p_soft[np.arange(len(yte)), yte],
                                     1e-300)).mean())
    return {
        "acc": curve[-1][2], "seconds": curve[-1][1],
        "nll_gauss_probe": nll_g, "nll_soft_probe": nll_s,
        "ece_gauss": ece(p_gauss, yte), "ece_soft": ece(p_soft, yte),
        "curve": curve,
    }


def main():
    d = np.load(HERE / "features.npz")
    Xtr, ytr = d["X_train"].astype(np.float64), d["y_train"]
    Xte, yte = d["X_test"].astype(np.float64), d["y_test"]
    # standardize features once
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - m) / sd; Xte = (Xte - m) / sd
    print(f"features {Xtr.shape}, test {Xte.shape}")
    rows = ["method,s,seconds,acc,nll_gauss,nll_soft,ece_gauss,ece_soft"]
    import sys as _sys
    methods = (_sys.argv[1],) if len(_sys.argv) > 1 else (
        "softmax", "fy_mc1", "fy_mc16", "fy_exact", "nll_exact")
    for method in methods:
        for s in ((1.0,) if method == "softmax" else (0.5, 1.0)):
            r = train(method, Xtr, ytr, Xte, yte, s=s)
            print(f"{method:>9} s={s}: {r['seconds']:6.1f}s "
                  f"acc {r['acc']:.4f} nllG {r['nll_gauss_probe']:.4f} "
                  f"nllS {r['nll_soft_probe']:.4f} "
                  f"eceG {r['ece_gauss']:.4f} eceS {r['ece_soft']:.4f}",
                  flush=True)
            rows.append(f"{method},{s},{r['seconds']:.1f},{r['acc']:.4f},"
                        f"{r['nll_gauss_probe']:.4f},{r['nll_soft_probe']:.4f},"
                        f"{r['ece_gauss']:.4f},{r['ece_soft']:.4f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
