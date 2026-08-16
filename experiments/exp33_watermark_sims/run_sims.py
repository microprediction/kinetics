"""Experiment 33: simulations for the high-value watermarking assertions.

Synthetic vocabulary of N tokens embedded in R^d with cluster (synonym)
structure; loadings V = leading PCA directions of the embeddings, so
semantic neighbors have similar loadings. Gaussian factor watermark
(exposure rho) vs a SynthID-style single tournament layer with
independent pseudorandom binary colors.

Tested assertions:
  A. EDIT ROBUSTNESS: replace a fraction s of tokens by (i) synonyms
     (same cluster) or (ii) random tokens. Metric: retention = z under
     attack / z clean. Claim: Gaussian retention under synonym attack
     >> color-scheme retention (which loses each substituted position
     entirely, regardless of semantic radius).
  B. SCRUBBING FRONTIER: z-retention against mean embedding distortion
     across attack strengths -- the quality cost of watermark removal.
  C. MULTI-BIT ATTRIBUTION: key one of C eigen-directions of
     M = E[V'J V] per "customer"; detector picks argmax channel score.
     Accuracy vs sequence length; top-eigen channels vs bottom.
  D. SHORT-TEXT POWER at n = 30: detection power at the exact-null 1%
     threshold for full-V score, top-eigen directional score, and a
     random direction.
  E. NULL CALIBRATION: empirical null z is N(0,1) (sanity, matches the
     paper's exact-null theorem).

All generation uses utilities directly (mu = model logits); marginal
preservation is the already-verified calibration layer and is not
re-tested here.

Run:  python experiments/exp33_watermark_sims/run_sims.py
Output: results.json, printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(33)

# ---------------- vocabulary with synonym clusters ----------------
C_CLUST, M_MEMB, DIM, K = 16, 4, 6, 4
N = C_CLUST * M_MEMB
centers = rng.normal(0, 1.0, (C_CLUST, DIM))
emb = np.repeat(centers, M_MEMB, axis=0) + 0.15 * rng.normal(0, 1, (N, DIM))
cluster = np.repeat(np.arange(C_CLUST), M_MEMB)
# loadings: leading K principal directions, column-centered, scaled
E = emb - emb.mean(0)
U_, S_, Vt_ = np.linalg.svd(E, full_matrices=False)
V = (U_[:, :K] * S_[None, :K]) * 0.45
V -= V.mean(0)
D = np.full(N, 0.7)
RHO = 0.6

def logits_stream(T, seed):
    r = np.random.default_rng(seed)
    base = r.normal(0, 1.2, C_CLUST)
    out = []
    for _ in range(T):
        base = 0.9 * base + 0.45 * r.normal(0, 1, C_CLUST)
        mu = np.repeat(base, M_MEMB) + 0.4 * r.normal(0, 1, N)
        out.append(mu - mu.mean())
    return out

def gen_gauss(mus, key_R, r):
    """Watermarked tokens under the factor watermark."""
    toks = []
    for t, mu in enumerate(mus):
        Z = r.standard_normal(K)
        eps = r.standard_normal(N)
        u = (mu + np.sqrt(RHO) * V @ key_R[t] + np.sqrt(1 - RHO) * V @ Z
             + np.sqrt(D) * eps)
        toks.append(int(np.argmax(u)))
    return np.array(toks)

def z_gauss(toks, key_R, a=None):
    """Exact-null z. a=None: full-V score; else directional."""
    if a is None:
        num = sum(key_R[t] @ V[y] for t, y in enumerate(toks))
        den = np.sqrt(sum(V[y] @ V[y] for y in toks))
    else:
        u = V @ a
        num = sum((key_R[t] @ a) * u[y] for t, y in enumerate(toks))
        den = np.sqrt(sum(u[y] ** 2 for y in toks))
    return num / max(den, 1e-12)

def gen_color(mus, colors, r):
    """SynthID-style: sample two candidates from softmax(mu), keyed
    binary color breaks the tie."""
    toks = []
    for t, mu in enumerate(mus):
        p = np.exp(mu - mu.max()); p /= p.sum()
        c1, c2 = r.choice(N, size=2, p=p)
        g = colors[t]
        if g[c1] > g[c2]: y = c1
        elif g[c2] > g[c1]: y = c2
        else: y = c1 if r.integers(2) == 0 else c2
        toks.append(int(y))
    return np.array(toks)

def z_color(toks, colors, mus):
    """Standardized green count against the exact conditional null."""
    num, var = 0.0, 0.0
    for t, y in enumerate(toks):
        p = np.exp(mus[t] - mus[t].max()); p /= p.sum()
        a = p @ colors[t]
        num += colors[t][y] - a
        var += a * (1 - a)
    return num / np.sqrt(max(var, 1e-12))

def attack(toks, kind, s, r):
    out = toks.copy()
    hit = r.uniform(size=len(toks)) < s
    for i in np.where(hit)[0]:
        if kind == "synonym":
            mates = np.where(cluster == cluster[toks[i]])[0]
            out[i] = int(r.choice(mates))
        else:
            out[i] = int(r.integers(N))
    dist = np.mean([np.linalg.norm(emb[a] - emb[b])
                    for a, b in zip(toks, out)])
    return out, float(dist)

results = {}

# ------------- A/B: edit robustness + scrubbing frontier -------------
T, TRIALS = 80, 300
rows = []
for kind in ("synonym", "random"):
    for s in (0.0, 0.2, 0.4, 0.6):
        zg, zc, dd = [], [], []
        for tr in range(TRIALS):
            r = np.random.default_rng(10_000 + tr)
            mus = logits_stream(T, 20_000 + tr)
            key_R = r.standard_normal((T, K))
            colors = (np.random.default_rng(30_000 + tr)
                      .integers(0, 2, size=(T, N)).astype(float))
            tg = gen_gauss(mus, key_R, r)
            tc = gen_color(mus, colors, r)
            ag, d1 = attack(tg, kind, s, r)
            ac, d2 = attack(tc, kind, s, r)
            zg.append(z_gauss(ag, key_R))
            zc.append(z_color(ac, colors, mus))
            dd.append(0.5 * (d1 + d2))
        rows.append({"kind": kind, "rate": s,
                     "z_gauss": float(np.mean(zg)),
                     "z_color": float(np.mean(zc)),
                     "dist": float(np.mean(dd))})
base_g = rows[0]["z_gauss"]; base_c = rows[0]["z_color"]
for rrow in rows:
    rrow["ret_gauss"] = rrow["z_gauss"] / base_g
    rrow["ret_color"] = rrow["z_color"] / base_c
results["edit_robustness"] = rows
print(f"{'attack':>8} {'rate':>5} {'zG':>6} {'retG':>6} {'zC':>6} "
      f"{'retC':>6} {'sem dist':>8}")
for rrow in rows:
    print(f"{rrow['kind']:>8} {rrow['rate']:>5.1f} {rrow['z_gauss']:>6.2f} "
          f"{rrow['ret_gauss']:>6.2f} {rrow['z_color']:>6.2f} "
          f"{rrow['ret_color']:>6.2f} {rrow['dist']:>8.3f}")

# ---------------- C: multi-bit attribution via eigenbasis ----------------
# estimate M by Stein regression: M = E[R u_Y'] / sqrt(rho), u = V rows
NS = 120_000
r = np.random.default_rng(7)
mus_flat = logits_stream(NS, 99)
Msum = np.zeros((K, K))
for t in range(NS):
    R1 = r.standard_normal(K)
    Z = r.standard_normal(K); eps = r.standard_normal(N)
    u = (mus_flat[t] + np.sqrt(RHO) * V @ R1 + np.sqrt(1 - RHO) * V @ Z
         + np.sqrt(D) * eps)
    y = int(np.argmax(u))
    Msum += np.outer(R1, V[y])
Mhat = Msum / NS / np.sqrt(RHO)
Mhat = 0.5 * (Mhat + Mhat.T)
ev, evec = np.linalg.eigh(Mhat)
print(f"\nM eigenvalues (est.): {ev}")

def attribution(dirs, n, trials=400):
    ok = 0
    for tr in range(trials):
        r = np.random.default_rng(50_000 + tr)
        c_true = tr % dirs.shape[1]
        a = dirs[:, c_true]
        mus = logits_stream(n, 60_000 + tr)
        # key only the a-direction: R = a * r_scalar + orth fresh
        key = []
        toks = []
        for t in range(n):
            rs = r.standard_normal()
            Rfull = a * rs + (np.eye(K) - np.outer(a, a)) @ r.standard_normal(K)
            key.append(rs)
            Z = r.standard_normal(K); eps = r.standard_normal(N)
            u = (mus[t] + np.sqrt(RHO) * V @ Rfull
                 + np.sqrt(1 - RHO) * V @ Z + np.sqrt(D) * eps)
            toks.append(int(np.argmax(u)))
        # detector: channel scores for each candidate direction
        scores = []
        for c in range(dirs.shape[1]):
            uc = V @ dirs[:, c]
            num = sum(key[t] * (dirs[:, c] @ a) * 0 + key[t] * uc[toks[t]]
                      for t in range(n))
            den = np.sqrt(sum(uc[toks[t]] ** 2 for t in range(n)))
            scores.append(num / max(den, 1e-12))
        ok += int(np.argmax(scores) == c_true)
    return ok / trials

top2 = evec[:, -2:]
bot2 = evec[:, :2]
attr = {}
for n in (30, 60, 120):
    attr[n] = {"top_channels": attribution(top2, n),
               "bottom_channels": attribution(bot2, n)}
    print(f"attribution n={n}: top-eigen channels {attr[n]['top_channels']:.3f}, "
          f"bottom {attr[n]['bottom_channels']:.3f}")
results["attribution"] = attr

# ---------------- D: short-text power at n=30 ----------------
n = 30; TRIALS2 = 600; thresh = 2.3263
a_star = evec[:, -1]; a_rand = np.array([1., 0, 0, 0])
pw = {"full": 0, "eigen": 0, "random": 0}
for tr in range(TRIALS2):
    r = np.random.default_rng(80_000 + tr)
    mus = logits_stream(n, 90_000 + tr)
    key_R = r.standard_normal((n, K))
    toks = gen_gauss(mus, key_R, r)
    pw["full"] += int(z_gauss(toks, key_R) > thresh)
    pw["eigen"] += int(z_gauss(toks, key_R, a_star) > thresh)
    pw["random"] += int(z_gauss(toks, key_R, a_rand) > thresh)
power = {k: v / TRIALS2 for k, v in pw.items()}
results["short_text_power_n30"] = power
print(f"\nshort-text power (n=30, 1% threshold): {power}")

# ---------------- E: null calibration ----------------
zs = []
for tr in range(2000):
    r = np.random.default_rng(200_000 + tr)
    mus = logits_stream(n, 210_000 + tr)
    toks = np.array([int(np.argmax(mu + 0.5 * r.standard_normal(N)))
                     for mu in mus])          # unwatermarked
    key_R = r.standard_normal((n, K))          # independent key
    zs.append(z_gauss(toks, key_R))
zs = np.array(zs)
results["null"] = {"mean": float(zs.mean()), "sd": float(zs.std()),
                   "frac_above_1pct": float((zs > thresh).mean())}
print(f"null: mean {zs.mean():.3f} sd {zs.std():.3f} "
      f"P(z>2.33) {(zs > thresh).mean():.4f}")

(HERE / "results.json").write_text(json.dumps(results, indent=2))
print("wrote results.json")
