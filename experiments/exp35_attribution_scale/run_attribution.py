"""Experiment 35: multi-recipient attribution at scale (clean channel).

The leak-attribution product question: a document of n tokens keyed to
one of K recipients; the detector scores all K keys. Wrong-key scores
are EXACTLY N(0,1) under the exact-null theorem (independent keys), so
false-attribution control is analytic; the true-key score is
approximately N(delta * sqrt(n), 1) with delta the per-token evidence
rate. This script:

  1. takes delta from experiment 34's real-GPT-2 measurement when
     available (fallback: the synthetic exp33 rate);
  2. computes the analytic operating envelope: tokens needed for
     attribution among K recipients at family FPR alpha with 95% power,
     n >= ((z_{alpha/K} + 1.645) / delta)^2;
  3. validates by direct simulation of the max-over-K-keys decision at
     representative points, including the identify-vs-abstain protocol
     (attribute only if best z exceeds the family threshold).

Run:  python experiments/exp35_attribution_scale/run_attribution.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
res34 = Path(__file__).resolve().parents[1] / "exp34_llm_watermark" / "results.json"
if res34.exists():
    d = json.loads(res34.read_text())
    delta = d["detection"]["z_wm_mean"] / np.sqrt(80)
    src = "exp34 (GPT-2, measured)"
else:
    delta = 6.72 / np.sqrt(80)
    src = "exp33 synthetic fallback"
print(f"per-token evidence rate delta = {delta:.4f} per sqrt-token [{src}]")

rows = ["K,alpha,n_required_95pct_power"]
print(f"{'recipients K':>13} {'family FPR':>11} {'tokens needed':>14}")
for K in (100, 500, 1000, 10000):
    for alpha in (1e-4, 1e-6):
        zthr = norm.ppf(1 - alpha / K)
        n_req = int(np.ceil(((zthr + 1.645) / delta) ** 2))
        rows.append(f"{K},{alpha},{n_req}")
        print(f"{K:>13} {alpha:>11.0e} {n_req:>14}")

# simulation validation at two representative points
rng = np.random.default_rng(35)
for K, alpha, n in ((500, 1e-6, None), (10000, 1e-6, None)):
    zthr = norm.ppf(1 - alpha / K)
    n = int(np.ceil(((zthr + 1.645) / delta) ** 2))
    T = 4000
    mu_true = delta * np.sqrt(n)
    correct = abstain = wrong = 0
    for _ in range(T):
        z_true = rng.normal(mu_true, 1)
        z_wrong_max = norm.ppf(rng.uniform() ** (1 / (K - 1)))  # max of K-1 exact
        best = max(z_true, z_wrong_max)
        if best < zthr:
            abstain += 1
        elif z_true >= z_wrong_max:
            correct += 1
        else:
            wrong += 1
    print(f"simulated K={K}, n={n}: correct {correct/T:.3f}, "
          f"abstain {abstain/T:.3f}, wrong-attribution {wrong/T:.5f}")
    rows.append(f"# sim K={K} n={n}: correct {correct/T:.4f} "
                f"abstain {abstain/T:.4f} wrong {wrong/T:.5f}")

# unwatermarked-document family FPR check (analytic, validated)
T2 = 200_000
K = 500; alpha = 1e-6
zthr = norm.ppf(1 - alpha / K)
hits = 0
for _ in range(T2):
    zmax = norm.ppf(rng.uniform() ** (1 / K))
    hits += int(zmax > zthr)
print(f"unwatermarked doc vs {K} keys at family FPR 1e-6: empirical "
      f"{hits/T2:.2e} (expected 1e-6; {T2} trials)")
(HERE / "results.csv").write_text("\n".join(rows) + "\n")
print("wrote results.csv")
