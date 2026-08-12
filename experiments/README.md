# Experiments

Numbered experiments follow the [research program](https://kinetics.microprediction.org/program.html);
flat scripts are identity checks. Everything here runs on numpy/scipy/matplotlib alone.

| What | File | Result |
|---|---|---|
| Rank-one cavity identity + timings | [`cavity_downdate_demo.py`](cavity_downdate_demo.py) | exact to 1e-15; leave-1/2/3-out verified |
| Multiplicative cavity identity + timings | [`race_field_demo.py`](race_field_demo.py) | exact; 26× at N=2000 |
| Shared race transforms (forward + inverse) | [`raceutil.py`](raceutil.py) | used by exp01 |
| **Exp 1** — surrogates on a real first-passage simulation (Brownian narrow escape) | [`exp01_narrow_escape/`](exp01_narrow_escape) | physics is non-IIA (TV 0.082) but *independent* races don't capture the geometric neighbor effect → motivates Q6 |
| **Exp 2** — rank-one cavity on a disordered elastic network | [`exp02_glass_cavity/`](exp02_glass_cavity) | all 784 site deletions in 9 ms (~870×); 4k defect-pair interactions in 35 ms; exponential screening recovered |
| **Exp 3** — geometry-informed **correlated race** (Q6) | [`exp03_correlated_race/`](exp03_correlated_race) | **~9× error reduction** on the blocked-window counterfactual (TV 0.009 vs 0.081), ℓ chosen on a held-out intervention |
| **Exp 4** — temperature transfer on a Kramers barrier-crossing simulation | [`exp04_kramers_transfer/`](exp04_kramers_transfer) | Arrhenius rescaling halves transfer error at kT 1.0→0.7 and →0.55; noise law (Gumbel vs Gaussian) second-order |
| **Exp 5** — correlated race applied to temperature transfer | [`exp05_correlated_transfer/`](exp05_correlated_transfer) | **informative negative**: TV flat in ℓ — correlation repairs deletion counterfactuals, not global parameter shifts |
| **Exp 6** — **fast transform for correlated fields** (Q6) | [`exp06_fast_correlated_transform/`](exp06_fast_correlated_transform) | factor-conditioned quadrature: exact given the model; deletion ensemble from one pass (1e-16); exp03 counterfactual reproduced with no Monte Carlo (TV 0.017 vs 0.082 IIA) |
| **Exp 10** — **substitution kernel from pure geometry** (Q3×Q7 bridge) | [`exp10_green_function_M/`](exp10_green_function_M) | geometric M beats empirical M at every depth; zero-trajectory model beats winner-only trajectory models on pairs/triples |
| **Exp 9** — **Green–Kubo on continuous physics** (Q7) | [`exp09_gk_narrow_escape/`](exp09_gk_narrow_escape) | reflected BM + Robin windows: theorem slopes 0.93/1.93; real simulation agrees with discretized generator at MC noise; two systematics caught by layer separation |
| **Exp 8** — **ranked narrow escape** (Q3, runner-up principle) | [`exp08_ranked_escape/`](exp08_ranked_escape) | winner-only degrades with block size (0.038→0.069); runner-up kernel + Markov resolvent hits the noise floor at every depth |
| **Exp 7** — **Green–Kubo theorem** on a finite-state chain (Q7 flagship) | [`exp07_green_kubo/`](exp07_green_kubo) | slopes 0.99/1.99 (theory 1/2); one (λ̄, K) corrects all blocked subsets; common mode = exact Luce at every ε; rank-r loadings ⟹ rank-r K |
