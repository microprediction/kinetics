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
| **Exp 15** — perturbation certificate | [`exp15_perturbation_certificate/`](exp15_perturbation_certificate) | Price triple-tie derivative identity **verified** (7.9e-6); two bounding conjectures **refuted** (ratio ≤1.27); certificate = empirical conservative estimate (held 100%, 6–30×) |
| **Exp 14** — **boundary studies** for the algorithm paper | [`exp14_boundaries/`](exp14_boundaries) | expected GHK-wins regime did not materialize (k=8 ≥ GHK R=10⁴ at all spectral decays); substitution (noise families standardized): probit misallocates 2.8% of redistributed mass vs 7.7% (mixed logit) vs 15.9% (IIA) on large deletions |
| **Exp 13** — **the GHK benchmark** (algorithm paper) | [`exp13_ghk_benchmark/`](exp13_ghk_benchmark) | factor-probit shares: flat err ~3e-4 vs GHK's growing 3–8e-3; N=5000 in 22 s vs ≥13 h matched-accuracy GHK; smooth unbiased derivatives; share inversion validated at N=1000; one-pass assortment ensemble |
| **Exp 12** — **real engineering data**: N−1/N−2 grid contingency screening (IEEE 30/118-bus) | [`exp12_grid_contingency/`](exp12_grid_contingency) | MLODF = the leave-k-out identity, exact to 2e-15; 17,205 N−2 pairs in 0.2 s; DC+baseQ screen recalls 95% of true worst AC outages at 22× less compute; reactive-blind screening fails (0.20) — the metric, not the linearization |
| **Exp 11** — **harmonic cavity screening for relaxed vacancies** (Paper 3) | [`exp11_vacancy_screening/`](exp11_vacancy_screening) | 360 vacancies from one inverse in 9 ms (exact to 5e-16); recall 1.0/0.85/0.70 as anharmonicity grows; 7.9× compute savings |
| **Exp 10** — **substitution kernel from pure geometry** (Q3×Q7 bridge) | [`exp10_green_function_M/`](exp10_green_function_M) | geometric M beats empirical M at every depth; zero-trajectory model beats winner-only trajectory models on pairs/triples |
| **Exp 9** — **Green–Kubo on continuous physics** (Q7) | [`exp09_gk_narrow_escape/`](exp09_gk_narrow_escape) | reflected BM + Robin windows: theorem slopes 0.93/1.93; real simulation agrees with discretized generator at MC noise; two systematics caught by layer separation |
| **Exp 8** — **ranked narrow escape** (Q3, runner-up principle) | [`exp08_ranked_escape/`](exp08_ranked_escape) | winner-only degrades with block size (0.038→0.069); runner-up kernel + Markov resolvent hits the noise floor at every depth |
| **Exp 7** — **Green–Kubo theorem** on a finite-state chain (Q7 flagship) | [`exp07_green_kubo/`](exp07_green_kubo) | slopes 0.99/1.99 (theory 1/2); one (λ̄, K) corrects all blocked subsets; common mode = exact Luce at every ε; rank-r loadings ⟹ rank-r K |
