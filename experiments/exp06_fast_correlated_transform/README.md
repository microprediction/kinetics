# Experiment 6: a fast transform for correlated fields

**The construction (program Q6).** Fit the correlation as Σ ≈ VVᵀ + diag(D) — k factors
plus idiosyncratic variance, by iterated principal-factor analysis. *Conditionally on
the factors the competitors are independent*, so the multiplicative cavity (field
product, divide one out) applies at every quadrature node:

> p_i = E_f [ ∫ f_i(x|f) · S_field(x|f)/S_i(x|f) dx ],  f ~ N(0, I_k)

with product Gauss–Hermite nodes for k ≤ 4 and scrambled-Sobol QMC beyond. The two
leave-one-out identities compose: the Gaussian/Schur side compresses the coupling into
factors; the field product prices the race in O(N·L) per node. Implementation in
[`../raceutil.py`](../raceutil.py) (`factor_model`, `win_probabilities_factor`,
`abilities_from_probabilities_factor`, `hermite_nodes`, `qmc_nodes`).

Relation to the `thurstone` package: this is `multiray` (ability = μ + Z·v per
condition) with the ray coordinate promoted to a latent Gaussian and integrated out;
the package's core transform is the independent limit V = 0.

**Results (seed 42, N = 16 windows, reference = 8M-draw MC with noise ~2e-4).**

- **Exact given the model**: known 2-factor model 2.8e-4; equicorrelated (k=1 exact)
  1.4e-4; C = I vs the independent transform 6.9e-9. All at reference noise.
- **Error is factor-model error**: max win-probability error tracks the off-diagonal
  residual of Σ̂, falling from ~1e-1 at k=1 to **7.5e-4 (exp kernel)** and **2.5e-4
  (chordal-SE kernel)** at k=8 (`figures/error_vs_rank.png`).
- **The deletion ensemble survives correlation**: one conditional field pass yields
  P(j wins | i removed) for all (i,j), agreeing with per-deletion recomputation to
  **1.1e-16** at 2.4× less time.
- **End-to-end, no Monte Carlo anywhere**: replicating exp03's blocked-window
  counterfactual with the fast transform (k=8): TV **0.0170** vs Harville 0.0823 —
  a ~5× improvement, fully deterministic, 36 s (exp03's MC pipeline reached 0.0094;
  the gap is the k=8 factor residual of the kinked exponential kernel).

**Two traps documented.** (1) Naive eigen-truncation *invents* off-diagonal correlation
— catastrophically near C = I (win-prob error 1e-1); iterated factor analysis fixes it.
(2) The squared-exponential of *geodesic* distance is not positive definite on a circle;
use the chordal version.

**Honest limits.** The exponential kernel's kink gives 1/m² eigenvalue decay, so global
factors are the wrong compression for it at large N: at N = 400, k=8 leaves 1.7e-2 error
and plain correlated MC is faster for one-shot forward pricing. The transform's value is
(a) small-to-moderate N (the 10–100 channel regime of the physics experiments) where it
is both fast and accurate, and (b) being smooth and deterministic in μ, which is what
the inverse fixed point needs. For Markov-type kernels the natural refinement is
sequential/Vecchia-style conditioning — tridiagonal precision instead of global factors
— which is the same Schur-complement machinery from the other end (see
[schur.microprediction.org](https://schur.microprediction.org)).

Run: `python run_fast_transform.py` (~3 min, numpy/scipy/matplotlib only).
