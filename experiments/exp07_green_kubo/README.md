# Experiment 7: softmax from fast mixing — the flagship theorem, verified exactly

**Claim (program Q7).** When a race's channel intensities λᵢ(Y_t) are driven by a fast
hidden Markov state (generator L/ε), the win probability solves a killed-resolvent
equation, and homogenization gives: **softmax is the leading-order choice law**
(p_i → λ̄ᵢ/Λ̄_A), with the first correction a **Green–Kubo** term built from integrated
rate autocovariances K_jk = ∫₀^∞ Cov_π(λⱼ(Y₀), λₖ(Y_t))dt:

> p_i^A = λ̄ᵢ/Λ̄_A − (ε/Λ̄_A)[Σ_{j∈A}K_jᵢ − (λ̄ᵢ/Λ̄_A)Σ_{j,k∈A}K_jk] + O(ε²)

Derived here independently from the resolvent expansion (solvability conditions at
orders ε⁰ and ε²; the deviation integral D = (Π−L)⁻¹(I−Π) supplies K) — the
coefficients match the review's formula exactly.

**Verification (6-state chain, 10 channels, all quantities exact linear solves —
seed 5, `results.csv`).**

- Convergence to softmax: slope **0.985** (theory 1). After the Green–Kubo
  correction: slope **1.985** (theory 2). `figures/convergence.png`.
- **One (λ̄, K) pair answers every blocked subset**: across 41 random availability
  sets at ε=0.05, max error drops from 2.8e-3 (softmax) to 4.0e-4 (GK) with no
  per-subset computation — the statistical analogue of "one global solve encodes all
  deletions", confirmed. `figures/subsets.png`.
- **Common-mode cancellation is stronger than the theorem**: λᵢ(y) = aᵢc(y) gives a
  GK coefficient of 2.6e-16 — and exact-minus-softmax of 1.1e-16 *at ε = 0.3*: a
  common-mode fluctuation is a common time change, so Luce holds **exactly at every
  ε** (the proportional-hazards normal form), not just asymptotically. The two theory
  pieces cross-validate.
- **Low-rank loadings ⟹ low-rank K**: rank-2 environmental loadings give K of rank
  exactly 2 (singular values [0.022, 0.002, 0, 0]) — the compressed correlated race
  has physical provenance.
- **Gillespie check**: a real event-driven simulation with the hidden state (40k
  trajectories, ε=0.3) agrees with the resolvent to 1.9e-3 = its own MC noise.

**Status of Paper 1** (*Softmax from Fast Mixing: Green–Kubo Corrections for
Counterfactual Races*): the central formula is now machine-verified on finite chains,
including the subset-uniformity and cancellation properties. What remains is the
write-up, the reversible-diffusion extension, and the narrow-escape application
(where the disk's boundary operator diagonalizes in Fourier modes).

Tests: `tests/test_green_kubo.py` locks in every property above, including K against
brute-force time integration of the covariance.

Run: `python run_green_kubo.py` (~5 s, numpy/matplotlib; tests need scipy).
