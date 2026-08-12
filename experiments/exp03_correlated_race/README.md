# Experiment 3: a correlated race captures the geometry that independent races miss

**Question (program Q6).** Experiment 1 showed real diffusive kinetics violates IIA
*geometrically* — flux from a blocked window is inherited by its neighbors — and that no
independent race captures it. Is correlation the missing object?

**Model.** Windows race with performances X = μ + Lε where the noise correlation is
geometry-informed: corr(i,j) = exp(−d_ij/ℓ), d_ij the angular distance between window
centers. In a correlated race the deleted competitor's wins pass disproportionately to
its correlated partners — exactly the physics' neighbor-inheritance. ℓ → 0 recovers the
independent Thurstone race. Win probabilities by Monte Carlo with common random numbers;
abilities fit by a damped fixed point with a correlation-aware step (the effective noise
between competitors i,j has variance 2(1−ρ_ij)σ², so strong correlation demands a small
step — with a naive fixed step the fit silently diverges for ℓ ≥ 0.4, which initially
masked the result).

**Protocol.** Fit abilities to open-geometry frequencies (30k trajectories). Calibrate ℓ
on one intervention (block the 4th–5th busiest windows, 40k-trajectory ground truth).
Test on a different intervention (block the 3 busiest, 60k ground truth). All ground
truths re-simulated Brownian dynamics, same geometry/seed as experiment 1.

**Result (seed 42).**

| surrogate | TV on test intervention |
|---|---|
| Harville / IIA | 0.0813 |
| independent Thurstone | 0.0850 |
| **correlated race, ℓ\* = 1.6 (chosen on calibration)** | **0.0094** |

A **~9× error reduction**, essentially at the simulation noise floor (~0.008), with the
correlation length chosen entirely on held-out data. The redistribution figure shows the
correlated race matching the truth at the starred neighbor windows where both
independent surrogates fail. The optimum is broad (ℓ ∈ [0.8, 3.2] all beat independent
races by ≥ 7×), consistent with the long-ranged (algebraic) correlations of harmonic
measure in a disk.

**Status of Q6.** Promoted from speculative to *first positive result*. Open: a fast
(non-Monte-Carlo) transform for correlated fields — the multiplicative cavity no longer
factorizes, which is precisely where the rank-one/Gaussian cavity should re-enter.

Run: `python run_correlated_race.py` (~3 min, numpy/scipy/matplotlib only).
