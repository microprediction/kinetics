# Experiment 14: the boundary studies

The algorithm paper's honesty sections, run before the claims were written.

**Anchor.** The generic-base factor forward implemented here (normal or Gumbel-min
idiosyncratic noise) reproduces softmax *to machine precision* (5.6e-17) with a
Gumbel base and zero loadings — the exact Luce nesting.

**Part A — full-covariance boundary** (N=50, dense correlation matrices with
eigenvalue decay λ_m ∝ m^−γ, truth = 8×10⁶-draw MC, GHK at the *exact* Σ as the
reference):

| γ (top-4 eig share) | lattice k=1 → k=8 | GHK R=10³ | GHK R=10⁴ |
|---|---|---|---|
| 0.5 (21%) | 3.8e-3 → 1.3e-3 | 8.0e-3 | 3.2e-3 |
| 1.5 (64%) | 2.4e-2 → 2.6e-3 | 1.3e-2 | 3.4e-3 |
| 3.0 (93%) | 5.6e-2 → 1.2e-3 | 2.2e-2 | 4.1e-3 |

**We expected a clear GHK-wins regime and did not find one at this size**: k=8
matches or beats GHK R=10⁴ at every decay rate. The factor floor decays slowest at
*intermediate* γ (the same mid-spectrum hardness as exp06's kinked kernels), which
is where a GHK advantage would first appear if accuracy demands exceeded the
affordable floor. The boundary is conditional, not territorial — reported as it
fell, with the original expectation corrected in the script docstring.

**Part B — substitution fidelity** (truth misspecified for *every* candidate:
t(5) factors + skew-normal idiosyncratic; candidates calibrated to identical menu
shares with supplied loadings; scored on deletion counterfactuals vs fresh MC).
Blocks whose deleted share sits at the MC noise floor are uninformative (all
models tie, as they must). On informative blocks, TV as a fraction of redistributed
mass:

| model | favorite removed (42.5% share) | mid blocks (2–10%) |
|---|---|---|
| plain logit (IIA) | 20.8% | 30.8% |
| factor mixed logit | 14.6% | 25.4% |
| **factor probit** | **7.0%** | **13.6%** |

Factor structure carries the first half of the correction; matching the
idiosyncratic noise *family* carries a further factor of two (caveat: the truth's
skew-normal noise is closer to Gaussian, so read this as "the noise family
matters," not "probit always wins").

Tests: `tests/test_boundaries.py` (softmax anchor, base parity with raceutil,
calibration roundtrips for both bases).

Run: `python run_boundaries.py` (~12 min, numpy/scipy/matplotlib only).
