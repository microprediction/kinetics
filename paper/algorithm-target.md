# The target: deterministic choice probabilities and share inversion for
# factor-structured probit with many alternatives

Decision (2026-08-12): the publishable-algorithm target for the winning/thurstone
lattice transform and its correlated extension is **discrete-choice econometrics**
(empirical IO, quantitative marketing, transportation).

## The field's acknowledged need

- Multinomial probit (MNP) choice probabilities are (N−1)-dimensional Gaussian
  integrals with no closed form — "intractable for moderate to large numbers of
  alternatives" is the field's own language. The incumbent is the GHK simulator
  (Geweke–Hajivassiliou–Keane): Monte Carlo, noisy, per-alternative cost, and the
  noise poisons derivative-based estimation. Whole subfields (mixed logit) exist
  substantially because probit is computationally avoided — IIA-conditional
  structure accepted as the price of tractability.
- Share inversion (market shares → mean utilities), the workhorse of demand
  estimation since BLP, is a contraction-mapped fixed point whose cost is dominated
  by the forward share computation. For probit demand it is essentially not
  attempted at scale, because the forward map is too expensive.
- Timeliness proof: arXiv 2603.24705 (March 2026) attacks exactly this with
  *neural amortization* — training equivariant networks to approximate MNP
  probabilities. Heavy, approximate, no inversion capability.

## What we have that answers it (all merged and tested in thurstone main)

1. **Forward**: factor-structured probit (Σ = VVᵀ + D, arbitrary — including
   non-Gaussian — idiosyncratic densities) choice probabilities for ALL N
   alternatives at once: O(Q · N · L) via the lattice field-product/divide-out at
   each quadrature node. Deterministic, smooth in utilities (usable inside
   gradient-based estimators), exact given the factor model. No training, no draws.
2. **Inverse**: shares → utilities under correlation (`solve_abilities`), the
   probit analogue of the BLP contraction, with the correlation-aware damping we
   learned the hard way.
3. **The assortment ensemble**: every single-removal (stockout/delisting)
   counterfactual from one conditional field pass — the retail question, with the
   correct scratch (marginal) semantics.

Prior-art status from the search: the 1-D reduction for *independent* probit is
classical; factor dimensional reduction is known in principle (Connors–Hess–Daly);
GHK and MACML are the benchmarks to beat. The specific assembly above — all-N inner
loop, correlated inversion, deletion ensemble, arbitrary base densities — appears
unclaimed in this literature. The 2021 SIAM paper (winning) is the independent-case
precedent by the same author.

## The paper

*Deterministic Choice Probabilities and Share Inversion for Factor Probit with Many
Alternatives.* Venue candidates: Journal of Choice Modelling, Transportation
Research B, Journal of Econometrics, Marketing Science.

Required evidence (exp13, the make-or-break benchmark):
- vs GHK: accuracy/time frontier at N ∈ {10, 100, 1000, 5000}, factor ranks 1–5;
  derivative smoothness comparison (finite-difference noise of GHK vs our exact
  smoothness) — the estimation-relevant metric;
- vs mixed logit: substitution-pattern fidelity on non-IIA ground truth;
- share inversion at N = 1000+ with convergence diagnostics (contraction evidence);
- one-pass assortment ensemble vs recompute-per-removal timing;
- honest limits: full-covariance (non-factor) cases, and where GHK's importance
  sampling wins.

Sources: GHK background and MNP intractability (Wikipedia GHK; Stata asmprobit/
cmmprobit docs; Connors–Hess–Daly, eprints.whiterose.ac.uk 77195); BLP inversion
practice and pain points (Nevo practitioner's guide; Conlon–Gortmaker; arXiv
1802.04444 demand inversion); neural competitor arXiv 2603.24705.
