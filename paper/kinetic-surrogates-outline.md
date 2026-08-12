# Kinetic Surrogates via Fast Ability Transforms — working-paper outline

Status: outline. Sections graduate to LaTeX once the corresponding experiment exists.

## 1. Introduction

- Kinetics as a race: KMC step = exponential race among escape channels; MD/fracture/glass
  first events = races with non-exponential, non-IIA structure.
- The surrogate loop: expensive simulation → first-event observations → latent propensities
  (fast ability transform) → cheap, transferable surrogate.
- Thesis: one global computation compresses the whole leave-one-out ensemble; two
  identities realize this (multiplicative for races, rank-one Schur downdate for quadratic
  systems).

## 2. The race model and the multiplicative cavity

- Thurstone race: X_i = μ_i + ε_i, argmin wins. p_i = ∫ f_i ∏_{j≠i} S_j.
- Exponential special case = Luce/Harville = KMC transition rule; IIA and where it fails.
- Fast ability transform (Cotton 2021): lattice, S_field = ∏ S_j once, S_{-i} = S_field/S_i,
  O(N) total; inverse transform for probabilities → abilities.

## 3. The rank-one cavity and leave-k-out

- G⁽ⁱ⁾_jk = G_jk − G_ji G_ik / G_ii ; interpretation: subtract interaction mediated
  through i. Cavity Green function of statistical mechanics; hat-matrix LOO of statistics.
- Leave-k-out: G⁽ˢ⁾ = G_{S̄S̄} − G_{S̄S} G_{SS}⁻¹ G_{SS̄}; compressed defect ensemble.
- Complexity accounting: one O(n³) solve + O(n²) per full cavity matrix + O(1) per scalar.

## 4. The dictionary and the organizing question

- Racing ↔ kinetics table (see docs/intro.html).
- Q: which remove-one-and-recompute calculations are done explicitly today although one
  global solve contains all the answers? (Audit table, Q1 of the program.)

## 5. Experiments

- 5.1 Identity verification and timings (experiments/cavity_downdate_demo.py,
  experiments/race_field_demo.py).
- 5.2 Synthetic KMC: Thurstone surrogate vs Harville surrogate on counterfactual races
  (channels blocked/merged) under non-exponential waiting times. [Q2]
- 5.3 Winner-only identifiability; value of the winning margin. [Q3]
- 5.4 Watermark races: context-conditioned race likelihood vs green-count detection
  (watermark_races/ — an "extremal observation" inference problem with fully known ground
  truth; doubles as a stress test of the truncated-field fast transform at vocabulary
  scale). [Q2/Q3 adjacent]

## 6. Discussion

- When the harmonic/independence assumptions break: relaxation after deletion,
  anharmonicity, correlated fields; the correlated-race open problem (Q6).
- Streaming maintenance (Q5) and links to the allocation/precise machinery.

## References

Cotton 2021 (SIAM JFM); Thurstone 1927; Luce 1959; Harville 1973; Mézard–Parisi–Virasoro
1987; Mézard–Montanari 2009; Erdős–Yau 2017; Voter 2007; Sherman–Morrison 1950; Hager 1989;
Kirchenbauer et al. 2023 (watermarking).
