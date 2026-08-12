# Experiment 11: harmonic cavity screening for relaxed vacancies

**The Paper 3 question.** The rank-one/Woodbury identities are classical; the
publishable question is whether a cheap **harmonic cavity** estimate can *screen*
expensive **relaxed, anharmonic** defect calculations. This experiment implements the
review's full prescription: vector (2D) elasticity, true vacancy semantics, nonlinear
ground truth, and screening metrics.

**Physics.** ~480-node jittered triangular network, boundary clamped, harmonic bonds
with frustrated natural lengths (quenched mismatch = the anharmonicity dial), relaxed
to a pre-stressed equilibrium. Removing a site leaves unbalanced forces g on its
neighbors; the lattice re-relaxes (geometric nonlinearity — finite rotations of
pre-stressed bonds), releasing energy ΔE(v).

**Harmonic prediction, all ~360 vacancies from ONE inverse.** ΔE_harm = ½gᵀH_v⁻¹g,
where H_v is reached from G = H⁻¹ by a 2×2 leave-out (the site's dofs) plus a
push-through Woodbury on the neighbor block: ΔE = ½gᵀ(I − SC)⁻¹Sg — no C⁻¹, so
degenerate bond states are safe. **9 ms for the whole sweep**, verified against
direct re-assembly to 5e-16.

**Results (seed 13, top-20 recall at a 40-relaxation budget, ~360 candidates).**

| mismatch | Spearman ρ | recall @ budget | harmonic/true ratio (5–95%) |
|---|---|---|---|
| 0.10 | 0.9995 | 1.00 | [0.98, 1.05] |
| 0.20 | 0.9896 | 1.00 | [0.96, 1.20] |
| 0.30 | 0.9399 | 0.85 | [0.87, 1.52] |
| 0.40 | 0.8504 | 0.70 | [0.56, 1.61] |

**Reading.** Screening is *perfect* through moderate frustration and degrades
gracefully — at mismatch 0.4 individual predictions are off by up to ±60%, yet rank
information survives (ρ = 0.85) and a 2× budget still recovers 70% of the true top-20
at **7.9× less compute** than relaxing everything. The exactness column never moves:
the failures are physics (anharmonic relaxation, occasional basin changes), never the
algebra — which is exactly the decomposition a screening tool needs. The regime map
(`figures/breakdown.png`) is the deliverable: it tells a practitioner *when* the
one-inverse compressed defect ensemble can be trusted as a first-stage filter.

Tests: `tests/test_vacancy_screening.py` (finite-difference gradient and Hessian
checks, Woodbury-vs-direct exactness, near-harmonic limit ratio → 1, graceful
degradation).

Run: `python run_vacancy_screening.py` (~1 min, numpy/scipy/matplotlib only).
