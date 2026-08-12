# Review notes (2026-08-12)

External review of the site and experiments 1–6. Actions taken are marked ▸; the rest
is preserved as the working agenda. These notes reframe the program significantly.

## 1. Semantics of deletion (site correction)

The two cavities encode DIFFERENT deletion semantics for X ~ N(μ, Σ):

| Operation | Surviving distribution |
|---|---|
| **Scratch** competitor i from the race | marginal: N(μ₋ᵢ, Σ₋ᵢ,₋ᵢ) |
| **Pin/condition** Xᵢ = c | conditional: Σ₋ᵢ,₋ᵢ − Σ₋ᵢ,ᵢ Σᵢᵢ⁻¹ Σᵢ,₋ᵢ |
| Delete coordinate of precision A = Σ⁻¹ | (A₋ᵢ,₋ᵢ)⁻¹ = the conditional covariance |

Scratching uses a **marginal**; Schur uses a **conditional**. A correlated Gaussian race
does not automatically invoke a Schur complement when a competitor is scratched; Schur
enters when the intervention pins, conditions, or re-equilibrates the field. Distinguish:
scratch a candidate / suppress a hazard / pin a coordinate / remove bonds / remove a
material site / recompute the background dynamics.
▸ docs/semantics.html added; intro and Q6 wording corrected.

## 2. Correct null model: proportional hazards ⟹ Luce (not "exponential ⟹ IIA")

Exponentiality is sufficient for Luce but NOT necessary. With cumulative hazards
Hᵢ(t) = aᵢ H₀(t) for ANY common baseline H₀, P(i wins) = aᵢ/Σaⱼ exactly — an arbitrary
common time change of exponential clocks still gives Luce. Non-exponential waiting
times break IIA only via different hazard SHAPES, dependence, history dependence, or
differential loadings on a shared environment. First-order perturbation theory in the
common clock u = H₀(t): with Hᵢ(u) = aᵢu + δbᵢ(u),
pᵢ = aᵢ/A + δ∫₀^∞ e^{−Au}[A bᵢ(u) − aᵢ B(u)] du + O(δ²);
a common perturbation bᵢ = aᵢ c(u) cancels exactly. (Compact note on its own.)
▸ intro.html corrected.

## 3. Flagship direction: softmax as homogenized kinetics (Paper 1)

Shared-state kinetic race: fast ergodic hidden state Y with generator L/ε; channel i
fires at intensity λᵢ(Y_t). Win probability solves the killed-generator equation
(L/ε − Λ_A) uᵢ^A = −λᵢ, i.e. uᵢ^A = (Λ_A − L/ε)⁻¹ λᵢ — a killed resolvent.

**Leading theorem**: p_i^A = λ̄ᵢ/Λ̄_A + O(ε) with λ̄ᵢ = ∫λᵢ dπ — softmax is the
homogenized choice law of fast hidden kinetics (θᵢ = log λ̄ᵢ).

**Green–Kubo correction**: with K_jk = ∫₀^∞ Cov_π(λⱼ(Y₀), λₖ(Y_t)) dt,
p_i^A = λ̄ᵢ/Λ̄_A − (ε/Λ̄_A)[Σ_{j∈A} K_jᵢ − (λ̄ᵢ/Λ̄_A) Σ_{j,k∈A} K_jk] + O(ε²).
Properties: (a) K is Green–Kubo — integrated dynamical correlations, not static
covariance; (b) the same (λ̄, K) answers EVERY blocked-set counterfactual by restricting
sums to A — the true statistical analogue of "one global computation encodes all
deletions"; (c) common-mode fluctuations λᵢ = aᵢc(y) cancel exactly; (d) low-rank
loadings λᵢ − λ̄ᵢ = bᵢᵀz(y) give K = BΓBᵀ — a compressed correlated race.

Title: *Softmax from Fast Mixing: Green–Kubo Corrections for Counterfactual Races.*
Prove first for finite-state irreducible chains; extend to reversible diffusions with a
spectral gap. Narrow escape is the ideal test case: hidden process = reflected position;
Dirichlet-to-Neumann operator = boundary-trace generator; disk diagonalizes in Fourier
modes, so the leading geometric substitution correction may be analytic.

## 4. Identifiability / the runner-up principle (Paper 2)

Winner-only nonidentifiability: q_j^{(−i)} = p_j + pᵢ M_ij with M_ij = P(π₂=j | π₁=i)
COMPLETELY unconstrained by p. Winner-only data yields only bounds
(q_j^{(−i)} ≥ p_j, sums to 1). Finite-choice version of classical competing-risks
nonidentifiability. Data hierarchy: winner-only → +time → **+runner-up identity**
(identifies every singleton scratch) → top-(k+1) (k-deletions) → multiple blocked sets.
The runner-up identity is more useful for deletion counterfactuals than the winning
margin. Independent Thurstone inversion = a canonical COMPLETION of p into a ranking
model, not identification of physical propensities; validity = held-out interventions.
▸ exp01 conclusion reframed: it demonstrates that full-menu winner frequencies contain
no geometric substitution information (not that "correlated Gaussian is the missing
model"). exp03 is consistent: its correlation structure was SUPPLIED by geometry and
calibrated on an intervention (blocked-set data), exactly what the hierarchy requires.

**Ranked narrow-escape experiment**: simulate all-reflecting particle, record the
sequence of distinct window encounters (boundary local time / excursion convention, or
Robin windows); one trajectory answers many blocked-set counterfactuals. Compare
information regimes (winner / +time / +runner-up / top-k / randomized interventions) ×
models (proportional renorm / independent Thurstone / low-rank factor race / Markov
substitution kernel / Green-function or Green–Kubo). Evaluate on random singleton,
pair, triple block sets across many geometries.

## 5. Substitution kernel / transfer resolvent

Markov substitution model: q_A = p_A + p_B (I − M_BB)⁻¹ M_BA. Known in OR choice
modeling (Markov chain choice); kinetic contribution = M from boundary encounter
sequences, Green-function approximation, geometric sparsity, estimation from few
interventions. Better unifying object for the site than a Gaussian covariance: the
**transfer resolvent** — (Λ_A − ε⁻¹L)⁻¹, (I − M_BB)⁻¹, A⁻¹ are all resolvents whose
response to suppression/deletion is Schur/Woodbury/perturbation theory.

## 6. Correlated Thurstone: demoted to principal computational baseline

Keep the exp06 factor machinery (conditional multiplicative identity survives), with
bᵢ = Fourier/geometric window features rather than free parameters — but it must be
trained across available sets or encounter data (not identifiable from one p-vector).
Baseline against shared-state and substitution-resolvent models; not the target.

## 7. Quadratic/defect direction (Paper 3): physical target needed

Rank-one downdate + fracture updates are classical (SMW + sparse Cholesky downdates,
selected inversion). The publishable question: **can a cheap harmonic cavity screen
expensive relaxed/nonlinear defect calculations?** Program: actual bond deletion
(H′ = H − kvvᵀ, Woodbury with positive denominator), true vacancies (small-rank block
of incident bonds), vector elasticity (2D/3D displacements, not scalar Laplacian),
anharmonic relaxation benchmark, screening metrics (top-K recall, rank correlation,
saved relaxation calls), error bounds from Hessian variation.
▸ exp02 README annotated: current experiment is pinning, not vacancy formation.

## 8. Paper sequence

1. *Softmax from Fast Mixing: Green–Kubo Corrections for Counterfactual Races* —
   the profound one (homogenization, killed resolvent, softmax layers, KMC).
2. *The Runner-Up Principle for Counterfactual Races* — fastest rigorous paper
   (nonidentifiability polytope, top-(k+1) sufficiency, substitution resolvent,
   ranked narrow escape).
3. *Harmonic Cavity Screening for Relaxed Defects* — keep separate from the races.

Bottom line: don't make "a full correlated Gaussian Thurstone transform" the flagship.
The stronger object: fast hidden dynamics ⟹ softmax at leading order + Green–Kubo
non-IIA correction, with identifiability determining what data (runner-up, encounter
sequences, interventions) is needed to LEARN the correction rather than assume it.
