# Softmax–Thurstone: finite-temperature races (notes for the next thread)

Distilled from discussion, 2026-08-12. Status: candidate next direction for the
thurstone package and possibly the strongest positioning of the algorithm line:
*inverse thermodynamics for finite races*. Everything here is pre-verification —
per house rules, no claim below graduates until machine-checked.

## 1. The model

Package convention: min wins, X_i = a_i + eps_i, smaller a stronger. The current
hard race is p_i^(0)(a) = Pr(X_i = min_j X_j). The **finite-temperature race** is

    p_i^(tau)(a) = E[ exp(-X_i/tau) / sum_j exp(-X_j/tau) ],   tau > 0,

with the expectation OUTSIDE the softmax (that placement is the whole problem —
E[softmax] != softmax(E)). Free-energy form: with

    F_tau(a) = -tau * E log sum_j exp(-(a_j+eps_j)/tau),

we have grad_a F_tau = p^(tau)(a), a concave potential; as tau -> 0,
F_tau -> E[min_j X_j] and p^(tau) -> the hard race. **The existing package is the
zero-temperature limit of the soft package.**

## 2. The organizing theorem (implementation route 1: Gumbel convolution)

Gumbel-max identity, conditioned on X then averaged: with G_i iid standard Gumbel,

    p_i^(tau)(a) = Pr( a_i + eps_i - tau*G_i = min_j { a_j + eps_j - tau*G_j } ).

So with reflected min-type Gumbel noise M = -tau*G (CDF 1 - exp(-exp(m/tau))):

    f_effective,tau = f_eps CONVOLVE g_{M,tau}

**Softmax is a Thurstone race whose base density has been convolved with
minimum-type Gumbel noise.** Exact in the continuum. This is the API design
principle and the principal regression test. Consequences: no second
order-statistics engine, inverse transform, clustering, global calibrator, or
pricing formula needed — only a well-engineered noise-convolution constructor plus
semantic wrappers.

Numerical care: the reflected Gumbel is very skewed (support ~23*tau below the
mean, ~4*tau above at 1e-10 tail tolerance): build lattice masses from **CDF bin
differences**, not center-sampled densities; size the noise lattice separately by
tail tolerance; convolve with keep_L = L_base + L_noise (default crop would
truncate the long tail); center() afterwards (harmless — common shift).

Neural convention: networks use larger-logit-better; set X_i = -z_i internally.

## 3. Native soft-race calculus (route 2, deferred): the Laplace field

Let W_i = exp(-X_i/tau), phi_i(t) = E e^{-t W_i}, psi_i(t) = -phi_i'(t),
r_i = psi_i/phi_i, P(t) = prod_j phi_j(t). Then, via 1/s = int_0^inf e^{-ts} dt:

    p_i^(tau) = int_0^inf P(t) r_i(t) dt.

The dictionary with the hard race is exact:

| hard race | soft race |
|---|---|
| performance coordinate x | Laplace coordinate t |
| survival S_i(x) | Laplace transform phi_i(t) |
| hazard f_i/S_i | tilted mean r_i = psi_i/phi_i |
| field prod_i S_i | field prod_i phi_i |
| remove runner by DIVISION | remove runner by DIVISION |
| min-probability | Gibbs occupancy |

Union/removal are simpler (Phi multiplicative, R additive; no dead-heat
bookkeeping). Normalization is automatic (P' = -P*sum r_i => sum p_i = P(0)-P(inf)
= 1). And abilities act as TRANSLATIONS on the s = log t grid (phi_i(t) =
phi_0(c_i t), c_i = e^{-a_i/tau}) — tailor-made for the package's
shift-and-interpolate architecture. This is the real package-level mathematical
contribution, but implement AFTER the convolution route establishes reference
behavior (it would duplicate working infrastructure and adds a semi-infinite
quadrature; later payoffs: fast temperature sweeps, direct tau-derivatives,
avoiding the long Gumbel tail).

## 4. The Laplacian structure survives exactly

With pi_i(X) the conditional softmax: dp_i/da_j = -(1/tau) E[pi_i (delta_ij -
pi_j)], so Dp(a) = -L(w), a weighted complete-graph Laplacian with w_ij =
(1/tau) E[pi_i pi_j] > 0 — the same structural form as the package's hard-race
laplacian.py. Fast form: w_ij = (1/tau) int t P(t) r_i r_j dt, and the HVP

    [L(w) v]_i = (1/tau) int t P(t) r_i(t) [ v_i R(t) - V(t) ] dt,

R = sum r_j, V = sum r_j v_j: **O(NQ) Hessian-vector products** on a Q-point grid;
Newton–CG inverse logic carries over almost verbatim. (Also: the package keeps TWO
forward maps deliberately — multiplicity-aware Race.state_prices() for production,
no-tie outright_win_probabilities() for the smooth Laplacian analytics; the soft
feature must preserve both, with lattice-refinement convergence tests rather than
exact-equality claims between them.)

## 5. Calibration: two problems, one identifiability warning

(i) **Fixed-tau inversion** p -> a: the package's job; not closed form once
Var(eps) > 0. (ii) **Temperature estimation**: NOT identifiable from one race
(K-1 numbers vs K-1 abilities + tau). Requires structure: many races with shared
abilities, repeated outcomes, fixed logits + held-out labels, class-removal
interventions, rankings, or ensemble/dropout samples. API consequence: single-race
inversion REQUIRES fixed tau; a separate dataset-level SoftmaxTemperatureCalibrator
does tau. (Deterministic-logit endpoint: a_i = -tau log p_i + c, closed form.)

## 6. Statistical mechanics reading

beta = 1/tau, E_i = X_i: softmax = Gibbs measure; hard Thurstone = zero-temperature
ground-state selection; random abilities => **disordered finite-state Gibbs
system**; F_tau is the QUENCHED free energy (not the annealed shortcut — E[softmax]
!= softmax(log E e^{...})). "One man's softmax is another man's noise": thermal
(Gumbel, tau*G) vs quenched (eps) randomness are indistinguishable from one
marginal probability vector; interventions/repetition separate them. Package
positioning across the continuum: tau=0 hard Thurstone; 0<tau<inf softmax–
Thurstone; Var(eps)=0 ordinary softmax/Luce; tau->inf uniform. **Inverse
thermodynamics for finite races** — observed occupancies -> effective fields,
the inverse counterpart of neural-network stat mech (which predicts mu(x),
Sigma(x) forward from architecture; soft Thurstone converts them to class
probabilities, and inverts). Only the contrast covariance P Sigma P matters
(common-mode logit noise cancels exactly).

Relation to exp07 (do not conflate): exp07's softmax arises from FAST-MIXING
shared-environment kinetics (homogenization); here softmax arises from THERMAL/
Gumbel choice noise at the terminal map. Two different physical routes to the same
law; the Green–Kubo correction and the quenched-disorder correction are different
expansions and both belong in the program.

## 7. The sharp choice theorem ("any choice -> softmax-Gauss?")

Correct statement is local + asymptotic, in two parts:

**(a) Exact representation (no assumptions).** softmax is a diffeomorphism from
the contrast space H = {x : sum x = 0} to the open simplex; every interior p
equals softmax(clr(p)) with clr(p) = Pi log p, Pi = I - (1/K) 1 1^T. So "softmax
representation != Luce/IIA assumption" — logits may depend on the menu; only
menu-invariance makes it Luce.

**(b) Softmax–Gaussian normal form (CLT).** If aggregate shares p_n satisfy
sqrt(n)(p_n - p) => N_H(0, Sigma), then sqrt(n)[clr(p_n) - clr(p)] =>
N_H(0, Gamma), Gamma = Pi D_p^{-1} Sigma D_p^{-1} Pi, i.e. p_n = softmax(theta +
G/sqrt(n)) + o_p(n^{-1/2}): **any regular aggregate choice is asymptotically
logistic-normal**. Categorical special case: Sigma = D_p - p p^T gives Gamma =
Pi D_p^{-1} Pi (alr form: 1{j=k}/p_j + 1/p_K).

**(c) Critical-race phase diagram.** Near a population tie (p_n = u + h/sqrt(n)),
with vote counts N_n and collective choice Q_n = softmax(N_n / T_n):

    T_n << sqrt(n)  =>  hard Thurstone: argmax_j (h_j + Z_j)
    T_n ~  sqrt(n)  =>  softmax–Gaussian: E softmax((h+Z)/tau), tau = lim T_n/sqrt(n)
    T_n >> sqrt(n)  =>  uniform.

Winner law converges to E[softmax((h+Z)/tau)] — expectation INSIDE stays outside
the softmax; symmetric categorical case = Thurstone Case V at tau=0.

**(d) Neural corollary.** If class scores are sums of many weak features with a
CLT, the terminal universality class is: softmax output => logistic-normal field;
argmax output => Thurstone field. "Gaussian contrast space is the universal
finite-variance limit; softmax is its canonical finite-temperature map to
probabilities, Thurstone its zero-temperature map to a winner."

Failure modes to respect: boundary probabilities (Poisson, not Gaussian, for rare
choices), K growing with n, common shocks/long-range dependence, single choices
without aggregation, races far from ties (deterministic winner).

## 8. Implementation plan for thurstone (from the source read)

Key trap found in source: the 2D location/scale path constructs
Density.skew_normal inside density_for() (not from self.base) — a soft calibrator
MUST override density_for or per-competitor scales silently drop the Gumbel noise.

Commit sequence:
1. **Primitives**: Density.from_cdf_bins (bin-mass construction, useful generally);
   Density.gumbel_min(lattice, scale, mean=0) centered via E[-tau G] = -gamma*tau;
   softmax_effective_base(base, temperature, tail_probability) with auto-sized
   noise lattice and keep_L full convolution; tail diagnostics; unit tests.
   [Note: a simpler Density.gumbel_min already landed via PR #14 — upgrade it to
   CDF-bin masses per the numerical qualification above.]
2. **Public API**: thurstone/softmax.py — SoftmaxRace(Race),
   SoftmaxAbilityCalibrator(AbilityCalibrator) (SUBCLASS, so Global/LS calibrators
   drop in), density_for override, exports.
3. **Mathematical validation tests**: point-mass base => exact softmax
   (the reference test); tau=0 delegates exactly to hard machinery; Monte Carlo
   identity (E softmax vs Gumbel-race vs lattice); forward-inverse roundtrips;
   Laplacian convergence to -(1/tau)[Diag(p) - p p^T] (point mass) and to MC
   (random base); candidate removal incl. IIA-holds (deterministic noise) and
   aggregate-IIA-violated (random noise) controls; translation invariance,
   permutation equivariance, tau->inf uniformity; small tau/u, ClusterSplitter
   extremes, 2D path.
4. **Integration**: global calibrators; dynamic calibrator gets
   calibrator_factory or choice_temperature (convolution associativity lets
   bookmaker-Gaussian and Gumbel noise compose in either order); benchmarks.
5. **JS/docs parity**: docs/js/thurstone/softmax.js + fixtures from the Python
   generator; one diagram: performance density + Gumbel choice noise -> effective
   Thurstone density; sign-convention docs.
6. **Neural calibration layer**: fixed-tau random-logit inversion; dataset-level
   temperature fitting (min -sum log E softmax(Z_r/tau)_{y_r}); heteroskedastic
   independent noise; class-removal counterfactuals.
7. **Correlated logits**: low-rank factor conditioning (THIS IS PR #14's
   FactorRace — version 3 is already merged; wire it in), empirical ensembles,
   optional native log-Laplace backend (Section 3).

## 9. Why this matters for the program

- Package positioning upgrades from "another link function" to a continuum:
  hard/soft choice probabilities <-> latent locations under separate performance
  and choice noise, tau in [0, inf].
- The potentially novel core: **fast inverse transform for disorder-averaged Gibbs
  races** + exact continuation of the lattice and Laplacian machinery from zero to
  positive temperature (the log-Laplace translation structure).
- Connects to: exp07 (second route to softmax — homogenization), the runner-up
  principle (temperature/disorder unidentifiable from one vector — interventions
  separate them), PR #14 (the correlated backend is the already-merged version 3),
  and the GHK/econometrics line (logistic-normal aggregate shares are the
  compositional-data workhorse; the inverse map is our transform).

## Verification log

2026-08-17: the organizing identity machine-checked. E[softmin(X/tau)]
= P(argmin(X + tau*g)), g iid min-Gumbel: max diff 3.7e-4 at R=4e6
common draws (MC noise 1.5e-3), N=8, tau=0.7. The softmin expectation
is exactly a hard race with the base convolved with tau-Gumbel, so it
drops into winning.factor.race_probabilities as a convolved base;
composes with factors since the g are iid (conditional independence
survives). Temperature still not identifiable from one race.

2026-08-17 (later): IMPLEMENTED. race_probabilities(..., temperature=)
in winning.factor.races via per-runner lattice convolution with the
tau-Gumbel kernel; envelope padded 30*tau left (heavy min-Gumbel tail;
the exp33 lesson applied). Tests: matches common-draw MC softmin at
3e-3 with and without factors; tau->0 recovers hard race; tau->inf
flattens to uniform; calibration roundtrips at fixed tau. The notes'
larger program (choice theorem, phase diagram) remains open.
