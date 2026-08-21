# Open items for this paper

## RESOLVED: Proposition 1 is prior art

Both flagged references were obtained and read. The verdict changed the paper.

**Marley & Colonius (1992)** contains Proposition 1 twice. Section 4 defines
the proportional hazard rate condition h^X_x(t) = C_X(x) h_X(t) and proves it
is EQUIVALENT to independence of the chosen option and the time of choice, with
the constants forced to equal the choice probabilities. Section 6 gives the
explicit representation Pr[t(x)>t] = exp(-u(x)Psi(t)) yielding u(x)/sum u(y)
for arbitrary increasing Psi. Full text recovered from a Wayback snapshot of
Colonius's Oldenburg faculty directory (the live URL is dead):
web.archive.org/web/20170705102702id_/http://www.uni-oldenburg.de/fileadmin/user_upload/psycho/ag/kogn/colonius/Marley_colonius_JMP92.pdf

**Elandt-Johnson (1976)** proves, per its abstract, that under proportional
hazard rates the cause-conditional failure time distribution equals the overall
one regardless of cause, and crucially WITHOUT assuming independent failure
times. Sixteen years earlier and in the pure competing-risks setting. Abstract
obtained via OpenAlex and the T&F landing page through a text proxy; the body
was not obtained, so whether she displays a_i/sum(a_j) explicitly is
unconfirmed, though her stated theorem entails it.

**Action taken**: Proposition 1 is now presented as a known result recovered in
a new setting, with all three citations (adding Kochar & Proschan 1991, which
Marley & Colonius call an equivalent result). The only thing claimed as added
is that the common factor may be a random functional of the hidden environment
rather than a deterministic function of time.

**Still worth doing**: Elandt-Johnson's 1979 review, "Equivalence and
nonidentifiability in competing risks: A review and critique", NCSU Institute
of Statistics Mimeo Series No. 1222, is by the same author reviewing exactly
this material and is the best single target for nailing down the 1976 body. It
sits behind NCSU's bot wall; a browser session would get it.
repository.lib.ncsu.edu/items/3d22862d-c325-4e81-9ebf-57c5e05a68fe

## Positioning that must survive editing

The diagonal of K is the classical motional-narrowing correction (Anderson
1954, Kubo 1954). The novelty is the OFF-DIAGONAL, non-symmetric K_ji acting on
the branching ratio, which the scalar theory cannot see because a common rate
shift cancels from a ratio. The remark before Theorem 1 says this. Do not cut
it; without it the paper reads as unaware of a seventy-year-old result.

Colantoni (2026), arXiv:2604.27901, is the closest structural neighbour and is
four months old: Markov-modulated killing, Feynman-Kac, fast-switching
averaging, single rate, leading order only. Cited in Related work. Keep it.

## Not yet done

- No comparison against an alternative method on the same problem. The audit
  ranked this second in importance after citations.
- Physics results are one geometry and one window arrangement; the chain
  results replicate over twenty environments but the continuum ones do not.
- No estimator for K from data. This is what a choice-modelling or chemical
  physics audience would want, and its absence rules out those venues.
- Venue undecided. The evidence base (exact linear algebra, measured orders,
  layered numerics, no data, no estimation) fits SIAM Multiscale Modeling and
  Simulation or Journal of Statistical Physics, and fails choice modelling and
  machine learning on the estimation gap alone.

## Saturation claim: justification and referee exposure

The intro says the order of arrivals "saturates at second place... at this
order of the expansion". Support: exp41 shows rank(winner+runner-up design) =
rank(all blocked-subset experiments) = N^2-N-1, the identified cap. The reason
deeper prefixes cannot exceed the cap is that, at first order, prefix
probabilities are functions of subset shares via chaining the inheritance
identity q_j^(-i) = p_j + p_i M_ij through successive deletions. That chaining
is not spelled out in the paper; a referee could ask for it. At second order
(the eps^2 term involves a third-order correlation tensor T_jkl) deeper places
plausibly DO add information — open question, connects to the higher-order
remark candidate.

## Cox-framing prior-art sweeps (2026-08-22)

Two agents swept credit/econometrics and point-process/ranking/physics under
the new framing. NO ANTICIPATION of either main result in any strand. All
added citations carry Crossref/OpenAlex-verified identifiers.

Flags absorbed into the paper: Beggs-Cardell-Hausman exploded logit +
Hausman-Ruud rejection (the ranked IIA null); Dansie/Bunch MNP normalization
removing exactly N+1 parameters (same count as our family — left open whether
structural); Kienker gauge classes; Zhao-Xia top-2 insufficiency for PL
mixtures (contrast); diversion-ratio familiarity of the deletion identity;
Rydén/Fredkin-Rice timestamps-vs-marks-only positioning; Duffie et al. frailty
filtering on one path; Ruan et al. dependent Poisson race.

Caveat: arXiv API and Semantic Scholar rate-limited both agents (429), so
2024-2026 arXiv-only preprints are undersampled. Rerun a preprint-focused
sweep before submission.

## Departure from Luce as a structural diagnostic (Peter's suggestion, 2026-08-22)

Where the arrival model applies (incl. neural temporal point processes, whose
softmax-mark x timing-density factorization IS the proportional-hazards
condition), departure from Luce is not just evidence of hidden structure but a
measurement of it:

- CERTIFICATE (exact, already in the paper): common-gain drivers produce
  identically zero departure (verified 1e-16 through the identified
  projection), so any nonzero departure certifies differential structure.
- MODE COUNTING (RESOLVED 2026-08-22, now experiment 42): the anomaly was
  geometry, as suspected. {K + d lam^T} is an N-dimensional subfamily whose
  members ALL have rank r+1 (verified: 100/100 random d), which is the fat
  set the optimizer finds; rank <= r needs d in col(K), codimension N-r,
  which it never hits. The estimator is algebraic, no optimization: one fat
  solve at rank r+1 recovers t to machine precision (the diag direction is
  not in the fat set), then the combinations lam_k M_j - lam_j M_k cancel d
  exactly and span col(K), so their numerical rank IS r. Exact for r=1..5
  with trailing singular values at 1e-13.
- Application sketch: fit the exacta board of a trained neural TPP (RMTPP /
  Neural Hawkes / Transformer Hawkes), read off effective latent dimension.

## Higher orders and the initial state (verified 2026-08-22)

From a stationary start the eps^2 coefficient m2 is a pure pi-functional
(iterated deviation solves: time-weighted covariance D^2 and triple
correlations); slope 2.99 verified. From a general start mu0, order eps^k
picks up exactly the mu0-average of that order's mean-zero corrector:
eps gets mu0.(-D(c LamT - lt)) (Prop 3, in the paper), eps^2 gets
mu0.(-D(Lam u1)) -- the start probed through two nested relaxation solves;
with both, point-mass starts reach slope 2.99, and both extras vanish to
4e-17 under pi. No boundary layer at any order: the observable solves a
static resolvent equation, so the expansion is regular and no matched
asymptotics is needed. Candidate for a short "higher orders" remark in the
paper alongside the earlier eps^2 verification (order-3 slope, mass
conserved at every order, subset-uniformity persists at O(N^3) tensor cost).

## Exacta-board estimation pipeline (status 2026-08-22, open)

The class-exact estimator (exp42) is exact. The FULL pipeline from sampled
exacta boards is not yet correct. Findings, all with exact (infinite-data)
inputs, so none of this is sampling noise:

1. Naive version (lam-bar := p-hat, fit K from the N deletion systems):
   design rank deficient. Spending the winner frequencies on lam-bar throws
   away the N-1 winner directions; spectrum tail ~0.3, eps-INDEPENDENT.
2. Corrected version (substitute the first-order relation
   lam-bar = p-hat - D_full K into the deletion equations): design rank
   reaches the full N^2-N-1, tail drops to ~0.08, but the fitted K-hat still
   only aligns with the projected truth at cosine 0.82 / best-scale residual
   0.58, eps-INDEPENDENT. A first-order term is missing from the linear
   model of the experiment. The Jc sign (both signs tried) is not it.
   Scale convention verified: pipeline estimates eps*K/s with s the total
   mean rate (alpha = 1.12 at the /s convention).
3. RESOLVED BY THE DECISIVE TEST (2026-08-22, late): the missing term was
   the c-shift Jacobian sign on the j-component (+1/(1-p_i), not -). With
   the correct model the linear-model test exposes the REAL finding: design
   rank drops to 24 = (N^2-N-1) - (N-1). With lam-bar as a nuisance
   estimated from the same board, the exacta board identifies only
   N^2-2N combinations of K; the rank-29 exacta-equals-interventions
   equivalence (exp41 Part D, and Section 6 of the paper) holds at KNOWN
   lam-bar. Consequences: (a) the paper's saturation claim needs the
   known-lam-bar qualifier or a trifecta remark; (b) trifecta data should
   restore exactly the missing N-1 directions, so deeper places carry
   estimation value even though they carry no design value at known
   lam-bar; (c) mode counting from a lone board reopens with the enlarged
   (N+1)+(N-1)-dim null space. NEXT: rank the trifecta-augmented design;
   redo mode counting in the enlarged class; then finite-data race counts.
   The paper edit should wait until the trifecta rank is computed, so the
   corrected statement can be positive rather than only a caveat.

3a. SUPERSEDED original next test: generate synthetic data exactly from the linear model
   (q = c + D*(epsK), p = lam + D_full*(epsK), known lam), run the pipeline.
   If K is recovered exactly, the design algebra is right and the residual
   is real second-order structure in the resolvent that the deletion
   identity linearization misses; if not, the bug is in the design rows.
   Exploratory code: experiments/exp42_mode_count/exploratory_exacta_pipeline.py

Finite-data footnote: at eps=0.05 the (still-wrong) pipeline fails at up to
1e8 races; do not quote any race-count requirement until the infinite-data
version is exact.
