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
- MODE COUNTING (empirical, one open anomaly): the driver's mode count r is
  NOT the rank of the projected departure (projection smears the spectrum;
  measured rank 7 for all r). The right object is minimum rank over the
  equivalence class K + d lam^T + t diag(lam). A class-constrained
  alternating low-rank fit flags r in every trial via "first machine-zero
  residual at rho = r+1"; the rank-r member (the true K) is in the search
  space yet is never found even with 12 restarts. Understand why before
  claiming anything: possibly the rank-r member is an isolated point of the
  feasible set while rank-(r+1) members form an attracting manifold.
  Scripts in the scratchpad; promote to exp42 when resolved.
- Application sketch: fit the exacta board of a trained neural TPP (RMTPP /
  Neural Hawkes / Transformer Hawkes), read off effective latent dimension.
