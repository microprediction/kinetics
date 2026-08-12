# Correlated races and trifecta prediction

How the correlated race (thurstone PR #14, developed at kinetics.microprediction.org)
helps horseracing prediction: implied and actual similarity between horses, factored
into trifecta prediction in both real and risk-neutral measures.

**Similarity, implied and actual.** Two horses are similar when their performances
co-move — same sire line, same running style, the same vulnerability to a fast early
pace or a rain-softened track. In the correlated race this is not a metaphor but a
parameter: each horse carries a vector of factor loadings, and similarity is the
alignment of those vectors. The crucial point, made precise by the runner-up principle,
is that win odds alone can never reveal it — an N-vector of win probabilities contains
no information about how probability redistributes when the field changes. But racing
supplies exactly the richer data the theory demands: exacta and trifecta pools are, in
effect, the market quoting its runner-up kernel, and historical finishing orders are
nature quoting hers. Fitting the loadings to exotic prices recovers *implied*
similarity — which horses the market believes are running the same race within the race
— while fitting them to observed top-k orderings recovers *actual* similarity. The gap
between the two fits is a map of structural mispricing: pairs the market treats as
independent that in fact share an engine, and vice versa.

**Trifecta prediction in both measures.** The industry-standard route to exotics is
Harville's formula — win probabilities chained by proportional renormalization — which
is IIA in sequence and systematically mishandles precisely the correlated cases: if the
favorite and second favorite are closers waiting on the same pace collapse, the trifecta
combinations where both run well (or both fail) are mispriced by any independence-based
chain. The correlated race prices the full ordering directly: conditional on the latent
factors the horses are independent, so P(i→j→k) is an ordinary order-statistics
computation at each quadrature node, integrated over factor scenarios — each node is
literally a "race shape" (pace scenario, track bias) under which the field is re-priced.
Calibrated to market prices this gives a *risk-neutral* exotic pricer that is
arbitrage-consistent across pools, so win-pool odds plus fitted correlations imply fair
trifecta prices to compare against the actual trifecta pool; calibrated to historical
results it gives *real-measure* probabilities. The wedge between the two measures — the
favorite–longshot bias and its lesser-known exotic cousins — then resolves horse-pair by
horse-pair rather than as one aggregate curve, and value sits exactly where the two
similarity structures disagree. As a practical bonus, the one-pass deletion ensemble
re-prices the entire card for every possible late scratching simultaneously, and
correctly: a scratch is a marginal, and the model treats it as one.

**How this generalizes the thurstone library.** The package's engine has always been
the fast ability transform: build the field's survival product once, divide each runner
back out, and price all N in O(N) — but only for *independent* fields. The new
`FactorRace` keeps that engine intact and wraps it in a latent-factor conditioning:
performance = ability + loadings·factors + idiosyncratic noise, where conditional on
the factors the field is independent again, so the same field-product identity runs
unchanged at each Gauss–Hermite or Sobol node. The generalization is strict and nests
the classical models exactly: zero loadings recover the package's `Race`; a Gumbel-min
base recovers Luce/Harville (softmax) in closed form, so nonzero loadings on a Gumbel
base give a *correlated softmax race* — a non-IIA generalization of the very model
Harville assumed; and any thurstone base density (skew-normal, heavy-tailed) carries
over untouched, because only the factors need to be Gaussian. Around the forward map
sit the pieces a practitioner needs: `factor_model` to compress any target correlation
matrix into loadings, `solve_abilities` to invert market prices into abilities *under*
correlation rather than pretending independence and correcting afterwards, and
`deletion_ensemble` for the scratch counterfactuals — all deterministic, all smooth,
all validated against Monte Carlo and against real first-passage physics before a
single horse was involved.
