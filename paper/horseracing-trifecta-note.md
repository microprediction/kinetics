# Correlated races and trifecta prediction

A note for people who already know the thurstone machinery and the exotic-pricing
literature. What the correlated race (thurstone PR #14, developed and stress-tested at
kinetics.microprediction.org) actually buys you.

**Implied versus realized similarity.** You cannot get substitution structure out of
the win pool. Not "it's hard" — it's unidentified: write the scratch counterfactual as
q_j = p_j + p_i·M_ij and the win vector tells you nothing whatsoever about M. Harville
just *assumes* M is proportional, and every position-discount patch assumes some other
M; none of them learned it, because there was nothing to learn it from. But the exacta
and trifecta boards are the market quoting M directly. So: give each horse a small
loading vector — pace posture, surface/distance aptitude, sire line, or just fit the
thing freely — and calibrate the correlated race two ways. Fit to the exotic board and
you have implied correlation, same object as implied correlation in index options
against single-name vols. Fit to the charts and you have realized correlation. The
spread between those two surfaces is the trade. Two need-the-lead types the board
prices as independent are a short in every combination where both run on; the market
knowing something the charts don't shows up as the opposite wedge. None of this is
visible from win odds, by theorem.

**Trifectas in both measures, without per-position patches.** The Henery/Stern/Benter
fixes shade Harville's second- and third-slot probabilities by position — the same
shade for every horse, whoever's involved. That repairs the average and leaves the
combinations wrong, and combinations are what a trifecta is. The correlated race fixes
it structurally: each quadrature node is a race shape — meltdown, crawl, golden rail —
conditional on which the field is independent and everything is the ordinary thurstone
order-statistics calculation; integrate over shapes and you've priced the ordering with
the correlations in, not painted on. And because a Gumbel base at zero loadings *is*
Harville exactly, the fitted loadings tell you where the chain breaks and for which
pairs, not just how much on average. Calibrate jointly to win + exacta + tri and you
get one internally consistent risk-neutral surface — cross-pool relative value net of
takeout falls out mechanically. Calibrate to results and you have the P-measure; the
Q-minus-P wedge then resolves pair-by-pair instead of as one favorite–longshot curve.
And late scratches finally get the right semantics: a scratch is a marginal, not a
renormalization — renormalizing is only correct under IIA, which is precisely the
assumption we just dropped — and the deletion ensemble reprices every singleton-scratch
card from one field pass.

**What actually changed in thurstone.** Nothing happened to the engine — it's still
the field survival product with the divide-one-out trick, O(N) a pass. `FactorRace`
just runs it conditionally: performance = ability + loadings·f + your usual base noise,
f standard Gaussian in k dimensions, Gauss–Hermite nodes for small k and Sobol past 4.
Only the factors have to be Gaussian; the base densities you already use — skew-normal,
heavy-tailed, whatever — compose with the correlation untouched. The nesting is exact
both ways: zero loadings reproduce `Race.state_prices`, and the new `Density.gumbel_min`
reproduces Luce in closed form, so correlated-Gumbel is a strict superset of the model
the exotics literature has been patching for forty years. `solve_abilities` inverts
prices to abilities *under* the correlation rather than inverting independent and
adjusting after — damped fixed point, with the step set by the smallest effective
pairwise noise, because a naive step diverges silently (that one cost us a result
before we caught it). One more trap worth knowing: `factor_model` is iterated factor
analysis, not eigen-truncation, because truncation invents off-diagonal correlation out
of thin air. Forward, inverse, and the all-scratches ensemble are deterministic and
smooth in the abilities, which is what a calibration loop in production wants and Monte
Carlo pricing never gives you.
