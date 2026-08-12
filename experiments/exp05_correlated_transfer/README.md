# Experiment 5: the correlated race does NOT fix temperature transfer (negative result)

**Question.** Experiment 4 left both independent surrogates with the same residual
transfer error (TV ~0.034 at kT 1.0 → 0.55), which we attributed to angular mixing
between adjacent channels. Experiment 3 showed a geometry-informed correlated race
repairs exactly that kind of error for blocked-window counterfactuals. Does it repair
the transfer residual too?

**Result (seed 3): no.** Sweeping the correlation length ℓ from 0.01 (independent) to
1.6 rad changes the transfer error by less than sampling noise — TV stays in
0.0338–0.0350 at kT 0.55 for every ℓ, indistinguishable from Harville (0.0342) and
independent Thurstone (0.0343). Calibrating ℓ at kT 0.7 selects a value that buys
nothing at kT 0.55. (Sanity check: the ℓ → 0 column reproduces experiment 4's numbers
exactly.)

**Why this is informative.** Correlation moves win probabilities strongly under
*structural* counterfactuals — deleting a competitor breaks the field asymmetrically,
and its correlated partners inherit its wins (experiment 3's ~9×). Under a *global*
parameter shift, by contrast, every ability scales together and the correlation's
effect on the marginal win probabilities largely cancels. The exp04 residual is
therefore probably not correlational after all: the better suspects are the transfer
law itself — temperature-dependent Kramers prefactors, and effective barrier heights
that shift with kT because the particle samples a smoothed angular barrier profile.
Claims in experiment 4's write-up have been revised accordingly.

**Moral for the program.** The correlated race (Q6) is a tool for deletion/blocking
counterfactuals, not a universal fix; transfer counterfactuals stress the *ability
scaling law* instead. These are orthogonal failure axes, and a surrogate can be good at
one and not the other.

Run: `python run_correlated_transfer.py` (~3 min, numpy/scipy/matplotlib only).
