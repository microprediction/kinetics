# Experiment 4: temperature transfer on a real barrier-crossing simulation

**Physics.** Overdamped Langevin dynamics in a 2D well enclosed by a barrier ring with
12 channels of quenched-random heights E_i ∈ [2, 3.5] kT₁ — a minimal model of competing
activated reaction pathways. Escape-channel frequencies are observed at kT₁ = 1 (25k
trajectories); surrogates predict the channel distribution at colder temperatures, with
ground truth re-simulated (40k trajectories each).

**Transfer laws compared.** Harville/Arrhenius (independent exponential clocks:
p_i^(kT₁/kT₂), renormalized) vs Thurstone/Arrhenius (performance = barrier + thermal
noise: abilities rescale by kT₁/kT₂, Gaussian race re-priced). Baseline: no transfer.

**Result (seed 3).**

| kT₁ → kT₂ | no transfer | Harville/Arrhenius | Thurstone/Arrhenius |
|---|---|---|---|
| 1.0 → 0.7 | 0.0342 | 0.0186 | 0.0186 |
| 1.0 → 0.55 | 0.0659 | 0.0342 | 0.0341 |

**Interpretation.** Arrhenius rescaling is what matters: both surrogates cut the
transfer error roughly in half, at both extrapolation depths. The noise law (Gumbel-like
vs Gaussian) is second-order — indistinguishable even at the deep step. The residual
error (0.034 at kT = 0.55, ~4× the sampling noise) was initially attributed to
channel-mixing correlation, but **experiment 5 tested that and found it false**: a
geometry-informed correlated race leaves the transfer error unchanged. The remaining
suspects are the transfer law itself — temperature-dependent Kramers prefactors, and
effective barrier heights that shift with kT as the particle samples the smoothed
angular barrier profile.

Run: `python run_kramers_transfer.py` (~90 s, numpy/scipy/matplotlib only).
