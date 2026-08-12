# Experiment 1: kinetic surrogates on a real first-passage simulation

**Physics.** Overdamped Brownian dynamics in the unit disk with 16 absorbing boundary
windows (the classical narrow-escape geometry) — 30,000 simulated trajectories, winner
recorded per trajectory. Counterfactual: the 3 busiest windows are blocked (made
reflecting) and the surviving win distribution is predicted, with ground truth from
60,000 re-simulated trajectories.

**Finding (seed 42).** The physics is measurably non-IIA: TV(truth, proportional
renormalization) = **0.082**. Blocking a window redistributes flux preferentially to its
*angular neighbors* (windows within 0.6 rad of a blocked window gain far more than
proportionally — see `figures/redistribution.png`). But the independent Gaussian
Thurstone race does **not** capture this either: both surrogates converge to essentially
the same error floor (Harville 0.081, Thurstone 0.085 at R = 30,000; `results.csv`).

**Interpretation.** The IIA violation in diffusive kinetics is *geometric* — it lives in
the correlation structure of the race (walkers rejected by one window are absorbed
nearby), not in the ability gaps that an independent race can bend. This is direct
motivation for program question Q6: a fast ability transform for **correlated** fields,
combining the multiplicative cavity (extremal structure) with the Schur/rank-one cavity
(coupling).

Run: `python run_narrow_escape.py` (~40 s, numpy/scipy/matplotlib only).
