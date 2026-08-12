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

**Interpretation (sharpened after review).** The deeper reading is an identifiability
fact, not a modeling gap: full-menu winner frequencies are an N-vector and *cannot*
contain the N×N substitution structure — after scratching i, the redistribution
q_j^(−i) = p_j + p_i·M_ij involves a runner-up kernel M that winner-only data leaves
completely unconstrained (program Q3, the runner-up principle). So the result here is
that **winner-only observation of this physics does not reveal its geometric
substitution structure**; no model could have recovered it from p alone. Experiment 3
supplies the structure from geometry and calibrates it on a held-out *intervention* —
exactly the extra data the identifiability hierarchy demands. A stronger follow-up is
*ranked narrow escape*: record the sequence of window encounters of a reflected
trajectory, so one trajectory answers every blocked-set counterfactual.

Run: `python run_narrow_escape.py` (~40 s, numpy/scipy/matplotlib only).
