# Experiment 8: ranked narrow escape — the runner-up principle on real physics

**Design (program Q3).** Same disk-and-windows geometry as experiments 1/3, but the
boundary is all-reflecting and each trajectory passively records its sequence of
distinct window encounters. The **truncation identity** makes this exact: absorption at
any set A merely truncates the reflecting path at its first A-encounter, so one
trajectory answers *every* blocked-set counterfactual, and held-out sequences are exact
ground truth for all of them simultaneously. (Cross-checked against a real absorbing
re-simulation: TV 0.012 ≈ the combined sampling noise ~0.013.)

**Result (24,000 trajectories, 12 random block sets per size, seed 42).**

| model (data regime) | singles | pairs | triples |
|---|---|---|---|
| Harville (winner only) | 0.0381 | 0.0570 | 0.0690 |
| independent Thurstone (winner only) | 0.0380 | 0.0562 | 0.0677 |
| **Markov M (winner + runner-up)** | **0.0154** | **0.0154** | **0.0160** |
| top-4 prefixes (upper bound) | 0.0154 | 0.0154 | 0.0157 |

**Findings.**

1. **Winner-only models degrade with deletion depth** and the two are
   indistinguishable from each other — as the nonidentifiability theorem requires: no
   winner-only model can know redistribution; its errors are the physics' non-IIA
   structure showing through.
2. **The runner-up identity is worth everything here**: the empirical kernel M with
   q_j = p_j + p_i·M_ij nails singleton scratches to the sampling-noise floor (~0.015
   with 12k train / 12k test trajectories).
3. **The Markov substitution resolvent composes**: q_A = p_A + p_B(I−M_BB)⁻¹M_BA,
   built from *pairs data alone*, matches the top-4-prefix upper bound on pair and
   triple blocks too. On this diffusive geometry, deeper prefixes buy nothing beyond
   the runner-up — the substitution process really is (near-)Markov, presumably
   because a rejected walker re-equilibrates locally before its next encounter.

**For Paper 2** (*The Runner-Up Principle for Counterfactual Races*): this is the
empirical half — the data hierarchy behaves exactly as the identification theory
predicts, with the bonus that the Markov composition (the transfer-resolvent object)
is empirically exact on real first-passage physics. Open refinements: geometries where
Markovianity of substitution *fails* (long-range flux correlations), and the
Green–Kubo/Green-function prediction of M from geometry alone.

Run: `python run_ranked_escape.py` (~1 min, numpy/scipy/matplotlib only).
