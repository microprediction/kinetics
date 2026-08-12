# Experiment 10: the substitution kernel, predicted from pure geometry

**Question (the Paper 1 ↔ Paper 2 bridge).** Experiment 8 showed that the runner-up
kernel M, *measured* from trajectories, answers blocked-set counterfactuals through
the Markov substitution resolvent. Can M — and the winner distribution p itself — be
**predicted with no trajectory data**, from Green-function hitting splits of the disk
generator validated in experiment 9?

**Method.** Disjoint window geometry (by construction — experiment 9 exposed the
overlap trap in the old seed-42 geometry). Sparse finite-volume Neumann generator
(invariants asserted); p_geo = hitting splits from the center; M_geo[i,·] = hitting
splits of the *other* windows from window i's boundary ring. One second of sparse
solves, total. Evaluated against 24k reflecting-trajectory encounter sequences
(12k train / 12k held-out truth).

**Results (mean TV over 12 random blocks per size).**

| model (data used) | singles | pairs | triples |
|---|---|---|---|
| Harville (trajectory p only) | 0.0369 | 0.0524 | 0.0705 |
| empirical M, first transitions (traj. p + M) | 0.0230 | 0.0207 | 0.0208 |
| empirical M, all pairs (traj. p + M) | 0.0240 | 0.0258 | 0.0298 |
| **hybrid: trajectory p + geometric M** | **0.0221** | **0.0201** | **0.0184** |
| **pure geometry (p_geo + M_geo, zero trajectories)** | 0.0322 | 0.0298 | 0.0276 |

**Findings.**

1. **The geometric kernel is the best substitution kernel available** — the hybrid
   beats both empirical variants at every depth. Green-function solves are noise-free
   and structurally consistent; empirical kernels carry sampling noise and
   start-profile mismatch.
2. **Pure geometry — zero trajectory data — beats winner-only trajectory models on
   pairs and triples.** Counterfactual structure really is a geometric object here.
3. **Two kernels, not one** (a Paper 2 point): the stationary encounter chain differs
   from the first-transition kernel by up to 0.21, and the counterfactual wants the
   latter — using all consecutive pairs *degrades* multi-block prediction.
4. **Where geometry and data disagree** (|M_geo − M_emp| up to 0.27) is entirely on
   **adjacent-window entries** (non-adjacent: 0.037, exactly the start-sensitivity
   bound). Adjacent transitions depend on the encounter-position profile *within* the
   window, which the π-weighted ring start cannot represent. This is the identified
   next-order refinement: solve for the encounter position profile (harmonic measure
   on the window) rather than assuming uniformity.

Tests: `tests/test_green_function_M.py` (disjointness, generator invariants, splits
sum to one and order by window size, symmetric-window 50/50).

Run: `python run_green_function_M.py` (~1 min, numpy/scipy/matplotlib only).
