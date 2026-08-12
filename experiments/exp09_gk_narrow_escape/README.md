# Experiment 9: the Green–Kubo theorem on continuous physics

**Setup (Q7 × the physics).** The hidden state is reflected Brownian motion in the
unit disk; the channels are Robin-style reaction windows — boundary bands in which
channel i fires at rate κ. First fire wins. κ is the time-scale separation (slow
reaction = fast mixing). Three layers separate every error source:

- **A. Real simulation** — Brownian dynamics with per-step independent firing.
- **B. Exact on the discretized generator** — finite-volume Neumann Laplacian on a
  polar grid (uniform invariant law, reversibility, and conservation *asserted*, not
  assumed), killed-resolvent solve per κ.
- **C. Theory** — band-area softmax + κ·(Green–Kubo coefficient), from one deviation
  solve, using the exact same code that passed exp07's chain tests.

**Results (seed-42 window geometry, n = 3520 cells, 50k trajectories/κ).**

- **B vs C (the theorem)**: softmax error slope **0.93** (theory 1); after the
  Green–Kubo correction slope **1.93** (theory 2). The homogenization formula holds
  on a continuous-physics generator, not just an abstract chain.
- **A vs B (consistency)**: max discrepancy 2.2e-3 / 1.6e-3 at κ = 5 / 20 with 50k
  trajectories (noise ~1e-3), and **9.6e-4 ≈ the noise floor** in a 200k-trajectory
  check (noise 7e-4).

**Two systematics found and fixed on the way** — the layer separation is what caught
them, and both would have silently corrupted a naive sim-vs-theory comparison:

1. **Band quantization** (5e-2!): sharp 0/1 cell indicators mis-size windows narrower
   than a grid cell by up to ~3×. Fixed with fractional (area-exact) cell coverage —
   band areas now match analytically to 1e-14.
2. **Overlap semantics** (1.4e-2): the window geometry contains three *overlapping*
   pairs, and the original simulation awarded an overlap to the lower-indexed window
   (argmax), while the model stacks intensities (λ = κΣg_i — independent receptors,
   the physically right convention). Diagnosed by the error's structure (−0.014 on
   window 9, +0.006 on its overlap partner 8), confirmed by its flatness in both dt
   and grid refinement, fixed by firing each covering window independently.

**For Paper 1**: this completes the numerical verification chain — exact finite
chains (exp07) → discretized continuous generator (here, B vs C) → real Brownian
dynamics (here, A vs B) — and stands as the worked example for the narrow-escape
section. Remaining for the paper: the Fourier-mode analytics of the disk's boundary
operator, and connecting the GK kernel to exp08's empirically-Markov substitution
kernel M.

Tests: `tests/test_gk_grid.py` (generator invariants, K symmetry, both convergence
orders, softmax limit = band-area share).

Run: `python run_gk_narrow_escape.py` (~3 min, numpy/matplotlib only).
