# Experiment 12: the deletion ensemble on a real engineering problem

**The direct connection.** N−1/N−2 transmission contingency analysis is a federally
mandated engineering workflow (NERC TPL standards): operators must continuously verify
the grid survives every credible outage. The industry's screening tool — line outage
distribution factors, generalized to multi-outages (MLODF) — **is the leave-k-out
Schur/Woodbury identity** on one inverse of the DC susceptance matrix:

> f′ = f₀ + PTDF[:,S] (I − PTDF[S,S])⁻¹ f₀[S],  islanding ⇔ the k×k block is singular.

This is not an analogy; it is the same equation, running in every energy-management
system. What this experiment adds is the exp11 question asked on real data: **how well
does the cheap linear deletion ensemble screen the expensive nonlinear truth** — full
AC Newton power flow — and where exactly does it fail?

**Data and method.** Alsac & Stott's 1974 30-bus system (the classic security-analysis
benchmark, all lines MVA-rated) and IEEE 118-bus, parsed directly from the public
MATPOWER files. DC layer: one inverse → all outages by MLODF. AC layer: a polar
Newton–Raphson written here (residual-validated; base case converges with max mismatch
< 1e-9), run on **every feasible N−2 pair** — full ground truth, no shortlist.

**Results (case30: 820 N−2 pairs; results.csv).**

- **Identity layer**: MLODF vs direct DC re-solve, max diff **2.2e-15**; all 143
  islanding pairs detected exactly via singularity of the 2×2 block.
- **Screening layer** (top-20 recall of the true worst AC loadings @ budget 40):
  - DC + base-Q severity (standard practice): **0.95**, at 22× less compute.
  - DC active-power-only severity: **0.20** — *reactive blindness*, not linearization
    error (base-flow DC-vs-AC correlation is 0.954). The classic practitioner trap,
    measured: MW screening against MVA ratings misses reactive-driven overloads;
    carrying base-case Q recovers the screen.
  - Spearman is only 0.60 — but it's a tie artifact: one line is at 109% loading in
    the base case and caps the severity of 633/677 pairs. Top-K recall is the
    operative metric for this workflow, not global rank correlation.
- **Blind spots, honestly**: zero AC non-convergences here, but voltage collapse is
  structurally invisible to any DC screen; islanding, by contrast, is caught exactly.
- **Scale (case118)**: all **17,205 N−2 pairs from one inverse in 0.2 s** (1,703
  islanding pairs), exact to 3e-14. N−3 on the same machinery is a 3×3 block per
  triple.

**What is and isn't a contribution.** The identity is 50-year-old industry practice —
which *validates* the project thesis (one global solve encodes the deletion ensemble;
engineers already bet the grid on it daily) rather than being novel. The transferable
contribution is the screening-quality methodology: exactness/screening/blind-spot
layers with recall-at-budget curves (exp11's harmonic-vs-relaxed, here DC-vs-AC), and
the measured warning that the *severity metric*, not the linear physics, is where
naive screening fails.

Tests: `tests/test_grid_contingency.py`.

Run: `python run_grid_contingency.py` (~30 s, numpy/scipy/matplotlib only).
