# Referee review of the factor-probit paper (2026-08-13) — distilled actions

Verdict: major revision; core contribution (conditional all-shares transform) survives.
Full review received from the user; this file records the action list and status.

## Accepted corrections (all valid)

1. **GHK/derivatives argument wrong as framed.** CRN-GHK is a smooth deterministic
   function conditional on draws; analytic GHK scores have existed for decades;
   "bias" is the wrong diagnosis for a fixed draw set; and our "analytic slope" is
   not the exact derivative of the returned map (undifferentiated adaptive
   endpoints, undifferentiated normalization T, floors/clipping → piecewise
   smooth). Correct claim: reproducible differentiable APPROXIMATION whose
   derivatives need no resampling; accuracy depends on L and Q. Figure needs an
   independent truth curve, matched budgets, multiple GHK seeds. STATUS: paper
   rewritten; upgraded figure queued (needs new experiment run).
2. **Inversion math.** Translation invariance ⟹ singular Jacobian; implicit-diff
   formula must use a reduced basis B spanning 1⊥: dθη = −(BᵀJB)⁻¹Bᵀp_θ. The
   6.3e-9 residual = solving our own equations (target floored/renormalized MC),
   NOT statistical accuracy — meaningful stat is utility recovery + identified
   count + independent evaluation + replication. Remove "intrinsically
   well-conditioned regardless of cond(Σ)". Rename: Jacobi-style diagonal-Newton,
   convergence empirical. Fix sign/notation (a = −μ). STATUS: rewritten.
3. **Substitution variance confound.** Truth skew-normal not standardized
   (mean δ√(2/π), var ≈ 0.427); Gumbel candidate variance π²/6·D. Fixed:
   all bases standardized to mean 0, var 1; rerun. RESULT: ordering SURVIVED and
   sharpened (mass>10%: IIA 15.9% / mixed 7.7% / probit 2.8% misallocated;
   2–10%: 25.7 / 17.1 / 7.8). Single calibration caveat retained; multi-seed
   replication queued. Also: prose favorite/mid-mass numbers had come from
   uncommitted diagnostics — mass-stratified reporting now IN the committed
   script (own footnote violated; fixed).
4. **Benchmark inconsistencies.** Truth = 2e6 draws only through N=200 (5e5 at
   1000/5000) — table footnote corrected. Abstract "flat 3e-4" vs 8-9e-4 —
   replaced with "below 1e-3 throughout; reference noise comparable at large N".
   Max-coordinate metric growth artifact — metric now defined; mean/TV metrics
   queued as an exp13 addendum. 12h explicitly labeled extrapolated. Baselines:
   direct utility MC (full-vector), QMC-GHK, seed bands, minimax tilting mention
   — QUEUED (exp13 addendum). Hardware/versions statement added.
5. **Boundary study.** "factor floor" → "rank-k approximation error for the
   stated fitting/integration procedure"; k>4 = fixed-seed scrambled Sobol
   (RQMC), so not "no simulation anywhere" — disclosed; "cheaper" unmeasured
   there — removed; single-realization caveat added; D_min/D_max, μ-span, L, Q
   sweeps QUEUED.
6. **Deletion ensemble complexity** is O(QN²L) with O(N²) output; "one
   conditional field construction" kept, "one pass" qualified; validation
   wording fixed (10 sampled recomputation checks; timing extrapolated).
7. Expository: numerical parameters stated (L=1501, GH(15) pruned 1e-7, QMC 2^13,
   normalization, floors); D_i > 0; non-Gaussian factors remark (opportunity);
   Gumbel/Luce equal-scale qualifier; identification paragraph (scale, rotation,
   differences-only); propositions (identity/complexity/identification) added
   compactly; related work: Butler–Moffitt 1982, Elrod–Keane 1995, Botev 2017
   minimax tilting; bib fixes (arXiv 2106.04636 = Andrew Chia; amortized = Huch
   & Keane); tone-down sweep ("cannot have", "unusable", "freezing noise in as
   bias", "needs no trade", "honest boundary", "prices" → "computes", etc.);
   precise open claim wording; pin immutable commit at submission.

## Related exp15 status (post-review, same discipline)

Price identity dp_i/dΣ_jk = triple-tie density VERIFIED (7.9e-6) with exact sum
rule; winner-term negativity REFUTED (+1.2e-2); the bound |dp|≤T_jk REFUTED
(ratio ≤1.27 observed); the certificate Σ|ΔΣ|·T held empirically in 100% of
tests (6–30× conservative) — status: practical conservative estimate, not a
theorem. Numerical trap fixed en route: 1−ndtr(z) underflows at z≳8; use
log_ndtr(−z) (hazards were exploding to 1e150).

## Queued experiment work (before submission)

- exp13 addendum: direct-utility-MC baseline, QMC-GHK, GHK seed bands, mean/TV
  metrics, replicated large-N references, hardware statement, derivative-truth
  figure.
- exp14 addendum: multi-seed/multi-design replication; D_min/D_max, μ-span, L, Q
  sweeps; several scrambles at k>4; timing of rank-8 vs GHK R=1e4.
- Multi-seed replication of the substitution study across skew/tail designs.
