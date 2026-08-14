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

---

## Second review (minor revisions) — executed 2026-08-13

- Prop 1 restated in max-wins form with proof; reflection caveat for asymmetric laws; "factors need not be Gaussian" contradiction fixed.
- Lattice interval now indexed over retained nodes f_q.
- Quadrature statement corrected per experiment: exp13 uses GH order 15/11/9 (k=2/3/4); exp14 uses order 15 for all k≤4, pruned to Q=145/1317/10929 (reviewer's counts reproduced exactly).
- Prop 3: nonsingularity hypothesis added, proof added, convex social-surplus remark (G(μ)=E max, p=∇G) as motivation.
- NEW Prop 4: reviewer's O(QNL) Jacobian–vector product — verified vs central FD to 7.7e-9, implemented in raceutil.jacobian_vector_product with regression test.
- Repo consistency: exp16 created and committed (mean/TV metrics, direct utility-MC baseline at matched wall time, inversion replication ×3, recovery-vs-share figure, replication over 10 problems/size at common spread with twin references). Footnote now "experiments 13–16".
- Direct-MC baseline (the "especially important" item): at matched wall time direct simulation is within 1.3–3× of lattice max error at the 1e-3 level; abstract and benchmark section reframed accordingly — case rests on error decay, reproducibility, derivatives, inversion.
- Substitution: reviewer's precise increment reading adopted (8.2 vs 4.9 high-mass; 8.6 vs 9.3 mid-mass, family slightly larger); strata counts disclosed (1 and 4 of 24 blocks); calibration residuals 4.9e-11 / 1.1e-9 printed by script and quoted.
- Wording sweep: plain multinomial logit, roughness statistic, discrepancy-between-approximations caption, "rank-k fitted-factor approximation vs GHK" figure title, RQMC phrasing, hardware metadata (Apple M4, 16 GB, Python 3.12.9, NumPy 2.4.6, SciPy 1.18.0; single-realization timings disclosed).

Still queued (declared in paper): QMC-GHK, seed bands, minimax tilting, derivative-truth figure, D_min/D_max and L,Q resolution sweeps, skewness/tail replication for substitution, commit pin at submission.

---

## Third review — executed 2026-08-13

Every mathematical claim verified numerically before adoption (standing rule):
w_ij Laplacian identity 6.4e-9, new JVP form 4.6e-9, coercivity G>=max mu,
common-shock invariance 2.8e-17, contrast-fit example 30x, reviewer's NaN case
reproduced exactly.

1. Jacobian section rewritten: three maps separated (exact max-wins / reflected
   min-wins / normalized lattice), normalization correction (v - p 1'v)/s and
   reflection stated, fixed-design convention declared, "damped Jacobi
   quasi-Newton preconditioner" naming. TAIL BUG FIXED: forward + inversion now
   use log_ndtr (1-ndtr underflows at z~8.3); JVP rewritten in log domain with
   the reviewer's integration-by-parts form; their failing case (N=8, D
   log-spaced, GH11) is a regression test. Deep-tail shares now genuine
   positives (5.7e-19 parity).
2. Theorem 1 (weighted Laplacian + global inversion) replaces the Remark, with
   proof; novelty repositioned (Chiong-Galichon-Shum, Norets-Takahashi,
   Loaiza-Maya & Nibbering cited); JVP is now the (h_i Λ − A) form.
3. Contrast-space factor fitting: factor_model_contrast (fit P Σ P, D from the
   quotient fit — first attempt refitting D against diag(C) was WRONG and
   caught by test; only the common factor direction is irrelevant), SVD
   canonicalization. exp14 Part A rerun: rank-1 errors improve at every gamma;
   k=8 still beats GHK R=1e4 on error but GHK is ~20x cheaper at N=50 —
   stated as not-wall-time-matched with the reviewer's exact sentence.
4. Substitution replaced by the 2x2 factorial (run_factorial.py): homoskedastic
   truth, common-draw top-3 coupled deletion truths, 105 informative blocks
   (26/29/28 strata). RESULT: factor increment dominant in both families
   (+0.115/+0.099 Gumbel, +0.165/+0.170 Gaussian); family increment small
   without factors (+0.019/+0.013), larger with (+0.069/+0.084); negative
   interaction — factors and Gaussian base are complements. Figure shows
   individual observations. Abstract conclusion removed.
5. exp17: L sweep (machine precision from L~200; spectral claim now measured,
   sentence replaced), GH order sweep 5.3e-4 -> 5.9e-9, RQMC across 8 scrambles
   with distributions, per-node interval implemented AND measured (no benefit
   at tested settings — reported honestly), |1-total| = 2.4e-8, Jacobian
   symmetry 1.7e-18 / reduced PD confirmed. Full accuracy-time frontier with
   direct MC, Sobol-QMC direct, GHK at several R: lattice 4.1e-5 @ 0.03s vs
   direct MC 1.2e-4 @ 15.7s. Fig 1 left = this frontier.
6. Plumbing: mc_shares chunk from 1.5GB memory budget (7.45GiB catch fixed),
   threadpoolctl in exp17, median-of-3 sub-second timings, run_all_paper.py
   manifest, exp13/16 rerun under fixed forward (all table numbers updated),
   figure title "Fixed-design smoothness along a utility path", hyperref
   pdftitle/pdfauthor, title changed to "Deterministic Approximation and...".
7. Intro rewritten per user: leads with the locked-probit story, then an
   explicit 4-item list (forward pass / calibration / derivatives /
   counterfactuals) so both directions are unmistakable.

---

## Fourth review — executed 2026-08-14

New items beyond rounds 2-3 (much of review 4 targeted an older draft):
- Jacobian property tests added: symmetry <h,Jk>=<Jh,k>, Laplacian sign,
  normalized-map FD in BOTH conventions (min-wins + reflected max-wins).
- Theorem strengthened to bijection onto the simplex interior.
- Algorithm box (rectangle rule, pruning absorbed by normalization, min-wins
  orientation, full inversion update spec) + generator in table caption.
- exp18 TOP-TWO DELETION BASELINE (reviewer's O(RN+N^2) construction): at
  matched 64s wall time, field ensemble 4.2e-5 vs top-two MC 1.2e-4 on top-20
  deletion rows vs independent 1e8-draw reference — field ~3x more accurate;
  2.8e-17 clarified as algebraic consistency only.
- exp17 additions: QMC-GHK (4.2e-4 @ 7.0s, 5-7x better than plain GHK, still
  off the lattice frontier), GHK 8-seed band (median 7.5e-3, [3.9e-3,1.3e-2]),
  refinement.png replaces the smoothness panel (roughness stat kept as text).
- exp14 Part A REDESIGN: shared eigenbasis across gamma; actual post-
  standardization spectra disclosed; decomposition shows integration error
  flat 2-7e-4 => rank-k error IS covariance-fit error. KEY REVERSAL: with the
  shared basis, rank-8 LOSES to GHK R=1e4 at gamma=1.5 (3.7e-3 vs 2.3e-3) —
  the GHK-wins regime exists at intermediate decay; paper says so.
- exp13: warmed median-of-3 timings; two-line GHK scaling figure (R=1000 +
  lattice-matched-error via R^{-1/2} scaling) per user request.
- exp16: 99%-mass-restricted error metric added.
- exp13 README stale claims scrubbed ("noise as bias", "unbiased derivatives").
- Style per user: title "Scalable Probit Calibration", 152-word abstract,
  short paragraphs, GHK defined at first use, no self-commentary
  ("Anchors first", "Scope honestly", "measured rather than asserted" etc.),
  intro reframed on universal logit tools + Thurstone 1927 + explicit
  forward/calibration list + prior-art table + scaling figure up front.
Remaining queued: minimax tilting baseline, substitution replication across
designs/heteroskedasticity axis, style-repo guidelines (repo not found —
possibly private; user to provide).
