# Fast-kernel research notes: the applied-math territory under the transform

*2026-08-14. Prompted by the observation that the calibration lookup trick
"puts us in familiar applied math territory." It does — three of them — and
one contains a measured, potentially order-of-magnitude algorithmic win.*

## The finding worth acting on: the pass is numerically low-rank

Both stages of the per-node O(NL) pass are kernel sums:

- **Field build**: logS_field(x) = Σ_i k(x; m_i, σ_i), kernel k = log Φ(−(x−m)/σ).
- **Distribute**: p_i = Σ_t h(x_t; m_i, σ_i) · w_t · Δx, kernel h = g/S (hazard),
  weights w_t = exp(field).

Both kernels are globally smooth in (x; m, σ). Measured on the exp13 N=1000
geometry (matrix 1000×1501, relative to top singular value):

| kernel matrix | rank @1e-6 | @1e-10 | @1e-13 |
|---|---|---|---|
| logS (field build) | 13 | 30 | 43 |
| hazard g/S (distribute) | 18 | 37 | 51 |
| density g | 28 | 41 | 50 |

A separated (Chebyshev/bbFMM-style) representation k(x; m, σ) ≈
Σ_c T_c(m, σ) k_c(x) with r ≈ 15–40 terms turns each stage into
O(r(N+L)) instead of O(NL):

- N=1000, L=1501, r=40: ~15× fewer operations per node.
- N=5000: ~29×. Stacks multiplicatively with the Rust kernel
  (fusion + SIMD + rayon over nodes) and applies to forward, slopes, and JVP.

Because the kernels are smooth *everywhere* (no near-field singularity), no
tree is needed — this is the degenerate, easy case of an FMM: one global
low-rank expansion. Caveats to validate before claiming: per-entry (not just
spectral) accuracy of the hazard expansion in the deep right tail, where the
hazard grows like z; and σ-range coverage when D is very heterogeneous.

## The three literatures

1. **Fast Gauss transform** (Greengard–Strain): sums of N Gaussians evaluated
   at M points in O(N+M); the improved FGT reduces the dimensional constant.
   Our density-side sums are exactly its object; the survival/hazard kernels
   are not Gaussian, which leads to:
2. **Kernel-independent / black-box FMM** (Fong–Darve bbFMM; accelerated 1D
   variants): Chebyshev interpolation of any smooth kernel, low-rank
   translation operators, SVD compression. The right vocabulary for the
   low-rank pass above; in our globally-smooth case it degenerates to a
   single expansion, i.e. trivially implementable (no tree, no M2L).
3. **Reduced basis methods / offline–online decomposition** (certified RBM,
   Patera–Quarteroni school): expensive offline surrogate build, cheap
   certified online queries in many-query settings — precisely the shape of
   calibration (many forward evaluations at moving parameters). The neural
   amortization line (Huch–Keane 2026) is the learned version of this; a
   Chebyshev-separated transform would be the *certified quadrature* version,
   with error bounds instead of training.

Also adjacent: the discrete order-statistics DP literature (Leemis et al.;
graph-DP variants) computes order-statistic distributions by bin-tracking
recursions — same family of objects, different (non-lattice) recursion.

## Status of the calibration lookup trick (winning's inversion carry-over)

Prototyped 2026-08-14 (frozen-field FFT cross-correlation curves, aggregated
over nodes by the same-shift observation; code in git history at the
"interpolation-inversion prototype" commit). **Negative result in NumPy**: at
N=300 it converged (forward-match 1.1e-5) but ran 23s vs 18s standard,
because the curve build costs about as much as the ~11 re-integrations it
avoids. The economics flip once the pass itself is cheap (low-rank and/or
Rust): revisit then. Removed from raceutil to keep the library honest.

## Validation (exp20, 2026-08-14): CONFIRMED

Prototype in `experiments/exp20_separated_pass/`. Exponential convergence;
speedup grows with N. At N=5000: 45x at 6.1e-5, 29x at 6.3e-7, 22x at
9.1e-9 (NumPy vs NumPy exact). The r needed is larger than the single-node
rank (the field sums N interpolation errors before exponentiating), but the
economics are decisive anyway.

## Suggested order of work

1. ~~Validate the separated expansion end-to-end at k=2~~ DONE (exp20).
2. Wire it into the Rust kernel (fastrace) — the r(N+L) inner loop is even
   more SIMD/cache-friendly than the fused O(NL) loop.
3. Re-run the calibration lookup on top (its economics become favorable).
4. If it holds, the paper's complexity claim improves from O(QNL) to
   O(Qr(N+L)) with measured r — a second-paper-sized result, or a strong
   revision item.

## Sources

- [Greengard & Strain, The Fast Gauss Transform](https://www.academia.edu/155540/The_Fast_Gauss_Transform_pdf_1_372_Kb_)
- [Yang, Duraiswami et al., Improved Fast Gauss Transform](http://users.umiacs.umd.edu/~ramanid/pubs/siam_fgt.pdf)
- [Fong & Darve, The black-box fast multipole method](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf)
- [Accelerated kernel-independent FMM in one dimension](https://www.researchgate.net/publication/228575392_An_Accelerated_Kernel-Independent_Fast_Multipole_Method_in_One_Dimension)
- [Certified Reduced Basis Methods for Parametrized PDEs (Hesthaven-Rozza-Stamm)](http://imm.dtu.dk/~pcha/JSH/CRBM.pdf)
- [Optimized M2L kernels for Chebyshev-interpolation FMM](https://arxiv.org/pdf/1210.7292)
- [Leemis et al., Distributions of order statistics for discrete RVs](https://www.math.wm.edu/~leemis/2005informsjoc.pdf)
- [DP for joint distributions of order statistics](https://arxiv.org/pdf/2111.10939)
