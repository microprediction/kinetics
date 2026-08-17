# Paper 3 program: Factor-Gaussian Transport on the Simplex

Notes from 2026-08-16 (user-supplied analysis; all identities verified in
experiment 27 part 3 before recording). Working title candidates:
"Factor-Gaussian Transport on the Simplex" / "Probit Mirror Maps and
Laplacian Newton Oracles".

## The central reframing

Factor-probit share calibration IS structured semi-discrete quadratic
optimal transport: on the contrast space, b_i = e_i - 1/N are regular
simplex vertices; argmax(x_i + mu_i) = argmin(0.5||x - b_i||^2 - mu_i)
(verified exactly), so choice cells are Laguerre cells; Kantorovich
duality gives Omega(q) = 0.5 W_2^2(contrast Gaussian, nu_q) + const.
The transform is a near-linear mass / gradient / Hessian-vector /
inverse-mirror-map ORACLE for this transport problem:

  value    G(mu)      : same-field reduction, O(QL) extra
  gradient p = grad G : one pass, O(QN(k+L))
  HVP      J h        : one pass (matrix-free facet Hessian -- no
                        geometric facet enumeration)
  inverse  mirror map : calibrated utilities

## Verified identities (exp27 --part3)

- Stein per-class: E[xi_i 1{I=i}] = (Sigma J)_ii within MC noise.
- G = mu'p + tr(Sigma J) to 3e-9.
- Omega(p) = -tr(Sigma J) = -sum_{i<j} w_ij s_ij^2,
  s_ij^2 = D_i + D_j + ||v_i - v_j||^2 to 3e-9.
  "Generalized entropy = negative variance-weighted total conductance."
- Binary: Omega(q) = -s phi(Phi^-1(q)) to 3e-13.
- w_ij <= 1/(sqrt(2 pi) s_ij) (part 2) => lambda_max(J) <= L explicit =>
  G globally L-smooth on quotient => Omega (1/L)-strongly convex.

## Theorems to write

1. The transport equivalence theorem (Laguerre cells, facet measure
   w_ij = (1/sqrt2) * Gaussian H^{N-2} mass of facet, Hessian =
   transport dual Hessian).
2. Global convergence of Armijo-damped Newton for the continuum
   calibration: H_q coercive on quotient (companion Prop), Hessian PD
   everywhere, compact sublevel sets, eigenvalue bounded below +
   Lipschitz Hessian on them => global + local quadratic. (Semi-discrete
   OT literature has analogues -- Kitagawa-Merigot-Thibert -- but their
   compact-source assumptions don't import verbatim; Gaussian source
   needs its own short proof.)
3. Newton-CG = electrical solve: residuals are current injections,
   corrections are voltages, w_ij conductances; matrix-free via JVP.
   Jacobi (companion Prop on normalized Laplacian) is the degree-only
   preconditioner; CG uses the whole network.
4. Mirror map / prox: p(mu) = argmax{mu'q - Omega(q)} gives a
   covariance-aware nonseparable simplex prox computable at N ~ 1e4;
   mirror-descent step q_{t+1} = p(mu_t - eta g_t) is a forward
   transform; FTPL/FTRL equivalence with an explicit strong-convexity
   constant 1/L.
5. Fenchel-Young: population FY risk = H_q exactly (share calibration =
   population FY risk minimization); links to Lin-Yin-Liu framework
   which needs exactly this oracle in the Gaussian case.

## What is known vs new

Known (frame as foundations): p = grad G, Omega = G*, assignment/OT form
of G* (Chiong-Galichon-Shum), FTPL/FTRL duality, semi-discrete OT
damped-Newton theory (compact sources), FY estimation (Lin-Yin-Liu).
Potentially new: the regular-simplex quadratic specialization with
factor-Gaussian source; O(QNL) all-cell mass oracle (vs one integral
per Laguerre cell); matrix-free transport-Hessian JVP without facet
enumeration; scalable inverse mirror map at N = 1e3-1e4; the
Omega = -tr(Sigma J) conductance identity (search literature before
claiming); explicit global smoothness/strong-convexity constants.

## Immediate implementables

- Newton-CG calibration variant (JVP-CG on B'JB d = -(p - q), Armijo):
  the reviewer-suggested robustness fallback for the JCGS paper AND the
  demonstration engine for paper 3.
- Omega evaluation at cost k JVPs + diagonal slopes (tr(Sigma J) =
  sum_r v_r' J v_r + sum_i D_i J_ii).
- Mirror-descent demo on a simplex-constrained problem with correlated
  costs vs entropic MD.

## Status in the JCGS paper

Compact "Transport form" remark added after the conjugate remark
(2026-08-16) with the verified conductance identity and binary check;
everything else deferred to this note.


## Referee round (2026-08-16) intersecting this program

A JCGS-style review of Paper 1 independently demanded the OT framing and
supplied the citation spine, now added to Paper 1's transport paragraph:
- Aurenhammer, Hoffmann, Aronov (1998): power-diagram formulation.
- Levy (2015): convex weight optimization.
- Kitagawa, Merigot, Thibert (2019): damped Newton, Laplacian Hessian
  with facet-integral weights, global convergence under compact-support
  regularity (does NOT import verbatim for an unbounded Gaussian source;
  a truncation/extension argument is theorem 2's opening move).
- Taskesen, Shafieezadeh-Abadeh, Kuhn (2023): discrete-choice smoothing
  as semi-discrete OT.
Paper 1 now claims only the oracle novelty (all cell masses + HVP in
O(QN(k+L)) for a factor-Gaussian source on simplex sites); the full
optimization story stays here for Paper 3. The reviewer also flagged
that grad Omega = calibration solve and Hess Omega = reduced inverse --
Paper 1's conjugate complexity claim was corrected accordingly; Paper 3
should state the oracle table (value: k JVPs + diagonal; gradient:
calibration; HVP of Omega: reduced solve) from the start.
