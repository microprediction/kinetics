# Neural / computer-vision program for the factor-probit transform

Notes from 2026-08-15 (user-supplied analysis, anchors verified against
primary sources). Status: two citations folded into the calibration paper
(Collier et al. 2021 in the intro and Related Work, Berthet et al. 2020 in
Related Work); everything below is companion-paper material.

## The smoking gun: the model already ships in vision

Collier, Mustafa, Kokiopoulou, Jenatton, Berent (CVPR 2021),
"Correlated Input-Dependent Label Noise in Large-Scale Image
Classification" (arXiv:2105.10305). Their output layer is literally our
model: latent utilities u(x) = mu(x) + V(x) f + d(x) eps with
low-rank-plus-diagonal covariance VV' + diag(d^2), observed class =
argmax. They state the winner probability has no closed form, relax the
argmax with a temperature softmax, and Monte Carlo it (up to ~10^4
samples). Deployed on ImageNet-1k (N = 1000), ImageNet-21k, WebVision,
JFT; covariance rank ~15 on ImageNet, ~50 on larger sets. Learned
covariance picks out semantically confusable / co-occurring classes.

Same problem, unsolved: their computational obstacle is exactly the map
the calibration paper computes deterministically.

Caveat to respect: their useful ranks are 15-50. Product Gauss-Hermite is
infeasible there; we would run QMC factor nodes. Whether we beat their MC
at their ranks is an EXPERIMENT, not a claim.

## The right ML frame: perturbed optimizers

Berthet, Blondel, Teboul, Cuturi, Vert, Bach (NeurIPS 2020), "Learning
with Differentiable Perturbed Optimizers": G(mu) = E max_i (mu_i + xi_i),
p = grad G = E[e_argmax], evaluated by Monte Carlo; Gumbel noise gives
log-sum-exp/softmax, general noise gives other regularizers via the
Fenchel conjugate Omega = G*. Our transform is a scalable deterministic
evaluator of the structured-Gaussian perturbed-max layer, with exact
Jacobian (the photo-finish Laplacian) and a global inverse.

The triad:

    softmax        <-> iid Gumbel     <-> Shannon entropy
    factor probit  <-> correlated Gaussian <-> structured nonseparable
                                                entropy-like regularizer

Calibration is the mirror map mu = grad Omega(p) on the mean-zero
quotient: forward map, inverse map, and Hessian geometry all in hand.

## Companion-paper theorems to work out

1. **Top-K / rank distributions via a generating-function field.**
   Conditional on f and U_i = x, i is in the top K iff at most K-1
   competitors exceed x. Build the single polynomial field
   H(z; x, f) = prod_j [F_j + (1 - F_j) z], truncate at degree K, divide
   out the i-th linear factor for each i:
   P(rank(i) = r) = E_f int g_i [z^{r-1}] H_i(z) dx, and
   P(i in TopK) = sum_{r<=K}. Cost ~ O(QNKL) for ALL alternatives, still
   linear in N at fixed K. Covers MoE routing (noisy top-k gating is
   almost literally a probit race), Recall@K under representation
   uncertainty, top-K token selection. Work this out regardless of the
   vision angle.

2. **Gaussian Fenchel-Young loss, nearly free.** The shared field already
   contains H_f(x) = prod_i F_i(x|f) = P(max <= x | f), so
   E[max | f] accumulates from one extra pass over the lattice and
   G(mu) = E_f E[max|f] costs essentially nothing extra. Then
   L(mu, y) = G(mu) - mu_y is the Fenchel-Young classification loss with
   grad = p(mu) - e_y. Cross-entropy is the Gumbel case; this is its
   correlated-Gaussian analogue, computed deterministically. Binary case:
   with t = delta/s, L = s [phi(t) - t Phi(-t)] -- a smooth Gaussian
   margin loss; gradient signal concentrates on photo finishes.
   Exact NLL alternative: grad_mu[-log p_y] = -J e_y / p_y, one JVP.
   Note: Fenchel-Young is fine with FIXED covariance; learning (V, D) by
   naive loss minimization can drive noise to zero -- use the actual
   likelihood for covariance learning.

3. **Pinsker certificate for covariance fit.** Winner events depend only
   on contrasts. With B a basis of 1-perp, C = B' Sigma B,
   Chat = B' Sighat B:
   |p_i(Sigma) - p_i(Sighat)| <= TV <= sqrt(KL(N(B'mu,C) || N(B'mu,Chat))/2),
   with the Gaussian KL explicit. A rigorous bridge from covariance-fit
   error to share error; candidate for the calibration paper's boundary
   section too.

4. **Clone robustness / multiplicity control.** m duplicates of one
   alternative at correlation rho get expected duplication bonus
   sigma sqrt(1-rho) E[max of m normals] ~ sigma sqrt((1-rho) 2 log m),
   which -> 0 as rho -> 1. Softmax adds log m regardless (IIA in
   neural-network clothing). Directly relevant to video temporal
   redundancy; the single-removal ensemble becomes a counterfactual
   token-substitution operator ("if token i disappeared, who takes
   over?") for interpretability/pruning.

## Where the applications rank

| target | why it fits | verdict |
|---|---|---|
| query-key matching / attention | q = qbar + Af gives v_j = A'k_j exactly; score covariance = KAA'K' + diag(D) | very strong; the clean derivation |
| VQ codeword selection | nearest-neighbor = argmax race exactly (common -||q||^2 drops) | surprisingly clean; straight-through gradient replaced by true probabilities |
| MoE / token routing | noisy top-k gating already perturbs scores with Gaussians | very strong, needs the top-K theorem |
| Bayesian 1000-class decision uncertainty | P(class wins) under Gaussian logit posterior; NOT E[softmax] -- a different, useful question | strong |
| heatmap soft-argmax (pose/keypoints) | pixels = alternatives, smooth spatial modes = factors; p_i = P(pixel is the true peak) | plausible; low rank natural for smooth error fields |
| randomized smoothing | Laplace-bridged RS discards off-diagonal logit covariance; our transform keeps it | strong if low-rank fit holds |
| NMS / detection | first selection is a race, but recursive suppression is extra structure | partial |
| segmentation / denoising / convolution | per-pixel argmaxes are COUPLED events; one-winner factorization does not give the joint mask | do not force |

## The falsification test (run before writing any vision claim)

Take a real many-way competition (ViT attention head, stereo cost volume,
VQ codebook, retrieval shortlist, 1000-way classifier). Perturb inputs a
few thousand times (augmentations, dropout, posterior draws); save
pre-argmax score vectors. Then:

1. Estimate the CONTRAST covariance P Sigma P; fit VV' + diag(D) at
   k = 1, 2, 4, 8. If the residual does not collapse with rank, the
   vision story is decorative. (This is the paper's own boundary
   criterion.)
2. On held-out perturbations, compare empirical winner frequencies vs
   softmax/Gumbel, independent probit, factor probit at matched means.
3. Delete candidates; measure actual redistribution vs the three maps'
   predictions (the IIA test at a neural layer).

## Experiment queue

1. **Collier head-to-head** (first): saved/generated (mu, V, D) at
   N = 1000, ranks 2/4/8/15; our transform vs their MC temperature
   softmax vs hard-argmax MC, all against a large CRN reference;
   probability error, NLL, gradient error, runtime.
2. **Berthet CIFAR-10 without Monte Carlo**: train with G(mu) - mu_y at
   k = 0 (iid Gaussian), then CIFAR-100 / ImageNet-1k.
3. **Correlated heatmap decoder**: pose/keypoints, mean heatmap + 2-8
   smooth uncertainty modes vs soft-argmax, especially under occlusion.
4. **Video clone test**: duplicate frames/patches deliberately; measure
   softmax vs factor-probit sensitivity to meaningless duplication.

## Positioning

Not "probit can process images." The claims are:

    large correlated visual hypothesis races -> factor-probit
    stochastic argmax

and

    softmax Jacobian p_i p_j  ~>  photo-finish Jacobian w_ij,

i.e. softmax's confusion geometry is fully determined by p; factor probit
distinguishes "similar marginal probability" from "direct competitors at
the decision boundary" -- an input-dependent confusion graph, matching
Collier et al.'s empirical finding that learned covariance identifies
confusable classes. The serious direction is a "correlated stochastic
attention / Thurstone attention" theory paper, not an application
paragraph.
