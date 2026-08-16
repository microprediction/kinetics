# The idea in plain English

(2026-08-16; framing worth preserving for the intro, a talk, or the
eventual blog post.)

## The loaded roulette wheel

Imagine a loaded roulette wheel: every slice has a different probability.
To watermark the outcomes, the obvious move is to quietly resize slices
-- but then you have changed the game, and frequency counts give you
away.

SynthID's move: leave the wheel alone; give every spin a secret coin
flip that paints slices green or red, and let green beat red when two
slices compete. Averaged over keys, the wheel is unchanged.

Our move: instead of painting slices independently, blow a SECRET WIND
across the wheel. Nearby slices feel similar wind; opposite slices feel
opposite wind. The detector knows the wind. It does not ask "was this
token painted green?" but "did the winner drift with the wind?"

## The enabling piece

The calibration paper answers the question this construction needs:
given the wheel you want, where do you place the starting positions so
that after Gaussian randomness every slice keeps EXACTLY its intended
probability? That inverse (unique, mean-zero, computable without
simulation) is what makes the Gaussian watermark exist in practice.

## The five properties

1. Marginals exactly preserved for every key-exposure level.
2. A knob (rho) trading detector evidence against output diversity,
   with the marginal untouched along the whole dial.
3. Smoothness: factor structure nudges semantically similar words
   together instead of painting each token independently (hypothesis --
   needs LLM experiments).
4. Watermark design becomes graph optimization. SHARPENING, not in the
   paper yet: the Stein drift is the Dirichlet energy of the exposed
   loading field on the photo-finish graph, so max signal per unit
   exposure is a Rayleigh quotient => the OPTIMAL watermark keys the
   principal eigendirection of the photo-finish Laplacian restricted to
   the loading space. Detection-optimal watermarking is an eigenvector
   problem. (Theorem candidate; verify numerically before claiming.)
5. Everything is exact: identities verified to 1e-10..1e-17 (paper's
   suite + independent intake checks).

## Honest caveats

- Non-distortion is per-token, averaged over keys. Within one keyed
  context, second-order statistics (co-occurrence, collision/diversity)
  ARE altered -- that is the quantified cost, and outsider
  detectability is the same second-order question SynthID faces.
- Not a SynthID killer yet: calibration at vocabulary scale per token
  is orders of magnitude off LLM latency. Routes: top-K version (the
  generating-function field), amortized inverse, caching, or a model
  whose output layer IS the Gaussian perturbed-max (the companion
  neural paper's layer -- in which case the watermark is free).

## The bigger claim

The watermark may not be the important application. The primitive is:
ANY desired distribution can be realized exactly as the marginal of a
correlated Gaussian race, with an invertible, differentiable map.
Watermarking, privacy, controlled randomness, steganography, keyed
communication protocols: all become "choose a different hidden Gaussian
factor." Calibration turned watermarking into a general coupling
problem.

## Computational efficiency (measured 2026-08-16)

- Eigen-direction M: offline, k JVPs per key draw + k x k eigenproblem;
  free at deployment. Detection: dot product per token (microseconds).
- Generation, deployment regime (top-k support, k=2, Q=49, L=129, warm
  start): NumPy measures 59 ms/token at N=64 and 256 ms/token at N=256
  vs a 1-10 ms LLM budget -- roughly 100x short. But the flop count
  (~4e5/pass at N=64) says NumPy small-array overhead dominates: a
  fused compiled kernel prices the pass at ~10-50 us, i.e. ~0.1-0.3
  ms/token at 2-7 warm iterations -- inside budget, unmeasured. The
  synthetic drift used here (0.15/logit/step) is adversarial; real
  decoding is smoother, so the iteration count should drop.
- Escape routes unchanged: amortized inverse + certified correction;
  native Gaussian head (companion neural layer) makes the watermark
  free.
