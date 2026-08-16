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

## High-value positioning (2026-08-16): where compute cost is irrelevant

Not "SynthID competitor pending a faster kernel" but the HIGH-ASSURANCE
TIER:

1. EDIT ROBUSTNESS WITH A MECHANISM: independent-color schemes lose the
   position's entire signal to a synonym swap (fresh coin flip); with
   embedding-derived loadings, v_{y'} ~ v_y for semantic neighbors, so
   drift degradation is BOUNDED BY SEMANTIC DISTANCE of the edit. An
   editor must damage meaning in proportion to watermark removed.
   Extends (speculatively) to translation via multilingual embedding
   loadings. Hypotheses -- but with a theorem-shaped mechanism that
   token-coloring schemes cannot formulate.
2. MULTI-BIT ATTRIBUTION VIA THE EIGENBASIS: k orthogonal wind
   directions = k channels, capacity-ranked by eigendecomposition of M;
   key directions to customers/tenants/versions; detection identifies
   WHICH wind. Per-user provenance.
3. FORENSIC-GRADE EXACT NULL: false-positive rate is a theorem, not an
   asymptotic or simulation -- survives hostile expert scrutiny (legal,
   misconduct, disputes). Requires the seed-independence idealization.
4. SHORT-TEXT EVIDENCE BUDGETING: rho-dial + eigen-direction maximize
   per-token evidence with marginal provably untouched -- exactly what
   30-token outputs need.
5. LOW-VOLUME HIGH-STAKES GENERATION: legal drafts, official statements,
   audited agentic actions, and TRAINING-DATA CANARIES (watermark
   benchmark answers / proprietary corpora; detect contamination when a
   trained model reproduces the drift). All tolerate 100ms-s/token.

Counterweights (honest): all robustness unexperimented; strong LLM
paraphrase attacks degrade everything and must be measured; public
loading rule lets adversaries attempt semantically-costly scrubbing --
the quality-vs-scrubbing frontier is itself quantifiable in this
geometry and is a natural experiment.

## Images and video (2026-08-16)

Two constructions:

1. TOKEN-BASED GENERATION (VQ/autoregressive/discrete diffusion): the
   text construction verbatim -- each patch's codeword selection is a
   categorical race; loadings = codebook vectors themselves (visual
   similarity IS loading similarity). Edit robustness is stronger than
   text: JPEG/resize/filter perturb re-tokenized codes to NEARBY
   codewords (small ||dv||, drift retained) where coloring schemes lose
   every flipped position. Video: temporal redundancy = clone
   robustness; smooth temporal key modes survive frame drops.
2. DIFFUSION: rho-split the initial noise z_T = sqrt(rho) K +
   sqrt(1-rho) Z -- marginal EXACTLY N(0,I) for all rho, output
   distribution untouched by construction (cleaner than fixed-pattern
   Tree-Ring). Detection via DDIM inversion + correlation. Factor V =
   smooth/low-frequency spatial modes; eigen-direction theorem picks
   detection-optimal modes. NO SHARE INVERSION NEEDED -- the compute
   objection vanishes for this branch.

Obstacles: geometric sync (as for all image watermarks); VQ
re-tokenization stability under heavy edits; DDIM inversion error;
dense prior art (Tree-Ring, Gaussian Shading claims
distribution-preserving noise keying, Stable Signature) -- novelty pass
REQUIRED before claiming the diffusion variant. Distinctive regardless:
continuous rho-dial with exactness, correlated factor keying with
capacity-ranked channels, eigen-optimal direction, exact null.

## Product positioning (2026-08-16): forensic fingerprinting, not AI detection

The killer application is NOT "was this AI-written" but ATTRIBUTION:
given leaked/disputed text, which model, tenant, session, or recipient
generated it, after ordinary metadata is gone. Concrete scenario: a
sensitive AI-generated briefing goes to 500 authorized recipients, each
copy statistically equivalent (same marginal law) but keyed with a
different Gaussian stream; a leak is scored against all 500 keys.

Why the controlled-enterprise setting fits this construction:
- generator AND detector controlled -> model-aware likelihood-ratio
  detection (evidence = I(R;Y)), not just the model-free score;
- exact marginal preservation matters in safety-critical text (no
  systematic bias toward a green subset when the token is a dosage or a
  "not");
- outputs valuable enough to fund specialized infrastructure (top-k,
  amortized inverse, native Gaussian head, offline forensics);
- per-recipient fingerprinting = the eigenbasis channel structure.

Four-layer architecture: digital signature/C2PA + provenance metadata +
Gaussian textual fingerprint (the durability layer that survives
copy-paste out of the credentialed container) + immutable audit logs.

Hard limitations to state in any product claim:
- NOT cryptographic authentication: spoofing/reverse-engineering
  attacks exist against distortion-free watermarks; never authorize
  consequential actions on a watermark hit alone;
- rewriting/translation/regeneration remain the central threats
  (semantic robustness is measured, not assumed -- exp33/34);
- short messages carry little evidence -> aggregate across documents,
  sessions, collections;
- operating point for high-value use is FPR 1e-5..1e-6, not academic
  1%; validated on wrong keys, human text, neighboring model versions,
  adversarial edits;
- collusion resistance (multiple recipients diffing copies) needs a
  coding layer on top of the statistical channel -- not yet designed;
- top-k deployment preserves the RENORMALIZED top-k law, not
  automatically the full-vocabulary distribution.

Benchmark spec: frozen logits; documents of 100/300/1000/3000 tokens;
thousands of simulated recipient keys; attack suite = paraphrase,
translation, truncation, reordering, copy-paste mixing, repeated
detector queries, key-stream estimation, collusion. exp35 gives the
clean-channel attribution scaling.
