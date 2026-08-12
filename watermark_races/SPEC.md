# Watermark races — experiment specification (queued)

Status: **queued** — the glass/physics experiments (see `../experiments/`) run first.
`config.py` is already in place; the modules below remain to be written. This file
preserves the full design so work can resume without re-derivation.

## Hypothesis

Watermark detection should weight token selections according to how competitive the
underlying next-token race was. A green-token selection in a high-entropy (competitive)
context carries more watermark evidence than one where a single token was already
overwhelmingly likely.

## Setup

- Open-weight causal LM via Hugging Face Transformers. Generator `Qwen/Qwen2.5-1.5B`
  (fallback `Qwen/Qwen2.5-0.5B` if GPU memory is short), float16/bfloat16 on GPU, all
  seeds fixed. **Do not use or attempt to reverse-engineer any proprietary watermark** —
  we implement the Kirchenbauer-style green-list watermark ourselves so key, strength,
  logits, and ground truth are all known.
- Watermark: secret key + previous token(s) → cryptographic hash → PRNG → green set of
  size γV; watermarked logits l + δ·1{green}. Defaults γ=0.25, δ ∈ {1,2,3}, temperature
  1.0 (later 0.7, 1.3), top-p=1, no top-k.

## Steps

1. **Generation** (`generate.py`): manual autoregressive loop (not `model.generate`) —
   at each position save raw logits info, sampled token, green membership, null and
   watermarked probability of the selected token, and enough to reconstruct the green
   list. Apply temperature and any truncation consistently to null and watermarked
   distributions.
2. **Matched corpora**: ≥500 diverse prompts (factual, creative, code, technical, email,
   summaries, history, science, dialogue, argument — `config.build_prompts()`); for each
   prompt and seed generate an unwatermarked and a watermarked continuation of ≥256
   tokens with identical settings apart from the perturbation. Prefer 1000 prompts.
   Store everything needed for reproduction.
3. **Baseline detector** (`detectors.py`): green count K, Binomial(T, γ) null,
   z-score Z = (K − γT)/√(Tγ(1−γ)); also raw green fraction.
4. **Exact LR detector** (`exact_lr`): s_t = log p₁(Y_t) − log p₀(Y_t) from the exact
   softmax under both hypotheses; S = Σ s_t. Neyman–Pearson benchmark that the race
   approximation must approach.
5. **Race informativeness analysis**: per-token s_t vs null probability of the selected
   token, entropy, top-1 probability, top-2 logit gap, N_eff = exp(H). Calibration plots
   testing the competitiveness prediction.
6. **Thurstone/race approximation** (`race_detector.py`): latent race U = a + ε with
   (A) Gumbel noise — must reproduce softmax, correctness test — and (B) Gaussian
   Thurstone noise, computed with the fast field-survival/divide-out operation (no naive
   leave-one-out products), truncating/aggregating to top-M logits, M ∈ {100, 500, 1000,
   5000}. S_Thurstone = Σ log q₁(Y_t)/q₀(Y_t). Tune ability map and σ on validation only.
7. **Misspecified proxy** (`proxy.py`): generate with 1.5B, detect with 0.5B logits only.
   Compare green-count z, proxy softmax LR, proxy Gaussian-Thurstone LR, entropy-weighted
   green counts. Key question: is the race model more robust under proxy misspecification?
8. **Detection experiments**: lengths {16, 32, 64, 128, 256} × δ ∈ {1, 2, 3}: ROC, AUROC,
   TPR at FPR 10%/1%/0.1%, bootstrap 95% CIs. Central metric: power at 16–64 tokens.
9. **Entropy ablation**: green-count detector restricted to entropy quintiles; then the
   weighted detector S_w = Σ w(H_t)(1{green} − γ) with w fit on the training split. If
   entropy weighting alone captures the improvement, the race machinery is not justified —
   report that clearly.
10. **Perturbation robustness**: delete 5/10/20% of tokens; synonym-replace 5/10/20%;
    paraphrase with a separate model; splice 25/50/75% watermarked into unwatermarked.
    Ordinary editing only — no attack optimization.
11. **Statistical tests**: paired bootstrap on shared documents. H1 exact_lr >
    green_count; H2 thurstone_proxy > proxy_softmax under mismatch; H3 advantage largest
    for short text; H4 per-token evidence increases with entropy. Held-out test data for
    all final comparisons.
12. **Outputs**: modules `config.py`, `generate.py`, `watermark.py`, `detectors.py`,
    `race_detector.py`, `proxy.py`, `experiment.py`, `analysis.py`; `tests/`, `data/`,
    `results/` (parquet/CSV), `figures/` (ROC, TPR-vs-length, s_t vs entropy, s_t vs
    top-1, exact-LR vs green-count scatter, proxy comparison), and a markdown report.

Unit tests required: deterministic green-list reconstruction; identical null
distributions at δ=0; exact LR ≡ 0 at δ=0; Gumbel race reproduces softmax numerically;
green fraction ≈ γ under unwatermarked generation; watermark raises green fraction;
truncation probabilities sum to one; reproducibility under fixed seeds.

## Decision criterion

Promising only if at least one of: (1) exact context-conditioned LR substantially beats
green counts at 16–64 tokens; (2) a Thurstone/race detector on a misspecified proxy beats
the proxy-softmax detector; (3) race conditioning materially improves robustness under
ordinary editing. If only (1) holds, context conditioning matters but Thurstone
specifically is unproven. Do not try to make the hypothesis win.
