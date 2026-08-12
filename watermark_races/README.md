# watermark_races

**Status: queued.** The glass/physics experiments in [`../experiments/`](../experiments)
run first; this study resumes afterwards.

Does a context-conditioned many-way race likelihood improve LLM watermark detection over
the standard green-token count / z-score detector — especially for short texts, weak
watermarks, and misspecified proxy models?

The full experimental design (12 steps, unit-test list, and decision criterion) is
preserved in [SPEC.md](SPEC.md). Configuration and the deterministic prompt corpus are
already implemented in [config.py](config.py).

Notes for when work resumes:

- Uses an open-weight model (`Qwen/Qwen2.5-1.5B`) and a self-implemented
  Kirchenbauer-style green-list watermark, so key, strength, and logits are all known.
  No proprietary watermark is used or reverse-engineered.
- The Gaussian-Thurstone race detector must use the fast field-survival/divide-out
  computation (the multiplicative cavity identity, see the
  [introduction](https://kinetics.microprediction.org/intro.html)) rather than naive
  leave-one-out products — the vocabulary is the field, top-M truncated.
- Requires `torch` and `transformers` (not needed elsewhere in this repository).
