# fastrace: Rust kernel for the factor-probit share transform

Fused, rayon-parallel implementation of the min-wins factor race
(the `thurstone`/`raceutil` shared-survival-field pass). Log-domain
throughout (log_ndtr via libm::erfc with asymptotic tail), x-tiled for
cache, parallel over factor nodes.

Measured (Apple M4, vs single-threaded NumPy reference, GH order 15, k=2,
L=1501; agreement 8e-17):

| N | NumPy | Rust | speedup |
|---|---|---|---|
| 1000 | 4.24 s | 0.71 s | 6.0x |
| 5000 | 21.6 s | 3.9 s | 5.5x |

Build: `pip install maturin && maturin develop --release` (needs Rust
toolchain). Exposes `fastrace.win_probabilities_factor(mu, V, D, F, W,
points=1501) -> (p, total)`.

Not yet ported: inversion slopes, JVP, deletion ensemble, and the
Chebyshev-separated low-rank pass (projected further ~15-29x; see
paper/fast-kernel-notes.md).
