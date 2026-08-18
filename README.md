# kinetics (view as [web page](https://kinetics.microprediction.org/))

Kinetic surrogates via fast ability transforms — exploring systems in which thousands of
latent stochastic processes compete to produce the first, fastest, weakest, or otherwise
extremal event, and inferring latent kinetic propensities from the winners using the
machinery of the [`thurstone`](https://github.com/microprediction/thurstone) package.

The connecting mathematics is a pair of leave-one-out identities saying the same thing:
one global computation is a compressed representation of the entire family of
single-deletion (cavity/defect) systems.

- **Multiplicative (races):** `S_field = ∏ S_j`, so the field faced by competitor *i* is
  `S_field / S_i` — one pass prices every competitor.
- **Rank-one (quadratic systems):** `G⁽ⁱ⁾_jk = G_jk − G_ji G_ik / G_ii` — one inverse
  contains every cavity Green function; leave-*k*-out costs only a *k*×*k* solve.

**Foundation paper:** Cotton, P. (2021). "Inferring Relative Ability from Winning
Probability in Multientrant Contests." [*SIAM Journal on Financial Mathematics* 12(1),
295–317](https://epubs.siam.org/doi/abs/10.1137/19M1276261)
**Implementation:** [`thurstone`](https://github.com/microprediction/thurstone)
([docs](https://thurstone.microprediction.org/))
**Sibling site:** [schur.microprediction.org](https://schur.microprediction.org) — the same
Schur-complement object in portfolio construction.

## Layout

- [`docs/`](docs) — the [web site](https://kinetics.microprediction.org/): introduction,
  research program, applications, bibliography.
- [`experiments/`](experiments) — identity verifications plus the first physics
  experiments: a Brownian narrow-escape first-passage study and the compressed defect
  ensemble of a disordered elastic network.
- [`watermark_races/`](watermark_races) — queued: does a context-conditioned many-way
  race likelihood improve LLM watermark detection over green-token counting?
- [`paper/`](paper) — working-paper outline.

## Cite

```bibtex
@article{cotton2021inferring,
  author  = {Cotton, Peter},
  title   = {Inferring Relative Ability from Winning Probability
             in Multientrant Contests},
  journal = {SIAM Journal on Financial Mathematics},
  volume  = {12},
  number  = {1},
  pages   = {295--317},
  year    = {2021}
}
```

## Note: factor-probit program moved

The factor multinomial probit paper and its experiment suite (13-38)
now live in the [winning repository](https://github.com/microprediction/winning)
(`papers/factor-probit-transform/`, `research/experiments/`, branch
`benchmark-arena`, tag `jcgs-v2`), alongside the production
implementation in `winning.factor`. The physics experiments and the
companion papers remain here; `experiments/raceutil.py` is now a shim
importing the canonical implementation from the winning package.
