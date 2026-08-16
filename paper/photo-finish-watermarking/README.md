# Photo-Finish Watermarking

**Working paper:** *Photo-Finish Watermarking: SynthID as a Laplacian Flow and Distortion-Free Gaussian Sampling*  
**Author:** Peter Cotton  
**Date:** August 2026

This package contains the LaTeX manuscript, bibliography, seeded synthetic experiments, numerical results, and all figure files.

## Contents

- `photo_finish_watermarking.tex` - manuscript source
- `photo_finish_watermarking.bib` - bibliography
- `photo_finish_watermarking.pdf` - compiled working paper
- `scripts/experiments.py` - seeded identity checks and synthetic experiments
- `data/results.json` - machine-readable numerical output
- `figures/` - PDF and PNG versions of the four figures
- `requirements.txt` - Python dependencies for reproducing the experiments

## Reproduce the experiments

```bash
python -m pip install -r requirements.txt
python scripts/experiments.py
```

The script uses the fixed seed `20260816`. It rewrites `data/results.json` and the files in `figures/`.

## Build the paper

A standard TeX installation with `pdflatex`, BibTeX, and `latexmk` is sufficient:

```bash
latexmk -pdf -bibtex photo_finish_watermarking.tex
```

The figures are already included, so Python is not required merely to compile the PDF.

## Scope

The numerical work is synthetic. It verifies the algebraic identities, Gaussian quadrature, share calibration, marginal non-distortion, Stein detector identity, and exact-null statistic. It is not a language-model benchmark and makes no production-latency, text-quality, robustness, or security claim.
