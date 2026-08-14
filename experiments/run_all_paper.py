"""Single-command manifest: regenerate every number and figure in the
factor-probit paper (experiments 13-17).

    python experiments/run_all_paper.py

Runs each committed experiment script in order and reports pass/fail. Every
table entry and figure in paper/factor-probit-transform/paper.tex traces to
one of these scripts' results.csv / figures/ outputs; the mapping is:

  exp13 run_ghk_benchmark.py   -> benchmark table, smoothness figure,
                                  inversion, removal ensemble
  exp14 run_boundaries.py      -> contrast-space rank-k boundary (Part A)
  exp14 run_factorial.py       -> 2x2 factorial substitution study
  exp15 run_certificate.py     -> Price-identity checks (perturbation section)
  exp16 run_addendum.py        -> mean/TV metrics, matched-time direct MC,
                                  inversion replication
  exp16 run_replication.py     -> multi-problem replication at common spread
  exp17 run_convergence.py     -> L/Q sweeps, RQMC scramble distribution,
                                  Jacobian diagnostics, accuracy-time frontier

Total wall time is roughly 1.5 hours on an Apple M4 (single-threaded BLAS).
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "exp13_ghk_benchmark/run_ghk_benchmark.py",
    "exp14_boundaries/run_boundaries.py",
    "exp14_boundaries/run_factorial.py",
    "exp15_perturbation_certificate/run_certificate.py",
    "exp16_benchmark_addendum/run_addendum.py",
    "exp16_benchmark_addendum/run_replication.py",
    "exp17_convergence/run_convergence.py",
]

def main():
    failures = []
    for rel in SCRIPTS:
        t0 = time.time()
        print(f"=== {rel}")
        r = subprocess.run([sys.executable, str(HERE / rel)])
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"=== {rel}: {status} ({time.time()-t0:.0f}s)\n")
        if r.returncode != 0:
            failures.append(rel)
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("all paper experiments regenerated")

if __name__ == "__main__":
    main()
