# Experiment 2: the rank-one cavity on a disordered elastic network

**Physics.** A 28×28 random spring network (jittered lattice, log-normal stiffness
disorder, weak substrate coupling) — a standard harmonic model of an amorphous solid.
One inversion of the Hessian yields the susceptibility G; every deletion question is
then a downdate.

**Results (seed 7, n = 784 sites).**

- **All 784 single-site pinning responses** (`drop_i = Tr G − Tr G⁽ⁱ⁾ = ‖G·ᵢ‖²/Gᵢᵢ`)
  from the one inverse in **9 ms**, vs ~7.7 s for 784 re-inversions — **~870×**, exact
  to 1.5e-13. The map (`figures/soft_spots.png`) shows the disorder-induced soft
  regions of the glass.
- **4,000 defect-pair deletions** priced by 2×2 Schur downdates
  (`drop_S = Tr[(G_SS)⁻¹ (G²)_SS]`) in **35 ms**, no re-solves, exact to 1.4e-13. The
  pair nonadditivity `I_ij = drop_i + drop_j − drop_ij` — the elastic interaction
  between two pinning defects — decays exponentially with distance over 2.5 decades
  (`figures/pair_interaction.png`), with screening length set by the substrate
  coupling.

**Point.** "One global inverse is a compressed representation of an enormous family of
defect systems" is not just an identity: the entire single-defect ensemble and a large
sample of the pair-defect ensemble of a physical model were computed in under 50 ms
after one solve. This is program Q1/Q4 in miniature; the open question is how far the
harmonic downdate tracks relaxed (anharmonic) defect calculations.

Run: `python run_glass_cavity.py` (~10 s, numpy/matplotlib only).
