# Astronomy applications for the factor-probit race (note, 2026-08-17)

Peter's hunch during the JCGS polish: "I bet there are astronomical
uses." A short sentence went into the paper's discussion (vision
cluster). The fuller mapping, for a future paper or collaborator
pitch:

1. Catalog cross-matching. Which of N candidate counterparts in survey
   B matches a source in survey A. Scores share astrometric-calibration
   and plate-distortion systematics (= common factors V), with
   idiosyncratic centroid noise (= D). Current practice: independence
   assumptions or expensive MCMC (cf. Budavari-Szalay probabilistic
   cross-identification -- VERIFY before citing).
2. Transient follow-up allocation (Rubin/LSST brokers). ~1e5-1e6
   alerts/night ranked by interest probability; observing a candidate
   removes it and the field re-ranks. Proportional renormalization =
   the IIA reflex; alerts correlate through CCD, airmass, crowding.
   The removal ensemble computes where probability should flow, all N
   in one pass.
3. Gravitational-wave counterpart tiling. Which sky tile hosts the
   counterpart; tiles share weather/extinction/instrument factors.
   ("Segment of the sky" in the paper already alludes to this.)
4. Host-galaxy association and photo-z template selection. Pick one of
   N hosts/templates under shared calibration error.

Regime match: huge N, modest k (a few shared systematics) -- exactly
the paper's sweet spot. Astronomers care about the counterfactuals
(candidate spectroscopically ruled out => match probabilities update).

Next steps if pursued: find a public cross-match or broker dataset;
one demo = inner calibration + removal ensemble only (no estimation
claim), mirroring the real-data suggestion in EDIT-TODO.
