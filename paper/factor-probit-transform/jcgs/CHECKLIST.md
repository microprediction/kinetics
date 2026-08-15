# JCGS submission checklist

Manuscript
- [x] Title: "Scalable Probit Share Calibration" (Share guards the fixed-(V,D) scope)
- [x] Abstract ~150 words, no citations/equations; keywords added (6)
- [x] Author-year citations via natbib (ASA style)
- [x] 12pt; double-spaced submission build (paper-jcgs.pdf, 29 pp; flip \jcgstrue)
- [x] "Supplementary Materials" section itemizing code/software/data
- [x] Disclosure statement (no conflicts, no external funding)
- [x] No "working draft" markings
- [ ] Pin an immutable git tag at the submission commit and update README

Supplement to upload
- [ ] Zip of experiments/ + raceutil.py + rust crate pointer + run_all_paper.py
      from the tagged commit (JCGS runs reproducibility review; the manifest
      regenerates every table and figure)
- [ ] Check portal for current template/anonymity requirements on submission
      (T&F blocks automated access to the author instructions page)

Portal
- [ ] https://rp.tandfonline.com (Taylor & Francis submission portal), journal code UCGS
- [ ] Suggested AE/keywords at submission: computational statistics,
      quadrature, discrete choice
