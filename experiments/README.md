# Experiments

Numbered experiments follow the [research program](https://kinetics.microprediction.org/program.html);
flat scripts are identity checks. Everything here runs on numpy/scipy/matplotlib alone.

| What | File | Result |
|---|---|---|
| Rank-one cavity identity + timings | [`cavity_downdate_demo.py`](cavity_downdate_demo.py) | exact to 1e-15; leave-1/2/3-out verified |
| Multiplicative cavity identity + timings | [`race_field_demo.py`](race_field_demo.py) | exact; 26× at N=2000 |
| Shared race transforms (forward + inverse) | [`raceutil.py`](raceutil.py) | used by exp01 |
| **Exp 1** — surrogates on a real first-passage simulation (Brownian narrow escape) | [`exp01_narrow_escape/`](exp01_narrow_escape) | physics is non-IIA (TV 0.082) but *independent* races don't capture the geometric neighbor effect → motivates Q6 |
| **Exp 2** — rank-one cavity on a disordered elastic network | [`exp02_glass_cavity/`](exp02_glass_cavity) | all 784 site deletions in 9 ms (~870×); 4k defect-pair interactions in 35 ms; exponential screening recovered |
