# Results

Computed from 40 runs in `results/`. Every number here and in the figures is read from those pickles.


## 0 ground truth

- measured k* per seed = [4, 4], predicted = 4; trivial = 1.0075, threshold = 0.02016
- loss by k: k=1: 0.75511, k=2: 0.50314, k=4: 0.00002, k=8: 0.00001, k=16: 0.00001, k=32: 0.00001

## 1 emergent B

- kappa_end=0.5 (n=5): B=5.60+/-1.00 (k*=4), converged loss=0.00025, frac of held phase at exactly k* = 0.20, role_coverage=0.90
- kappa_end=1 (n=5): B=5.40+/-1.18 (k*=4), converged loss=0.00022, frac of held phase at exactly k* = 0.40, role_coverage=0.85
- kappa_end=1.5 (n=5): B=4.00+/-0.00 (k*=4), converged loss=0.00066, frac of held phase at exactly k* = 1.00, role_coverage=0.80
- kappa_end=2 (n=5): B=3.00+/-1.24 (k*=4), converged loss=0.25297, frac of held phase at exactly k* = 0.61, role_coverage=0.60
- kappa_end=2.5 (n=5): B=1.40+/-0.78 (k*=4), converged loss=0.65559, frac of held phase at exactly k* = 0.00, role_coverage=0.30
- closest to k*=4: kappa_end=1.5 gives B=4.00

## 2 territory

- T=1 (n=3): role_coverage=0.25+/-0.00, territory_span=1.00, perfused=1.00
- T=2 (n=3): role_coverage=0.50+/-0.00, territory_span=2.00, perfused=2.00
- T=4 (n=3): role_coverage=0.75+/-0.00, territory_span=4.00, perfused=4.00
- T=8 (n=3): role_coverage=0.92+/-0.16, territory_span=7.67, perfused=8.00
- T=16 (n=3): role_coverage=1.00+/-0.00, territory_span=15.33, perfused=16.00
