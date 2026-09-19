# Results

Computed from 60 runs in `results/`. Every number here and in the figures is read from those pickles.


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

- T=1 (n=3): role_coverage=0.75+/-0.00, territory_span=1.00, perfused=4.00
- T=2 (n=3): role_coverage=0.83+/-0.16, territory_span=2.00, perfused=4.67
- T=4 (n=3): role_coverage=1.00+/-0.00, territory_span=3.67, perfused=6.33
- T=8 (n=3): role_coverage=0.92+/-0.16, territory_span=6.67, perfused=9.33
- T=16 (n=3): role_coverage=1.00+/-0.00, territory_span=13.00, perfused=16.00
- T=32 (n=3): role_coverage=1.00+/-0.00, territory_span=32.00, perfused=32.00

## 3 delay

- tau=0 (n=5): converged loss=0.00066+/-0.00051, flips/step=0.00, role_coverage=0.80
- tau=1 (n=3): converged loss=0.00073+/-0.00092, flips/step=0.00, role_coverage=0.75
- tau=5 (n=3): converged loss=0.00073+/-0.00091, flips/step=0.00, role_coverage=0.75
- tau=20 (n=3): converged loss=0.00064+/-0.00074, flips/step=0.00, role_coverage=0.75
- tau=100 (n=3): converged loss=0.04918+/-0.09588, flips/step=0.00, role_coverage=0.75

## 4 pool

- pool_beta=0 (n=3): role_coverage=0.92, territory_span=6.67, converged loss=0.00007+/-0.00005
- pool_beta=0.25 (n=3): role_coverage=1.00, territory_span=7.67, converged loss=0.00004+/-0.00001
- pool_beta=0.5 (n=2): role_coverage=0.88, territory_span=7.00, converged loss=0.00010+/-0.00015
