# Results

Computed from 88 runs in `results/`. Every number here and in the figures is read from those pickles.


## 0 ground truth

- measured k* per seed = [4, 4], predicted = 4; trivial = 1.0075, threshold = 0.02016
- loss by k: k=1: 0.75511, k=2: 0.50314, k=4: 0.00002, k=8: 0.00001, k=16: 0.00001, k=32: 0.00001

## 1 emergent B

- kappa_end=0.5 (n=5): B=5.60+/-1.00 (k*=4), converged loss=0.00025, frac of held phase at exactly k* = 0.20, role_coverage=0.90
- kappa_end=1 (n=5): B=5.40+/-1.18 (k*=4), converged loss=0.00022, frac of held phase at exactly k* = 0.40, role_coverage=0.85
- kappa_end=1.5 (n=5): B=4.00+/-0.00 (k*=4), converged loss=0.00066, frac of held phase at exactly k* = 1.00, role_coverage=0.80
- kappa_end=2 (n=5): B=3.00+/-1.24 (k*=4), converged loss=0.25297, frac of held phase at exactly k* = 0.61, role_coverage=0.60
- kappa_end=2.5 (n=5): B=1.40+/-0.78 (k*=4), converged loss=0.65559, frac of held phase at exactly k* = 0.00, role_coverage=0.30
- VERDICT prediction 'emergent B converges on k*': SUPPORTED. kappa_end=1.5 gives B=4.00+/-0.00 against k*=4, with 100% of the held phase at exactly k*. B is monotone in kappa_end over 1.40 to 5.60, so the mechanism does not sit at k* for free: kappa_end selects it.

## 2 territory

- T=1 (n=3): role_coverage=0.75+/-0.00, territory_span=1.00, perfused=4.00
- T=2 (n=3): role_coverage=0.83+/-0.16, territory_span=2.00, perfused=4.67
- T=4 (n=3): role_coverage=1.00+/-0.00, territory_span=3.67, perfused=6.33
- T=8 (n=3): role_coverage=0.92+/-0.16, territory_span=6.67, perfused=9.33
- T=16 (n=3): role_coverage=1.00+/-0.00, territory_span=13.00, perfused=16.00
- T=32 (n=3): role_coverage=1.00+/-0.00, territory_span=32.00, perfused=32.00
- VERDICT prediction 'coverage collapses below T=R': NOT SUPPORTED. mean coverage below T=R is 0.79 vs 0.98 at or above; a decline, not a collapse.
- VERDICT prediction 'territory_span falls well below T': NOT SUPPORTED. span/T = 0.81 to 1.00 (mean 0.93); span tracks T.
- CAUSE the never-fully-infarct fallback is applied per territory, so every territory keeps at least one perfused head and perfused >= T by construction. Compaction below T cannot occur for this mechanism as written.

## 3 delay

- tau=0 (n=5): converged loss=0.00066+/-0.00051, flips/step=0.0004, role_coverage=0.80
- tau=1 (n=3): converged loss=0.00073+/-0.00092, flips/step=0.0000, role_coverage=0.75
- tau=5 (n=3): converged loss=0.00073+/-0.00091, flips/step=0.0000, role_coverage=0.75
- tau=20 (n=3): converged loss=0.00064+/-0.00074, flips/step=0.0000, role_coverage=0.75
- tau=100 (n=3): converged loss=0.04918+/-0.09588, flips/step=0.0037, role_coverage=0.75
- VERDICT prediction 'an optimal tau exists': NOT SUPPORTED. tau 0 to 20 spans only 0.00010 in loss with overlapping CIs; only tau=100 is worse.
- VERDICT prediction 'long tau causes gate oscillation': NOT SUPPORTED. max perfused-set changes per step over the held phase is 0.0037 at any tau (one set change per ~272 steps at worst), so there is no oscillation to trade against.

## 4 pool

- pool_beta=0 (n=3): role_coverage=0.92, territory_span=6.67, converged loss=0.00007+/-0.00005
- pool_beta=0.25 (n=3): role_coverage=1.00, territory_span=7.67, converged loss=0.00004+/-0.00001
- pool_beta=0.5 (n=3): role_coverage=0.92, territory_span=7.33, converged loss=0.00008+/-0.00010
- pool_beta=1 (n=3): role_coverage=1.00, territory_span=7.67, converged loss=0.00029+/-0.00007
- VERDICT pooling changes nothing. role_coverage spans 0.92 to 1.00 and territory_span 6.67 to 7.67 across pool_beta 0 to 1; full pooling has the worst converged loss (0.00029).

## 5 demand arms

- outnorm_ema, leak=0 (n=3): loss@B=k*=0.00132+/-0.00160
- outnorm_ema, leak=0.05 (n=3): loss@B=k*=0.00122+/-0.00152
- outnorm_inst, leak=0 (n=3): loss@B=k*=0.00145+/-0.00186
- outnorm_inst, leak=0.05 (n=3): loss@B=k*=0.00124+/-0.00151
- qnorm, leak=0 (n=3): loss@B=k*=0.00098+/-0.00128
- qnorm, leak=0.05 (n=3): loss@B=k*=0.00089+/-0.00123
- random, leak=0 (n=3): loss@B=k*=0.08007+/-0.15430
- random, leak=0.05 (n=3): loss@B=k*=0.00271+/-0.00338
- VERDICT ordering at leak=0, best to worst: qnorm 0.00098, outnorm_ema 0.00132, outnorm_inst 0.00145, random 0.08007
- VERDICT the three signal arms are not separable: they span 0.00098 to 0.00145 while every one of their 95% CIs is wider than that spread. EMA vs instantaneous is a tie.
- VERDICT random is separable and worse: 0.08007 vs 0.00098 for the best signal arm, a factor of 82. Demand-ranked supply beats chance at B = k*.
- VERDICT leak=0.05 does not rescue the EMA arm specifically. It helps random most (0.08007 -> 0.00271); outnorm_ema moves 0.00132 -> 0.00122.

## 5 ablation

- rho(demand, ablation delta) = +0.379+/-0.066 over 88 runs, at full perfusion
- rho(outnorm, ablation delta) = +0.295+/-0.097 over 88 runs, at full perfusion
- rho(neg_entropy, ablation delta) = +0.452+/-0.083 over 88 runs, at full perfusion
- rho(qnorm, ablation delta) = +0.484+/-0.085 over 88 runs, at full perfusion
- VERDICT the gating signal is not the best predictor of causal head importance. demand is +0.379 while qnorm reaches +0.484 on the same runs. Measured at d_k=32, ABOVE the capacity bound, so this does not contradict the reported qnorm sign flip below it.
