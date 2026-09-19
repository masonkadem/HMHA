# Results

Computed from 100 runs in `results/`. Every number here and in the figures is read from those pickles.


## 1 task

- true offsets [1, 5, 9, 13]; a dense 4-head model implements [1, 5, 9, 13]
- under supply, 4 of 32 heads survive and cover 75% of the true offsets, final loss 0.00027 against trivial 1.007

## 2 k* tracking

- R=k*=2 (n=3): B=2.67+/-0.65, error +0.67, role_coverage=1.00, held phase at exactly k* = 0.30, measured k*=[2]
- R=k*=3 (n=3): B=2.00+/-1.13, error -1.00, role_coverage=0.44, held phase at exactly k* = 0.33, measured k*=[3]
- R=k*=4 (n=5): B=4.00+/-0.00, error +0.00, role_coverage=0.80, held phase at exactly k* = 1.00, measured k*=[4, 4]
- R=k*=6 (n=3): B=4.00+/-1.13, error -2.00, role_coverage=0.61, held phase at exactly k* = 0.02, measured k*=[6]
- R=k*=8 (n=3): B=4.00+/-2.99, error -4.00, role_coverage=0.38, held phase at exactly k* = 0.00, measured k*=[8]
- VERDICT B does NOT track k*. Over k* = 2 to 8 (a span of 6) B spans only 2.00 and saturates: the fitted slope dB/dk* is 0.29, against 1.00 for tracking. Mean absolute error 1.53 heads, exact on 1 of 5 settings.
- VERDICT experiment 1 does not generalise. theta = mean + kappa*std over H standardized demands admits a nearly fixed number of heads regardless of the task, so B is set by (kappa_end, H) and the earlier B = 4 = k* was kappa_end = 1.5 matching k* = 4, not the mechanism finding it.
- CAVEAT the task is only solved at k* in [2, 4]; elsewhere the hemo run ends above the solved threshold, so those B values describe a failed run. The hemo arm gets cfg.steps while the dense k* check gets scratch_mult x that, so under-training at large R is a live alternative explanation and is worth one control before this is written up.

## 1 emergent B

- kappa_end=0.5 (n=5): B=5.60+/-1.00 (k*=4), converged loss=0.00025, frac of held phase at exactly k* = 0.20, role_coverage=0.90
- kappa_end=1 (n=5): B=5.40+/-1.18 (k*=4), converged loss=0.00022, frac of held phase at exactly k* = 0.40, role_coverage=0.85
- kappa_end=1.5 (n=5): B=4.00+/-0.00 (k*=4), converged loss=0.00066, frac of held phase at exactly k* = 1.00, role_coverage=0.80
- kappa_end=2 (n=5): B=3.00+/-1.24 (k*=4), converged loss=0.25297, frac of held phase at exactly k* = 0.61, role_coverage=0.60
- kappa_end=2.5 (n=5): B=1.40+/-0.78 (k*=4), converged loss=0.65559, frac of held phase at exactly k* = 0.00, role_coverage=0.30
- VERDICT prediction 'emergent B converges on k*': SUPPORTED. kappa_end=1.5 gives B=4.00+/-0.00 against k*=4, with 100% of the held phase at exactly k*. B is monotone in kappa_end over 1.40 to 5.60, so the mechanism does not sit at k* for free: kappa_end selects it.

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

- rho(demand, ablation delta) = +0.369+/-0.059 over 100 runs, at full perfusion
- rho(outnorm, ablation delta) = +0.294+/-0.086 over 100 runs, at full perfusion
- rho(neg_entropy, ablation delta) = +0.478+/-0.075 over 100 runs, at full perfusion
- rho(qnorm, ablation delta) = +0.505+/-0.077 over 100 runs, at full perfusion
- VERDICT the gating signal is not the best predictor of causal head importance. demand is +0.369 while qnorm reaches +0.505 on the same runs. Measured at d_k=32, ABOVE the capacity bound, so this does not contradict the reported qnorm sign flip below it.

## 0 ground truth

- measured k* per seed = [4, 4], predicted = 4; trivial = 1.0075, threshold = 0.02016
- loss by k: k=1: 0.75511, k=2: 0.50314, k=4: 0.00002, k=8: 0.00001, k=16: 0.00001, k=32: 0.00001
