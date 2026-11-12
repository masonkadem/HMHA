# Sensitivity of the rule to its untuned settings: predictions, written before the runs (2026-11-12)

experiments/run.py, retrieval task, k* = 4, seeds 0-2, --skip-redundancy. Everything as in the paper
(supply local, upkeep 0.03 L_triv, fade 100 steps, warm-up 10%, check every 25 steps, conserve 0) except one setting:

  fade (taper)          25, 400 steps        (paper: 100)
  warm-up               5%, 25% of training  (paper: 10%)
  check interval        10, 50 steps         (paper: 25)

18 runs. Scored as in the paper: exact count (open heads over the last 40% of training equal 4) and stayed solved.

1. Exact count in at least 16 of 18 runs.
2. Stayed solved in at least 16 of 18 runs.
3. If anything fails, it is the shortest fade (25 steps), which is closest to the no-fade ablation that lost the
   solution in every run.
Failure: either count below 14 of 18. Then the paper says the settings matter and names which.

## Result (2026-11-12, 18 runs, run.log)
| setting | exact count | stayed solved |
|---|---|---|
| fade 25 steps | 3/3 | 2/3 |
| fade 400 steps | 3/3 | 3/3 |
| warm-up 5% | 3/3 | 3/3 |
| warm-up 25% | 3/3 | 3/3 |
| check every 10 steps | 3/3 | 3/3 |
| check every 50 steps | 3/3 | 3/3 |
Total: exact count 18/18, stayed solved 17/18. All final losses below 0.0002 L_triv.
1. Exact count in at least 16 of 18: holds (18).
2. Stayed solved in at least 16 of 18: holds (17).
3. Any failure is the shortest fade: holds (the one loss is fade 25, seed 0).
