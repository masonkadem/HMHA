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
