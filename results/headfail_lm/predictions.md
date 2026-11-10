# Whole-head failure in a real language model: predictions, written before the runs (2026-11-10)

experiments/headfail_lm.py. Character-level transformer on Tiny Shakespeare, 2 layers x 16 heads, no ordinary
dropout. During training every head is switched off independently with probability p (per sequence, per step),
p = 0, 0.1, 0.2, 0.3, 3 seeds. Then, frozen, the rule (upkeep = 0.05 of the closed-open loss gap) decides which
heads stay, valued two ways on the same model:
  aware  loss averaged over random failures at the training rate p
  blind  loss with nothing failing

1. Aware heads kept (mean over seeds) rises with p: p=0 < 0.1 < 0.2 < 0.3.
2. At every p > 0, aware keeps more heads than blind.
3. Rough count (expected to be a lower bound, as in the induction pilot): with R = heads kept at p = 0 and
   v = mean value of those heads at the stopping point, aware keeps about the largest B with
   (1-p) * P[Bin(B-1, 1-p) < R] * v >= upkeep. Within 2 heads in at least half the runs.

Failure: aware count flat or falling with p, or aware no larger than blind.
Contrast: with ordinary dropout the count fell as p rose (results/dropout_lm).
