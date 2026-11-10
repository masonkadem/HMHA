# Removing an ability so that backups cannot restore it: predictions, written before the runs (2026-11-10)

experiments/gpt2_remove.py. Frozen GPT-2 small, IOI on the prompts of Wang et al. (as results/gpt2_ioi_wang),
seeds 0-4. A removed head is replaced by its mean output on three-name prompts. Goal: bring held-out
accuracy (correct name over the subject) to 50% or below, chance for this pair, with as few removed heads
as possible. Heads are chosen on 300 fit prompts and scored on 1500 held-out prompts.

Ways of choosing which heads to remove, for k = 1..20:
  one-shot    the k heads whose single removal from the full model raises the loss most
  re-valued   one at a time: each time, the head whose removal raises the loss most GIVEN those already removed
  published   the three name movers; then name movers plus the eight backup name movers
  random      20 random sets of k heads

1. Self-repair: removing the three name movers alone leaves accuracy above 80% in every seed.
2. Re-valuing needs fewer heads than one-shot to reach 50% or below, in at least 4 of 5 seeds.
3. At equal k (5 and 10 heads removed), held-out loss is higher for re-valued than for one-shot in every seed.
4. Within its first 10 removals, re-valuing picks at least one head that is not in the one-shot top 10.

Failure: one-shot reaches 50% with as few heads as re-valuing in 3 or more seeds.

Limit, stated in advance: this tests re-routing through frozen weights only. It does not test whether
fine-tuning brings the ability back, which is what "stays removed" needs in the end.
