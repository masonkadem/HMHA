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

## Result (2026-11-10, 5 seeds, 1500 held-out prompts each; run.log)
Full model 99.1-99.8%. Neither method reached 50% within 20 heads in any seed.

| heads removed | one-shot acc | re-valued acc | random acc | one-shot loss | re-valued loss |
|---|---|---|---|---|---|
| 3 | 87-89% | 93-94% | 99% | | |
| 5 | 77-82% | 90-91% | 99% | 2.4-2.8 | 3.6 |
| 10 | 81-83% | 82-84% | 97-98% | 4.7 | 6.2 |
| 20 | 68-70% | 77-80% | 93-96% | | |

Published sets: 3 name movers removed 98.1-99.3%; name movers + 8 backups (11 heads) 85-87%.

1. Self-repair: **holds** (98.1-99.3% after removing the three name movers, 5/5 seeds).
2. Fewer heads to 50%: **fails**. Neither reaches 50%; one-shot accuracy is lower than re-valued at 3, 5 and 20 heads.
3. Higher loss at 5 and 10 heads: **holds** in 5/5 seeds (3.6 vs 2.4-2.8; 6.2 vs 4.7).
4. Picks outside the one-shot top 10: **holds** (9.6, 10.10, 11.6, 10.1 in every seed).

Reading: the target was wrong for the goal. Re-valuing was told to raise the loss on the correct name, and it
did (prediction 3). It did so by removing the heads that write names to the output (name movers and their
backups: 10.0, 9.9, 9.6, 10.10, 11.6, 10.1). That lowers both names together, so the correct name still wins
the comparison. One-shot happened to include 5.5 and 7.9, which decide WHICH name is written, and so hurt
accuracy more. The ability was not removed by either method.

## Second test: target the comparison itself (written 2026-11-10, after the result above, before this run)
Same script with --target margin: removal now maximises minus (correct-name logit - subject logit) on the fit
prompts, for both one-shot ranking and re-valuing. Files remove_margin_s*.pkl. This test was designed after
seeing the first result and should be reported as such.

1. Re-valuing on the margin reaches 50% or below within 20 heads in every seed.
2. It needs fewer heads than one-shot ranking on the margin in at least 4 of 5 seeds.
3. Its first 5 removals include at least one S-inhibition head (7.3, 7.9, 8.6, 8.10) in every seed.
Failure: re-valuing does not reach 50% within 20 heads, or one-shot needs as few heads in 3 or more seeds.
