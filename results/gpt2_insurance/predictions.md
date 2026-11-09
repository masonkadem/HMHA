# Overnight predictions, written before the runs (2026-11-09)

## Direction 1: backups as insurance against inputs on which a head fails (experiments/gpt2_insurance.py)

Frozen GPT-2 small, 600 held-out IOI prompts, no ablation. A head "works" on a prompt when, at the final
position, it attends more to the indirect object than to either copy of the subject.

1. Each of the three name movers (9.9, 9.6, 10.0) fails on a non-trivial share of prompts (> 5%).
2. Insurance: the backup movers' summed direct logit attribution is larger on prompts where more name
   movers fail (rises from 0 to 1 to 2 failing name movers).
3. Across prompts, name movers' and backups' attributions are negatively correlated.
4. The backups kept by hemodynamic attenuation at the highest upkeep (10.10, 11.2) are among the backups
   with the highest empirical tree value P(backup works and too few name movers work).

Failure: backups' attribution does not rise when name movers fail (no insurance on natural inputs).

## Direction 2: one-shot head scores are biased by backups (experiments/gpt2_core.py)

Tasks: IOI, greater-than, repeated random tokens. Upkeep 0.05, 0.02, 0.01 of the closed-open loss gap.

1. In every task, at least one head that the rule keeps has a one-shot value (loss rise when removed
   from the full model) below the upkeep, i.e. a one-shot rule would have dropped it ("masked" heads).
2. In every task, the rule's heads score higher on held-out prompts than the one-shot top-n heads and
   than random sets of the same size.
3. In every task the kept sets are nested across the three upkeeps.

Failure: no masked heads, or one-shot top-n as good as the rule, in two of the three tasks.
