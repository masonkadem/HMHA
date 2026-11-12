# Cross-task test and logit difference for the IOI heads: written before the run (2026-11-12)

experiments/gpt2_cross.py, frozen GPT-2 small, CPU. Head sets come from results/gpt2_ioi_wang (seeds 0-4).

## A. Logit difference on IOI (a second measure, no hypothesis beyond agreement with accuracy)
Mean (correct-name logit - subject logit) on the 1500 held-out prompts, as a fraction of the full model's:
rule at the three upkeeps, the same number of heads ranked once, the published 26 heads.
Expectation: the rule at upkeep 0.01 recovers at least 80% of the full model's logit difference; ranked once
at 17-18 heads recovers less than 30%.

## B. Other tasks (greater-than and repeated tokens, task seed 0)
1. Keep ONLY the IOI heads the rule kept at upkeep 0.01 (others replaced by their mean on that task):
   accuracy on the other tasks falls well below the full model (the kept heads are a circuit for one task,
   not a smaller general model). Compared with 20 random sets of the same size.
2. Close ONLY those IOI heads, keep the other ~124: I do not have a confident prediction. Head 6.9 and 0.1
   are also kept for greater-than, so some loss is expected. Reported against closing 20 random sets.
Either outcome is reported; B answers the supervisor's question "does pruning hurt other tasks".
