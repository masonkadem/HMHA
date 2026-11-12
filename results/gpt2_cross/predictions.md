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

## Result (2026-11-12, 5 seeds, run.log)
A. Logit difference on IOI, as a fraction of the full model's (full 3.62-3.67):
| | seed 0-4 |
|---|---|
| published 26 heads | 0.85, 0.86, 0.88, 0.88, 0.86 |
| rule, upkeep 0.05 (10-11 heads) | 0.26-0.29 |
| rule, upkeep 0.02 (17-18 heads) | 0.90, 0.92, 1.09, 0.94, 1.07 |
| rule, upkeep 0.01 (20-21 heads) | 1.21, 1.23, 1.21, 1.24, 1.23 |
| ranked once, 17-18 heads | -0.63 to -0.53 |
Expectations hold: rule at 0.01 recovers >= 80% (121-124%); ranked once at 17-18 < 30% (negative).
Above 100%: the rule never keeps the negative name movers, which push against the answer (not tested directly).

B. Other tasks (task seed 0), with the 20-21 heads kept at upkeep 0.01:
- only those heads open: greater-than 42% (random sets of the same size 33-44%); repeated tokens 17.5-19% (random 0%).
  Prediction 1 holds: the kept heads are a circuit for one task, not a smaller general model.
- only those heads closed: greater-than 100% (random 98.5-99.5%); repeated tokens 10.7-12.2%, against
  75.5-78.3% for random sets closed (lowest of 20: 52-68%). Full model 83%.
  The induction heads the rule keeps for IOI (5.5, 5.8, 5.9, 6.9) are the ones repeated-token prediction needs.
