# IOI on the prompts of Wang et al., 5 seeds: results (2026-11-10)

Predictions: predictions.md in this folder (committed before the runs, 883e590).
Frozen GPT-2 small; 30 templates, 99 names; fit on 300 prompts, tested on 1500 held-out prompts per seed.
Full model 99.1-99.8% correct; the published 26-head circuit 99.0-99.4% on the same prompts.

| upkeep | heads kept (5 seeds) | correct name, rule | one-shot top-n | random | in published circuit | kept in all 5 seeds |
|---|---|---|---|---|---|---|
| 0.05 | 10, 10, 11, 11, 10 | 67-69% | 10-13% | 37-46% | 100% | 10 heads |
| 0.02 | 17, 17, 18, 17, 18 | 94-97% | 8-13% | 38-42% | 94% | 17 heads |
| 0.01 | 20, 20, 21, 20, 20 | 98-99% | 33-38% | 35-42% | 94% | 19 heads |

Heads kept in all 5 seeds at upkeep 0.01:
0.1 2.2 3.0 4.11 5.5 5.8 5.9 6.9 7.9 8.6 8.10 9.6 9.9 10.0 10.2 10.6 10.10 11.2 11.3
(11.3 is the one head outside the published circuit that is kept every time.)

## Against the predictions: all five hold
1. Accuracy at upkeep 0.01: 20-21 heads, 0.6 to 1.3 points below the published 26 heads, in 5/5 seeds.
2. Overlap with the published circuit: 100%, 94%, 94% (predicted at least 75%).
3. Stability: 90-100% of each run's heads are kept in at least 4 of 5 seeds (predicted at least 70%).
   Name movers 9.6, 9.9, 10.0 kept in 15/15 runs; negative name movers in 0/15.
4. Nested across the three upkeeps in 5/5 seeds.
5. Rule beats one-shot top-n at upkeep 0.02 and 0.05 in every seed (94-97% vs 8-13%; 67-69% vs 10-13%).

## Differences from the earlier run on our own six templates
Counts are 10-11 / 17-18 / 20-21 here against 11 / 17 / 22. Accuracy at 17-18 heads is 94-97% here against 89%.
The paper's GPT-2 paragraph should be rewritten on these numbers.
