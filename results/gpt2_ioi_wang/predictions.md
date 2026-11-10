# IOI on the prompts of Wang et al., 5 seeds: predictions, written before the runs (2026-11-10)

experiments/gpt2_core.py --tasks ioiwang. Frozen GPT-2 small. Prompts from the published generator
(experiments/ioi_wang_data.py): 15 templates in both name orders, 99 names, 8 places, 8 objects.
Fit on 300 prompts, tested on 1500 held-out prompts, closed heads mean-ablated on three-name prompts.
Seeds 0-4 (different prompt samples), upkeep 0.05, 0.02, 0.01 of the closed-open loss gap. 15 runs.
Only a 3-decision smoke test was run before writing this (full model 99.3%, published circuit 99.0%).

1. Accuracy: at upkeep 0.01 the rule keeps fewer than 26 heads and is within 3 points of the published
   26-head circuit on the same held-out prompts, in at least 4 of 5 seeds.
2. Overlap: at every upkeep, on average at least 75% of the kept heads are in the published circuit.
3. Stability: at each upkeep, the heads kept in at least 4 of 5 seeds make up at least 70% of each run's
   kept set. The three name movers (9.6, 9.9, 10.0) are kept in all 15 runs; the two negative name movers
   (10.7, 11.10) in none.
4. Nesting: kept sets are nested across the three upkeeps in at least 4 of 5 seeds.
5. Re-valuing: at upkeep 0.02 and 0.05 the rule's heads beat the one-shot top-n heads in every seed.

Failure: more than 5 points below the published circuit at upkeep 0.01 in 3 or more seeds, or fewer than
half of the kept heads recurring across seeds.
