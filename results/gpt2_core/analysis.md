# Direction 2 results (overnight run finished 2026-11-10)

Predictions: results/gpt2_insurance/predictions.md (committed before the run, a3c1cb9).
Frozen GPT-2 small, seed 0, upkeep = 0.05 / 0.02 / 0.01 of the closed-open loss gap.
Held-out accuracy of the same number of heads chosen three ways.

| task | upkeep | heads | rule | one-shot top-n | random (mean) | masked heads |
|---|---|---|---|---|---|---|
| greater-than | 0.05 | 3 | 96% | 96% | 27% | 2 |
| | 0.02 | 5 | 99% | 96% | 29% | 4 |
| | 0.01 | 6 | 99% | 98% | 27% | 3 |
| IOI | 0.05 | 11 | 69% | 9% | 48% | 7 |
| | 0.02 | 17 | 89% | 9% | 49% | 6 |
| | 0.01 | 22 | 97% | 57% | 43% | 4 |
| repeated tokens | 0.05 | 11 | 18% | 2% | 0% | 11 |
| | 0.02 | 23 | 73% | 27% | 0% | 20 |
| | 0.01 | 29 | 83% | 56% | 0% | 19 |

"Masked" = kept by the rule, but its one-shot value (loss rise when removed from the full model) is below the upkeep.

## Against the predictions
1. Masked heads in every task: **yes** (2-20 per run).
2. Rule beats one-shot and random in every task: **2 of 3**. Clear on IOI and repeated tokens;
   on greater-than the one-shot ranking is as good (within 3 points, 100 test prompts, so a tie).
3. Kept sets nested across upkeeps in every task: **yes** (3/3).

Pre-registered failure condition (no masked heads, or one-shot as good, in two of three tasks): **not met**.
IOI numbers reproduce the earlier results/gpt2_ioi runs exactly (same kept sets).

## Reading
Re-valuing after each step matters when the task uses a multi-step circuit with backups (IOI, induction-like
repeated tokens). For a task carried by a few late heads (greater-than: 7.10, 8.11, 9.1), ranking once is enough.
Single seed, single prompt set; the one-shot baseline is top-n by the first-step values.
