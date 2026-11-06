# Backup heads: predictions, committed before any of these runs started

Written 2026-11-06, before `results/backup/jobs.txt` and `experiments/backup_robust.py` were
launched. The git commit that adds this file comes before every result file in `results/backup/`.

## Theory (same as `results/confirm/predictions.md`)

Each head fails independently with probability p. Any R working heads do the whole task once the
read-out is re-fitted, and each missing working head costs L_triv / R. The expected value of the
B-th head is

    v(B) = (1 - p) * P( Binomial(B - 1, 1 - p) < R ) / R        (units of L_triv)

which is the one branch of the failure tree where the head works and the others fall short.
v(B) falls with B, so the rule keeps the largest B with v(B) >= price.

## 1. Heads kept, 10 seeds per cell

Seeds 3 to 9 are new for the cells already run with seeds 0 to 2 (`results/confirm`); the other
cells are new for all 10 seeds. k* = 8 has never been run under damage.

| k* | upkeep (L_triv) | p    | predicted heads | next head's value / upkeep |
|----|-----------------|------|-----------------|----------------------------|
| 2  | 0.03            | 0.05 | 3               | 0.11 |
| 2  | 0.03            | 0.1  | 3               | 0.42 |
| 2  | 0.03            | 0.2  | 4               | 0.36 |
| 2  | 0.03            | 0.3  | 4               | 0.98 (borderline) |
| 4  | 0.03            | 0.05 | 5               | 0.18 |
| 4  | 0.03            | 0.1  | 5               | 0.61 |
| 4  | 0.03            | 0.2  | 6               | 0.66 |
| 4  | 0.03            | 0.3  | 7               | 0.74 |
| 4  | 0.01            | 0.05 | 5               | 0.54 |
| 4  | 0.01            | 0.1  | 6               | 0.36 |
| 4  | 0.01            | 0.2  | 7               | 0.67 |
| 8  | 0.01            | 0.1  | 10              | 0.79 |
| 8  | 0.01            | 0.2  | 12              | 0.73 |

Where the next head's value is close to the upkeep (k* = 2, p = 0.3: 0.98), the measured value is
noisy and we expect the count to be predicted + 1 in some runs. The value is an average over 32
random failure patterns per check, so it carries sampling noise of roughly 10 to 20%.

Success: at least 80% of the new runs exactly at the prediction, and every miss within one head.
Failure: counts that do not grow with p, or a mean miss above one head in any cell.

## 2. Do the backups pay off? (`experiments/backup_robust.py`, k* = 4, upkeep 0.03, 10 seeds)

Arms: `aware` (trained with p = 0.2, values heads under failure), `blind` (trained with p = 0.2,
values heads as if nothing failed), `lean` (no failures in training).

Predicted heads kept: aware 6, blind 4, lean 4.

Test damage: each kept head fails with p_test. With the read-out re-fitted to the survivors, the
loss is the expected shortfall, sum_w P(w heads work) * max(0, R - w) / R:

| p_test | 4 heads | 6 heads |
|--------|---------|---------|
| 0      | 0       | 0       |
| 0.1    | 0.100   | 0.004   |
| 0.2    | 0.200   | 0.029   |
| 0.3    | 0.300   | 0.085   |

So at p_test = 0.2 the aware arm should have about 7 times lower re-fitted loss than blind and
lean, and blind should be no better than lean on this measure (the same 4 heads).

Without a re-fit (the model as trained, failed heads switched off) we predict only the order:
aware lowest; blind lower than lean, because training under failure teaches the read-out to cope.

Failure: aware not clearly lower than blind under damage, or blind keeping backups as well.
