# Induction under random head failure: pilot predictions, written before the runs

Pilot (2 seeds per setting), so this is a check of direction, not a confirmation.

Setup: 2 layers x 8 heads, induction task, same rule as the paper's induction runs
(probe "logit", price_mode "bar", price_frac 0.05, 6000 steps). Each head fails with probability
p at every training step; the probe averages each head's value over 8 random failure patterns.

The "bar" rule keeps a head unless the others hold the expected loss under the bar
(0.05 x ln 64 = 0.21 nats). If every head in a layer fails, the model is at about chance
(~4.2 nats), so with B heads in that layer the expected loss is at least p^B x 4.2, and the
rule needs p^B x 4.2 <= 0.21, that is p^B <= 0.05, in each of the two layers.

| p    | smallest B with p^B <= 0.05 | predicted heads (layer 1 + layer 2) |
|------|-----------------------------|-------------------------------------|
| 0    | -                           | 2 + 1 (as in the paper)             |
| 0.1  | 2                           | 2 + 2                               |
| 0.2  | 2                           | 2 + 2 (borderline: 0.2^2 = 0.04)    |
| 0.3  | 3                           | 3 + 3                               |

The bound ignores the partial loss when some (not all) heads of a layer fail, so the counts
may come out higher. Direction we expect regardless: layer-2 heads (backups of the induction
head) rise with p. Failure of the pilot: no rise with p.
