# Predictions for the damage (backup-head) runs, written before any result was read

Written 2026-09-30, while the runs in `jobs.txt` were executing and before any of their
outputs were opened.

Assumptions, from the rank argument: trained heads blend all R jobs, so any R working heads
(with the read-out refitted) do the whole task, and each missing working head costs 1/R of
the trivial loss. Each head fails independently with probability p. The collateral rule
keeps a head while its value, averaged over failures, is at least the price.

Value of one of B symmetric heads, in units of the trivial loss:

    value(B) = (1 - p) * P( Binomial(B - 1, 1 - p) <= R - 1 ) / R

Predicted heads kept = the largest B with value(B) >= price (the rule closes heads one at a
time from 32; reopening needs twice the price, so it stops there).

| p    | R = 2, price 0.03 | R = 4, price 0.03 | R = 4, price 0.01 |
|------|-------------------|-------------------|-------------------|
| 0    | 2                 | 4                 | 4                 |
| 0.05 | 3                 | 5                 | 5                 |
| 0.1  | 3                 | 5                 | 6                 |
| 0.2  | 4                 | 6                 | 7                 |
| 0.3  | 4                 | 7                 | 9                 |

(The 0.05 and 0.3 columns at price 0.01 were not run.)

Earlier rough version, now superseded: "backups appear only when p > price x R". It assumed
one head per job; the heads blend jobs, so the number kept grows gradually with p instead
of switching at one point.

What would count against the prediction: heads kept that do not grow with p, or that miss
these values by more than one head on average.
