# Grokking pilot: predictions for seeds 1 to 3, written before those runs

Written 2026-11-10 after looking at seed 0 only (`results/grok_P97_f0.4_H8_s0.json`), and
committed before seeds 1 to 3 were started. Seed 0 is exploratory and is not counted as
evidence for any of these.

Setting: (a + b) mod 97, 40% of pairs for training, one layer, 8 heads, weight decay 1.0,
20000 full-batch steps (`grok.py`). Each head is scored every 100 steps by zero ablation,
mean ablation and the own-layer re-fit, on training pairs not used for the fit and on test
pairs. "Sum" is the sum over the 8 heads of the loss rise. "Grokking step" is the first
step with test accuracy at least 0.5.

What seed 0 showed: test accuracy reached 0.5 at step 8700. The train-side sums fell well
before that. After generalisation the re-fit sum went to zero while the ablation sums did
not. On test pairs, removing a head and re-fitting LOWERED the loss throughout memorisation.

Predictions:

1. **Train-only early signal.** The train-side re-fit sum falls below half of its
   memorisation plateau (its mean over steps 1000 to 3000) at least 500 steps before the
   grokking step. Counts against: it crosses later than that, or never.
2. **Same for zero ablation**, stated separately so the two can be compared: if zero
   ablation leads by as much as the re-fit, the re-fit adds nothing as an early signal.
3. **Signature of the generalising solution.** Averaged over the last 2000 steps, the
   train-side ratio (re-fit sum / zero-ablation sum) is below 0.05; averaged over steps
   1000 to 3000 it is between 0.2 and 0.5. Counts against: a final ratio above 0.1.
4. **Sign change on test pairs.** The test-side re-fit sum is negative on average over
   steps 1000 to 3000 and becomes positive at least 500 steps before the grokking step.

Baselines that must be reported beside these: the weight norm and the test loss itself.

If a seed does not reach test accuracy 0.5 within 20000 steps it is reported as such and
predictions 1, 2 and 4 are not scored for it.
