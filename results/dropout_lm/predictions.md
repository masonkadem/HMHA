# Ordinary dropout and head redundancy in a small language model: pilot prediction

Written before the runs. A 5-10 minute pilot, 2 seeds per setting: a check of direction only.

Setup: character-level transformer on Tiny Shakespeare, 2 layers x 8 heads, d_model 128,
trained with ordinary dropout p in {0, 0.1, 0.2, 0.3} (attention weights and residual
outputs, as in GPT-2). Not head dropout: whole heads are never switched off in training.
After training, the model is frozen and the rule is run on its 16 heads: close a head while
its value (rise in held-out loss when it is switched off) is below the upkeep, reopen if the
gain exceeds 2 x upkeep. Upkeep = 0.02 x (loss with all heads off - loss with all on).

Prediction: the number of heads kept rises with p. Dropout makes any single path unreliable,
so a model trained with more dropout should spread the work over more heads, the same way
head failure produced more backups. No exact count is predicted: ordinary dropout does not
fail whole heads, so the failure tree does not apply directly.

Failure of the pilot: heads kept do not rise with p.
