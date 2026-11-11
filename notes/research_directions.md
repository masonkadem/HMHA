# Open research directions

Ideas and pilot status, kept separate from the main claim. Each entry says what has been
run and what has not. No result here is established.

## 1. Functional connectivity of attention heads

Treat each head as a "region" and its activity over tokens as a time series, then analyse
it the way resting-state fMRI analyses brain networks.

Definitions used (`experiments/shakespeare_pilot/fc_heads.py`):
- signal: the size (L2 norm) of what a head writes into the residual stream at each token
- scales: per token, and averaged over 8 and 32 tokens
- connectivity: Pearson correlation between every pair of heads
- reliability: ICC(3,1) of the pairwise correlations across four segments of text

Pilot results, 4-layer 4-head character model on Shakespeare, three seeds per model:
- Co-activity patterns are reproducible: ICC 0.84 to 0.92 per token.
- Without layer norm, late-layer heads move together (mean r 0.20 to 0.31); with layer
  norm they do not (0.02 to 0.09). Removing the residual-stream size cuts the block by a
  little over half (`fc_check.py`). The rest is unexplained.
- Removing the global signal makes 73% to 82% of pairs negative, the same artefact as
  global signal regression in fMRI.
- Before any removal, about 10 of 120 head pairs are reliably anticorrelated with layer
  norm and 2 to 5 without, against a circular-shift null (`fc_neg.py`). Most are between
  layers. There are no two opposed networks of the kind reported in fMRI.
- Sync barely predicts which heads back each other up (rank correlation +0.26).

Not done: what drives the anticorrelated pairs (token type is the first thing to check);
other activity signals (attention entropy, effect on the output); any model larger than
16 heads.

## 2. Do transformers "slow down" on ambiguous words?

Question: humans read ambiguous words (homographs such as "bank", "bat", "lead") more
slowly when the context does not settle the meaning. Does a transformer show an analogue?

A standard transformer spends the same compute on every token, so it cannot literally
slow down. Candidate measures of "more processing":
1. Settling depth: the layer from which the next-token prediction stops changing, read
   with the logit lens (unembed each layer's residual stream). Prediction: later for an
   ambiguous word in neutral context than for the same word in disambiguating context.
2. Representation change per layer: how far the word's residual vector moves between
   consecutive layers. Prediction: larger and later movement when ambiguous.
3. Attention: entropy of the heads reading from the ambiguous word, and how much
   attention it pays back to the disambiguating context word.
4. Surprisal and prediction entropy at the next token, which are known to predict human
   reading times and are the baseline any new measure must beat.
5. In a model with adaptive depth (early exit), the actual number of layers used.

Design: matched sentence pairs from the psycholinguistics literature, same homograph with
context before it that is neutral, biased to the dominant meaning, or biased to the
subordinate meaning. The human "subordinate bias effect" (slowest when context favours
the rarer meaning) gives a specific prediction to test.

Feasibility: needs a word-level pretrained model (GPT-2 small runs on CPU). The character
model here is not suitable. Hugging Face is blocked from the current cloud environment,
so this needs either a network allowance for huggingface.co or a run on another machine.

Prior work to check before starting: surprisal and reading times; layer-wise word-sense
probing and contextual embeddings (Ethayarajh 2019); logit lens and tuned lens; early-exit
transformers.
