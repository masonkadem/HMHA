# Shakespeare pilot (not part of the synthetic benchmark)

Exploratory runs on a 4-layer, 4-head character transformer. Kept apart from `src/hemo`
because the main repo is synthetic only. The text is not committed: place tiny Shakespeare
(`input.txt`, from karpathy/char-rnn) in `data/` next to the scripts. One seed per process.

| script | question |
|---|---|
| `shakes.py` | head failure, element dropout or none during training: head importance, the rule during training |
| `shakes2.py` | dose-response in the failure rate; which cheap score predicts the loss after the network adapts |
| `shakes3.py` | three seeds, undamaged: own-layer re-fit against zero ablation, as score and as pruning with compensation |
| `shakes4.py` | control: mean ablation |
| `shakes5.py` | longer fine-tuning and a gradient baseline |

`results/` holds the JSON output of each run.
