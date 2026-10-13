# HMHA: finding the minimal attention-head circuit during training

A rule inspired by how the brain manages blood flow. Each attention head has a supply
valve. During training, a head is closed when the other heads can take over its job
(as collateral vessels take over when one vessel is pruned), and it fades out gradually
so the others can adjust. The question: does this rule keep exactly the heads the task
needs, without ever breaking the model on the way down?

## Start here

Open [`notebooks/walkthrough_by_hand.ipynb`](notebooks/walkthrough_by_hand.ipynb). It
rebuilds the whole project in small steps: the PyTorch basics, the task, one attention
head, why the task needs exactly R heads, the real model run line by line, the valve,
the collateral value, the rule during training, and every result. Each step has a
by-hand example and a check against the real code. It runs on a CPU in a few minutes.

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace walkthrough_by_hand.ipynb
```

## The task

A memory of N = 16 slots, each holding a random item. A query carries a position p and
must return the items at p + 1, p + 5, p + 9 and p + 13 (mod 16). One attention head
returns one weighted average, which is one equation, so R = 4 offsets need exactly 4
heads. The true circuit size is known, so the count kept by any rule can be scored.

## Main results (from the saved runs, shown in the notebook, section 8)

| rule | runs | kept exactly R heads | never broke the model |
|---|---|---|---|
| collateral rule | 50 | 47/50 | 50/50 |
| ordinary importance (remove the head that hurts least alone) | 30 | 2/30 | 30/30 |
| random choice | 30 | 30/30 | 6/30 |
| standard pruning (Michel et al. 2019) | 36 | 34/36 | 0/36 |
| learned hard-concrete gates (Voita et al. 2019), best penalty | 9 | 3/9 | 0/9 |

When heads can fail at random during training, the rule keeps a predictable number of
backups, written down before the runs (section 9). On a two-layer induction task the
rule recovers the known previous-token plus induction circuit, with a trade-off between
an exact count and stability (section 10).

## Layout

| path | what |
|---|---|
| `src/hemo/` | task (`tasks.py`), model and supply rules (`model.py`), training loop (`train.py`), settings (`config.py`), induction benchmark (`induction.py`) |
| `tests/` | ground-truth checks for the benchmark and the rule: `pytest -q tests/` |
| `experiments/run.py` | one training run; `induction_run.py` the same for induction |
| `experiments/figure_*.py` | the main figures in `figures/` |
| `experiments/build_walkthrough_by_hand.py` | writes the walkthrough notebook |
| `results/` | saved runs used by the notebook and figures; each folder has its `jobs.txt` |
