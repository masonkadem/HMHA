# Running this locally

The compute in this repo is small but not trivial: a single 4000-step run is about
3.4 minutes on four CPU cores and roughly 20 to 40 seconds on one modern GPU. The full
remaining sweep is 189 runs. On CPU that is most of a day; on one GPU it is under an hour.
Move it to a GPU if you have one.

## Setup

```bash
git clone <your-remote> hemo-attn && cd hemo-attn
git checkout claude/latest-experiment-ideas-fualsp

python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt                        # torch, numpy, matplotlib, pytest
```

For a GPU, install the CUDA build of torch for your driver instead of the default wheel.
`pick_device` auto-detects CUDA and then MPS (Apple Silicon), so nothing else changes.

## Always run the tests first

```bash
pytest -q tests/
```

Five tests, about 20 seconds. Two of them assert the benchmark's ground truth: that one
head fails the task and two heads solve it, and that a trained model's heads implement
distinct offsets. **If either fails, no downstream number means anything.** The others
assert supply conservation for every mechanism and the territory threshold's scale
invariance.

## One run

```bash
python experiments/run.py --seed 0 --supply autoreg --stall_gate 0.02 --n_rel 4
```

Every field of `Cfg` is exposed as a flag automatically, so anything in
`src/hemo/config.py` can be set from the command line. Results land in
`results/<tag>.pkl`, where the tag encodes every swept field. Add `--device cuda` to force
a device, `--skip-redundancy` to skip the dense-from-scratch k* check (which trains six
extra models and roughly quadruples the cost).

## The sweep

```bash
python experiments/sweep.py --list                       # what exists, what is left
python experiments/sweep.py --run --workers 8            # resumable
python experiments/sweep.py --run --workers 8 --only stall_eps,slowgain_ctl
```

It skips any job whose pickle already exists, so it is safe to interrupt and restart. Use
`--workers` equal to your core count on CPU. On a GPU this model is small enough that 8 to
16 concurrent processes keep one card busy, so oversubscribe.

**Priority order for what is left**, most informative first:

| `--only` | jobs | what it decides |
|---|---|---|
| `stall_eps` | 12 | is the stall threshold itself tuned? Proposition 2 says no |
| `slowgain_ctl` | 6 | is the stall gate just a slower loop? the sharpest critique |
| `watershed_auto`, `watershed_compete`, `compaction` | 36 | can the spatial claim be revived |
| `qsa_kstar` | 24 | second benchmark, k\* from capacity not function |
| `arm_reserve`, `arm_reserve_auto`, `baseline_gradient` | 27 | the novel sensor vs its baseline |
| `sensors`, `arm_marginal`, `arm_poiseuille` | 73 | lower value, run last |

The first two are 18 runs and settle the only open question in the paper's limitations
section. Start there.

## Do not shorten the runs

`steps` is load-bearing. Measured, with the controller gain scaled to hold control
authority constant: at 400 steps every head stays perfused at every circuit size (slope
-0.09) because the task is not learned yet and the loop never sees a reason to constrict;
at 1000 steps it over-perfuses by one to two heads; 4000 tracks exactly. Note these are
optimizer steps on a freshly generated batch each time, not epochs, so 4000 steps is less
than one epoch in the usual sense.

## Figures, tables and the paper

```bash
python experiments/aggregate.py --results results --out figures --readme README.md
python experiments/paper_assets.py
```

The first writes `figures/*.png`, `figures/summary.md`, and the generated block in
`README.md`. The second writes the paper's two tables and its comparison figure. Between
them, every number in `paper/` is computed from the pickles: `results_table.tex`,
`target_table.tex`, and `numbers.tex` (LaTeX macros for the figures quoted in prose).
Nothing is transcribed, which is the repo's standing rule.

```bash
cd paper && pdflatex chapter.tex && pdflatex chapter.tex   # twice, for refs
pdflatex main.tex && pdflatex main.tex                     # the conference draft
```

`figures/fig1_task.png` additionally needs a cached trained model:

```bash
python experiments/task_panel.py      # about 10 minutes on CPU, writes figures/task_panel.pkl
```

## Layout

```
src/hemo/config.py       one Cfg dataclass, every knob
src/hemo/tasks.py        multi_relation, qsa_cross, predicted k*
src/hemo/model.py        CrossAttn, HemoAttn; demand and supply are separate axes
src/hemo/train.py        schedule, train loop, the autoregulation controller
src/hemo/analysis.py     redundancy_check, circuit_recovery, head_ablation
experiments/run.py       one seed per invocation
experiments/sweep.py     the roadmap as a resumable job list
experiments/aggregate.py figures and the README block
experiments/paper_assets.py  the paper's tables, figure and prose macros
paper/                   chapter.tex, main.tex, and the generated .tex fragments
tests/                   ground truth. Run these first.
```

## Conventions worth knowing before you change anything

Each was adopted after a measurement error, and each is enforced somewhere.

1. **Ablation runs at full perfusion.** At a starved gate most heads sit at zero and every
   rank correlation becomes a tie artefact.
2. **Thresholds are a fraction of the dense-to-trivial gap**, never of the dense loss
   alone, which becomes unreachable once a task is solved exactly.
3. **Where B emerges, report the median over the held phase**, not the final step. A
   closed loop's last gate is one sample of its trajectory.
4. **Any field a sweep varies must be in `TAG_FIELDS` or `TAG_OPTIONAL`** in
   `experiments/run.py`, or two arms silently overwrite each other's pickle. This has
   bitten four times.
5. **Add every new supply mechanism to `test_supply_is_conserved`.** Conservation is the
   property that separates this from pruning; it is an assertion, not a comment.
