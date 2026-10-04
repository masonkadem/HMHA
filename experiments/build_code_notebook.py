"""Writes notebooks/the_code.ipynb: only the project's real code, in the order it runs (settings, task,
model, valve and collateral rule, training, one experiment, the results), printed from src/ and
experiments/ so it is identical to what ran, each followed by a cell that runs it.

  python experiments/build_code_notebook.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/the_code.ipynb
"""
import os
import nbformat as nbf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# The code, start to finish

Only the code that produces the results, in the order it runs. Every listing is printed from the
repo files (`src/hemo/`, `experiments/`), so it is exactly what ran. After each listing, a cell runs
it on a small version of the task (8 slots, 2 distances, 8 heads) so you can see what it produces.

| step | what | file |
|---|---|---|
| 1 | the settings | `src/hemo/config.py` |
| 2 | the task | `src/hemo/tasks.py` |
| 3 | the model | `src/hemo/model.py`, `CrossAttn` |
| 4 | the valves and the collateral rule | `src/hemo/model.py`, `HemoAttn` |
| 5 | training | `src/hemo/train.py` |
| 6 | one experiment | `experiments/run.py` |
| 7 | the results | `results/`, `figures/` |

For the same steps explained slowly and by hand, see `walkthrough_by_hand.ipynb` and `hmha_by_hand.xlsx`.
""")

code(r"""
import glob, inspect, pickle, sys, textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, display
from dataclasses import replace

sys.path.insert(0, "../src")
from hemo.config import Cfg
from hemo.tasks import make_batch, make_val, offsets_for, trivial_loss
from hemo.model import CrossAttn, HemoAttn
from hemo.train import train
torch.set_num_threads(4)


def _cut(lines, start=None, end=None):
    if start:
        lines = lines[next(i for i, l in enumerate(lines) if start in l):]
    if end:
        lines = lines[:next(i for i, l in enumerate(lines) if end in l) + 1]
    out, inside = [], False
    for l in lines:                                   # leave docstrings out
        t = l.strip()
        if not inside and t.startswith('\"\"\"'):
            inside = not (t.endswith('\"\"\"') and len(t) > 3)
            continue
        if inside:
            inside = not t.endswith('\"\"\"')
            continue
        out.append(l)
    print(textwrap.dedent("\n".join(out)))


def show(obj, start=None, end=None):                # print a function or class from the repo
    _cut(inspect.getsource(obj).splitlines(), start, end)


def show_file(path, start=None, end=None):          # print part of a file from the repo
    _cut(open(path, encoding="utf-8").read().splitlines(), start, end)


SMALL = replace(Cfg(), seq_len=8, n_rel=2, m_content=4, d_model=16, num_heads=8, d_k=8,
                steps=2000, batch_size=128, val_size=512, val_every=100, device="cpu", seed=0)
print("ready")
""")

# ---------------------------------------------------------------- 1 settings
md(r"""
## 1. The settings (`src/hemo/config.py`)

One settings object, `Cfg`, holds every number. These are the lines this project uses (the file
also holds settings for older rules that are switched off). `SMALL` above is a copy with smaller numbers.

The values printed are the file's **defaults**. The collateral-rule experiments override five of them
on the command line (step 6): `supply=local` (the collateral rule), `price_frac=0.03`, `taper=100`,
`budget_hold_frac=0.1` (first decision after 10% of training) and `conserve=0`.
""")
code(r"""
used = ("d_model:", "seq_len:", "n_rel:", "m_content:", "num_heads:", "d_k:", "supply:", "price_frac:",
        "probe_every:", "probe_batch:", "taper:", "budget_hold_frac:", "conserve:", "local_value:",
        "batch_size:", "steps:", "lr:", "seed:")
for line in inspect.getsource(Cfg).splitlines():
    if line.strip().startswith(used):
        print(line.strip())
""")
code(r"""
for name in ("seq_len", "n_rel", "m_content", "d_model", "num_heads", "d_k", "steps"):
    print(f"{name:<10} SMALL {getattr(SMALL, name):>5}   real {getattr(Cfg(), name):>5}")
""")

# ---------------------------------------------------------------- 2 task
md(r"""
## 2. The task (`src/hemo/tasks.py`)

Memory `Y`: one row per slot, `[one-hot label | random item | noise]`. Queries `X`: one row per query,
`[one-hot of p | noise]`. Target `T`: the items at `(p + offset) mod N`, side by side.
""")
code(r"""
show(offsets_for)
show(make_batch, 'elif cfg.task == "multi_relation"', 'aux = {"p": p')
show(make_val)
show(trivial_loss)
""")
code(r"""
X, Y, T, aux = make_batch(2, SMALL, "cpu", gen=torch.Generator().manual_seed(0))
print("X", tuple(X.shape), " Y", tuple(Y.shape), " T", tuple(T.shape), "  offsets", offsets_for(SMALL))
N, m = SMALL.seq_len, SMALL.m_content
p0 = int(aux["p"][0, 0])
for r, off in enumerate(offsets_for(SMALL)):        # check one answer by hand
    j = (p0 + off) % N
    print(f"query 0: p = {p0}, ({p0} + {off}) mod {N} = {j}, target block {r} = item in slot {j}:",
          torch.equal(T[0, 0, r * m:(r + 1) * m], Y[0, j, N:N + m]))
""")

# ---------------------------------------------------------------- 3 model
md(r"""
## 3. The model (`src/hemo/model.py`, `CrossAttn`)

One cross-attention layer, no MLP. All heads' weights are stored together and cut into heads by
`_split`. `heads` computes every head's blend; `combine` multiplies each head's blend by its valve
(`out * gate`) and applies the output weights.
""")
code(r"""
show(CrossAttn)
""")
code(r"""
torch.manual_seed(0)
layer = CrossAttn(SMALL)
Q, attn, out = layer.heads(X, Y)
pred, g = layer(X, Y)
print("Q", tuple(Q.shape), " attention", tuple(attn.shape), " each head's blend", tuple(out.shape),
      " prediction", tuple(pred.shape), " valves", tuple(g.shape))
""")

# ---------------------------------------------------------------- 4 valve and rule
md(r"""
## 4. The valves and the collateral rule (`src/hemo/model.py`, `HemoAttn`)

`HemoAttn` is `CrossAttn` with valves controlled by a rule. With `supply="local"` (the collateral rule):
- `gate`: each head's valve is its tone, 1 = open, 0 = closed, in between while fading;
- `probe_ischemia`: each head's collateral value, the loss rise if it is removed and the other heads
  re-fit the output weights (one matrix inverse, then a small correction per head);
- `local_step`: close the cheapest head if it is worth less than the price; reopen a closed head
  worth more than twice the price;
- `relax_tone`: move each valve toward open or closed by 1/taper per step.
""")
code(r"""
show(HemoAttn.forward)
show(HemoAttn.gate, 'if self.supply_kind == "local"', "return self.H * self.tone")
""")
code(r"""
show(HemoAttn.probe_ischemia, "H, dk = self.H", "self.head_value.copy_(value")
""")
code(r"""
show(HemoAttn.local_step)
show(HemoAttn.relax_tone)
""")

# ---------------------------------------------------------------- 5 training
md(r"""
## 5. Training (`src/hemo/train.py`)

The training loop with the lines the collateral rule uses: every step, a fresh batch, a forward pass,
a gradient step, and the valves move toward their targets; every 25 steps after the warm-up (`hold`),
value the heads and close or reopen one. (The full function also runs older rules and controls.)
""")
code(r"""
keep = ("def train(", "model = (HemoAttn", "opt = torch.optim", "hold = int(", "local = hemo and",
        "price = cfg.price_frac", "for step in range(total)", "X, Y, T, _ = make_batch(cfg.batch_size",
        "pred, g = model(X, Y", "loss = F.mse_loss(pred, T)", "opt.zero_grad", "loss.backward()",
        "opt.step()", "if local:", "model.relax_tone()", "if (step - hold) % cfg.probe_every == 0",
        "model.probe_ischemia(*make_batch", "model.local_step(price)", 'h["ledger"].append(g[0]',
        'h["val_loss"].append(evaluate', "return model, h")
seen = set()
for line in inspect.getsource(train).splitlines():
    if any(k in line for k in keep) and line.strip() not in seen:
        seen.add(line.strip())
        print(line)
""")

# ---------------------------------------------------------------- 6 one experiment
md(r"""
## 6. One experiment (`experiments/run.py`)

Every result in `results/` is one call of this script, e.g.
`python experiments/run.py --n_rel 4 --seed 0 --supply local --price_frac 0.03 --taper 100 --budget_hold_frac 0.1 --conserve 0`.
It builds the settings from the command line, trains with the rule, and saves everything to a `.pkl`.
""")
code(r"""
show_file("../experiments/run.py", "cfg = Cfg()", 'out["final_loss"]')
""")
md(r"""
The same thing, run here on the small task. First the dense model with 1 and 2 heads (the task needs
2), then the collateral rule starting from 8 heads, three seeds.
""")
code(r"""
val = make_val(SMALL, "cpu")
triv = trivial_loss(val)
for H in (1, 2):
    _, hist = train(SMALL, val, "cpu", hemo=False, num_heads=H)
    print(f"dense, {H} head(s): final loss {hist['val_loss'][-1]:.4f}   (know-nothing loss {triv:.3f})")
""")
code(r"""
RULE_SMALL = replace(SMALL, supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.2,
                     conserve=0, probe_batch=256)
runs = []
for seed in range(3):
    model, hist = train(replace(RULE_SMALL, seed=seed), val, "cpu", hemo=True)
    runs.append(hist)
    kept = (hist["ledger"][-1] > 0).nonzero()[0].tolist()
    print(f"collateral rule, seed {seed}: heads kept {kept} ({len(kept)} of 8, task needs 2), "
          f"final loss {hist['val_loss'][-1]:.4f}")

hist = runs[0]
fig, ax = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True)
ax[0].semilogy(hist["val_step"], hist["val_loss"], color="#b2182b", lw=0.8); ax[0].set_ylabel("loss")
ax[1].plot((hist["ledger"] > 0).sum(1), color="#b2182b"); ax[1].axhline(2, color="#888", ls=":")
ax[1].set_ylabel("heads open"); ax[1].set_xlabel("training step")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 7 results
md(r"""
## 7. The results of the full experiments

The saved runs (`results/`, 16 slots, 32 heads, 4000 steps), read back and counted. *Exact count*:
the heads kept equal the number the task needs. *Never broke*: once solved, the loss never went back
above the solved bar.
""")
code(r"""
D = Cfg()
RUNS = [pickle.load(open(f, "rb")) for d in ("../results", "../results/proposal", "../results/confirm", "../results/l0")
        for f in glob.glob(d + "/*.pkl")]
RUNS = [r for r in RUNS if isinstance(r, dict) and "hist" in r and "ledger" in r.get("hist", {})]
PIN = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0, probe_every=25,
           prune_stop=1, conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0,
           trial=0, target_frac=0.02, budget_hold_frac=0.25, taper=0, price_frac=0.01, l0_lambda=0.01)
get = lambda r, k: r["cfg"].get(k, getattr(D, k))
runs_of = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PIN, **w}.items())]
kept = lambda r: float(np.median(((r["hist"]["ledger"] > 0).sum(1))[int(0.6 * len(r["hist"]["ledger"])):]))


def never_broke(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
arms = {"collateral rule": (RULE, (2, 3, 4, 6, 8)),
        "ordinary importance": ({**RULE, "local_value": "ablate"}, (2, 4, 8)),
        "random choice": ({**RULE, "local_value": "random"}, (2, 4, 8)),
        "standard pruning (Michel et al.)": (dict(supply="prune"), (2, 3, 4, 6, 8))}
print(f"{'rule':<34}{'runs':>5}{'exact count':>13}{'never broke':>13}")
for name, (kw, sizes) in arms.items():
    rs = [r for k in sizes for r in runs_of(**kw, n_rel=k)]
    print(f"{name:<34}{len(rs):>5}{sum(kept(r) == get(r, 'n_rel') for r in rs):>9}/{len(rs)}"
          f"{sum(never_broke(r) for r in rs):>9}/{len(rs)}")
""")
code(r"""
display(Image("../figures/fig_setup.png", width=900))
display(Image("../figures/fig_main.png", width=900))
""")
md(r"""
The figures are made by `experiments/figure_setup.py` and `experiments/figure_main.py` from the same
saved runs; the controls in the paper (one-shot pruning, near-copies) by `experiments/review_analysis.py`.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(ROOT, "notebooks", "the_code.ipynb")
nbf.write(nb, out)
print("wrote", os.path.relpath(out, ROOT))
