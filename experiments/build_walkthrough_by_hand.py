"""Writes notebooks/walkthrough_by_hand.ipynb: the whole project rebuilt from scratch in small
steps, each with the idea, a by-hand example, instructions to code it yourself, a solution,
a check, and then the real result from the saved runs. Runs on a CPU in about two minutes.

  python experiments/build_walkthrough_by_hand.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough_by_hand.ipynb
"""
import os
import nbformat as nbf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# How many attention heads does a task need? Built by hand

This notebook rebuilds the whole project from scratch, in small steps you can do yourself.
Every section has the same five parts:

1. **Idea**: what we are doing, in plain words.
2. **By hand**: the same thing on numbers small enough for pen and paper.
3. **Your turn**: step-by-step instructions to write the code yourself. Try it in a new cell
   before reading the solution.
4. **Solution and check**: short code, and a check that it matches the by-hand numbers.
5. **Real result**: what the full experiments (hundreds of saved runs) found.

Everything here runs on a CPU in about two minutes. Only numpy, PyTorch and matplotlib are
needed for sections 1 to 8; the real-result cells read the saved runs in `results/`.

New to PyTorch? Section 0 shows every PyTorch function the project uses, one at a time, on
numbers small enough to check by eye. Later cells point back to it, e.g. "(0.5)".

| section | what you build |
|---|---|
| 0 | PyTorch, one function at a time |
| 1 | the task: a memory, a query, and its answers (the real generator) |
| 2 | one attention head |
| 3 | why the task needs exactly $R$ heads (one head, one equation) |
| 4 | the model: the real code, line by line, then trained |
| 5 | the supply valve on each head |
| 6 | the collateral value: what nobody else can cover |
| 6b | everything as matrices, by hand: every shape and every number |
| 6c | where the valve and the collateral value live in the real code |
| 6d | the whole task in L-shapes, spreadsheet style |
| 7 | hemodynamic attenuation, running during training |
| 8 | what the full experiments found |
| 9 | backup heads when heads can fail |
| 10 | a second circuit: two-layer induction |
| 11 | what holds and what does not |
""")

code(r"""
import glob, pickle, sys                           # files, saved results, import paths
from math import comb                              # binomial coefficients (section 9)
import numpy as np                                 # plain arrays and least squares
import torch                                       # tensors, automatic gradients, training
import torch.nn.functional as F                    # softmax, one-hot, loss functions
import matplotlib.pyplot as plt                    # plots

RED, GREY = "#b2182b", "#888888"                   # colours used in the plots
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})              # plot style
torch.set_num_threads(4)                           # use 4 CPU cores
sys.path.insert(0, "../src")                       # so we can import the repo's own code
import inspect                                     # lets us print a function's source code
from dataclasses import replace                    # copy a settings object with a few changes
from hemo.config import Cfg                        # all settings in one place (src/hemo/config.py)
from hemo.tasks import make_batch, make_val, offsets_for, trivial_loss   # the task (src/hemo/tasks.py)
from hemo.model import CrossAttn, HemoAttn         # the model (src/hemo/model.py)
from hemo.train import train                       # the training loop (src/hemo/train.py)

# The experiments use 16 slots, 4 offsets, 32 heads. To run in seconds on a laptop, this
# notebook uses the SAME code with smaller settings: 8 slots, 2 offsets (so 2 heads needed),
# 8 heads, smaller vectors. Only the numbers change, never the code.
SMALL = replace(Cfg(), seq_len=8, n_rel=2, m_content=4, d_model=16, num_heads=8, d_k=8,
                steps=2000, batch_size=128, val_size=512, val_every=100, device="cpu", seed=0)


def annotated(lines, notes):                       # print code lines, adding a note where a key matches
    for line in lines:
        note = next((n for key, n in notes.items() if key in line), None)
        print(f"{line}    # <- {note}" if note else line)


def show(func, notes):                             # print one function from the repo, with notes
    annotated(inspect.getsource(func).splitlines(), notes)


print("ready")
""")

md(r"""
### The settings: `cfg`

Every number the task, the model and the training use lives in ONE settings object, `Cfg`, in
`src/hemo/config.py`. The notebook does not copy it (two copies could drift apart, and then the
notebook would no longer run what the experiments ran). It imports `Cfg` and changes a few numbers
with `replace` to make the small version `SMALL`. Wherever the code says `cfg.something`, it reads
one of these settings. The cell below prints the ones this notebook uses, next to the real
experiments' values.
""")

code(r"""
REAL = Cfg()                                        # the settings of the real experiments
settings = [                                        # (name in the code, what it means)
    ("seq_len", "N: memory slots"),
    ("n_rel", "R: distances = heads the task needs"),
    ("m_content", "m: numbers per stored item"),
    ("d_model", "d: length of every vector (label + item + noise)"),
    ("num_heads", "heads the model starts with"),
    ("d_k", "size of each head"),
    ("batch_size", "examples per training step"),
    ("steps", "training steps"),
    ("lr", "learning rate"),
]
print(f"{'cfg.' + 'name':<16}{'SMALL':>8}{'real':>8}   meaning")
for name, meaning in settings:                      # one line per setting
    print(f"{'cfg.' + name:<16}{getattr(SMALL, name):>8}{getattr(REAL, name):>8}   {meaning}")
print()
print("Hemodynamic attenuation's own settings (price, fade, first decision) are set where the rule is used: section 7.")
print("To try a different size, make another copy:  mine = replace(SMALL, seq_len=4, n_rel=2)")
""")

# ------------------------------------------------------------------ 0
md(r"""
## 0. PyTorch, one function at a time

The real code is built from about twenty PyTorch functions. Each step below shows one or two
of them on tiny numbers. Run the cell, then compare every printed line with its comment: the
comment says what you should see and why. Nothing here is specific to this project yet.

### 0.0 Python bits that appear everywhere
""")

code(r"""
a, b = 3, 4                                        # two names at once: a = 3 and b = 4 ("unpacking")
_, b = 3, 4                                        # "_" is a name for a value we do not need
print(f"a is {a} and a + b is {a + b}")            # f-string: each {...} is replaced by its value
for i, letter in enumerate(["x", "y", "z"]):       # enumerate gives (position, item) pairs: (0, x), (1, y), (2, z)
    print(i, letter)
print(list(range(4)))                              # range(4) counts 0, 1, 2, 3 (it stops BEFORE 4)
squares = [k * k for k in range(4)]                # a list built by a loop, written in one line
print(squares)                                     # [0, 1, 4, 9]
kw = {"sep": " | "}                                # a dictionary: names -> values
print("one", "two", **kw)                          # **kw passes the dictionary as named settings: same as sep=" | "
""")

md(r"""
### 0.1 A tensor is a box of numbers with a shape

A **tensor** is PyTorch's array. Its **shape** says how many numbers it has along each
direction (each *dimension*). Read a shape left to right as "how many of ..., each holding
how many of ...". In this project `X` has shape `(B, N, d)`: **B** examples, each with **N**
tokens, each token a list of **d** numbers. Dimension `0` is the first entry of the shape;
dimension `-1` means the last one.
""")

code(r"""
torch.set_printoptions(precision=2, sci_mode=False, linewidth=140)   # print 2 decimals, no "1e-01" style
v = torch.tensor([1.0, 2.0, 3.0])                  # a list of 3 numbers
M = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])                # a table: 2 rows, 3 columns
print(v.shape, M.shape)                            # torch.Size([3]) and torch.Size([2, 3])
S = torch.zeros(2, 3, 4)                           # a stack of 2 tables, each 3 rows x 4 columns, all zeros
print(S.shape, " number of dimensions:", S.dim())  # (2, 3, 4) and 3
""")

md(r"""
### 0.2 Making tensors

`torch.randn` draws random numbers from a bell curve (average 0, spread 1). A **generator**
with a fixed seed makes the "random" numbers the same every time you run, which is how we can
check the real code against a by-hand copy later.
""")

code(r"""
print(torch.ones(2, 3))                            # all ones, shape (2, 3)
print(torch.eye(3))                                # identity: 1 on the diagonal. Row j is the "one-hot" code of j
print(torch.arange(5))                             # 0, 1, 2, 3, 4
g = torch.Generator().manual_seed(0)               # a random-number source with a fixed seed
print(torch.randn(2, 3, generator=g))              # 6 random numbers from the bell curve
print(torch.randint(0, 8, (2, 4), generator=g))    # random whole numbers from 0 to 7 (8 is excluded), shape (2, 4)
print(torch.randn(2, 3, generator=g) * 0.1)        # times 0.1: small noise
""")

md(r"""
### 0.3 Reading and writing parts of a tensor (slicing)

Inside the square brackets there is one entry per dimension. `:` means "all of them",
`a:b` means "from a up to but **not including** b", `-1` means "the last one", and `...`
means "all the earlier dimensions". Writing into a slice changes only those entries.
""")

code(r"""
A = torch.arange(24).view(2, 3, 4)                 # the numbers 0..23 laid out as 2 tables of 3 rows x 4 columns
print(A)
print(A[0])                                        # the first table, shape (3, 4)
print(A[0, 1])                                     # first table, second row: tensor([4, 5, 6, 7])
print(A[:, :, :2])                                 # every table, every row, columns 0 and 1
print(A[:, :, 2:4].shape)                          # columns 2 and 3: shape (2, 3, 2)
print(A[..., -1])                                  # the last column of every row: shape (2, 3)
Z = torch.zeros(2, 3)
Z[:, :2] = 7                                       # write 7 into columns 0 and 1 of every row
print(Z)                                           # column 2 is untouched
""")

md(r"""
### 0.4 Arithmetic on whole tensors, mod, and one-hot

`+`, `*`, `%` act on every entry at once, so no loop is needed. `%` is the remainder after
division ("mod"): it is how positions wrap round the circle of slots (section 1).
`F.one_hot(p, N)` turns each whole number into a row of N zeros with a single 1 at that place.
""")

code(r"""
p = torch.tensor([6, 2, 7])                        # three query positions
print(p + 5)                                       # 11, 7, 12: added to every entry at once
print((p + 5) % 8)                                 # 3, 7, 4: the remainder after dividing by 8 (11 = 8 + 3)
print(F.one_hot(p, 8))                             # row i has a single 1 in column p[i]
""")

md(r"""
### 0.5 Picking items by index: `torch.gather`

The simple version is `content[j]`: "give me rows j". `torch.gather` does the same when there
is an extra batch dimension in front (every example has its own slots and its own indices).
It needs the index repeated for every number in an item, which is what
`.unsqueeze(-1).expand(-1, -1, m)` does: `unsqueeze(-1)` adds a size-1 dimension at the end,
and `expand` repeats it `m` times (`-1` means "keep this size").
""")

code(r"""
content = torch.tensor([[10., 11.], [20., 21.], [30., 31.], [40., 41.]])   # 4 slots, each holding an item of 2 numbers
j = torch.tensor([3, 0, 3])                        # three queries want slots 3, 0 and 3
print(content[j])                                  # simple version: rows 3, 0, 3 -> [40, 41], [10, 11], [40, 41]
c = content.unsqueeze(0)                           # add an example dimension in front: (4, 2) -> (1, 4, 2)
jj = j.unsqueeze(0)                                # (3,) -> (1, 3): one example, three queries
idx = jj.unsqueeze(-1).expand(-1, -1, 2)           # (1, 3) -> (1, 3, 1) -> (1, 3, 2): each index written twice
print(idx)
print(torch.gather(c, 1, idx))                     # along dimension 1 (the slots): the same rows 3, 0, 3
""")

md(r"""
### 0.6 Gluing: `torch.cat`
""")

code(r"""
a = torch.tensor([[1., 2.]])                       # shape (1, 2)
b = torch.tensor([[3., 4., 5.]])                   # shape (1, 3)
print(torch.cat([a, b], dim=-1))                   # side by side along the last dimension: [[1, 2, 3, 4, 5]]
print(torch.cat([a, a], dim=0))                    # on top of each other along dimension 0: shape (2, 2)
""")

md(r"""
### 0.7 Changing shape: `view`, `transpose`, `reshape`

`view` reads the same numbers with a different shape (nothing is copied or changed).
`transpose(i, j)` swaps two dimensions. `reshape` is like `view` but also works after a
transpose. This is exactly how the model cuts each token's $H \cdot d_k$ numbers into $H$
heads (`_split`, section 4) and glues them back (`combine`).
""")

code(r"""
x = torch.arange(12)                               # 12 numbers in a row
print(x.view(3, 4))                                # the same 12 numbers read as 3 rows of 4
y = x.view(1, 3, 4)                                # think: 1 example, 3 tokens, 4 numbers per token
h = y.view(1, 3, 2, 2)                             # cut each token's 4 numbers into 2 heads of 2 numbers
print(h[0, :, 0])                                  # head 0's part of every token: the first 2 numbers of each
t = h.transpose(1, 2)                              # swap dimensions 1 and 2: (1, 3 tokens, 2 heads, 2) -> (1, 2 heads, 3 tokens, 2)
print(t.shape)
back = t.transpose(1, 2).reshape(1, 3, 4)          # swap back, then glue the 2 heads side by side again
print(torch.equal(back, y))                        # True: nothing was lost
""")

md(r"""
### 0.8 Stretching to match (broadcasting), `None` and `unsqueeze`

When two tensors with different shapes are multiplied, any dimension of size 1 is copied to
match the other. Writing `None` inside the brackets (or calling `unsqueeze`) adds a size-1
dimension. The valve line of the model, `out * gate[:, :, None, None]`, is exactly this.
""")

code(r"""
out = torch.ones(1, 2, 3, 2)                       # (examples, heads, tokens, numbers): every head's output, all ones
gate = torch.tensor([[1.0, 0.0]])                  # (examples, heads): head 0 on, head 1 off
g4 = gate[:, :, None, None]                        # (1, 2) -> (1, 2, 1, 1)
print(g4.shape)
print(out * g4)                                    # the size-1 dimensions stretch to (3, 2): head 1 becomes all zeros
v1 = torch.tensor([1.0, 0.5])                      # one gain per head, shape (2,)
print(v1.unsqueeze(0).expand(3, -1))               # (2,) -> (1, 2) -> (3, 2): the same gains for 3 examples
""")

md(r"""
### 0.9 Matrix multiplication: `@`

For two tables, `A @ B` is "row times column": each entry is a sum of products. With more
than two dimensions, `@` multiplies the **last two** dimensions and repeats that for every
earlier one (every example, every head) at once. `K.transpose(-2, -1)` swaps the last two
dimensions, which is $K^\top$, so `Q @ K.transpose(-2, -1)` puts every score
$q_i \cdot k_j$ in row $i$, column $j$.
""")

code(r"""
A = torch.tensor([[1., 2.], [3., 4.]])
x = torch.tensor([[5.], [6.]])
print(A @ x)                                       # 1*5 + 2*6 = 17 and 3*5 + 4*6 = 39
print(A.T)                                         # the transpose: rows become columns
Qs = torch.randn(1, 2, 3, 4, generator=g)          # (examples, heads, 3 queries, 4 numbers)
Ks = torch.randn(1, 2, 5, 4, generator=g)          # (examples, heads, 5 slots,   4 numbers)
print((Qs @ Ks.transpose(-2, -1)).shape)           # (1, 2, 3, 5): every query against every slot, for each head
""")

md(r"""
### 0.10 Softmax: scores to weights

`F.softmax(s, dim=-1)` works along the last dimension: each **row** becomes positive weights
that add up to 1, with the biggest score getting the biggest weight (section 2 does it by hand).
""")

code(r"""
s = torch.tensor([[2.0, 0.0, 0.0],
                  [0.0, 0.0, 9.0]])
w = F.softmax(s, dim=-1)
print(w)                                           # row 1: 0.79, 0.11, 0.11 (as in section 2); row 2: almost all on the third
print(w.sum(-1))                                   # every row adds up to 1
""")

md(r"""
### 0.11 A layer of weights: `nn.Linear`

`nn.Linear(n_in, n_out)` holds a weight table $W$ (shape `(n_out, n_in)`) and a bias $b$, and
computes $y = x W^\top + b$. It acts on the **last** dimension, so on a `(B, N, d)` tensor it
transforms every token separately. Its weights start random; training changes them.
A class that holds layers like this is an `nn.Module`; calling `layer(x)` runs its `forward`.
""")

code(r"""
import torch.nn as nn
torch.manual_seed(0)                               # fixed random starting weights
lin = nn.Linear(3, 2)                              # turns 3 numbers into 2
print(lin.weight.shape, lin.bias.shape)            # (2, 3) and (2,): one row of W per output number
x = torch.tensor([1.0, 2.0, 3.0])
by_hand = x @ lin.weight.T + lin.bias              # the formula written out
print(lin(x).detach(), by_hand.detach())           # the same two numbers (.detach() hides tracking info, 0.13)
X3 = torch.randn(1, 5, 3, generator=g)             # 1 example, 5 tokens, 3 numbers each
print(lin(X3).shape)                               # (1, 5, 2): every token transformed on its own
""")

md(r"""
### 0.12 Adding up

`sum(dim)` and `mean(dim)` add along one dimension, which then disappears from the shape.
""")

code(r"""
A = torch.tensor([[1., -2.], [3., 4.]])
print(A.sum())                                     # everything: 6
print(A.sum(0), A.sum(1))                          # down each column: [4, 2]; along each row: [-1, 7]
print(A.abs(), A.mean(), (A ** 2).mean())          # sizes without sign; the average 1.5; average of squares 7.5
""")

md(r"""
### 0.13 Learning: loss, gradient, one step

`F.mse_loss(pred, target)` is the mean squared error. `loss.backward()` works out, for every
weight marked `requires_grad=True`, how much the loss changes when that weight grows (the
**gradient**), and stores it in `.grad`. An optimizer then moves each weight a little
**against** its gradient. Training repeats these three calls thousands of times.
`torch.no_grad()` turns tracking off (used when we only measure), `.detach()` gives a copy
that is cut off from tracking, and `.item()` turns a one-number tensor into a plain number.
""")

code(r"""
w = torch.tensor(3.0, requires_grad=True)          # a weight, tracked
x, target = torch.tensor(2.0), torch.tensor(10.0)
loss = F.mse_loss(w * x, target)                   # (3*2 - 10)^2 = 16
loss.backward()                                    # the gradient d loss / d w
print(loss.item(), w.grad.item())                  # by hand: 2 * (w*x - target) * x = 2 * (-4) * 2 = -16
opt = torch.optim.SGD([w], lr=0.1)                 # an optimizer with step size 0.1 (training uses Adam, a smarter cousin)
opt.step()                                         # w <- w - 0.1 * (-16) = 4.6
print(round(w.item(), 4))                          # 4.6 (computers store decimals slightly inexactly, so we round)
opt.zero_grad()                                    # clear the stored gradient before the next step
with torch.no_grad():                              # measure only, track nothing
    print(round(F.mse_loss(w * x, target).item(), 4))   # (4.6*2 - 10)^2 = 0.64: smaller, so the step helped
print((w * x).detach().requires_grad)              # False: a detached copy is not tracked
""")

md(r"""
That is all the PyTorch this project needs:

| function | what it does | step |
|---|---|---|
| `torch.tensor`, `.shape` | a box of numbers and its sizes | 0.1 |
| `torch.zeros/ones/eye/arange/randn/randint` | make tensors | 0.2 |
| `x[:, :, a:b]` | read or write a part | 0.3 |
| `%`, `F.one_hot` | wrap round; a number as a row with one 1 | 0.4 |
| `torch.gather`, `unsqueeze`, `expand` | pick items by index, for every example | 0.5 |
| `torch.cat` | glue side by side | 0.6 |
| `view`, `transpose`, `reshape` | cut into heads and glue back | 0.7 |
| `x[:, :, None, None]`, broadcasting | one gain per head, stretched over its outputs | 0.8 |
| `@`, `.T`, `transpose(-2, -1)` | matrix multiplication; scores | 0.9 |
| `F.softmax` | scores to weights | 0.10 |
| `nn.Linear`, `nn.Module` | a layer of weights | 0.11 |
| `sum`, `mean`, `abs` | adding up | 0.12 |
| `F.mse_loss`, `backward`, `.grad`, `opt.step`, `no_grad`, `detach` | learning | 0.13 |
""")

# ------------------------------------------------------------------ 1
md(r"""
## 1. The task

**Idea.** A memory has $N$ slots. Slot $j$ holds a random vector $c_j$ (its *content*). A
query arrives holding a position $p$. The model must return the contents found $\delta_1,
\dots, \delta_R$ slots ahead of $p$, wrapping around like a clock:

$$\text{answer} = (c_{p+\delta_1},\ \dots,\ c_{p+\delta_R}), \qquad \text{positions taken mod } N.$$

**By hand.** $N = 8$ slots, $R = 2$ offsets $\delta = (1, 5)$, query $p = 6$:
$(6 + 1) \bmod 8 = 7$ and $(6 + 5) \bmod 8 = 3$. The answer is $(c_7, c_3)$.

**Why "mod" (wrap around)?** Without it, a query near the end would point past the last
slot: $p = 6$ plus 5 is 11, and there is no slot 11. Wrapping round the circle means **every**
query position has all $R$ answers inside the memory, and the rule "look 5 ahead" is the same
for every position, so one head can learn it once and use it everywhere. Without wrapping you
would have to throw away the queries near the end, or make the memory longer.

**The real code.** This is the actual task generator from `src/hemo/tasks.py` (the part for
this task), with a note on each line. Two things the by-hand version leaves out: every example
has $N$ query tokens at once (one card per position, each with its own random $p$), and a
little noise is added to every vector. Note the loop: it runs over the $R$ **offsets** (2 here,
4 in the experiments), not over the 16 slots. All slots and all queries are handled at once as
arrays. First the whole function with notes; then we run it one line at a time.
""")

code(r"""
show(offsets_for, {"return [(1 + r": "offsets spread evenly: 1, 1 + N/R, 1 + 2N/R, ... -> (1, 5) here"})
print()
src = inspect.getsource(make_batch).splitlines()   # the whole generator ...
a = next(i for i, l in enumerate(src) if 'cfg.task == "multi_relation"' in l)
b = next(i for i, l in enumerate(src) if 'aux = {"p": p' in l)
annotated(src[a:b + 1], {                           # ... printed from the multi_relation part
    'elif cfg.task == "multi_relation"': "this task",
    "m = cfg.m_content": "m = numbers per stored item",
    "assert d >= N + m": "each vector must have room for a slot label (N) and an item (m)",
    "X = torch.randn(B, N, d": "query tokens: start as small noise, (B, N tokens, d)",
    "Y = torch.randn(B, N, d": "memory slots: start as small noise, (B, N slots, d)",
    "Y[:, :, :N] = torch.eye(N": "slot label: slot j gets the one-hot code of j",
    "content = torch.randn(B, N, m": "the random item stored in every slot",
    "Y[:, :, N:N + m] = content": "put the item next to the label",
    "p = torch.randint(0, N, (B, N)": "each of the N query tokens gets its own random position p",
    "X[:, :, :N] = F.one_hot(p, N)": "query token = the one-hot code of its p",
    "blocks = []": "the answer, built one offset at a time",
    "for off in offsets_for(cfg)": "loop over the R offsets (not over slots)",
    "j = (p + off) % N": "which slot: p + offset, wrapped round (mod N)",
    "blocks.append(torch.gather(": "take the item in that slot, for every query at once",
    "T = torch.cat(blocks": "the correct answer: the R items side by side",
    'aux = {"p": p': "extra information kept for checking roles later",
})
""")

md(r"""
### Run `make_batch` one line at a time

Each cell below holds lines **copied exactly** from `make_batch`, run on one example with a
fixed seed, and prints what the line made. The last cell calls the real function with the
same seed and checks that we got the very same numbers.

**Step 1: the settings.** `B` is the number of examples; `gen` is the fixed random source (0.2).
""")

code(r"""
B, cfg, device = 1, SMALL, "cpu"                   # what we pass to make_batch: 1 example, the small settings
gen = torch.Generator().manual_seed(0)             # fixed random source, so we can repeat this exactly
# --- copied from make_batch ---
N, d = cfg.seq_len, cfg.d_model                    # N = 8 slots; d = 16 numbers per vector
dev = "cpu" if gen is not None else device         # a generator works on the CPU
kw = {"generator": gen} if gen is not None else {} # use the generator in every random call (0.0, **kw)
m = cfg.m_content                                  # m = 4 numbers per stored item
assert d >= N + m, "d_model must hold the position block plus the content block"   # room for label (8) + item (4)
print("N =", N, " d =", d, " m =", m)
""")

md(r"""
**Step 2: start every vector as small noise** (0.2). `X` holds the $N$ query tokens of each
example and `Y` the $N$ memory slots; each is a list of $d = 16$ numbers.
""")

code(r"""
X = torch.randn(B, N, d, device=dev, **kw) * 0.1   # query tokens: (1 example, 8 tokens, 16 numbers)
Y = torch.randn(B, N, d, device=dev, **kw) * 0.1   # memory slots: (1 example, 8 slots, 16 numbers)
print("X", tuple(X.shape), " Y", tuple(Y.shape))
print(Y[0])                                        # 8 rows (slots) of 16 small numbers
""")

md(r"""
**Step 3: write a label into each slot.** Columns 0 to 7 of slot $j$ become the one-hot code
of $j$ (`torch.eye`, 0.2; writing into a slice, 0.3). Look for the diagonal of 1s on the left.
""")

code(r"""
Y[:, :, :N] = torch.eye(N, device=dev)             # every example, every slot, columns 0..7 <- the identity table
print(Y[0])
""")

md(r"""
**Step 4: store a random item in each slot**, in columns 8 to 11. Columns 12 to 15 stay noise.
""")

code(r"""
content = torch.randn(B, N, m, device=dev, **kw)   # one item of 4 numbers per slot: (1, 8, 4)
Y[:, :, N:N + m] = content                         # columns 8..11 <- the items
print(Y[0])                                        # label | item | noise
""")

md(r"""
**Step 5: give each query token a position $p$**, written as a one-hot code in its first 8
columns (`torch.randint`, 0.2; `F.one_hot`, 0.4). `.float()` turns the whole-number 0s and 1s
into decimals so they can be stored in `X`.
""")

code(r"""
p = torch.randint(0, N, (B, N), device=dev, **kw)  # 8 query tokens, each with a random position 0..7: (1, 8)
X[:, :, :N] = F.one_hot(p, N).float()              # columns 0..7 of token i <- the one-hot code of p[i]
print("p =", p[0].tolist())
print(X[0, :, :N])                                 # row i has its single 1 in column p[i]
""")

md(r"""
**Step 6: build the answer.** For each offset, work out which slot every query needs
(`+` and `%`, 0.4), fetch those items (`torch.gather`, 0.5), and finally glue the $R$ items side
by side (`torch.cat`, 0.6). The loop runs twice here (offsets 1 and 5), never over slots.
""")

code(r"""
blocks = []                                        # the answer, one offset at a time
for off in offsets_for(cfg):                       # off = 1, then off = 5
    j = (p + off) % N                              # the slot each of the 8 queries needs, wrapped round
    print(f"offset {off}:  p + {off} mod 8 = {j[0].tolist()}")
    blocks.append(torch.gather(content, 1, j.unsqueeze(-1).expand(-1, -1, m)))   # those slots' items: (1, 8, 4)
T = torch.cat(blocks, dim=-1)                      # item for offset 1 | item for offset 5: (1, 8, 8)
print("T", tuple(T.shape))
print(T[0])
""")

md(r"""
**Step 7: check against the real function.** Same seed, so the same random draws in the same
order; every number must match.
""")

code(r"""
X2, Y2, T2, _ = make_batch(1, SMALL, "cpu", gen=torch.Generator().manual_seed(0))   # the real function
print("same X:", torch.equal(X, X2), "  same Y:", torch.equal(Y, Y2), "  same T:", torch.equal(T, T2))
""")

md(r"""
**Check the answers by hand.** Pick query token 0, read its $p$, add each offset mod 8, and
check the answer holds the items stored in those slots:
""")

code(r"""
X, Y, T, aux = make_batch(1, SMALL, "cpu", gen=torch.Generator().manual_seed(0))   # one example, fixed seed
N, R, m = SMALL.seq_len, SMALL.n_rel, SMALL.m_content   # 8 slots, 2 offsets, 4 numbers per item
content = Y[0, :, N:N + m]                         # the items stored in the 8 slots
print("shapes: X", tuple(X.shape), " Y", tuple(Y.shape), " T", tuple(T.shape))
p0 = int(aux["p"][0, 0])                           # the position carried by query token 0
print("offsets", offsets_for(SMALL), "  query token 0 carries p =", p0)
for r, d in enumerate(offsets_for(SMALL)):         # for each offset ...
    slot = (p0 + d) % N                            # ... the slot it points to
    print(f"  ({p0} + {d}) mod {N} = {slot}:  answer block {r} equals the item in slot {slot}:",
          torch.equal(T[0, 0, r * m:(r + 1) * m], content[slot]))   # check the answer holds that item
""")

# ------------------------------------------------------------------ 2
md(r"""
## 2. One attention head

**Idea.** A head turns the query into a *query vector* $q$, each memory slot into a *key*
$k_j$ and a *value* $v_j$. It scores each slot by $q \cdot k_j$, turns the scores into
weights that are positive and sum to one (the softmax), and returns the weighted average of
the values:

$$a_j = \frac{e^{q\cdot k_j}}{\sum_i e^{q \cdot k_i}}, \qquad \text{output} = \sum_j a_j\, v_j .$$

**By hand.** Three slots with scores $(2, 0, 0)$: $e^2 = 7.389$, $e^0 = 1$, so the weights are
$7.389/9.389 = 0.787$ and $1/9.389 = 0.107$ twice. With values $(10, 20, 30)$ the output is
$0.787 \cdot 10 + 0.107 \cdot 20 + 0.107 \cdot 30 = 7.87 + 2.13 + 3.20 = 13.2$: mostly slot 1,
a little of the others. (Keep three decimals: rounding the weights to 0.79 and 0.11 first
gives 13.4, which is wrong.)

**Your turn.** Write `attend(scores, values)`: softmax the scores, then return the weighted sum.
Check it gives 13.2 on the example above.
""")

code(r"""
def attend(scores, values):
    w = torch.softmax(torch.as_tensor(scores, dtype=torch.float), -1)   # scores -> weights that sum to 1
    return w, w @ torch.as_tensor(values, dtype=torch.float)            # weighted average of the values


w, out = attend([2.0, 0.0, 0.0], [10.0, 20.0, 30.0])  # the by-hand example
print("weights", w.numpy().round(3), "  output", round(float(out), 1))
assert abs(float(out) - 13.2) < 0.05                  # must match the hand calculation
""")

md(r"""
The key fact for everything that follows: **a head returns one weighted average**. If it
splits its attention between two slots it returns a *blend* of their contents, never the two
contents separately.
""")

# ------------------------------------------------------------------ 3
md(r"""
## 3. One head, one equation: why the task needs exactly $R$ heads

**Idea.** The answer contains $R$ unknown vectors. Each head hands back one blend of them,
which is one equation. To solve for $R$ unknowns you need $R$ *different* equations, so at
least $R$ heads.

**By hand.** Two unknowns $x, y$. One equation, $x + y = 10$: cannot be solved. Add
$x - y = 2$: $x = 6, y = 4$. Add instead $2x + 2y = 20$: still stuck, because it is the first
equation again. The number of different equations is the **rank**.

If each missing equation leaves one of the $R$ unknowns unknown, the best possible error is

$$\text{loss}(k) = \Big(1 - \frac{k}{R}\Big) \times \text{loss of a model that learned nothing.}$$

At $R = 4$: one head leaves $3/4$, two heads $2/4$, three $1/4$, four heads 0.

**Your turn.** Check the formula numerically, without training anything:
1. Draw $R = 4$ unknown vectors per example (many examples), unit variance.
2. Make $k$ random blends: `U = A @ C` with a random `k x 4` matrix `A`.
3. Fit the best linear guess of `C` from `U` with least squares, and measure the error.
4. Compare with $1 - k/4$. Then make one blend a copy of another and see the error.
""")

code(r"""
rng = np.random.default_rng(0)                     # random numbers, fixed seed
Rn, n = 4, 20000                                   # 4 unknowns per example, 20000 examples
C = rng.normal(size=(n, Rn))                       # the unknowns (one number each), unit variance


def best_error(A):                                 # A: one row per head = that head's mixing recipe
    U = C @ A.T                                    # what the heads hand back: one blend per head
    W = np.linalg.lstsq(U, C, rcond=None)[0]       # best linear guess of the unknowns from the blends
    return np.mean((C - U @ W) ** 2)               # how wrong that best guess still is


print("blends  error   predicted 1 - k/4")
for k in range(1, 5):                              # 1, 2, 3, 4 heads (random recipes)
    print(f"{k:>6}  {best_error(rng.normal(size=(k, Rn))):.3f}   {1 - k / Rn:.3f}")
A = rng.normal(size=(3, Rn))                       # 3 different recipes
A_copy = np.vstack([A, A[0]])                      # add a 4th that repeats the 1st (a copied head)
print(f"4 blends with one copy: {best_error(A_copy):.3f}  (same as 3 blends: {1 - 3 / Rn:.3f})")
""")

code(r"""
E1 = pickle.load(open("../results/proposal/equations.pkl", "rb"))   # saved by experiments/equations_test.py
print("REAL RESULT: a trained 4-head model, heads removed or copied, read-out re-fitted")
print(f"{'heads used':<22}{'different':>10}{'loss':>8}{'predicted':>11}")
for row in E1["rows"]:                             # one line per combination of heads
    print(f"{row['name']:<22}{row['different']:>10}{row['loss']:>8.3f}{row['predicted']:>11.3f}")
""")

# ------------------------------------------------------------------ 4
md(r"""
## 4. The model: the real code, line by line

**Idea.** Put $H$ heads side by side. Each head has its own query, key and value weights and
its own output weights; the layer's answer is the sum of what the heads write.

The experiments use `CrossAttn` from `src/hemo/model.py`. It stores all heads in one big
weight matrix and then cuts it into heads, the usual way in PyTorch. Below, every line is
printed straight from the source file, with a plain note after it (`# <-`), so what you read
is exactly what runs.
""")

code(r"""
show(CrossAttn.__init__, {
    "super().__init__()": "standard PyTorch set-up",
    "self.cfg = cfg": "keep the settings",
    "self.H = num_heads": "number of heads H (8 here, 32 in the experiments)",
    "self.d_k, self.d_model": "d_k = size of one head, d_model = size of each input vector",
    "self.d_out = d_out(cfg)": "size of the answer: R items x m numbers",
    "inner = self.H * self.d_k": "all heads side by side: H x d_k numbers",
    "self.W_q = nn.Linear": "query weights for ALL heads at once: d_model -> H*d_k",
    "self.W_k = nn.Linear": "key weights for all heads at once",
    "self.W_v = nn.Linear": "value weights for all heads at once",
    "self.W_o = nn.Linear": "output weights: read every head's blend, write the answer",
})
print()
show(CrossAttn._split, {
    "B, N, _ = x.shape": "batch size and number of tokens",
    "return x.view": "cut the H*d_k numbers into H heads of d_k: shape (B, H, N, d_k)",
})
""")

code(r"""
HEADS_NOTES = {
    "Q, K, V =": "queries from the query tokens X; keys and values from the memory Y; split into heads",
    "if attn_override is None": "the normal case (the override is used by one analysis only)",
    "attn = F.softmax": "score every query against every slot, scale by sqrt(d_k), softmax -> weights",
    "else:": "analysis only:",
    "attn = attn_override": "force a chosen attention pattern",
    "return Q, attn, attn @ V": "attn @ V = each head's blend of the values",
}
COMBINE_NOTES = {
    "B = out.size(0)": "batch size",
    "out = (out * gate": "THE VALVE: multiply each head's blend by its gain g_h, then lay the heads side by side",
    "return self.W_o(out)": "output weights turn the gated blends into the answer",
}
show(CrossAttn.heads, HEADS_NOTES)
print()
show(CrossAttn.combine, COMBINE_NOTES)
print()
show(CrossAttn.forward, {
    "_, _, out = self.heads": "run every head",
    "if gate is None:": "no valves given ...",
    "gate = torch.ones": "... so every head is fully on (gain 1)",
    "if gate.dim() == 1:": "one gain per head given ...",
    "gate = gate.unsqueeze": "... copy it for every example in the batch: (H,) -> (B, H)",
    "return self.combine": "apply the valves and the output weights; also return the gains used",
})
""")

code(r"""
# HemoAttn is CrossAttn plus the supply rules. Its forward pass is where the valves come from:
show(HemoAttn.forward, {
    "Q, attn, out = self.heads": "run every head, exactly as in CrossAttn",
    "if gate is None:": "no valves given: ask the supply rule",
    "self.update_demand": "older rules only: track how much each head writes",
    "gate = self.gate(": "every head's valve from the rule (for hemodynamic attenuation: its tone, section 6c)",
    "if self.needs_gate_grad()": "pruning baseline only: let the loss send a gradient to the valves",
    "gate = gate.detach().requires_grad_": "(pruning baseline) make the valves a tensor that collects gradients",
    "self._gate_leaf = gate": "(pruning baseline) keep it to read the gradient later",
    "if gate.dim() == 1:": "one gain per head given ...",
    "gate = gate.unsqueeze": "... copy it for every example in the batch",
    "used = gate": "the gains actually applied this step",
    "if self.head_dropout > 0": "damage experiment only (section 9):",
    "keep = (torch.rand": "each head fails at random this step",
    "used = gate * keep": "failed heads get gain 0 (the others are rescaled, as in dropout)",
    "return self.combine(out, used), gate": "apply the valves (THE attachment point) and the output weights",
})
""")

md(r"""
### Run the layer one line at a time

Now push the example from section 1 through a real 2-head layer, one line of `heads` and
`combine` at a time, printing the shape after each. The weights are untrained (random), so the
attention is still spread out; training is what makes it sharp. The last cell checks we got
exactly what `layer(X, Y)` gives.

**Step 1: queries for all heads at once** (`nn.Linear`, 0.11). `W_q` turns each token's 16
numbers into $H \cdot d_k = 2 \times 8 = 16$ numbers: the first 8 belong to head 0, the next 8
to head 1.
""")

code(r"""
import math                                        # for the square root
torch.manual_seed(0)                               # fixed random starting weights
layer = CrossAttn(SMALL, num_heads=2)              # the real layer, 2 heads
B, N, H, d_k = X.size(0), X.size(1), layer.H, layer.d_k   # 1 example, 8 tokens, 2 heads, 8 numbers per head
print("W_q.weight", tuple(layer.W_q.weight.shape), "= (H*d_k, d_model): every head's query weights stacked")
q_all = layer.W_q(X)                               # every query token through the query weights
print("W_q(X)", tuple(q_all.shape), "= (examples, tokens, H*d_k)")
""")

md(r"""
**Step 2: cut into heads** (`view` and `transpose`, 0.7). This is `_split`. Keys and values are
made the same way, but from the memory `Y`.
""")

code(r"""
q4 = q_all.view(B, N, H, d_k)                      # each token's 16 numbers -> 2 heads of 8: (1, 8, 2, 8)
Q = q4.transpose(1, 2)                             # heads before tokens: (1, 2, 8, 8)
print("view", tuple(q4.shape), " -> transpose", tuple(Q.shape), "= (examples, heads, tokens, d_k)")
print("same as the real _split:", torch.equal(Q, layer._split(layer.W_q(X))))
K, V = layer._split(layer.W_k(Y)), layer._split(layer.W_v(Y))   # keys and values from the memory slots
print("K", tuple(K.shape), " V", tuple(V.shape))
""")

md(r"""
**Step 3: score, softmax, blend** (`@`, 0.9; `F.softmax`, 0.10). Each head scores every query
against every slot, turns each row of scores into weights, and returns each query's weighted
average of the values (section 2). This is `heads`.
""")

code(r"""
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # every query vs every slot, per head; / sqrt(8) keeps numbers moderate
print("scores", tuple(scores.shape), "= (examples, heads, queries, slots)")
attn = F.softmax(scores, dim=-1)                   # each query's row -> 8 weights that add up to 1
print("head 0, query 0, weights on the 8 slots:", attn[0, 0, 0].detach())   # nearly even: untrained
print("every row adds up to 1:", torch.allclose(attn.sum(-1), torch.ones(1)))
out = attn @ V                                     # weighted average of the values: one blend per head per query
print("out", tuple(out.shape), "= (examples, heads, queries, d_k)")
""")

md(r"""
**Step 4: the valve, glue, read out** (broadcasting, 0.8; `transpose` and `reshape`, 0.7;
`nn.Linear`, 0.11). This is `combine`: each head's blend is multiplied by its gain $g_h$, the
heads are glued back side by side, and the output weights write the answer.
""")

code(r"""
gate = torch.ones(B, H)                            # every head fully on (what forward uses when no gate is given)
gated = out * gate[:, :, None, None]               # THE VALVE: each head's blend times its gain: (1, 2, 8, 8)
side = gated.transpose(1, 2).reshape(B, -1, H * d_k)   # tokens first again, heads glued side by side: (1, 8, 16)
pred = layer.W_o(side)                             # the output weights read both heads and write the answer: (1, 8, 8)
print("side by side", tuple(side.shape), "  prediction", tuple(pred.shape), "  answer T", tuple(T.shape))
print("same as the real layer(X, Y):", torch.allclose(pred, layer(X, Y)[0]))
""")

md(r"""
**Step 5: close a valve.** With head 1's gain at 0, its 8 columns of `side` are all zero, so the
output weights never see head 1. Section 5 checks that it then also gets no gradient.
""")

code(r"""
gate0 = torch.tensor([[1.0, 0.0]])                 # head 0 on, head 1 off
side0 = (out * gate0[:, :, None, None]).transpose(1, 2).reshape(B, -1, H * d_k)   # the same two lines as step 4
print("token 0, head 0's columns:", side0[0, 0, :d_k].detach())
print("token 0, head 1's columns:", side0[0, 0, d_k:].detach())   # all zeros ("-0." is still zero)
""")

md(r"""
**Train it.** The real training function, `train` in `src/hemo/train.py`, with every head on
(`hemo=False` means a plain layer with no supply rule). One head against two, on the small
task that needs two:
""")

code(r"""
val = make_val(SMALL, "cpu")                       # a fixed validation set
triv = trivial_loss(val)                           # loss of a model that always guesses the average
for H in (1, 2):                                   # one head, then two heads
    _, hist = train(SMALL, val, "cpu", hemo=False, num_heads=H)   # the real training loop
    print(f"{H} head(s): loss {hist['val_loss'][-1]:.4f}   (formula (1 - k/R) x trivial: {max(0, 1 - H / R) * triv:.3f})")
""")

md(r"""
One head cannot solve a two-offset task; two heads can. The one-head loss lands a little below
the formula, and the reason is worth knowing: the formula assumes a head's attention depends
only on positions, but the memory vectors also contain the stored items, so the attention can
lean on them and leak a little extra information. In the full-size task this effect is
negligible: the measured one-head loss there is 0.755 against a predicted 0.756 (section 3).
""")

# ------------------------------------------------------------------ 5
md(r"""
## 5. The supply valve on each head

**Idea.** Each head's contribution is multiplied by a gain $g_h$, its *supply*. $g_h = 1$ is
fully on, $g_h = 0$ is off. A head at $g_h = 0$ is **removed exactly**: the layer is then an
ordinary layer of the remaining heads, and the starved head receives no gradient.

**By hand.** Four heads write the numbers $(3, 7, -1, 5)$. All on: $3 + 7 - 1 + 5 = 14$.
Gains $(1, 0, 1, 0)$: $3 - 1 = 2$, which is exactly a two-head layer of heads 1 and 3.

**Check on the real layer.** With a 4-head `CrossAttn`, (a) gates `(1, 0, 1, 1)` give the same
output whatever head 2's weights are, and (b) head 2 gets zero gradient.
""")

code(r"""
torch.manual_seed(0)                               # fixed starting weights
layer = CrossAttn(SMALL, num_heads=4)              # the real layer, 4 heads
X, Y, T, _ = make_batch(64, SMALL, "cpu")          # 64 examples of the real task
gate = torch.tensor([1.0, 0.0, 1.0, 1.0])          # valves: head 2 (index 1) switched off
dk = SMALL.d_k                                     # rows dk..2dk of W_v belong to head 2
before = layer(X, Y, gate=gate)[0].detach()        # the layer's output now
with torch.no_grad():                              # change weights without tracking gradients
    layer.W_v.weight[dk:2 * dk] += 100.0           # change head 2's value weights drastically
print("output unchanged:", torch.allclose(before, layer(X, Y, gate=gate)[0]))  # head 2 cannot affect it
loss = F.mse_loss(layer(X, Y, gate=gate)[0], T)    # a loss to differentiate
loss.backward()                                    # compute gradients
per_head = layer.W_q.weight.grad.abs().view(4, dk, -1).sum((1, 2))   # gradient on each head's query rows
print("gradient reaching each head's query weights:", per_head.numpy().round(4))   # head 2 gets 0
""")

# ------------------------------------------------------------------ 6
md(r"""
## 6. The collateral value: what nobody else can cover

**Idea.** In the brain, losing a blood vessel does no harm if neighbouring vessels can take
over its territory (*collateral circulation*). So we value a head by what is lost when it is
switched off **and the other heads are allowed to adjust** (their output weights re-fitted):

$$v_h = \text{error}(\text{all open heads except } h,\ \text{re-fitted}) - \text{error}(\text{all open heads}).$$

**By hand.** Heads A and B output the same signal $z$, head C outputs $w$, the target is
$z + w$. The best fit uses $\tfrac12 A + \tfrac12 B + C$.
- *Ordinary importance* (switch A off, nobody adjusts): the fit loses $\tfrac12 z$, so the
  error rises by $\mathrm{var}(z)/4 = 0.25$. A looks important, and so does B.
- *Collateral value* (switch A off, B adjusts): B takes A's weight, nothing is lost,
  $v_A = 0$. So one of the two copies can go.

**Your turn.** Write `collateral_values(outputs, target, open_heads)`:
1. `err(heads)`: least-squares fit of `target` from the chosen heads' outputs (plus a column
   of ones), return the mean squared error.
2. For each open head $h$: `err(open without h) - err(open)`.
Check it on the A, B, C example, then compare with ordinary importance.
""")

code(r"""
def collateral_values(outputs, target, open_heads):
    # outputs: one array per head, (examples x numbers); target: (examples x answer size)
    def err(heads):                                # error of the best read-out that uses only these heads
        if not heads:                              # no heads: best guess is the average answer
            return float(np.mean((target - target.mean(0)) ** 2))  # error of always guessing the average
        Z = np.concatenate([outputs[h] for h in heads] + [np.ones((len(target), 1))], 1)  # heads side by side + a column of ones
        W = np.linalg.lstsq(Z, target, rcond=None)[0]   # best output weights (least squares): the "re-fit"
        return float(np.mean((target - Z @ W) ** 2))    # how wrong that best read-out is
    base = err(open_heads)                         # error with every open head
    return {h: err([g for g in open_heads if g != h]) - base for h in open_heads}  # rise in error without h, others re-fitted


z, w = rng.normal(size=(50000, 1)), rng.normal(size=(50000, 1))  # two independent signals
outs = {"A": z, "B": z.copy(), "C": w}             # A and B are copies; C is different
print("collateral values:", {h: round(v, 3) for h, v in collateral_values(outs, z + w, ["A", "B", "C"]).items()})
full = np.hstack([z, z, w]); coef = np.linalg.lstsq(full, z + w, rcond=None)[0]  # fit with all three heads
print("ordinary importance of A (B does not adjust):",
      round(float(np.mean((z + w - full[:, 1:] @ coef[1:]) ** 2)), 3))  # drop A, keep B's and C's old weights
""")

md(r"""
**The real code does exactly this, faster.** `HemoAttn.probe_ischemia` (printed in section 6c)
computes every head's collateral value with one matrix inverse and a shortcut formula. Here it
is on a real 4-head layer with one head closed, next to the slow way: re-fit the read-out
without each head, one at a time. The numbers are the same.
""")

code(r"""
torch.manual_seed(0)                               # fixed starting weights
small_rule = replace(SMALL, supply="local", num_heads=4, d_k=4, m_content=8, d_model=32)  # a 4-head supplied layer
m4 = HemoAttn(small_rule)                          # the real supplied layer
X, Y, T, _ = make_batch(256, small_rule, "cpu")    # a batch of the task
m4.open_mask[3] = False                            # close head 3
m4.probe_ischemia(X, Y, T)                         # the real collateral values (shortcut formula)

with torch.no_grad():
    _, _, out = m4.heads(X, Y)                     # every head's blends
Z = out.permute(0, 2, 1, 3).reshape(-1, 16).double(); Z -= Z.mean(0)   # heads side by side, centred
t = T.reshape(-1, T.size(-1)).double(); t -= t.mean(0)                  # answers, centred
n = Z.size(0)
A = Z.T @ Z / n                                    # least-squares ingredients, as in the real code
A += 1e-4 * torch.diagonal(A).mean() * torch.eye(16, dtype=A.dtype)     # the same tiny ridge
Bm = Z.T @ t / n


def fit_loss(hs):                                  # error of the best read-out from heads hs (slow way)
    idx = torch.cat([torch.arange(h * 4, h * 4 + 4) for h in hs])       # the columns of those heads
    explained = torch.trace(Bm[idx].T @ torch.linalg.solve(A[idx][:, idx], Bm[idx]))
    return float(((t ** 2).sum() / n - explained) / t.size(1))


open_ = [0, 1, 2]                                  # the open heads
for h in range(4):
    if h in open_:                                 # open head: loss rise if removed, others re-fit
        brute, kind = fit_loss([o for o in open_ if o != h]) - fit_loss(open_), "open, loss if removed"
    else:                                          # closed head: loss fall if fed again
        brute, kind = fit_loss(open_) - fit_loss(open_ + [h]), "closed, gain if fed"
    print(f"head {h} ({kind:<21}): real code {float(m4.head_value[h]):.6f}   slow way {brute:.6f}")
""")

# ------------------------------------------------------------------ 6b
md(r"""
## 6b. Everything as matrices, by hand

This section puts every piece together with real matrices small enough to compute with a
pencil: **one query, 3 memory slots, 2 heads, head size $d_k = 1$, one output number.** The
numbers are chosen so the softmax comes out in quarters and halves.

### The shapes

In the real model ($N$ slots, $H$ heads) and in this example ($N = 3$, $H = 2$, $d_k = 1$):

| object | meaning | real shape | here |
|---|---|---|---|
| $x$ | the query: one-hot of its position $p$ | $1 \times N$ (plus content) | $1 \times 3$ |
| $Y$ | the memory: one row per slot | $N \times d_\text{model}$ | $3 \times 3$ (identity: slot $j$ is $e_j$) |
| $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$ | head $h$'s query, key, value weights | $d_\text{model} \times d_k$ | $3 \times 1$ |
| $q_h = x\,W_Q^{(h)}$ | head $h$'s query | $1 \times d_k$ | $1 \times 1$ |
| $K_h = Y W_K^{(h)}$, $V_h = Y W_V^{(h)}$ | keys and values, one row per slot | $N \times d_k$ | $3 \times 1$ |
| $s_h = K_h\, q_h^{\top}$ | score of each slot | $N \times 1$ | $3 \times 1$ |
| $a_h = \mathrm{softmax}(s_h)$ | attention weights, sum to 1 | $N \times 1$ | $3 \times 1$ |
| $o_h = a_h^{\top} V_h$ | the head's blend | $1 \times d_k$ | $1 \times 1$ |
| $W_O^{(h)}$ | head $h$'s output weights | $d_k \times d_\text{out}$ | $1 \times 1$ |
| $g_h$ | head $h$'s supply valve | number | number |
| $y = \sum_h g_h\, o_h W_O^{(h)}$ | the layer's output | $1 \times d_\text{out}$ | $1 \times 1$ |

(The real code scales the scores by $1/\sqrt{d_k}$; with $d_k = 1$ that is 1.)

### The numbers

Query at position 0, so $x = (1, 0, 0)$. Memory $Y = I_3$.

| | $W_Q$ | $W_K$ | $W_V$ | $W_O$ |
|---|---|---|---|---|
| head 1 | $(\ln 2,\ 0,\ 0)^\top$ | $(0,\ 0,\ 1)^\top$ | $(4,\ 8,\ 2)^\top$ | $0.5$ |
| head 2 | $(\ln 2,\ 0,\ 0)^\top$ | $(1,\ 0,\ 0)^\top$ | $(6,\ 2,\ 2)^\top$ | $0.25$ |

**Your turn, with a pencil (head 1).**
1. $q_1 = x W_Q^{(1)}$: the query picks row 0 of $W_Q$, so $q_1 = \ln 2$.
2. $K_1 = Y W_K^{(1)} = (0, 0, 1)^\top$, because $Y$ is the identity.
3. Scores $s_1 = K_1 q_1 = (0,\ 0,\ \ln 2)$.
4. $e^{s} = (1,\ 1,\ 2)$, total 4, so $a_1 = (\tfrac14,\ \tfrac14,\ \tfrac12)$.
5. $V_1 = (4, 8, 2)^\top$, so $o_1 = \tfrac14\cdot 4 + \tfrac14 \cdot 8 + \tfrac12 \cdot 2 = 1 + 2 + 1 = 4$.

**Head 2:** scores $(\ln 2, 0, 0)$, $e^s = (2, 1, 1)$, $a_2 = (\tfrac12, \tfrac14, \tfrac14)$,
$o_2 = 3 + 0.5 + 0.5 = 4$.

**Output with the valves.** $y = g_1 \cdot 4 \cdot 0.5 + g_2 \cdot 4 \cdot 0.25 = 2g_1 + g_2$.
Both on, $g = (1, 1)$: $y = 3$. Head 2 off, $g = (1, 0)$: $y = 2$, exactly what a one-head
layer of head 1 gives. The valve is just a number multiplying the head's contribution.

(Notice that the score of slot $j$ for a query at $p$ is $W_Q[p] \cdot W_K[j]$, one entry of
the matrix $W_Q W_K^{\top}$. That matrix is what draws the stripes in the attention maps of
the setup figure.)
""")

code(r"""
x = np.array([[1.0, 0, 0]])                      # (1, 3): the query at position 0 (one-hot)
Y = np.eye(3)                                    # (3, 3): memory, slot j = row j of the identity
heads = {1: dict(WQ=[[np.log(2)], [0], [0]], WK=[[0], [0], [1]], WV=[[4], [8], [2]], WO=0.5),   # head 1's weights
         2: dict(WQ=[[np.log(2)], [0], [0]], WK=[[1], [0], [0]], WV=[[6], [2], [2]], WO=0.25)}  # head 2's weights
outs = {}                                        # each head's output, filled below
for h, w in heads.items():                       # for each head:
    q = x @ np.array(w["WQ"])                    #   query = x W_Q: picks row 0 of W_Q -> ln 2   (1, 1)
    K, V = Y @ np.array(w["WK"]), Y @ np.array(w["WV"])   #   keys and values of the 3 slots      (3, 1) each
    s = (K @ q.T).ravel()                        #   one score per slot                          (3,)
    a = np.exp(s) / np.exp(s).sum()              #   softmax: e^score / total                    (3,)
    outs[h] = float(a @ V.ravel())               #   the blend: weights times values, added up
    print(f"head {h}: scores {s.round(3)}  weights {a.round(3)}  output {outs[h]:.3f}")
for g in ((1, 1), (1, 0)):                       # two settings of the valves
    y = sum(g[i] * outs[h] * heads[h]["WO"] for i, h in enumerate(heads))  # gain x output x output weight, summed
    print(f"valves g = {g}: y = {y:.3f}")
assert np.isclose(outs[1], 4) and np.isclose(outs[2], 4)  # must match the pencil answers
""")

md(r"""
### The collateral value, by hand

Now three heads observed on **three examples** (one number per head per example), and a
target the layer must produce:

| example | head 1 | head 2 | head 3 | target $t$ |
|---|---|---|---|---|
| 1 | 1 | 2 | 1 | 2 |
| 2 | 2 | 4 | 0 | 2 |
| 3 | 3 | 6 | 1 | 4 |

Head 2 is exactly twice head 1 (a copy, scaled). The target is head 1 + head 3.

**All three heads on.** The fit $t = w_1 h_1 + w_2 h_2 + w_3 h_3$ is perfect whenever
$w_1 + 2w_2 = 1$ and $w_3 = 1$; least squares picks the smallest such weights,
$(w_1, w_2, w_3) = (0.2,\ 0.4,\ 1)$. Error 0.

**Collateral value of head 1** (remove it, let the others re-fit). Heads 2 and 3 can still
make the target: $w_2 = 0.5$, $w_3 = 1$. Error stays 0, so **value = 0**. Head 2 covers for it.

**Collateral value of head 3.** Only heads 1 and 2 remain, and both point along $(1, 2, 3)$.
The best single weight on $u = (1, 2, 3)$ is
$w = \dfrac{t \cdot u}{u \cdot u} = \dfrac{2 + 4 + 12}{1 + 4 + 9} = \dfrac{18}{14} = \dfrac97.$
The leftover is $t - \tfrac97 u = (\tfrac57,\ -\tfrac47,\ \tfrac17)$, whose mean square is
$\tfrac13 \cdot \tfrac{25 + 16 + 1}{49} = \tfrac{2}{7} \approx 0.286$. **Value = 0.286**:
nobody can cover head 3.

**Ordinary importance of head 1** (remove it, nobody adjusts). The fit loses $0.2 \times$
head 1, so the error is the mean of $(0.2 \cdot (1, 2, 3))^2 = 0.04 \cdot \tfrac{14}{3}
\approx 0.187$. Ordinary importance calls head 1 important; the collateral value correctly
says it is covered.

**Your turn.** Redo these four numbers with `np.linalg.lstsq` (no intercept column here).
""")

code(r"""
h1, h2, h3 = np.array([1, 2, 3.]), np.array([2, 4, 6.]), np.array([1, 0, 1.])  # each head's output on the 3 examples
t = np.array([2, 2, 4.])                         # the target (= head 1 + head 3)
cols = {1: h1, 2: h2, 3: h3}                     # look up a head's outputs by its number


def fit_error(names):                            # best weights and error using only these heads
    Z = np.stack([cols[n] for n in names], 1)    # the chosen heads as columns
    w = np.linalg.lstsq(Z, t, rcond=None)[0]     # least-squares weights (the smallest, if several fit)
    return w, float(np.mean((t - Z @ w) ** 2))   # weights, and the mean squared error


w_all, e_all = fit_error([1, 2, 3])              # all heads on
print("all heads: weights", w_all.round(3), " error", round(e_all, 4))
print("collateral value of head 1:", round(fit_error([2, 3])[1] - e_all, 4))   # remove head 1, others re-fit
print("collateral value of head 3:", round(fit_error([1, 2])[1] - e_all, 4), "  (2/7 =", round(2 / 7, 4), ")")
no_refit = np.stack([h2, h3], 1) @ w_all[1:]     # remove head 1 but keep the others' old weights
print("ordinary importance of head 1:", round(float(np.mean((t - no_refit) ** 2)) - e_all, 4))
""")

# ------------------------------------------------------------------ 6c
md(r"""
## 6c. Where this lives in the real code

Below are the actual functions from `src/hemo/model.py`, printed straight from the source so
they can never go out of date. Under each one: what every part does, and which step of 6b it
is.

### (1) The heads: `CrossAttn.heads`
""")

code(r"""
# the same function as in section 4: steps 1 to 5 of 6b for all heads at once
show(CrossAttn.heads, HEADS_NOTES)
""")

md(r"""
- `self.W_q(X)`, `self.W_k(Y)`, `self.W_v(Y)` compute $q$, $K$, $V$ for **all heads at once**
  (one big matrix), and `_split` cuts the result into $H$ heads of size $d_k$. Shapes:
  `(batch, H, N, d_k)`.
- `Q @ K.transpose(-2, -1) / sqrt(d_k)` is the score of every query against every slot
  (step 3 of 6b), and `softmax` turns scores into weights (step 4).
- `attn @ V` is each head's blend (step 5). The function returns the query vectors, the
  attention weights and the blends.

### (2) The valve: `CrossAttn.combine`
""")

code(r"""
show(CrossAttn.combine, COMBINE_NOTES)              # y = sum over heads of g_h o_h W_O
""")

md(r"""
- `out * gate[:, :, None, None]` multiplies **each head's blend by its valve $g_h$**. This one
  multiplication is where the supply attaches to the heads.
- `.transpose(1, 2).reshape(...)` lays the $H$ blends side by side, and `self.W_o(...)` applies
  the output weights. Together that is $y = \sum_h g_h\, o_h W_O^{(h)}$ from 6b.

### (3) Where the valve values come from: the `local` branch of `HemoAttn.gate`
""")

code(r"""
src = inspect.getsource(HemoAttn.gate).splitlines()             # the whole gate function ...
start = next(i for i, l in enumerate(src) if 'supply_kind == "local"' in l)  # ... find hemodynamic attenuation's branch
annotated(src[start:start + 4], {
    'supply_kind == "local"': "hemodynamic attenuation",
    "if not self.conserve": "the version we use (no fixed total)",
    "return self.tone.clone()": "valve = tone: 1 on, 0 off, in between while fading",
    "return self.H * self.tone / self.tone.sum()": "original version: rescale so the valves add up to H",
})
""")

md(r"""
- `self.tone` holds one number per head between 0 and 1: 1 is fully on, 0 is off, values in
  between mean the head is fading.
- With `conserve = 0` (the version we use, since the fixed total did not earn its place), the
  valve **is** the tone. With `conserve = 1` the tones are rescaled so they always add up to
  $H$ (the original "fixed total blood supply").

### (4) The collateral value: `HemoAttn.probe_ischemia`
""")

code(r"""
show(HemoAttn.probe_ischemia, {
    "H, dk = self.H, self.d_k": "number of heads and size of each head",
    "_, _, out = self.heads(X, Y)": "run every head on a fresh batch; keep each head's blends",
    "Z = out.permute": "all heads' blends side by side: one row per token",
    "t = T.reshape": "the correct answers, one row per token",
    "Z, t = Z - Z.mean(0)": "centre both (plays the role of the intercept)",
    "n, d_out = Z.size(0)": "number of tokens, answer size",
    "A, Bm = Z.T @ Z / n": "the two ingredients of a least-squares fit",
    "A += ridge": "a tiny ridge so the inverse always exists",
    "open_ = self.open_mask": "which heads are open now",
    "blk = torch.arange": "which columns of Z belong to which head",
    "if self.head_dropout > 0": "damage experiment: average the values over random failures",
    "self._probe_under_damage": "(damage experiment)",
    "S = blk[open_]": "the columns of the open heads",
    "M = torch.linalg.inv(A[S][:, S])": "one inverse for the fit that uses all open heads",
    "W = M @ Bm[S]": "best read-out from the open heads (the 'all heads on' fit)",
    "value = torch.zeros": "one collateral value per head, filled below",
    "ablate = torch.zeros": "ordinary importance per head (for the control)",
    "heads_open = torch.nonzero(open_)": "the list of open heads",
    "for i, h in enumerate(heads_open": "for every open head:",
    "J = slice(i * dk": "its columns inside the open-heads fit",
    "value[h] = torch.trace(W[J].T @ torch.linalg.solve(M[J, J], W[J]))": "rise in error if it is removed and the others re-fit (shortcut formula)",
    "Jg = blk[h]": "its columns in the full matrix",
    "ablate[h] = torch.trace": "rise in error if it is removed and nobody re-fits",
    "self.head_ablate.copy_": "store the ordinary importance",
    "shut = torch.nonzero(~open_)": "the closed (starved) heads",
    "if len(shut):": "if there are any closed heads:",
    "for i, h in enumerate(shut.tolist())": "for every closed head:",
    "Jall = blk[shut]": "their columns",
    "C = A[Jall][:, S] @ M": "how much each closed head overlaps the open ones ...",
    "R = Bm[Jall] - C @ Bm[S]": "... what it could add that the open heads do not already give",
    "Sc = A[Jall][:, Jall]": "... and its new information (a Schur complement)",
    "value[h] = torch.trace(R[J].T": "fall in error if this closed head were fed again (used to reopen)",
    "self.head_value.copy_": "store the values for the decision step",
    "return": "(damage experiment) done",
})
""")

md(r"""
This is section 6 and the second half of 6b, done for 32 heads at once and fast:
- `_, _, out = self.heads(X, Y)` runs every head on a fresh batch and keeps the blends.
- `Z` puts all the heads' blends side by side (one row per token); `t` is the target. Both are
  centred, which plays the role of the intercept.
- `A = Z.T @ Z / n` and `Bm = Z.T @ t / n` are the pieces of the least-squares fit, and
  `W = M @ Bm[S]` is the best read-out using the open heads (the "all heads on" fit of 6b).
- For each open head, `trace(W[J].T @ solve(M[J, J], W[J]))` is **the rise in error when
  that head is removed and the others re-fit**. It is a shortcut formula: refitting
  once per head would give the same number (the older walkthrough checks this numerically),
  but this needs only one matrix inverse.
- For each starved head (the `shut` block), the same algebra gives how much the error would
  *fall* if the head were fed again; that is what reopening uses.
- `ablate[h]` is the ordinary importance (removed, nobody re-fits), kept for the control.

### (5) The decision and the fade: `local_step` and `relax_tone`
""")

code(r"""
show(HemoAttn.local_step, {
    "v, open_ = self.head_value": "the values from the probe, and which heads are open",
    "if int(open_.sum()) > 1:": "never close the last head",
    "score = self.head_ablate if": "controls only: the ordinary-importance version",
    "cheapest = torch.where(open_, score": "each open head's value (closed heads count as infinity)",
    "h = int(cheapest.argmin())": "the cheapest open head",
    "if float(cheapest[h]) < price:": "worth less than the price?",
    'if self.local_value == "random"': "control only: pick a random open head instead",
    "idx = torch.nonzero(open_)": "(random control) the open heads",
    "h = int(idx[": "(random control) one of them at random",
    "open_[h] = False": "close it; its tone then fades to 0",
    "best = torch.where(~open_": "otherwise: each closed head's value if fed again",
    "h = int(best.argmax())": "the most valuable closed head",
    "if float(best[h]) > 2 * price:": "worth more than twice the price? (factor 2 stops flickering)",
    "open_[h] = True": "reopen it",
})
print()
show(HemoAttn.relax_tone, {
    "target = self.open_mask.to": "1 for open heads, 0 for closed heads",
    "if self.taper <= 0:": "no fading: jump straight to the target",
    "self.tone.copy_(target)": "(no fading)",
    "speed = torch.full_like": "how far a tone may move per step: 1/taper",
    "if self.trial_head is not None": "trial-closure experiment only: fade that head more slowly",
    "speed[self.trial_head]": "(trial closure)",
    "self.tone.add_(": "move each tone toward its target by at most the speed",
})
""")

md(r"""
- `local_step`: find the cheapest open head; if it is worth less than the price, close it
  (set `open_mask[h] = False`). Otherwise, if some closed head would now be worth more than
  twice the price, reopen it. One change per check. The `ablate` and `random` lines are the
  two controls.
- `relax_tone`: every training step, move each head's tone toward 1 (open) or 0 (closed) by
  at most `1/taper`. A closing head fades out over `taper` steps (100 in the experiments), so
  the others take over its job gradually.

**How the pieces connect during training** (in `src/hemo/train.py`): every step runs the
model with the current valves and updates the weights, then calls `relax_tone`; every
`probe_every = 25` steps after the start, it calls `probe_ischemia` on a fresh batch and then
`local_step` with the price. Section 7 below is exactly this loop, written from scratch.
""")

# ------------------------------------------------------------------ 6d
md(r"""
## 6d. The whole task in L-shapes (spreadsheet style)

The same computation on our task, with every matrix on the page: **4 memory slots, 4 queries
(one per position), 2 heads**. Head 1 should look 1 slot ahead, head 2 three slots ahead.

**How to read an L-shape.** To multiply $A \times B$, put $B$ above and $A$ to the left; the
result sits where they meet, and each result cell is "its row of $A$" times "its column of
$B$". In the figure: $Q$ above, $K^\top$ to the left, the scores $S$ in the corner; then the
softmax above, $V$ to the left, the head's output in the corner.

**What to check by hand** (head 1, query at position 0):
1. $Q = W_q X$: $W_q$ is 12 times a shift, so query $p$ becomes a 12 at row $p + 1$.
2. $K = W_k Y$ keeps only the slot positions, so $K$ is the identity.
3. $S = K^\top Q / \sqrt{4}$: the score of slot $j$ for query $p$ is 6 if $j = p + 1$, else 0.
   That is the **stripe**.
4. Softmax of a column $(0, 6, 0, 0)$: $e^6 = 403.4$, total $406.4$, so the weights are
   $(0.002,\ 0.993,\ 0.002,\ 0.002)$.
5. Output for query 0: $0.002 \cdot 3 + 0.993 \cdot 1 + 0.002 \cdot 4 + 0.002 \cdot 2 \approx 1.01$,
   the content of slot 1, which is the target.
6. Valve 2 closed: head 2's row of the combined output becomes 0, exactly.
""")

code(r"""
from IPython.display import Image, display      # show a picture inside the notebook
display(Image("../figures/fig_matrices.png", width=900))  # the L-shape figure
""")

code(r"""
# the same numbers, recomputed: change SCALE, the contents or the shifts and rerun
Nn, dd, SCALE = 4, 4, 12.0                              # 4 slots, head size 4, sharpness of the query weights
contents = np.array([3.0, 1.0, 4.0, 2.0])               # the item stored in each slot
Xq = np.eye(Nn)                                         # queries: column t = one-hot of position t
Ym = np.vstack([np.eye(Nn), contents])                  # memory: column j = [one-hot j ; content j]
shift = lambda k: np.roll(np.eye(Nn), k, axis=0)        # matrix that moves position p to p + k
for h, k in ((1, 1), (2, 3)):                           # head 1 looks 1 ahead, head 2 looks 3 ahead
    Q = SCALE * shift(k) @ Xq                           # Q = Wq X: query p becomes "slot p + k"
    K = np.hstack([np.eye(Nn), np.zeros((Nn, 1))]) @ Ym # K = Wk Y: keep the slot positions only
    V = np.array([[0, 0, 0, 0, 1.0]]) @ Ym              # V = Wv Y: keep the contents only
    S = K.T @ Q / np.sqrt(dd)                           # scores, slots x queries: the stripe of 6s
    A = np.exp(S) / np.exp(S).sum(0, keepdims=True)     # softmax down each column (each query's weights)
    print(f"head {h}: output {np.round(V @ A, 2).ravel()}   target {contents[(np.arange(Nn) + k) % Nn]}")  # V x softmax
""")

# ------------------------------------------------------------------ 7
md(r"""
## 7. Hemodynamic attenuation, running during training

**Idea.** Train with all heads on for a while. Then, every 25 steps:
1. compute every open head's collateral value on a fresh batch (`probe_ischemia`);
2. if the cheapest head is worth less than a **price**, close it (`local_step`);
3. a closed head **fades** out gradually (`relax_tone`, its valve falls by 1/100 per step), so
   the others take over its job smoothly, as vessels constrict gradually.

This is the real `train` function with `supply="local"`, the settings used in the experiments
(price 0.03 of the trivial loss, fade over 100 steps, no fixed total), on the small task: 8
heads, and the task needs 2. Here are the lines of `train` that run the rule:
""")

code(r"""
src = inspect.getsource(train).splitlines()
keep = [l for l in src if any(k in l for k in (
    "for step in range(total)", "X, Y, T, _ = make_batch(cfg.batch_size", "pred, g = model(X, Y", "loss = F.mse_loss(pred, T)",
    "opt.step()", "model.relax_tone()", "model.probe_ischemia(*make_batch", "model.local_step(price)",
    "price = cfg.price_frac", "if (step - hold) % cfg.probe_every == 0"))]
annotated(keep, {
    "price = cfg.price_frac": "the price of one head (fraction of the trivial loss)",
    "for step in range(total)": "each training step:",
    "X, Y, T, _ = make_batch": "a fresh batch of the task",
    "pred, g = model(X, Y": "run the model with the current valves",
    "loss = F.mse_loss": "how wrong it is",
    "opt.step()": "update the weights",
    "model.relax_tone()": "move every valve toward open (1) or closed (0) by at most 1/taper",
    "if (step - hold) % cfg.probe_every": "every 25 steps after the start:",
    "model.probe_ischemia(": "value every head: what nobody else can cover",
    "model.local_step(price)": "close the cheapest head if it is worth less than the price",
})
""")

code(r"""
RULE_SMALL = replace(SMALL, supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.2,
                     conserve=0, probe_batch=256)  # hemodynamic attenuation, experiment settings, small task
results = []
for seed in range(3):                              # three runs with different seeds
    model, hist = train(replace(RULE_SMALL, seed=seed), val, "cpu", hemo=True)   # the real training loop
    kept_now = int((hist["ledger"][-1] > 0).sum()) # heads whose valve is above 0 at the end
    results.append(hist)
    print(f"seed {seed}: heads kept {kept_now} of 8 (task needs {R});  final loss {hist['val_loss'][-1]:.4f}")

hist = results[0]                                  # plot the first run
fig, ax = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True)   # two plots stacked, same x axis
ax[0].semilogy(hist["val_step"], hist["val_loss"], color=RED, lw=0.8); ax[0].set_ylabel("loss")   # loss, log scale
ax[1].plot((hist["ledger"] > 0).sum(1), color=RED); ax[1].axhline(R, color=GREY, ls=":")         # heads with supply
ax[1].set_ylabel("heads on"); ax[1].set_xlabel("training step")
ax[0].set_title("The real hemodynamic attenuation on the small task (seed 0)", loc="left")
plt.tight_layout(); plt.show()                     # draw it
""")

md(r"""
The same function, with 32 heads and the full task, produced every result in section 8.
""")

# ------------------------------------------------------------------ 8
md(r"""
## 8. What the full experiments found (main task, 32 heads)

The repo's version: 32 heads, $N = 16$, tasks needing $k^\star = 2, 3, 4, 6, 8$ heads, 10 seeds
each. The controls change one ingredient at a time:
- **ordinary importance**: value a head *without* letting the others adjust;
- **random choice**: close a head at the same moments, but chosen at random;
- **standard pruning** (Michel et al. 2019): remove the least important head until the loss
  gets worse.

Two numbers matter: did it keep **exactly** $k^\star$ heads, and did it **never break the
model** (never go back above the "solved" loss) while removing heads?
""")

code(r"""
from hemo.config import Cfg                       # the settings every run was made with

DEFAULT = Cfg()                                    # default settings (older runs did not store newer ones)
RUNS = [pickle.load(open(f, "rb")) for d in ("../results", "../results/proposal", "../results/confirm", "../results/l0")
        for f in glob.glob(d + "/*.pkl")]          # load every saved run
RUNS = [r for r in RUNS if isinstance(r, dict) and "hist" in r and "ledger" in r.get("hist", {})]  # keep training runs
PIN = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0, probe_every=25,
           prune_stop=1, conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0,
           trial=0, target_frac=0.02, budget_hold_frac=0.25, taper=0, price_frac=0.01, l0_lambda=0.01)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))   # a run's setting (default if not stored)
runs_of = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PIN, **w}.items())]  # runs matching settings


def kept(r):                                       # heads with supply, median over the last 40% of training
    on = (r["hist"]["ledger"] > 0).sum(1)          # heads with a valve above 0, at every step
    return float(np.median(on[int(0.6 * len(on)):]))  # median over the last 40% of training


def stayed_solved(r):                              # never back above the solved bar after first reaching it
    h, bar = r["hist"], 0.02 * r["trivial"]        # the bar: 2% of a know-nothing model's loss
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]  # losses after pruning starts
    first = next((i for i, l in enumerate(ls) if l < bar), None)   # first time it was solved
    return first is not None and max(ls[first:]) <= bar             # never above the bar afterwards


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)  # hemodynamic attenuation's settings
arms = {"hemodynamic attenuation": (RULE, (2, 3, 4, 6, 8)),     # each rule: (its settings, task sizes run)
        "ordinary importance": ({**RULE, "local_value": "ablate"}, (2, 4, 8)),
        "random choice": ({**RULE, "local_value": "random"}, (2, 4, 8)),
        "standard pruning": (dict(supply="prune"), (2, 3, 4, 6, 8))}
print(f"{'rule':<34}{'runs':>5}{'exact count':>13}{'never broke':>13}")
for name, (kw, sizes) in arms.items():             # one line per rule
    rs = [r for k in sizes for r in runs_of(**kw, n_rel=k)]   # all its runs, every task size
    print(f"{name:<34}{len(rs):>5}{sum(kept(r) == get(r, 'n_rel') for r in rs):>9}/{len(rs)}"
          f"{sum(stayed_solved(r) for r in rs):>9}/{len(rs)}")
for lam in (0.05, 0.2, 1.0):                       # the learned-gate baseline at three penalty strengths
    rs = [r for k in (2, 4, 8) for r in runs_of(supply="l0", budget_hold_frac=0.1, l0_lambda=lam, n_rel=k)]  # its runs
    print(f"{'learned gates, penalty ' + str(lam):<34}{len(rs):>5}{sum(kept(r) == get(r, 'n_rel') for r in rs):>9}/{len(rs)}"
          f"{sum(stayed_solved(r) for r in rs):>9}/{len(rs)}")
""")

md(r"""
Only hemodynamic attenuation does both. Ordinary importance keeps redundant heads; random
choice and standard pruning get the count but break the model on the way. The classic
learned-gate method (Voita et al. 2019: a learnable on/off gate per head plus a penalty for
every open head) keeps too many heads at a weak penalty and cuts too far at a strong one;
no single penalty works for every task size. Two more checks
from the same runs: started with every head duplicated, hemodynamic attenuation keeps exactly 4
heads and never both copies of a pair (10 of 10 seeds), and the right count holds at every
price from 0.01 to 0.1.
""")

# ------------------------------------------------------------------ 9
md(r"""
## 9. Backup heads when heads can fail

**Idea.** If heads fail at random during training (each with chance $p$), a spare head
becomes worth something: it saves the task when another fails. Any $R$ working heads can do
the job (they blend all the answers), so a head is worth something only when, without it,
fewer than $R$ heads would be working:

$$v(B) = \frac{1-p}{R}\ \Pr\big[\text{fewer than } R \text{ of the other } B-1 \text{ heads work}\big].$$

The rule keeps adding heads while $v(B)$ is at least the price.

**By hand**, $R = 4$, $p = 0.1$, price $0.03$: with $B = 5$ heads, the other 4 all work with
chance $0.9^4 = 0.656$, so fewer than 4 work with chance $0.344$, and
$v(5) = 0.9 \times 0.344 / 4 = 0.077 > 0.03$: keep 5. With $B = 6$:
$v(6) = 0.9 \times 0.081 / 4 = 0.018 < 0.03$: stop. Prediction: **5 heads**.

**Your turn.** Write `predict(R, p, price)` with `math.comb` for the binomial probability,
and check it returns 5 for the example.
""")

code(r"""
def predict(R, p, price=0.03):                   # heads the rule should keep when heads fail with chance p
    q = 1 - p                                      # chance a head works
    def value(B):                                  # worth of one head when B heads are kept
        fewer = sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1))  # P[fewer than R of the others work]
        return q * fewer / R                       # it must work itself, and it saves 1/R of the loss
    return max(B for B in range(1, 33) if value(B) >= price)   # the largest B still worth its price


print("predicted heads kept, R = 4, p = 0.1:", predict(4, 0.1))
assert predict(4, 0.1) == 5                        # must match the hand calculation
print("\nREAL RESULT (predictions written before the runs):")
print(f"{'R':>2} {'p':>5} {'predicted':>10} {'measured (per seed)':>22}")
for Rr in (2, 4):                                  # tasks needing 2 and 4 heads
    for pp in (0.05, 0.1, 0.2, 0.3):               # four failure rates
        rs = runs_of(**RULE, n_rel=Rr, head_dropout=pp)  # the runs at this task size and failure rate
        print(f"{Rr:>2} {pp:>5} {predict(Rr, pp):>10} {str([int(kept(r)) for r in rs]):>22}")
""")

md(r"""
**Do the spares actually protect the model?** After training with failures, switch off one
kept head at a time (nobody re-fits) and measure the loss, in units of the solved bar.
""")

code(r"""
ROB = [pickle.load(open(f, "rb")) for f in glob.glob("../results/robust/*.pkl")]   # runs with the knockout check
print(f"{'p fail':>7}{'heads kept':>16}{'worst head removed (x bar)':>30}")
for pp in sorted({r["cfg"].get("head_dropout", 0.0) for r in ROB}):          # each failure rate
    rs = [r for r in ROB if r["cfg"].get("head_dropout", 0.0) == pp]          # the runs at this failure rate
    bar = 0.02 * rs[0]["trivial"]                                             # the solved bar
    worst = np.median([max(r["knockout"]["per_head"].values()) / bar for r in rs])   # worst single head removed
    print(f"{pp:>7}{str([len(r['knockout']['per_head']) for r in rs]):>16}{worst:>30.1f}")
""")

md(r"""
With no spares, losing one head breaks the model (about 31 times the bar). With spares the
damage is about 2.4 times smaller, but still well above the bar: the spares reduce, but do
not remove, the damage of losing a head.
""")

# ------------------------------------------------------------------ 10
md(r"""
## 10. A second circuit: two-layer induction

**Idea.** In a sequence like `... 45 26 38 57 ... 45 26 38 ?`, the next token is `57`: find
the earlier copy of the current token and copy what came after it. Language models learn
this with two heads in two layers:
- a **previous-token head** (layer 1): each position looks one step back, so position `57`
  learns "the token before me was `38`";
- an **induction head** (layer 2): at the current `38`, look for the position whose previous
  token was `38`, which is `57`, and copy it.

So the known answer is **one head in each layer**. One layer alone cannot do it, whatever its
size (checked: about 16% accuracy against 99.8% with one head per layer).

**By hand.** Sequence positions: `0:11 1:45 2:26 3:38 4:57 5:14 6:45 7:26 8:38`. At position 8
(token 38) the induction head should attend to position 4: the earlier 38 is at 3, plus one.

**The model** (`src/hemo/induction.py`) is a small but complete attention-only transformer:
token and position embeddings, two causal self-attention layers added into a "residual
stream", and an unembedding that scores every possible next token. Every head has a valve,
exactly as in the main task. Here is its code with a note on every line:
""")

code(r"""
from hemo.induction import AttnLayer, InductionNet   # the two-layer induction model

show(AttnLayer.__init__, {
    "super().__init__()": "standard PyTorch set-up",
    "self.H, self.d_head = H, d_head": "heads in this layer, and the size of each head",
    "self.W_q = nn.Linear": "query weights for all heads of this layer",
    "self.W_k = nn.Linear": "key weights for all heads",
    "self.W_v = nn.Linear": "value weights for all heads",
    "self.W_o = nn.Parameter": "each head's own output weights: (H, d_head, d_model)",
})
print()
show(AttnLayer.forward, {
    "B, T, _ = x.shape": "batch size and sequence length",
    "split = lambda": "cut H*d_head numbers into H heads",
    "q, k, v = split": "queries, keys and values all from the same sequence (self-attention)",
    "scores = q @ k.transpose": "every position's score for every position",
    "mask = torch.ones": "causal mask: a position may look only at itself and earlier positions",
    "attn = F.softmax(scores.masked_fill": "later positions get minus infinity, so weight 0",
    "return torch.einsum": "each head's blend, mapped into the residual stream by its own W_o",
})
""")

code(r"""
show(InductionNet.__init__, {
    "super().__init__()": "standard PyTorch set-up",
    "h1 = cfg.heads if": "heads in layer 1",
    "h2 = cfg.heads if": "heads in layer 2",
    "self.sizes = [h1, h2]": "remember both",
    "self.embed = nn.Embedding": "token embedding: each token id -> a vector",
    "self.pos = nn.Parameter": "position embedding: each position -> a learned vector",
    "self.layers = nn.ModuleList": "the two attention layers",
    "self.unembed = nn.Linear": "turns the final vector into a score for every possible next token",
    "self.H = h1 + h2": "total heads; valves are numbered layer 1 first",
})
print()
show(InductionNet.residual, {
    "x = self.embed(tokens) + self.pos": "start of the residual stream: token + position",
    "gate = torch.ones(self.H": "no valves given: every head on",
    "attns, i = [], 0": "attention patterns (if asked), and which valve comes next",
    "for layer in self.layers:": "layer 1, then layer 2",
    "out, attn = layer(x)": "every head's contribution",
    "x = x + (out * gate": "THE VALVE: each head's contribution times its gain, added into the stream",
    "attns.append": "keep the attention patterns if asked (for the role scores)",
    "i += layer.H": "move on to the next layer's valves",
    "return x, attns": "the final stream (and patterns)",
})
print()
show(InductionNet.forward, {"return self.unembed": "scores for the next token"})
""")

md(r"""
**Real result** (8 heads per layer, 10 seeds):
""")

code(r"""
IND = [pickle.load(open(f, "rb")) for f in glob.glob("../results/induction/*.pkl")]   # every induction run


def kept_layers(r):                                # heads kept in layer 1 and in layer 2
    h1 = r["sizes"][0]                             # number of heads in layer 1
    return int(r["kept"][:h1].sum()), int(r["kept"][h1:].sum())   # (kept in layer 1, kept in layer 2)


def ind_group(**w):                                # runs with these settings (older runs lack "probe": it was "mse")
    return [r for r in IND if all(r["cfg"].get(k, v if k != "probe" else "mse") == v for k, v in w.items())]  # matching runs


groups = {                                         # the three versions of the rule we tried
    "first version (linear read-out values)": dict(rule="collateral", probe="mse", price_frac=0.03, steps=3000),
    "loss-based values, keep it solved": dict(rule="collateral", probe="logit_tone", price_mode="bar",
                                              price_frac=0.02, steps=6000, trial=0, squeeze_at=0.0),
    "... plus trial closure": dict(rule="collateral", probe="logit_tone", price_mode="bar",
                                   price_frac=0.02, steps=6000, trial=1),
}
for name, w in groups.items():                     # one line per version of the rule
    rs = ind_group(**w)                            # its 10 runs
    ks = [kept_layers(r) for r in rs]              # heads kept per layer in each run
    print(f"{name:<42} n={len(rs):>2}  exactly (1, 1): {sum(k == (1, 1) for k in ks)}/{len(rs)}"
          f"  solved: {sum(r['final_loss'] <= r['bar'] for r in rs)}/{len(rs)}  kept: {ks}")
""")

md(r"""
Every kept head has the right role. The loss-based version keeps **one** induction head but
**two** previous-token heads, because those two share the job and the network needs time to
move it onto one. Letting it try (trial closure) reaches exactly one head per layer in 8 of
10 runs, but the failed tries push the loss above the solved bar, so the model no longer
stays solved throughout. On induction the rule finds the right roles with one spare; the
exact minimum costs stability.
""")

# ------------------------------------------------------------------ 11
md(r"""
## 11. What holds, and what does not

**Holds**
- The task needs exactly $R$ heads, and the loss of every smaller model follows
  $(1 - k/R)$ (sections 3, 4).
- A head at zero supply is removed exactly (section 5).
- Valuing a head by what the others cannot cover removes duplicates that ordinary importance
  keeps (sections 6, 8).
- On the main task hemodynamic attenuation keeps exactly the needed heads and never breaks the
  model while pruning, unlike standard pruning (section 8).
- With random head failures it keeps the number of backups a simple formula predicts
  (section 9).

**Does not hold, or is not claimed**
- The "fixed total supply" from autoregulation is not needed (it was tested and dropped).
- Final loss is a few times higher than a fully trained dense model, at about a third of
  the compute.
- On induction it keeps one spare previous-token head; forcing the minimum costs stability
  (section 10).
- Only small synthetic tasks so far; no real language model yet.
- The biology is inspiration (collateral circulation, gradual vessel pruning), not evidence
  about the brain.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(ROOT, "notebooks", "walkthrough_by_hand.ipynb"))
print("wrote notebooks/walkthrough_by_hand.ipynb")
