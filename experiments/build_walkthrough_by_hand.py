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

| section | what you build |
|---|---|
| 1 | the task: a memory, a query, and four answers |
| 2 | one attention head |
| 3 | why the task needs exactly $R$ heads (one head, one equation) |
| 4 | a layer of many heads, trained from scratch |
| 5 | the supply valve on each head |
| 6 | the collateral value: what nobody else can cover |
| 6b | everything as matrices, by hand: every shape and every number |
| 6c | where the valve and the collateral value live in the real code |
| 6d | the whole task in L-shapes, spreadsheet style |
| 7 | the collateral rule, running during training |
| 8 | what the full experiments found |
| 9 | backup heads when heads can fail |
| 10 | a second circuit: two-layer induction |
| 11 | what holds and what does not |
""")

code(r"""
import glob, pickle, sys
from math import comb
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

RED, GREY = "#b2182b", "#888888"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
torch.set_num_threads(4)
print("ready")
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

**Your turn.** Write `make_batch(B)` that returns `query, memory, answer`:
1. Draw `B` positions `p` uniformly from `0..N-1`.
2. Draw contents `content` of shape `(B, N, m)` from a standard normal.
3. The query is the one-hot vector of `p`: shape `(B, N)`.
4. The memory row for slot `j` is `[one-hot of j, content of j]`: shape `(B, N, N + m)`.
5. The answer stacks `content[(p + delta_r) mod N]` for each offset: shape `(B, R * m)`.
""")

code(r"""
N, R, m = 8, 2, 4                                   # slots, offsets, content size
OFFSETS = [1 + r * N // R for r in range(R)]          # evenly spread: (1, 5)


def make_batch(B, gen=None):
    p = torch.randint(0, N, (B,), generator=gen)
    content = torch.randn(B, N, m, generator=gen)
    query = F.one_hot(p, N).float()
    memory = torch.cat([torch.eye(N).expand(B, N, N), content], 2)
    idx = (p[:, None] + torch.tensor(OFFSETS)[None]) % N
    answer = torch.gather(content, 1, idx[..., None].expand(B, R, m)).reshape(B, R * m)
    return query, memory, answer, p, content


q, mem, ans, p, content = make_batch(1, torch.Generator().manual_seed(0))
p0 = int(p[0])
print("offsets", OFFSETS, "  query position p =", p0)
for r, d in enumerate(OFFSETS):
    slot = (p0 + d) % N
    print(f"  ({p0} + {d}) mod {N} = {slot}:  answer block {r} equals content of slot {slot}:",
          torch.equal(ans[0, r * m:(r + 1) * m], content[0, slot]))
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
    w = torch.softmax(torch.as_tensor(scores, dtype=torch.float), -1)
    return w, w @ torch.as_tensor(values, dtype=torch.float)


w, out = attend([2.0, 0.0, 0.0], [10.0, 20.0, 30.0])
print("weights", w.numpy().round(2), "  output", round(float(out), 1))
assert abs(float(out) - 13.2) < 0.05
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
rng = np.random.default_rng(0)
Rn, n = 4, 20000
C = rng.normal(size=(n, Rn))                       # one number per unknown, many examples


def best_error(A):
    U = C @ A.T                                    # each row of A is one head's blend
    W = np.linalg.lstsq(U, C, rcond=None)[0]       # best linear guess of C from the blends
    return np.mean((C - U @ W) ** 2)


print("blends  error   predicted 1 - k/4")
for k in range(1, 5):
    print(f"{k:>6}  {best_error(rng.normal(size=(k, Rn))):.3f}   {1 - k / Rn:.3f}")
A = rng.normal(size=(3, Rn))
A_copy = np.vstack([A, A[0]])                      # 4 blends, but the 4th repeats the 1st
print(f"4 blends with one copy: {best_error(A_copy):.3f}  (same as 3 blends: {1 - 3 / Rn:.3f})")
""")

code(r"""
E1 = pickle.load(open("../results/proposal/equations.pkl", "rb"))
print("REAL RESULT: a trained 4-head model, heads removed or copied, read-out re-fitted")
print(f"{'heads used':<22}{'different':>10}{'loss':>8}{'predicted':>11}")
for row in E1["rows"]:
    print(f"{row['name']:<22}{row['different']:>10}{row['loss']:>8.3f}{row['predicted']:>11.3f}")
""")

# ------------------------------------------------------------------ 4
md(r"""
## 4. A layer of many heads, trained from scratch

**Idea.** Put $H$ heads side by side. Each head has its own query, key and value weights and
its own output weights; the layer's answer is the sum of what the heads write.

**Your turn.** Write a `Heads(H)` module with four weight tensors:
`Wq (H, N, d)`, `Wk (H, N+m, d)`, `Wv (H, N+m, d)`, `Wo (H, d, R*m)`. In `outputs`, compute for
every head $q$, $k_j$, $v_j$, the softmax weights over slots and the blend (shape
`(B, H, d)`). In `forward`, multiply each head's blend by its output weights and by a gain
`gate[h]`, and add over heads. Then train it with Adam on mean squared error for 1 and for
2 heads.
""")

code(r"""
class Heads(torch.nn.Module):
    def __init__(self, H, d=8):
        super().__init__()
        self.H, self.d = H, d
        self.Wq = torch.nn.Parameter(torch.randn(H, N, d) * 0.5)
        self.Wk = torch.nn.Parameter(torch.randn(H, N + m, d) * 0.5)
        self.Wv = torch.nn.Parameter(torch.randn(H, N + m, d) * 0.5)
        self.Wo = torch.nn.Parameter(torch.randn(H, d, R * m) * 0.3)

    def outputs(self, query, memory):
        q = torch.einsum("bn,hnd->bhd", query, self.Wq)
        k = torch.einsum("bjn,hnd->bhjd", memory, self.Wk)
        v = torch.einsum("bjn,hnd->bhjd", memory, self.Wv)
        a = torch.softmax(torch.einsum("bhd,bhjd->bhj", q, k) / self.d ** 0.5, -1)
        return torch.einsum("bhj,bhjd->bhd", a, v)           # one blend per head

    def forward(self, query, memory, gate):
        return torch.einsum("bhd,hdo,h->bo", self.outputs(query, memory), self.Wo, gate)


def train_dense(H, steps=2000, seed=0):
    torch.manual_seed(seed)
    model = Heads(H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(steps):
        q, mem, ans, *_ = make_batch(128)
        loss = F.mse_loss(model(q, mem, torch.ones(H)), ans)
        opt.zero_grad(); loss.backward(); opt.step()
    q, mem, ans, *_ = make_batch(2000, torch.Generator().manual_seed(99))
    with torch.no_grad():
        return model, F.mse_loss(model(q, mem, torch.ones(H)), ans).item()


trivial = 1.0                                      # contents have unit variance
for H in (1, 2):
    _, loss = train_dense(H)
    print(f"{H} head(s): loss {loss:.4f}   (formula 1 - k/R: {max(0, 1 - H / R):.2f})")
""")

md(r"""
One head cannot solve a two-offset task; two heads can. The one-head loss (about 0.37) is
below the formula's 0.5, and the reason is worth knowing: the formula assumes a head's
attention depends only on positions, but here the keys also contain the stored contents, so
the attention weights can depend on them and leak a little extra information. In the full
task (more slots, larger contents, noisy inputs) this effect is negligible: the measured
one-head loss there is 0.755 against a predicted 0.756 (section 3).
""")

# ------------------------------------------------------------------ 5
md(r"""
## 5. The supply valve on each head

**Idea.** Each head's contribution is multiplied by a gain $g_h$, its *supply*. $g_h = 1$ is
fully on, $g_h = 0$ is off. A head at $g_h = 0$ is **removed exactly**: the layer is then an
ordinary layer of the remaining heads, and the starved head receives no gradient.

**By hand.** Four heads write the numbers $(3, 7, -1, 5)$. All on: $3 + 7 - 1 + 5 = 14$.
Gains $(1, 0, 1, 0)$: $3 - 1 = 2$, which is exactly a two-head layer of heads 1 and 3.

**Your turn.** With a 4-head `Heads` model, (a) check that gates `(1, 0, 1, 1)` give the same
output whatever head 2's weights are, and (b) check that head 2 gets zero gradient.
""")

code(r"""
torch.manual_seed(0)
model = Heads(4)
q, mem, ans, *_ = make_batch(64)
gate = torch.tensor([1.0, 0.0, 1.0, 1.0])
before = model(q, mem, gate).detach()
with torch.no_grad():
    model.Wv[1] += 100.0                           # change head 2 drastically
print("output unchanged:", torch.allclose(before, model(q, mem, gate)))
loss = F.mse_loss(model(q, mem, gate), ans)
loss.backward()
print("gradient reaching each head's query weights:", model.Wq.grad.abs().sum((1, 2)).numpy().round(4))
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
    # outputs: list of (n, d) arrays, one per head; target: (n, t)
    def err(heads):
        if not heads:
            return float(np.mean((target - target.mean(0)) ** 2))
        Z = np.concatenate([outputs[h] for h in heads] + [np.ones((len(target), 1))], 1)
        W = np.linalg.lstsq(Z, target, rcond=None)[0]
        return float(np.mean((target - Z @ W) ** 2))
    base = err(open_heads)
    return {h: err([g for g in open_heads if g != h]) - base for h in open_heads}


z, w = rng.normal(size=(50000, 1)), rng.normal(size=(50000, 1))
outs = {"A": z, "B": z.copy(), "C": w}
print("collateral values:", {h: round(v, 3) for h, v in collateral_values(outs, z + w, ["A", "B", "C"]).items()})
full = np.hstack([z, z, w]); coef = np.linalg.lstsq(full, z + w, rcond=None)[0]
print("ordinary importance of A (B does not adjust):",
      round(float(np.mean((z + w - full[:, 1:] @ coef[1:]) ** 2)), 3))
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
x = np.array([[1.0, 0, 0]])                      # (1, 3): the query at position 0
Y = np.eye(3)                                    # (3, 3): memory, slot j = e_j
heads = {1: dict(WQ=[[np.log(2)], [0], [0]], WK=[[0], [0], [1]], WV=[[4], [8], [2]], WO=0.5),
         2: dict(WQ=[[np.log(2)], [0], [0]], WK=[[1], [0], [0]], WV=[[6], [2], [2]], WO=0.25)}
outs = {}
for h, w in heads.items():
    q = x @ np.array(w["WQ"])                    # (1, 1)
    K, V = Y @ np.array(w["WK"]), Y @ np.array(w["WV"])   # (3, 1) each
    s = (K @ q.T).ravel()                        # (3,)  scores
    a = np.exp(s) / np.exp(s).sum()              # (3,)  attention weights
    outs[h] = float(a @ V.ravel())               # the blend (V is 3 x 1; ravel makes it 3)
    print(f"head {h}: scores {s.round(3)}  weights {a.round(3)}  output {outs[h]:.3f}")
for g in ((1, 1), (1, 0)):
    y = sum(g[i] * outs[h] * heads[h]["WO"] for i, h in enumerate(heads))
    print(f"valves g = {g}: y = {y:.3f}")
assert np.isclose(outs[1], 4) and np.isclose(outs[2], 4)
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
h1, h2, h3 = np.array([1, 2, 3.]), np.array([2, 4, 6.]), np.array([1, 0, 1.])
t = np.array([2, 2, 4.])
cols = {1: h1, 2: h2, 3: h3}


def fit_error(names):
    Z = np.stack([cols[n] for n in names], 1)
    w = np.linalg.lstsq(Z, t, rcond=None)[0]
    return w, float(np.mean((t - Z @ w) ** 2))


w_all, e_all = fit_error([1, 2, 3])
print("all heads: weights", w_all.round(3), " error", round(e_all, 4))
print("collateral value of head 1:", round(fit_error([2, 3])[1] - e_all, 4))
print("collateral value of head 3:", round(fit_error([1, 2])[1] - e_all, 4), "  (2/7 =", round(2 / 7, 4), ")")
no_refit = np.stack([h2, h3], 1) @ w_all[1:]
print("ordinary importance of head 1:", round(float(np.mean((t - no_refit) ** 2)) - e_all, 4))
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
from IPython.display import Image, display
display(Image("../figures/fig_matrices.png", width=900))
""")

code(r"""
# the same numbers, recomputed: change SCALE, the contents or the shifts and rerun
Nn, dd, SCALE = 4, 4, 12.0
contents = np.array([3.0, 1.0, 4.0, 2.0])
Xq = np.eye(Nn)                                         # queries: column t = position t
Ym = np.vstack([np.eye(Nn), contents])                  # memory: column j = [position j; content j]
shift = lambda k: np.roll(np.eye(Nn), k, axis=0)        # moves position p to p + k
for h, k in ((1, 1), (2, 3)):
    Q = SCALE * shift(k) @ Xq
    K = np.hstack([np.eye(Nn), np.zeros((Nn, 1))]) @ Ym
    V = np.array([[0, 0, 0, 0, 1.0]]) @ Ym
    S = K.T @ Q / np.sqrt(dd)
    A = np.exp(S) / np.exp(S).sum(0, keepdims=True)
    print(f"head {h}: output {np.round(V @ A, 2).ravel()}   target {contents[(np.arange(Nn) + k) % Nn]}")
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
import inspect
sys.path.insert(0, "../src")                       # the repo's own code
from hemo.model import CrossAttn, HemoAttn
print(inspect.getsource(CrossAttn.heads))
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
print(inspect.getsource(CrossAttn.combine))
""")

md(r"""
- `out * gate[:, :, None, None]` multiplies **each head's blend by its valve $g_h$**. This one
  multiplication is where the supply attaches to the heads.
- `.transpose(1, 2).reshape(...)` lays the $H$ blends side by side, and `self.W_o(...)` applies
  the output weights. Together that is $y = \sum_h g_h\, o_h W_O^{(h)}$ from 6b.

### (3) Where the valve values come from: the `local` branch of `HemoAttn.gate`
""")

code(r"""
src = inspect.getsource(HemoAttn.gate).splitlines()
start = next(i for i, l in enumerate(src) if 'supply_kind == "local"' in l)
print("\n".join(src[start:start + 4]))
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
print(inspect.getsource(HemoAttn.probe_ischemia))
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
print(inspect.getsource(HemoAttn.local_step))
print(inspect.getsource(HemoAttn.relax_tone))
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

# ------------------------------------------------------------------ 7
md(r"""
## 7. The collateral rule, running during training

**Idea.** Train with all heads on for a while. Then, every 25 steps:
1. compute every open head's collateral value on a fresh batch;
2. if the cheapest head is worth less than a **price**, close it;
3. a closed head **fades** out gradually (its gain falls by 1/50 per step), so the others
   take over its job smoothly, as vessels constrict gradually.

**Your turn.** Extend `train_dense` into `train_rule(H=8)`:
1. keep a list `open_` of booleans and a tensor `tone` of gains, all ones;
2. use `tone` as the gate in the forward pass;
3. after step 400, every 25 steps, get each head's outputs on a fixed probe batch, call
   `collateral_values`, and close the cheapest open head if its value is below `price = 0.03`;
4. every step, move `tone` toward the open/closed targets by at most 1/50.
Record the loss and the number of heads with supply at every step.
""")

code(r"""
def train_rule(H=8, steps=2000, start=400, every=25, price=0.03, taper=50, seed=0):
    torch.manual_seed(seed)
    model = Heads(H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    open_, tone = [True] * H, torch.ones(H)
    pq, pmem, pans, *_ = make_batch(512, torch.Generator().manual_seed(1))
    history = []
    for step in range(steps):
        q, mem, ans, *_ = make_batch(128)
        loss = F.mse_loss(model(q, mem, tone), ans)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= start and (step - start) % every == 0:
            with torch.no_grad():
                o = model.outputs(pq, pmem).numpy()
            values = collateral_values([o[:, h] for h in range(H)], pans.numpy(),
                                       [h for h in range(H) if open_[h]])
            cheapest = min(values, key=values.get)
            if sum(open_) > 1 and values[cheapest] < price:
                open_[cheapest] = False
        goal = torch.tensor([1.0 if o_ else 0.0 for o_ in open_])
        tone += (goal - tone).clamp(-1 / taper, 1 / taper)
        history.append((loss.item(), int((tone > 0).sum())))
    return open_, np.array(history)


runs = [train_rule(seed=s) for s in range(3)]
for s, (open_, hist) in enumerate(runs):
    print(f"seed {s}: heads kept {sum(open_)} of 8 (task needs {R});  final loss {hist[-50:, 0].mean():.4f}")

hist = runs[0][1]
fig, ax = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True)
ax[0].semilogy(hist[:, 0], color=RED, lw=0.6); ax[0].set_ylabel("loss")
ax[1].plot(hist[:, 1], color=RED); ax[1].axhline(R, color=GREY, ls=":"); ax[1].set_ylabel("heads on")
ax[1].set_xlabel("training step")
ax[0].set_title("The collateral rule on the small task (seed 0)", loc="left")
plt.tight_layout(); plt.show()
""")

md(r"""
That is the whole method in about 30 lines. On this small task it usually lands on the
right number of heads and occasionally keeps one spare. The full experiments below use the
same idea at a larger size, with more seeds and the controls.
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
sys.path.insert(0, "../src")
from hemo.config import Cfg

DEFAULT = Cfg()
RUNS = [pickle.load(open(f, "rb")) for d in ("../results", "../results/proposal", "../results/confirm")
        for f in glob.glob(d + "/*.pkl")]
RUNS = [r for r in RUNS if isinstance(r, dict) and "hist" in r and "ledger" in r.get("hist", {})]
PIN = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0, probe_every=25,
           prune_stop=1, conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0,
           trial=0, target_frac=0.02, budget_hold_frac=0.25, taper=0, price_frac=0.01)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs_of = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PIN, **w}.items())]


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int(0.6 * len(on)):]))


def stayed_solved(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
arms = {"collateral rule": (RULE, (2, 3, 4, 6, 8)),
        "ordinary importance": ({**RULE, "local_value": "ablate"}, (2, 4, 8)),
        "random choice": ({**RULE, "local_value": "random"}, (2, 4, 8)),
        "standard pruning": (dict(supply="prune"), (2, 3, 4, 6, 8))}
print(f"{'rule':<22}{'runs':>5}{'exact count':>13}{'never broke':>13}")
for name, (kw, sizes) in arms.items():
    rs = [r for k in sizes for r in runs_of(**kw, n_rel=k)]
    print(f"{name:<22}{len(rs):>5}{sum(kept(r) == get(r, 'n_rel') for r in rs):>9}/{len(rs)}"
          f"{sum(stayed_solved(r) for r in rs):>9}/{len(rs)}")
""")

md(r"""
Only the collateral rule does both. Ordinary importance keeps redundant heads; random
choice and standard pruning get the count but break the model on the way. Two more checks
from the same runs: started with every head duplicated, the collateral rule keeps exactly 4
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
def predict(R, p, price=0.03):
    q = 1 - p
    def value(B):
        fewer = sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1))
        return q * fewer / R
    return max(B for B in range(1, 33) if value(B) >= price)


print("predicted heads kept, R = 4, p = 0.1:", predict(4, 0.1))
assert predict(4, 0.1) == 5
print("\nREAL RESULT (predictions written before the runs):")
print(f"{'R':>2} {'p':>5} {'predicted':>10} {'measured (per seed)':>22}")
for Rr in (2, 4):
    for pp in (0.05, 0.1, 0.2, 0.3):
        rs = runs_of(**RULE, n_rel=Rr, head_dropout=pp)
        print(f"{Rr:>2} {pp:>5} {predict(Rr, pp):>10} {str([int(kept(r)) for r in rs]):>22}")
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

**Real result** (`src/hemo/induction.py`, 8 heads per layer, 10 seeds):
""")

code(r"""
IND = [pickle.load(open(f, "rb")) for f in glob.glob("../results/induction/*.pkl")]


def kept_layers(r):
    h1 = r["sizes"][0]
    return int(r["kept"][:h1].sum()), int(r["kept"][h1:].sum())


def ind_group(**w):
    return [r for r in IND if all(r["cfg"].get(k, v if k != "probe" else "mse") == v for k, v in w.items())]


groups = {
    "first version (linear read-out values)": dict(rule="collateral", probe="mse", price_frac=0.03, steps=3000),
    "loss-based values, keep it solved": dict(rule="collateral", probe="logit_tone", price_mode="bar",
                                              price_frac=0.02, steps=6000, trial=0, squeeze_at=0.0),
    "... plus trial closure": dict(rule="collateral", probe="logit_tone", price_mode="bar",
                                   price_frac=0.02, steps=6000, trial=1),
}
for name, w in groups.items():
    rs = ind_group(**w)
    ks = [kept_layers(r) for r in rs]
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
- On the main task the collateral rule keeps exactly the needed heads and never breaks the
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
