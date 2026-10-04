"""Writes notebooks/the_code.ipynb: the whole project written from scratch in about 80 lines of plain,
readable code (task, attention with valves, collateral value, the rule, training), run on a small task.
It does what src/hemo does; the last section lists the differences.

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
# The whole project in about 80 lines

**The story.** The memory is a list of N items. A question names an index *p* and asks for the
items at index p + 1 and p + 5 (wrapping round to the start, mod N). An attention head is a
helper that can fetch from one fixed distance, so this task needs exactly **2 helpers** (one per
distance). We start with **8**, and while training we keep asking of each helper: *if it went home,
could the others cover its job?* If yes, it fades out. The goal: end with exactly 2, without ever
breaking the model. (These are the starting sizes; section 1 shows how to change them.)

The code below is written to be read. It does the same thing as the project code in `src/hemo/`
(the last section lists the small differences).
""")

code(r"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.set_num_threads(4)
""")

# ---------------------------------------------------------------- 1
md(r"""
## 1. The sizes

Change **N**, **R** or **M**, then run the notebook from the top (Run All). The vector length D, the
distances and the number of starting heads are worked out from them. A smaller version that still
behaves like the real task: `N = 4, R = 2, M = 4`. Keep R at most N/2, so the distances differ.

**Keep M at 4 or more.** With tiny items (try `M = 1`), one head can cheat: it lets the item values
steer its own attention, so the attention weight itself carries information, and one head almost
solves a two-distance task. "One head, one equation" only holds when items are big enough that this
trick cannot carry them. The real experiments use M = 16.
""")
code(r"""
# change these three ...
N = 8                # length of the memory list: positions with index 0 to 7
R = 2                # how many distances to look ahead; this is also how many heads the task needs
M = 4                # how many numbers make up each item in the list

# ... and everything below follows from them
NOISE = 4                                      # extra noise numbers in every vector (can be 0)
D = N + M + NOISE                              # length of every vector: label + item + noise
DK = 8                                         # size of each head: how many numbers each head works with
HEADS = 4 * R                                  # heads the model starts with: 4 times what it needs
DISTANCES = [1 + r * (N // R) for r in range(R)]   # R distances spread evenly round the list
print(f"memory of {N}, {R} distances {DISTANCES}, items of {M}, vectors of {D}, starting with {HEADS} heads")
""")

# ---------------------------------------------------------------- 2
md(r"""
## 2. The task

Each position in the memory is one vector: `[its index as a one-hot label | its item | a little noise]`.
Each question is `[the one-hot of p | noise]`. The answer is the items at index p + 1 and p + 5 (mod N).
""")
code(r"""
def make_batch(B):                                             # B examples at once
    labels = torch.eye(N).expand(B, N, N)                      # (B, N, N)   position j's label = one-hot of j
    items = torch.randn(B, N, M)                               # (B, N, M)   a random item at every position
    memory = torch.cat([labels, items, 0.1 * torch.randn(B, N, D - N - M)], -1)   # (B, N, D)  X_m
    p = torch.randint(0, N, (B, N))                            # (B, N)      the index each question asks about
    questions = torch.cat([F.one_hot(p, N).float(), 0.1 * torch.randn(B, N, D - N)], -1)   # (B, N, D)  X_q
    rows = torch.arange(B)[:, None]                            # (B, 1)      example b reads its own memory
    answer = torch.cat([items[rows, (p + d) % N] for d in DISTANCES], -1)   # (B, N, R*M)  Y: the items at (p+d) mod N
    return questions, memory, answer
""")
code(r"""
questions, memory, answer = make_batch(1)
print("memory", tuple(memory.shape), "questions", tuple(questions.shape), "answer", tuple(answer.shape))
p = int(questions[0, 0, :N].argmax())                          # the index question 0 asks about
needed = [(p + d) % N for d in DISTANCES]                     # the indices this question needs
print(f"question 0 asks about index {p}: needs the items at index {needed}")
print("its answer is exactly those items:",
      torch.equal(answer[0, 0], torch.cat([memory[0, j, N:N + M] for j in needed])))
""")

# ---------------------------------------------------------------- 3
md(r"""
## 3. Attention heads, each with a valve

Every head scores each position against the question, turns the scores into weights that add up to 1
(softmax), and returns the weighted average of the items. Each head's output is then multiplied by
its **valve** (1 = open, 0 = closed) before the output weights `W_o` combine the heads.
""")
code(r"""
class Attention(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.H = heads
        self.W_q = nn.Linear(D, heads * DK)            # D -> H*DK   every head's W_Q, stored side by side
        self.W_k = nn.Linear(D, heads * DK)            # D -> H*DK   every head's W_K
        self.W_v = nn.Linear(D, heads * DK)            # D -> H*DK   every head's W_V
        self.W_o = nn.Linear(heads * DK, R * M)        # H*DK -> R*M  combines the heads into the answer

    def head_outputs(self, questions, memory):         # questions X_q, memory X_m: (B, N, D)
        split = lambda t: t.view(t.shape[0], N, self.H, DK).transpose(1, 2)   # (B, N, H*DK) -> (B, H, N, DK)
        q = split(self.W_q(questions))                 # (B, H, N, DK)  Q = X_q W_Q: what each question looks for
        k = split(self.W_k(memory))                    # (B, H, N, DK)  K = X_m W_K: each position's key
        v = split(self.W_v(memory))                    # (B, H, N, DK)  V = X_m W_V: each position's value
        scores = q @ k.transpose(-2, -1) / DK ** 0.5   # (B, H, N, N)   S = Q K^T / sqrt(DK): question i vs position j
        weights = torch.softmax(scores, dim=-1)        # (B, H, N, N)   A = softmax(S): each row adds up to 1
        return weights @ v                             # (B, H, N, DK)  O = A V: each question's weighted average

    def forward(self, questions, memory, valves):      # valves: (H,), one number per head
        out = self.head_outputs(questions, memory)                                 # (B, H, N, DK)
        valved = []
        for h in range(self.H):                        # THE VALVE: head h's whole block times its number g_h
            valved.append(out[:, h] * valves[h])       # (B, N, DK) x one number
        out = torch.stack(valved, dim=1)                                           # (B, H, N, DK) again
        out = out.transpose(1, 2).reshape(len(questions), N, -1)                   # (B, N, H*DK)   heads side by side
        return self.W_o(out)                                                       # (B, N, R*M)    Y-hat: the answer
""")

md(r"""
**Every step's size, for one example.** Letters: B examples, H heads, N questions (= memory
positions), D vector length, DK head size, R*M answer length. Read `(B, H, N, DK)` as "for every
example, for every head, an N x DK matrix".
""")
code(r"""
model = Attention(HEADS)
questions, memory, answer = make_batch(1)                       # one example: B = 1
split = lambda t: t.view(1, N, HEADS, DK).transpose(1, 2)
q, k, v = split(model.W_q(questions)), split(model.W_k(memory)), split(model.W_v(memory))
scores = q @ k.transpose(-2, -1) / DK ** 0.5
weights = torch.softmax(scores, dim=-1)
out = weights @ v
side_by_side = out.transpose(1, 2).reshape(1, N, -1)
prediction = model.W_o(side_by_side)
steps = [("X_q  questions", questions, "(B, N, D)"), ("X_m  memory", memory, "(B, N, D)"),
         ("Q = X_q W_Q", q, "(B, H, N, DK)"), ("K = X_m W_K", k, "(B, H, N, DK)"), ("V = X_m W_V", v, "(B, H, N, DK)"),
         ("S = Q K^T / sqrt(DK)", scores, "(B, H, N, N)"), ("A = softmax(S)", weights, "(B, H, N, N)"),
         ("O = A V", out, "(B, H, N, DK)"), ("[O_1 | ... | O_H]", side_by_side, "(B, N, H*DK)"),
         ("Y-hat = [...] W_o", prediction, "(B, N, R*M)"), ("Y  answer key", answer, "(B, N, R*M)")]
for name, t, letters in steps:
    print(f"{name:<22}{letters:<16}= {tuple(t.shape)}")
print("same as model(...):", torch.allclose(prediction, model(questions, memory, torch.ones(HEADS))))
""")

# ---------------------------------------------------------------- 4
md(r"""
## 4. The collateral value of a head

Remove the head, let the remaining heads re-fit the output weights (least squares), and measure how
much the loss goes up. A head whose job another head can do is worth about 0, however busy it is.
""")
code(r"""
@torch.no_grad()
def collateral_values(model, open_heads):
    # 1. run the model on a fresh batch and keep every head's output
    questions, memory, answer = make_batch(256)
    outs = model.head_outputs(questions, memory)              # (B, H, N, DK)
    target = answer.reshape(-1, R * M).double()               # one row per question: (B*N, R*M)

    # 2. the best loss we can reach using only some heads: re-fit the read-out by least squares
    def best_loss(heads):
        columns = []
        for h in heads:
            head_out = outs[:, h].reshape(-1, DK).double()    # head h's output, one row per question
            columns.append(head_out)
        columns.append(torch.ones(len(target), 1, dtype=torch.double))   # a constant column (the bias)
        Z = torch.cat(columns, dim=1)                         # the chosen heads side by side
        W = torch.linalg.lstsq(Z, target).solution            # the best read-out weights
        prediction = Z @ W
        return ((prediction - target) ** 2).mean().item()

    # 3. a head's value = how much the best loss rises when it is left out
    loss_with_all = best_loss(open_heads)
    values = {}
    for h in open_heads:
        others = []
        for o in open_heads:
            if o != h:
                others.append(o)
        values[h] = best_loss(others) - loss_with_all
    return values
""")

# ---------------------------------------------------------------- 5
md(r"""
## 5. Training, with the collateral rule

An ordinary training loop. With `rule=True`, after a warm-up and then every 25 steps: value the open
heads, and close the cheapest one if it is worth less than the **price** (3% of the loss of a model
that knows nothing, which is about 1 here). A closed head's valve fades to 0 over 100 steps, so the
other heads can take over its job while it goes.
""")
code(r"""
def train(heads, steps=2000, rule=False, price=0.03, warmup=400, every=25, fade=100, seed=0):
    torch.manual_seed(seed)
    model = Attention(heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    valves = torch.ones(heads)       # each head's valve: 1 = fully open, 0 = closed
    is_open = torch.ones(heads)      # the rule's decision for each head: 1 = keep, 0 = close
    log = []

    for step in range(steps):
        # --- an ordinary training step ---
        questions, memory, answer = make_batch(128)
        prediction = model(questions, memory, valves)
        loss = F.mse_loss(prediction, answer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rule:
            # --- a closed head's valve fades by 1/fade per step until it reaches 0 ---
            for h in range(heads):
                if is_open[h] == 0 and valves[h] > 0:
                    valves[h] = max(float(valves[h]) - 1 / fade, 0.0)

            # --- every `every` steps after the warm-up: value the open heads, maybe close one ---
            if step >= warmup and step % every == 0:
                open_heads = []
                for h in range(heads):
                    if is_open[h] == 1:
                        open_heads.append(h)

                values = collateral_values(model, open_heads)

                cheapest = open_heads[0]                      # find the open head worth least
                for h in open_heads:
                    if values[h] < values[cheapest]:
                        cheapest = h

                if values[cheapest] < price and len(open_heads) > 1:
                    is_open[cheapest] = 0                     # close it: its valve starts fading

        heads_on = int((valves > 0).sum())
        log.append((loss.item(), heads_on))

    return model, valves, log
""")

# ---------------------------------------------------------------- 6
md(r"""
## 6. Run it

First, how many heads does the task really need? Train with 1 head and with 2 (no rule).
""")
code(r"""
for heads in range(1, R + 1):                                 # 1 head, 2 heads, ... up to R
    _, _, log = train(heads)
    print(f"{heads} head(s): final loss {sum(l for l, _ in log[-100:]) / 100:.4f}")
""")
md(r"""
With fewer than R heads the loss stays high (a head fetches from one distance; the task has R).
With R heads it is solved. Now start with `HEADS` heads (4 times too many) and let the rule decide,
three times with different random starts:
""")
code(r"""
logs = []
for seed in range(3):
    _, valves, log = train(HEADS, rule=True, seed=seed)
    logs.append(log)
    print(f"seed {seed}: heads left {int((valves > 0).sum())} of {HEADS} (task needs {R}),  final loss "
          f"{sum(l for l, _ in log[-100:]) / 100:.4f},  worst loss after step 1000 {max(l for l, _ in log[1000:]):.4f}")
""")
code(r"""
loss, heads = zip(*logs[0])
fig, ax = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True)
ax[0].semilogy(loss, color="#b2182b", lw=0.5); ax[0].set_ylabel("loss")
ax[0].axhline(0.02, color="#888", ls=":", lw=0.8)              # the 'solved' bar
ax[1].plot(heads, color="#b2182b"); ax[1].axhline(R, color="#888", ls=":")
ax[1].set_ylabel("heads open"); ax[1].set_xlabel("training step")
plt.tight_layout(); plt.show()
""")
md(r"""
With the starting sizes, every run ends with exactly the R = 2 heads the task needs, and the loss
stays far below the solved bar (dotted) the whole way down. Try other sizes in section 1.
""")

# ---------------------------------------------------------------- 7
md(r"""
## 7. How this compares with the project code

| here | in `src/hemo/` | difference |
|---|---|---|
| `make_batch` | `tasks.make_batch` | the project also adds noise to the labels |
| `Attention` | `model.CrossAttn` / `HemoAttn` | same; the project stores more for the analyses |
| `collateral_values` | `HemoAttn.probe_ischemia` | the project gets the same numbers with one matrix inverse instead of one re-fit per head (faster) |
| the `if rule:` block | `HemoAttn.local_step`, `relax_tone` | the project also **reopens** a closed head worth more than twice the price |
| `train` | `train.train` | the project adds a learning-rate schedule and gradient clipping |
| sizes 8 / 2 / 8 heads | `Cfg` | the experiments use a memory of 16, 4 distances, 32 heads, 4000 steps |

The full experiments (hundreds of runs, see `walkthrough_by_hand.ipynb` section 8 and the paper)
found the same as here, at full size: the rule kept exactly the needed number of heads in 47 of 50
runs and never broke the model in 50 of 50, which no other rule tested managed.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(ROOT, "notebooks", "the_code.ipynb")
nbf.write(nb, out)
print("wrote", os.path.relpath(out, ROOT))
