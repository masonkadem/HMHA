"""One figure that tells the story: the task, why it needs R heads, the collateral rule
against the alternatives, planted copies, and backup heads under damage. Every number is
read from the result pickles.

  python experiments/story_figure.py   ->  figures/story_figure.png
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY, LIGHT, INK, DARK = "#9c9a93", "#e9e8e4", "#52514e", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.titlelocation": "left", "axes.titleweight": "bold",
                     "axes.titlesize": 10, "legend.frameon": False, "savefig.dpi": 300})

RUNS = []
for d in ("results", "results/proposal", "results/confirm"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r:
            RUNS.append(r)
DEFAULT = Cfg()
PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", leak=0.0,
              probe_every=25, prune_stop=1, conserve=1, local_value="refit", plant_copies=0,
              head_dropout=0.0, probe_masks=32, target_frac=0.02, budget_hold_frac=0.25,
              taper=0, price_frac=0.01)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


def stayed_solved(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.05], hspace=0.45, wspace=0.38)

# ---------------------------------------------------------------- a: the task
ax = fig.add_subplot(gs[0, :2])
ax.set_title("a   The task: one question, four answers", pad=6)
ax.axis("off")
N, p, offs, cols = 16, 8, (1, 5, 9, 13), (BLUE, ORANGE, AQUA, VIOLET)
ax.set(xlim=(-4.2, 16.2), ylim=(-3.3, 2.6))
ax.add_patch(Rectangle((p - 0.42, 1.35), 0.84, 0.75, color=DARK))
ax.text(p, 1.72, str(p), color="white", ha="center", va="center", fontweight="bold", fontsize=10)
ax.text(-4.1, 1.72, "question\nholds p", fontsize=8.5, color=INK, va="center")
ax.text(-4.1, 0, "16 memory\nslots", fontsize=8.5, color=INK, va="center")
for j in range(N):
    r = next((i for i, d in enumerate(offs) if (p + d) % N == j), None)
    ax.add_patch(Rectangle((j - 0.42, -0.36), 0.84, 0.72, facecolor=LIGHT,
                           edgecolor=cols[r] if r is not None else "#c9c7c0", lw=2.2 if r is not None else 0.8))
    ax.text(j, 0, str(j), ha="center", va="center", fontsize=8, color=INK)
for r, d in enumerate(offs):
    j = (p + d) % N
    ax.add_patch(FancyArrowPatch((p, 1.32), (j, 0.42), connectionstyle=f"arc3,rad={0.28 if j < p else -0.28}",
                                 arrowstyle="-|>", mutation_scale=10, color=cols[r], lw=1.5))
    ax.add_patch(Rectangle((-0.5 + r * 4.05, -2.75), 3.8, 1.05, color=cols[r]))
    ax.text(1.4 + r * 4.05, -2.22, f"item in slot {j}", color="white", ha="center",
            va="center", fontsize=8.5, fontweight="bold")
ax.text(-4.1, -2.22, "answer:\nR = 4 items", fontsize=8.5, color=INK, va="center")

# ---------------------------------------------------------------- b: one head, one equation
ax = fig.add_subplot(gs[0, 2])
E1 = pickle.load(open(os.path.join(ROOT, "results", "proposal", "equations.pkl"), "rb"))
triv = E1["trivial"]
ax.plot([0, 4], [1, 0], "--", color=GREY, lw=1.5, label="prediction: 1 - k/4")
plain = [r for r in E1["rows"] if len(set(r["heads"])) == len(r["heads"])]
copies = [r for r in E1["rows"] if len(set(r["heads"])) < len(r["heads"])]
ax.plot([r["different"] for r in plain], [r["loss"] / triv for r in plain], "o", color=BLUE, ms=8,
        label="heads removed")
ax.plot([r["different"] for r in copies], [r["loss"] / triv for r in copies], "x", color=ORANGE, ms=11,
        mew=2.5, label="with a copied head")
ax.set(xlabel="different heads k", ylabel="error (1 = learned nothing)", xticks=[0, 1, 2, 3, 4],
       xlim=(-0.2, 4.3), ylim=(-0.05, 1.05))
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("b   Each head is one equation;\n     a copy adds none")

# ---------------------------------------------------------------- c: count and stability
ax = fig.add_subplot(gs[1, 0])
arms = [("collateral rule\n(this work)", RULE, BLUE, (2, 3, 4, 6, 8)),
        ("ordinary importance", {**RULE, "local_value": "ablate"}, GREY, (2, 4, 8)),
        ("random choice", {**RULE, "local_value": "random"}, GREY, (2, 4, 8)),
        ("standard pruning", dict(supply="prune"), GREY, (2, 3, 4, 6, 8))]
ax.add_patch(Rectangle((80, 80), 25, 25, color=BLUE, alpha=0.08, lw=0))
ax.text(81.5, 82, "goal", color=BLUE, fontsize=8, ha="left")
for name, kw, color, sizes in arms:
    rs = [r for k in sizes for r in runs(**kw, n_rel=k)]
    x = 100 * np.mean([kept(r) == get(r, "n_rel") for r in rs])
    y = 100 * np.mean([stayed_solved(r) for r in rs])
    ax.scatter(x, y, s=90 if color == BLUE else 60, color=color, zorder=3, edgecolor="white", lw=1.5)
    ha, dx, dy = {"collateral rule\n(this work)": ("center", 0, -38), "ordinary importance": ("left", 5, -9),
                  "random choice": ("right", -5, 4), "standard pruning": ("right", -5, 4)}[name]
    ax.text(x + dx, y + dy, f"{name}\n{len(rs)} runs", ha=ha, fontsize=7.5,
            color=DARK if color == BLUE else INK, fontweight="bold" if color == BLUE else "normal")
ax.set(xlim=(-5, 105), ylim=(-8, 108), xlabel="runs with the exact number of heads (%)",
       ylabel="runs that never lost the task\nwhile heads were removed (%)")
ax.set_title("c   Only one rule does both")

# ---------------------------------------------------------------- d: planted copies
ax = fig.add_subplot(gs[1, 1])
res = {}
for value, name in (("refit", "collateral rule"), ("ablate", "ordinary importance")):
    out = []
    for r in runs(**RULE, n_rel=4, plant_copies=1, local_value=value):
        on = r["hist"]["ledger"][-1] > 0
        out.append((int(on.sum()), sum(bool(on[h] and on[h + 16]) for h in range(16))))
    res[name] = np.array(out)
rng = np.random.default_rng(0)
for i, (name, color) in enumerate((("collateral rule", BLUE), ("ordinary importance", GREY))):
    for j in range(2):
        v = res[name][:, j]
        x0 = j * 3 + i
        ax.bar(x0, v.mean(), 0.8, color=color, alpha=0.9, label=name if j == 0 else None)
        ax.scatter(x0 + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=DARK, zorder=3)
ax.plot([-0.5, 1.5], [4, 4], ":", color=DARK, lw=1.2)
ax.text(1.55, 4, "needed: 4", fontsize=7.5, va="center")
ax.set_xticks([0.5, 3.5], ["heads kept", "pairs of copies\nboth kept"])
ax.set_ylabel(f"count ({len(res['collateral rule'])} seeds each)")
ax.set_ylim(0, 18)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("d   Start with every head copied")

# ---------------------------------------------------------------- e: backups under damage
ax = fig.add_subplot(gs[1, 2])


def predict(R, p, price=0.03):
    q = 1 - p
    val = lambda B: q * sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1)) / R
    return max(B for B in range(1, 33) if val(B) >= price)


ps = [0.0, 0.05, 0.1, 0.2, 0.3]
hit = n = 0
for R, color in ((4, BLUE), (2, ORANGE)):
    ax.plot(ps, [predict(R, p) for p in ps], "--", color=color, lw=1.5, alpha=0.8)
    for p in ps:
        B = [kept(r) for r in runs(**RULE, n_rel=R, head_dropout=p)][:10]
        ax.scatter(p + rng.uniform(-0.008, 0.008, len(B)), B, s=16, color=color, zorder=3,
                   label=f"needs {R}" if p == 0 else None)
        if p > 0:
            hit += sum(b == predict(R, p) for b in B)
            n += len(B)
ax.text(0.31, 1.6, f"dashed: predicted before the runs\n{hit}/{n} damaged runs exact",
        fontsize=7.5, ha="right", color=INK)
ax.set(xlabel="chance each head fails during training", ylabel="heads kept", ylim=(1, 8), xticks=ps)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("e   More damage, more backup heads,\n     exactly as predicted")

out = os.path.join(ROOT, "figures", "story_figure.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
