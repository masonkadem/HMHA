"""Backup heads as a probability tree, for one setting (R jobs, failure rate p, price).

Panels a and b: the tree for the last head the rule keeps and the first one it closes.
Panel c: the value of the B-th head against the price, with the predicted count.
Panel d: predicted against measured heads kept across failure rates, from results/confirm.
Every number drawn is computed here from the formula in results/confirm/predictions.md or
read from the saved runs; nothing is typed in.

  python experiments/figure_backup_tree.py [R] [p] [price]
      ->  figures/fig_backup_tree.png and figures/fig_backup_tree.pdf
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
PRICE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03

RED = "#b2182b"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})


# ---------------------------------------------------------------- the formula
def p_needed(B, R, p):
    """Chance that at most R - 1 of the other B - 1 heads are alive, so this one is needed."""
    q = 1 - p
    return sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(0, min(R - 1, B - 1) + 1))


def value(B, R, p):
    """Value of one of B heads, in units of the loss of a model that learned nothing."""
    return (1 - p) * p_needed(B, R, p) / R


def predict(R, p, price, H=32):
    return max(B for B in range(1, H + 1) if value(B, R, p) >= price)


# ---------------------------------------------------------------- the saved runs
DEFAULT = Cfg()
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
RULE = dict(task="multi_relation", supply="local", taper=100, budget_hold_frac=0.1, conserve=0,
            local_value="refit", plant_copies=0, oneshot=0, steps=4000)


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    start = int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on))
    return float(np.median(on[start:]))


measured = {}                                            # failure rate -> heads kept per seed
for d in ("results", "results/proposal", "results/confirm"):
    for path in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(path, "rb"))
        if not (isinstance(r, dict) and "hist" in r):
            continue
        if all(get(r, k) == v for k, v in RULE.items()) and get(r, "n_rel") == R \
                and get(r, "price_frac") == PRICE:
            measured.setdefault(get(r, "head_dropout"), []).append(kept(r))

B_keep = predict(R, P, PRICE)


# ---------------------------------------------------------------- drawing
def tree(ax, B, letter):
    q, need, v = 1 - P, p_needed(B, R, P), value(B, R, P)
    keep = v >= PRICE
    ax.set_xlim(0, 10), ax.set_ylim(0.6, 10), ax.axis("off")
    verdict = "keep" if keep else "close"
    ax.set_title(f"{letter}   Is head number {B} worth keeping?  ({R} jobs, a head fails {P:.0%} of the time)",
                 fontweight="bold", pad=6)
    box = dict(boxstyle="round,pad=0.35", fc="white", ec=K, lw=0.8)
    leaf = dict(boxstyle="round,pad=0.35", fc=G3, ec="none")
    win = dict(boxstyle="round,pad=0.35", fc="white", ec=RED, lw=1.4)
    nodes = dict(root=(0.8, 5.2), alive=(3.4, 7.2), dead=(3.4, 2.9), needed=(8.0, 8.8), spare=(8.0, 5.4))
    edge = lambda a, b, c=G1, w=1.0: ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                                             color=c, lw=w, zorder=1, solid_capstyle="round")
    edge("root", "alive", K, 1.4), edge("root", "dead"), edge("alive", "needed", RED, 1.8), edge("alive", "spare")
    lab = lambda a, b, t, dy=0.45, c=K, f=0.5: ax.text(nodes[a][0] + f * (nodes[b][0] - nodes[a][0]),
                                                       nodes[a][1] + f * (nodes[b][1] - nodes[a][1]) + dy,
                                                       t, ha="center", va="center", color=c, fontsize=8.5)
    lab("root", "alive", f"it works\n{q:.2f}", 0.75)
    lab("root", "dead", f"it fails\n{P:.2f}", -0.8, G1)
    lab("alive", "needed", f"{need:.3g}", 0.5, K, 0.36)
    lab("alive", "spare", f"{1 - need:.3g}", -0.5, G1, 0.36)
    ax.text(*nodes["root"], f"head {B}", ha="center", va="center", bbox=box, zorder=3)
    ax.text(*nodes["alive"], "do the other\n" + f"{B - 1} cover all\n{R} jobs?", ha="center", va="center", bbox=box, zorder=3)
    ax.text(*nodes["dead"], "saves nothing", ha="center", va="center", bbox=leaf, color=G1, zorder=3)
    ax.text(*nodes["needed"], f"no: fewer than {R} of them\nwork, so it is needed.\nIt saves 1/{R} of the loss",
            ha="center", va="center", bbox=win, zorder=3)
    ax.text(*nodes["spare"], "yes: it is spare,\nsaves nothing", ha="center", va="center", bbox=leaf, color=G1, zorder=3)
    sign = "is at least" if keep else "is below"
    ax.text(5.0, 1.2, f"worth = {q:.2f} × {need:.3g} × 1/{R} = {v:.3f},  which {sign} the price {PRICE:g}:  {verdict}",
            ha="center", va="center", fontsize=9, fontweight="bold", color=RED if keep else K)


fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.22, wspace=0.16,
                      left=0.05, right=0.98, top=0.94, bottom=0.08)
tree(fig.add_subplot(gs[0, 0]), B_keep, "a")
tree(fig.add_subplot(gs[0, 1]), B_keep + 1, "b")

ax = fig.add_subplot(gs[1, 0])
Bs = np.arange(1, B_keep + 4)
vals = np.array([value(int(b), R, P) for b in Bs])
ax.bar(Bs, vals, width=0.6, color=[RED if v >= PRICE else G2 for v in vals], zorder=2)
ax.axhline(PRICE, color=K, lw=0.9, ls=(0, (4, 3)), zorder=3)
ax.text(Bs[-1] + 0.45, PRICE, f"price {PRICE:g}", va="bottom", ha="right")
ax.set_yscale("log"), ax.set_xticks(Bs)
ax.set_xlabel("number of heads open"), ax.set_ylabel("worth of one head (fraction of the loss)")
for b, v in zip(Bs, vals):
    ax.text(b, v * 1.12, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)
ax.set_ylim(vals.min() / 2.5, vals.max() * 2.2)
ax.set_title(f"c   Heads close one at a time until the next one is worth the price: {B_keep} stay",
             fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
ps = sorted(measured)
grid = np.linspace(0, max(ps + [P]) if ps else P, 61)
ax.step(grid, [predict(R, float(x), PRICE) for x in grid], where="post", color=K, lw=1.2,
        label="predicted beforehand", zorder=2)
n_runs = hit = 0
for x in ps:
    ys = np.array(measured[x])
    for y in sorted(set(ys)):
        ax.scatter([x], [y], s=28 + 34 * (int((ys == y).sum()) - 1), color=RED, zorder=3,
                   edgecolor="white", linewidth=1.0)
    n_runs += len(ys)
    hit += int((ys == predict(R, x, PRICE)).sum())
ax.scatter([], [], s=28, color=RED, label="measured (larger dot = more runs agree)")
ax.set_xlabel("chance that a head fails during training")
ax.set_ylabel("heads kept")
ax.yaxis.get_major_locator().set_params(integer=True)
ax.legend(loc="upper left")
ax.set_title(f"d   Measured against the prediction, {R} jobs, price {PRICE:g}: "
             f"{hit} of {n_runs} runs exact", fontweight="bold")

out = os.path.join(ROOT, "figures", "fig_backup_tree")
fig.savefig(out + ".png"), fig.savefig(out + ".pdf")
print(f"R={R} p={P} price={PRICE}: predicted {B_keep} heads; value({B_keep})={value(B_keep, R, P):.4f}, "
      f"value({B_keep + 1})={value(B_keep + 1, R, P):.4f}; measured {measured.get(P)}; "
      f"exact {hit}/{n_runs} across p in {ps}")
