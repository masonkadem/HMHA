"""Murray's law and the head rule, explained in four panels (illustrations, not data).

  a  thin vessel: little blood to maintain, but hard to push blood through
  b  wide vessel: easy to push, but a lot of blood to maintain
  c  Murray (1926): total cost = pumping work + upkeep of blood, lowest at one size
  d  heads: total cost = loss + upkeep x open heads; each closed head is one step downhill,
     and the rule stops at the bottom, which is the number of heads the task needs

  python experiments/figure_murray.py  ->  paper/neuroai/fig_murray.pdf (and .png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RED, V = "#a50f15", "#7f0000"
K, G1, G2, G3 = "#1a1a1a", "#666666", "#a6a6a6", "#e6e6e6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=5)

fig = plt.figure(figsize=(7.0, 1.85))
gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.8, 1, 1.15], wspace=0.35)


def vessel(ax, half_width, n_cells, work_text, upkeep_text, work_hi):
    ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 10)); ax.set_aspect("equal")
    y0, y1 = 5 - half_width, 5 + half_width
    ax.add_patch(Rectangle((0.5, y0), 9, y1 - y0, fc="#f3d9d9", ec=V, lw=1.0))
    rng = np.random.default_rng(1)
    for _ in range(n_cells):
        ax.add_patch(plt.Circle((rng.uniform(1, 9), rng.uniform(y0 + 0.35, y1 - 0.35)), 0.28, fc=RED, ec="none"))
    ax.add_patch(FancyArrowPatch((0.6, 5), (9.6, 5), arrowstyle="-|>", mutation_scale=7, color=K, lw=0.8))
    ax.text(5, 1.4, work_text, ha="center", fontsize=6.3, color=RED if work_hi else G1)
    ax.text(5, 0.2, upkeep_text, ha="center", fontsize=6.3, color=G1 if work_hi else RED)


ax = fig.add_subplot(gs[0])
vessel(ax, 0.55, 5, "hard to push: high work", "little blood: low upkeep", True)
title(ax, "a", "Thin vessel")
ax = fig.add_subplot(gs[1])
vessel(ax, 2.6, 45, "easy to push: low work", "lots of blood: high upkeep", False)
title(ax, "b", "Wide vessel")

ax = fig.add_subplot(gs[2])
r = np.linspace(0.55, 2.4, 200)
work, up = 1 / r ** 4, 0.5 * r ** 2
ax.plot(r, work, color=G1, lw=0.9)
ax.plot(r, up, color=G1, lw=0.9, ls=(0, (3, 2)))
ax.plot(r, work + up, color=RED, lw=1.4)
i = np.argmin(work + up)
ax.plot(r[i], (work + up)[i], "o", color=RED, ms=4)
ax.text(1.0, 3.7, "pumping work", fontsize=6, color=G1, va="top")
ax.text(2.05, 1.45, "upkeep", fontsize=6, color=G1, ha="left", va="top")
ax.annotate("best size", xy=(r[i], (work + up)[i]), xytext=(1.85, 0.45), fontsize=6, color=RED,
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
ax.text(2.4, 3.7, "total", fontsize=6, color=RED, ha="right", va="top")
ax.set(xlim=(0.5, 2.45), ylim=(0, 4), xticks=[], yticks=[], xlabel="vessel width", ylabel="cost")
title(ax, "c", "Murray's law (1926)")

ax = fig.add_subplot(gs[3])
k = np.arange(0, 13)
loss, upk = np.maximum(0, 1 - k / 4), 0.06 * k
tot = loss + upk
ax.plot(k, loss, color=G1, lw=0.9)
ax.plot(k, upk, color=G1, lw=0.9, ls=(0, (3, 2)))
ax.plot(k, tot, color=RED, lw=1.4)
for a in range(12, 4, -1):                                   # each closed head: one step downhill
    ax.add_patch(FancyArrowPatch((a, tot[a] + 0.07), (a - 1, tot[a - 1] + 0.07), arrowstyle="-|>",
                                 mutation_scale=5, color=K, lw=0.6))
ax.plot(4, tot[4], "o", color=RED, ms=4)
ax.text(12, tot[12] + 0.16, "start: all open", fontsize=6, color=K, ha="right")
ax.text(4.5, 0.1, "stops at the heads needed", fontsize=6, color=RED, ha="left", va="center")
ax.text(0.5, 0.42, "loss", fontsize=6, color=G1)
ax.text(1.0, 0.0, "upkeep", fontsize=6, color=G1, ha="left", va="bottom")
ax.set(xlim=(-0.3, 12.6), ylim=(-0.05, 1.2), xticks=[0, 4, 8, 12], yticks=[], xlabel="open heads", ylabel="cost")
title(ax, "d", "Heads: close while cost falls")

out = os.path.join(ROOT, "paper", "neuroai", "fig_murray")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
