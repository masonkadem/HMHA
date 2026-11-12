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
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=5)

fig = plt.figure(figsize=(3.35, 1.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.28)

ax = fig.add_subplot(gs[0])
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
ax.text(2.4, 3.7, "sum", fontsize=6, color=RED, ha="right", va="top")
ax.set(xlim=(0.5, 2.45), ylim=(0, 4), xticks=[], yticks=[], xlabel="vessel width", ylabel="cost")
title(ax, "a", "Vessel: work + upkeep")

ax = fig.add_subplot(gs[1])
k = np.arange(0, 13)
loss, upk = 0.06 + 0.94 * np.maximum(0, 1 - k / 4), 0.06 * k
tot = loss + upk
ax.plot(k, loss, color=G1, lw=0.9)
ax.plot(k, upk, color=G1, lw=0.9, ls=(0, (3, 2)))
ax.plot(k, tot, color=RED, lw=1.4)
for a in range(12, 4, -1):                                   # each closed head: one step downhill
    ax.add_patch(FancyArrowPatch((a, tot[a] + 0.07), (a - 1, tot[a - 1] + 0.07), arrowstyle="-|>",
                                 mutation_scale=5, color=K, lw=0.6))
ax.plot(4, tot[4], "o", color=RED, ms=4)
ax.text(12, tot[12] + 0.14, "all open", fontsize=6, color=K, ha="right")
ax.text(4.6, 0.19, "stops at the heads needed", fontsize=5.3, color=RED, ha="left", va="center")
ax.text(2.6, 0.28, "loss", fontsize=6, color=G1, ha="right")
ax.text(12.4, 0.52, "upkeep", fontsize=6, color=G1, ha="right", va="top")
ax.set(xlim=(-0.3, 12.6), ylim=(-0.05, 1.2), xticks=[0, 4, 8, 12], yticks=[], xlabel="open heads", ylabel="cost")
title(ax, "b", "Heads: loss + upkeep")

out = os.path.join(ROOT, "paper", "neuroai", "fig_murray_col")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
