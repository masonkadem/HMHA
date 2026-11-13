"""Figure 1, from vessels to heads (illustration, not data).

  a  a vessel tree: branches that carry flow stay (red),
     branches that carry little regress (dashed gray)
  b  an attention layer in the style of figures/fig_architecture_dims: head bars, a valve under each
     (red = open), summed and read out
  c  Murray (1926): pumping work + upkeep of blood, lowest at one width
  d  heads: loss + upkeep x open heads; closing heads moves downhill until the heads needed

  python experiments/figure_hero.py  ->  paper/neuroai/fig_hero.pdf (and .png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.interpolate import splprep, splev

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RED = "#b2182b"
K, G1, G2, G3 = "#1a1a1a", "#666666", "#a6a6a6", "#e6e6e6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)

fig = plt.figure(figsize=(3.35, 2.75))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.42, wspace=0.22)

# ---------------------------------------------------------------- a  vessel tree
ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set(xlim=(-0.95, 0.95), ylim=(-0.02, 1.02))
rng = np.random.default_rng(3)
FADE = {(0, 1), (1, 1, 0), (0, 0, 0, 1)}                            # low-flow branches that regress


def branch(path, x, y, ang, length, width, depth, faded):
    x2, y2 = x + length * np.cos(ang), y + length * np.sin(ang)
    faded = faded or tuple(path) in FADE
    if faded:
        ax.plot([x, x2], [y, y2], color=G2, lw=max(0.6, width * 0.55), ls=(0, (1.6, 1.1)), zorder=2)
    else:
        ax.plot([x, x2], [y, y2], color=RED, lw=width, solid_capstyle="round", zorder=3)
    if depth == 0:
        return
    for i, d in enumerate((-1, 1)):
        branch(path + [i], x2, y2, ang + d * (0.42 + 0.12 * rng.random()), length * 0.72, width * 0.68, depth - 1, faded)


branch([], 0, 0, np.pi / 2, 0.3, 4.5, 4, False)
ax.text(0.95, 0.1, "little flow:\nregresses", fontsize=5.3, color=G1, ha="right", va="bottom")
ax.text(-0.95, 0.1, "flow:\nkept", fontsize=5.3, color=RED, ha="left", va="bottom")
title(ax, "a", "Brain vessels")

# ---------------------------------------------------------------- b  attention heads, architecture style
ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 8.6))
LW = 0.7


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=K, lw=LW, mutation_scale=4.5, shrinkA=0, shrinkB=0))


xs = np.linspace(1.4, 8.6, 8)
OPEN = {0, 2, 3, 6}
arrow(5.0, 8.4, 5.0, 7.55)
ax.plot([xs[0], xs[-1]], [7.55, 7.55], color=K, lw=LW)
for i, x in enumerate(xs):
    on = i in OPEN
    arrow(x, 7.55, x, 7.0)
    ax.add_patch(Rectangle((x - 0.32, 5.0), 0.64, 2.0, fc="white", ec=K if on else G2, lw=LW,
                           ls="-" if on else (0, (2, 1.5))))
    ax.plot([x, x], [5.0, 4.6], color=K if on else G2, lw=LW)
    ax.add_patch(Rectangle((x - 0.2, 3.95), 0.4, 0.65, fc=RED if on else "white", ec=RED if on else G2, lw=0.6))
    ax.plot([x, x], [3.95, 3.45], color=K if on else G2, lw=LW)
ax.plot([xs[0], xs[-1]], [3.45, 3.45], color=K, lw=LW)
arrow(5.0, 3.45, 5.0, 2.75)
ax.add_patch(Polygon([(2.6, 2.75), (7.4, 2.75), (5.7, 1.45), (4.3, 1.45)], closed=True, fc="white", ec=K, lw=LW))
arrow(5.0, 1.45, 5.0, 0.6)
ax.text(9.15, 6.0, "heads", fontsize=5.2, color=G1, ha="left", va="center")
ax.text(9.15, 4.27, "valves", fontsize=5.2, color=RED, ha="left", va="center")
ax.text(7.7, 2.1, "read-out", fontsize=5.2, color=G1, ha="left", va="center")
title(ax, "b", "Attention heads")

# ---------------------------------------------------------------- c  Murray's law
ax = fig.add_subplot(gs[1, 0])
r = np.linspace(0.55, 2.4, 200)
work, up = 1 / r ** 4, 0.5 * r ** 2
ax.plot(r, work, color=G1, lw=0.8)
ax.plot(r, up, color=G1, lw=0.8, ls=(0, (3, 2)))
ax.plot(r, work + up, color=RED, lw=1.3)
i = np.argmin(work + up)
ax.plot(r[i], (work + up)[i], "o", color=RED, ms=3.5)
ax.text(1.0, 3.8, "pumping work", fontsize=5.3, color=G1, va="top")
ax.text(2.42, 1.35, "upkeep", fontsize=5.3, color=G1, ha="right", va="top")
ax.text(r[i] + 0.1, (work + up)[i] - 0.45, "best width", fontsize=5.3, color=RED, va="top")
ax.set(xlim=(0.5, 2.45), ylim=(0, 4), xticks=[], yticks=[], xlabel="vessel width", ylabel="cost")
title(ax, "c", "Murray's law")

# ---------------------------------------------------------------- d  heads
ax = fig.add_subplot(gs[1, 1])
k = np.arange(0, 13)
loss, upk = 0.06 + 0.94 * np.maximum(0, 1 - k / 4), 0.06 * k
tot = loss + upk
ax.plot(k, loss, color=G1, lw=0.8)
ax.plot(k, upk, color=G1, lw=0.8, ls=(0, (3, 2)))
ax.plot(k, tot, color=RED, lw=1.3)
for a in range(12, 4, -1):
    ax.add_patch(FancyArrowPatch((a, tot[a] + 0.08), (a - 1, tot[a - 1] + 0.08), arrowstyle="-|>",
                                 mutation_scale=4.5, color=K, lw=0.5))
ax.plot(4, tot[4], "o", color=RED, ms=3.5)
ax.text(12.3, tot[12] + 0.16, "all open", fontsize=5.3, ha="right")
ax.text(4.5, 0.18, "heads needed", fontsize=5.3, color=RED, ha="left", va="center")
ax.text(2.4, 0.3, "loss", fontsize=5.3, color=G1, ha="right")
ax.text(12.4, 0.5, "upkeep", fontsize=5.3, color=G1, ha="right", va="top")
ax.set(xlim=(-0.3, 12.6), ylim=(-0.05, 1.25), xticks=[0, 4, 8, 12], yticks=[], xlabel="open heads", ylabel="cost")
title(ax, "d", "Our rule")

out = os.path.join(ROOT, "paper", "neuroai", "fig_hero")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
