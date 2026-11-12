"""Figure 1, from vessels to heads (illustration, not data).

  a  a lateral view of the brain with a cortical artery tree: branches that carry flow stay (red),
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

# ---------------------------------------------------------------- a  brain with a cortical artery tree
ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set(xlim=(0.2, 9.8), ylim=(-0.2, 7.0)); ax.set_aspect("equal")
P = np.array([(0.6, 3.3), (0.9, 4.7), (1.8, 5.8), (3.2, 6.5), (4.8, 6.75), (6.4, 6.55), (7.8, 5.95), (8.9, 4.95),
              (9.45, 3.8), (9.35, 2.75), (8.85, 2.15), (8.75, 1.45), (8.2, 0.85), (7.2, 0.8), (6.5, 1.2),
              (6.15, 0.95), (6.05, 0.1), (5.45, 0.1), (5.5, 1.25), (4.6, 1.35), (3.3, 1.55), (2.55, 2.0),
              (2.35, 2.55), (1.5, 2.65), (0.85, 2.85)])
tck, _ = splprep([np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]]], s=0.3, per=True)
bx, by = splev(np.linspace(0, 1, 400), tck)
brain = Polygon(np.c_[bx, by], closed=True, fc="#f2f2f2", ec=G1, lw=0.7, zorder=1)
ax.add_patch(brain)


def curve(pts, lw=0.5, color=G2):
    pts = np.array(pts)
    t, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
    x, y = splev(np.linspace(0, 1, 60), t)
    ln, = ax.plot(x, y, color=color, lw=lw, zorder=2)
    ln.set_clip_path(brain)


curve([(2.5, 2.75), (3.8, 3.35), (5.2, 3.75), (6.3, 3.95)], lw=0.7)          # lateral (Sylvian) fissure
curve([(5.6, 6.7), (5.1, 5.6), (5.0, 4.9), (4.6, 4.2)], lw=0.6)              # central sulcus
for pts in ([(3.0, 6.3), (2.9, 5.3), (2.2, 4.6)], [(7.4, 6.0), (7.0, 5.1), (7.4, 4.3)],
            [(3.6, 2.3), (4.8, 2.5), (6.0, 2.7)], [(8.4, 4.8), (7.9, 3.9), (8.5, 3.1)],
            [(1.5, 4.3), (2.2, 3.9), (3.0, 4.2)]):
    curve(pts, lw=0.45, color="#c4c4c4")

rng = np.random.default_rng(7)
FADE = {(1, 0), (0, 1, 1), (1, 1, 0, 1), (0, 0, 0)}                          # branches with little flow


def branch(path, x, y, ang, length, width, depth, faded):
    x2, y2 = x + length * np.cos(ang), y + length * np.sin(ang)
    faded = faded or tuple(path) in FADE
    if faded:
        ln, = ax.plot([x, x2], [y, y2], color=G1, lw=max(0.5, width * 0.6), ls=(0, (1.4, 1.0)), zorder=3)
    else:
        ln, = ax.plot([x, x2], [y, y2], color=RED, lw=width, solid_capstyle="round", zorder=4)
    ln.set_clip_path(brain)
    if depth == 0:
        return
    for i, d in enumerate((-1, 1)):
        branch(path + [i], x2, y2, ang + d * (0.45 + 0.2 * rng.random()), length * 0.74, width * 0.7, depth - 1, faded)


branch([], 2.6, 2.8, np.radians(28), 1.45, 2.6, 4, False)                    # an artery from the lateral fissure
ax.text(0.25, 6.95, "kept", fontsize=5.3, color=RED, ha="left", va="top")
ax.text(9.75, 6.95, "regresses", fontsize=5.3, color=G1, ha="right", va="top")
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
ax.text(7.7, 2.1, "sum", fontsize=5.2, color=G1, ha="left", va="center")
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
