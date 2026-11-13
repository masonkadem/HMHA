"""Changing demand: the number of needed heads goes 2 -> 4 -> 2 during training (results/switch, 10 seeds each).
  a  the rule with reopening only        b  the rule with sprouting
  python experiments/figure_demand.py  ->  paper/neuroai/fig_demand.pdf (and .png)
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED, K, G1, G2 = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 450, "pdf.fonttype": 42})
D = Cfg()
get = lambda r, k: r["cfg"].get(k, getattr(D, k))
RUNS = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "switch", "*.pkl"))]

fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.5), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, sp, letter, name in zip(axes, (0, 4), "ab", ("Reopening only", "With sprouting")):
    rs = [r for r in RUNS if get(r, "sprout") == sp]
    ok = 0
    for r in rs:
        on = (np.asarray(r["hist"]["ledger"]) > 0).sum(1)
        n = len(on)
        ax.plot(np.arange(n), on, color=RED if sp else K, lw=0.7, alpha=0.6, drawstyle="steps-post")
        mid = on[n // 3:2 * n // 3]
        ok += int(np.median(mid[int(0.6 * len(mid)):]) == 4)
    ph = [int(c) for c in get(rs[0], "phases")]
    edges = np.linspace(0, n, len(ph) + 1)
    for a, b, need in zip(edges[:-1], edges[1:], ph):
        ax.plot([a, b], [need, need], color=G2, lw=0.7, ls=":")
    ax.text(0.97, 0.95, f"reaches 4 heads:\n{ok} of {len(rs)} runs", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.6, color=G1)
    ax.set(yscale="log", ylim=(1.5, 45), yticks=[2, 4, 8, 16, 32], xlim=(0, n), xticks=edges,
           xticklabels=[f"{e / 1000:g}k" for e in edges], xlabel="training step")
    ax.set_yticklabels(["2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_title(rf"$\mathbf{{{letter}}}$   {name}", pad=4)
axes[0].set_ylabel("open heads")
out = os.path.join(ROOT, "paper", "neuroai", "fig_demand")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
