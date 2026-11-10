"""GPT-2 small, indirect object identification: the heads kept at three upkeeps are nested.
  (a) 12 x 12 head map. Shade = the highest upkeep at which the head is still kept (darkest = core).
      Letters = role in the published circuit (Wang et al. 2023); grey outline = published head never kept.
  (b) Accuracy (IO name above subject) and logit difference against heads kept, held-out prompts.
  python experiments/figure_gpt2_nested.py  ->  paper/neuroai/fig_gpt2_nested.pdf (and .png)
"""
import os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RED, K, G1, G2, G3 = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)
WANG = {
    "name mover": [(9, 9), (9, 6), (10, 0)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
ABBR = {"name mover": "N", "backup name mover": "B", "negative name mover": "−", "S-inhibition": "S",
        "induction": "I", "duplicate token": "D", "previous token": "P"}
role = {h: r for r, hs in WANG.items() for h in hs}
FR = ("0.05", "0.02", "0.01")                                   # high to low upkeep
SHADE = {"0.05": RED, "0.02": "#e0787f", "0.01": "#f6cdd0"}
R = {f: pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", f"ioi_pf{f}_s0.pkl"), "rb")) for f in FR}
kept = {f: set(map(tuple, R[f]["kept"])) for f in FR}
assert kept["0.05"] <= kept["0.02"] <= kept["0.01"], "not nested"

fig = plt.figure(figsize=(7.0, 2.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.55, 1.0], wspace=0.15)

# ---------------------------------------------------------------- a  nested map
ax = fig.add_subplot(gs[0])
for l in range(12):
    for h in range(12):
        tier = next((f for f in FR if (l, h) in kept[f]), None)
        ax.add_patch(Rectangle((h, l), 1, 1, facecolor=SHADE[tier] if tier else "white", edgecolor="white", lw=0.6))
        if not tier:
            ax.add_patch(Rectangle((h + 0.04, l + 0.04), 0.92, 0.92, facecolor="#f7f7f7" if (l, h) not in role else "white",
                                   edgecolor=G2 if (l, h) in role else "none", lw=0.6))
        if (l, h) in role:
            ax.text(h + 0.5, l + 0.53, ABBR[role[(l, h)]], ha="center", va="center", fontsize=4.6,
                    color="white" if tier == "0.05" else K if tier else G1)
        elif tier:
            ax.text(h + 0.5, l + 0.53, "·", ha="center", va="center", fontsize=6, color=K)
ax.set(xlim=(0, 12), ylim=(12, 0), aspect="equal", xticks=[0.5, 5.5, 11.5], xticklabels=["0", "5", "11"],
       yticks=[0.5, 5.5, 11.5], yticklabels=["0", "5", "11"], xlabel="head", ylabel="layer")
ax.tick_params(length=0, pad=1)
for s in ax.spines.values():
    s.set_visible(False)
title(ax, "a", "GPT-2 small: heads kept are nested")

# ---------------------------------------------------------------- key
ax = fig.add_subplot(gs[1]); ax.axis("off")
n = {f: len(kept[f]) for f in FR}
ins = {f: sum(k in role for k in kept[f]) for f in FR}
hand = [Patch(facecolor=SHADE["0.05"], label=f"core: high upkeep\n{n['0.05']} kept, {ins['0.05']} in circuit"),
        Patch(facecolor=SHADE["0.02"], label=f"+ middle upkeep\n{n['0.02']} kept, {ins['0.02']} in circuit"),
        Patch(facecolor=SHADE["0.01"], label=f"+ low upkeep\n{n['0.01']} kept, {ins['0.01']} in circuit"),
        Patch(facecolor="white", edgecolor=G2, label="published head\nnever kept")]
ax.legend(handles=hand, loc="upper left", fontsize=5.4, handlelength=1.0, handleheight=1.0, labelspacing=0.7,
          borderaxespad=0, bbox_to_anchor=(-0.05, 1.0))
ax.text(-0.02, 0.0, "N name mover   B backup\n− negative N   S S-inhibition\nI induction   D duplicate\n"
        "P previous token   · not in circuit", fontsize=4.8, color=G1, va="bottom", linespacing=1.3)

# ---------------------------------------------------------------- b  performance vs size
ax = fig.add_subplot(gs[2])
full, empty, wt = R["0.02"]["full"], R["0.02"]["empty"], R["0.02"]["wang_test"]
rec = lambda x: (x[1] - empty[1]) / (full[1] - empty[1])
xs = [n[f] for f in FR]
acc = [R[f]["test"][2] for f in FR]
ld = [rec(R[f]["test"]) for f in FR]
ax.plot(xs, acc, "-o", color=RED, ms=3, lw=1.0, label="right name picked")
ax.plot(xs, ld, "--s", color=K, ms=2.6, lw=0.8, label="logit difference (full = 1)")
ax.plot(26, wt[2], "o", mfc="white", mec=RED, ms=3.5, mew=0.9)
ax.plot(26, rec(wt), "s", mfc="white", mec=K, ms=3.2, mew=0.9)
ax.annotate("published\ncircuit (26)", (26, wt[2]), (26, 0.22), ha="center", fontsize=5.2, color=G1,
            arrowprops=dict(arrowstyle="-", color=G2, lw=0.5))
for x, a, f in zip(xs, acc, FR):
    ax.text(x, a + 0.07, f"{a:.0%}", ha="center", fontsize=5.4, color=RED)
ax.axhline(1, color=G2, lw=0.5, ls=":")
ax.set(xlim=(8, 29), ylim=(0, 1.62), xticks=[11, 17, 22, 26], yticks=[0, 0.5, 1],
       yticklabels=["0", ".5", "1"], xlabel="heads kept (of 144)")
ax.text(9.5, 0.03, "upkeep: high → low", fontsize=5.2, color=G1)
ax.legend(loc="upper left", fontsize=5.4, handlelength=1.6, borderaxespad=0)
title(ax, "b", "Upkeep is a dial")

out = os.path.join(ROOT, "paper", "neuroai", "fig_gpt2_nested")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out, n, ins, [f"{a:.2f}" for a in acc], [f"{v:.2f}" for v in ld], f"circuit {wt[2]:.2f} {rec(wt):.2f}")
