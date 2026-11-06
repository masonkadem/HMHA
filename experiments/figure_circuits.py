"""Known circuits, for the 4-page NeuroAI paper (full width).

  (a) induction: open heads per layer during training, 10 runs (results/induction)
  (b) induction: final loss of models trained from scratch with a fixed number of heads per layer
  (c) GPT-2 small, indirect object identification: the 144 heads, the published circuit (Wang et al.
      2023) shaded by role, the heads hemodynamic attenuation kept outlined (results/gpt2_ioi)

  python experiments/figure_circuits.py [--price 0.02]  ->  paper/neuroai/fig_circuits.pdf (and .png)
"""
import argparse, glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.transforms import blended_transform_factory

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import ICfg

ap = argparse.ArgumentParser()
ap.add_argument("--price", type=float, default=0.02)
args = ap.parse_args()

RED = "#b2182b"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text="", pad=4: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=pad)

# ---------------------------------------------------------------- induction runs
RUNS = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))]
SAME = dict(probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0)


def pick(heads=(8, 8), **kw):
    base = ICfg()
    match = lambda r: all(r["cfg"].get(k, getattr(base, k)) == v for k, v in kw.items())
    return sorted([r for r in RUNS if match(r) and tuple(r["sizes"]) == heads], key=lambda r: r["cfg"]["seed"])


fig = plt.figure(figsize=(7.0, 1.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.5], wspace=0.5)

# ---------------------------------------------------------------- a  induction heads closing
ax = fig.add_subplot(gs[0])
rs = pick(rule="collateral", **SAME)
h1 = 8
for r in rs:
    led = np.asarray(r["hist"]["ledger"]) > 0
    for cols, color, label in ((slice(0, h1), RED, "layer 1"), (slice(h1, None), K, "layer 2")):
        ax.plot(led[:, cols].sum(1), color=color, lw=0.8, alpha=0.6, label=label if r["cfg"]["seed"] == 0 else None)
ax.axhline(1, color=G2, lw=0.7, ls="--", zorder=0)
ax.text(5950, 0.8, "needed per layer", ha="right", va="top", fontsize=5.2, color=G1)
ax.set(xlim=(0, 6000), ylim=(0, 8.6), yticks=[0, 1, 2, 4, 8], xticks=[0, 3000, 6000],
       xticklabels=["0", "3k", "6k"], xlabel="training step", ylabel="open heads")
ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.4, borderaxespad=0, fontsize=5.6)
title(ax, "a", f"Induction ({len(rs)} runs)")

# ---------------------------------------------------------------- b  minimal induction circuit
ax = fig.add_subplot(gs[1])
rng = np.random.default_rng(2)
groups = (((1, 0), "1"), ((8, 0), "8"), ((1, 1), "1 + 1"), ((2, 1), "2 + 1"))
for i, (heads, name) in enumerate(groups):
    rr = pick(heads=heads, rule="dense", steps=6000)
    loss = np.array([r["final_loss"] for r in rr])
    ok = loss <= rr[0]["bar"]
    ax.plot(i + rng.uniform(-0.15, 0.15, len(loss)), loss, "o", ms=3.0, mec=K, mew=0.6,
            mfc=(0.698, 0.094, 0.169, 0.25) if heads[1] > 0 else (1, 1, 1, 0.6), ls="none")
    ax.text(i, 40, f"{ok.sum()}/{len(rr)}", ha="center", va="center", fontsize=5.4, color=RED if heads[1] else K)
ax.axhline(rr[0]["bar"], color=G2, lw=0.6, ls=":")
ax.text(3.45, rr[0]["bar"] * 1.3, "solved", ha="right", va="bottom", fontsize=5.2, color=G1)
ax.set(yscale="log", ylim=(1e-4, 150), xlim=(-0.5, 3.5), xticks=range(4), ylabel="final loss",
       yticks=[1e-4, 1e-2, 1e0])
ax.set_xticklabels([g[1] for g in groups], fontsize=5.8)
under = blended_transform_factory(ax.transData, ax.transAxes)
for x, text in ((0.5, "1 layer"), (2.5, "2 layers")):
    ax.text(x, -0.2, text, transform=under, ha="center", va="top", fontsize=6)
    ax.plot([x - 0.75, x + 0.75], [-0.17, -0.17], color=G2, lw=0.6, transform=under, clip_on=False)
ax.minorticks_off()
title(ax, "b", "Heads trained alone")

# ---------------------------------------------------------------- c  GPT-2 head map
sys.path.insert(0, os.path.join(ROOT, "experiments"))
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
r = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", f"ioi_pf{args.price:g}_s0.pkl"), "rb"))
kept = set(map(tuple, r["kept"]))
inside = sum(k in role for k in kept)
ax = fig.add_subplot(gs[2])
for l in range(12):
    for h in range(12):
        rl = role.get((l, h))
        ax.add_patch(Rectangle((h, l), 1, 1, facecolor=G3 if rl else "white", edgecolor="#ececec" if not rl else "white", lw=0.4))
        if rl:
            ax.text(h + 0.5, l + 0.52, ABBR[rl], ha="center", va="center", fontsize=4.2, color=G1)
for (l, h) in kept:
    ax.add_patch(Rectangle((h + 0.06, l + 0.06), 0.88, 0.88, facecolor="none",
                           edgecolor=RED if (l, h) in role else K, lw=1.0))
ax.set(xlim=(0, 12), ylim=(12, 0), aspect="equal", xticks=[0.5, 5.5, 11.5], xticklabels=["0", "5", "11"],
       yticks=[0.5, 5.5, 11.5], yticklabels=["0", "5", "11"], xlabel="head", ylabel="layer")
ax.tick_params(length=0, pad=1)
for s in ax.spines.values():
    s.set_visible(False)
full, test, wt, empty = r["full"], r["test"], r["wang_test"], r["empty"]
rec = lambda x: (x[1] - empty[1]) / (full[1] - empty[1])
ax.legend(handles=[Patch(facecolor=G3, edgecolor="none", label="published circuit"),
                   Patch(facecolor="none", edgecolor=RED, lw=1.0, label="kept, in circuit"),
                   Patch(facecolor="none", edgecolor=K, lw=1.0, label="kept, outside")],
          loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.0, handleheight=1.0, borderaxespad=0,
          fontsize=5.6, labelspacing=0.4)
key = "N name mover\nB backup N\n− negative N\nS S-inhibition\nI induction\nD duplicate\nP previous token"
ax.text(1.04, 0.52, key, transform=ax.transAxes, fontsize=5.0, color=G1, va="top", linespacing=1.25)
ax.text(1.04, 0.0, f"kept {len(kept)}, {inside} in circuit\nlogit diff. {rec(test):.0%} (circuit {rec(wt):.0%})",
        transform=ax.transAxes, fontsize=5.4, color=RED, va="bottom")
title(ax, "c", "GPT-2 small, indirect object")
print(f"GPT-2 price {args.price}: kept {len(kept)}, {inside} in circuit, logit diff {rec(test):.3f} vs {rec(wt):.3f}")

out = os.path.join(ROOT, "paper", "neuroai", "fig_circuits")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out + ".pdf")
