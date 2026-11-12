"""Figure 4, known circuits.
  a  induction model, architecture style (as figures/fig_architecture_dims): 2 layers x 8 heads, 2 + 1 valves open
  b  induction runs: open heads per layer, 10 runs
  c  GPT-2 small, IOI: the published circuit of Wang et al. (2023); heads our rule keeps at the lowest upkeep in all
     5 seeds are filled; one summary line of held-out results (results/gpt2_cross)

  python experiments/figure_circuits_v8.py  ->  paper/neuroai/fig_circuits_v8.pdf (and .png)
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import ICfg

RED, K, G1, G2, G3 = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text="": ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)
LW = 0.7

fig = plt.figure(figsize=(7.0, 2.15))
gs = fig.add_gridspec(1, 3, width_ratios=[0.6, 0.66, 3.1], wspace=0.26)


def arrow(ax, x0, y0, x1, y1, color=K, lw=LW, ms=4.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- a  induction model, architecture style
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set(xlim=(0, 6.2), ylim=(0, 12.6))
xr = 0.9                                                    # residual stream
ax.text(xr, 12.15, "tokens", ha="center", va="center", fontsize=5.4)
arrow(ax, xr, 11.8, xr, 11.25)
ax.add_patch(Polygon([(xr - 0.45, 11.25), (xr + 0.45, 11.25), (xr + 0.8, 10.45), (xr - 0.8, 10.45)], closed=True,
                     fc="white", ec=K, lw=LW, zorder=3))
ax.plot([xr, xr], [10.45, 1.6], color=K, lw=2.0, solid_capstyle="butt", zorder=1)
hx = [1.95 + 0.55 * i for i in range(8)]


def layer(yb, open_set, name):
    ax.plot([xr, hx[-1]], [yb, yb], color=K, lw=LW)
    for i, x in enumerate(hx):
        on = i in open_set
        arrow(ax, x, yb, x, yb - 0.35, ms=3.5)
        ax.add_patch(Rectangle((x - 0.17, yb - 1.55), 0.34, 1.2, fc="white", ec=K if on else G2, lw=LW,
                               ls="-" if on else (0, (2, 1.5)), zorder=3))
        ax.plot([x, x], [yb - 1.55, yb - 1.75], color=K if on else G2, lw=LW)
        ax.add_patch(Rectangle((x - 0.12, yb - 2.1), 0.24, 0.35, fc=RED if on else "white",
                               ec=RED if on else G2, lw=0.6, zorder=3))
        ax.plot([x, x], [yb - 2.1, yb - 2.4], color=K if on else G2, lw=LW)
    ax.plot([xr + 0.22, hx[-1]], [yb - 2.4, yb - 2.4], color=K, lw=LW)
    arrow(ax, hx[0], yb - 2.4, xr + 0.22, yb - 2.4, ms=3.5)
    ax.add_patch(Circle((xr, yb - 2.4), 0.22, fc="white", ec=K, lw=LW, zorder=4))
    ax.plot([xr - 0.14, xr + 0.14], [yb - 2.4] * 2, color=K, lw=LW, zorder=5)
    ax.plot([xr, xr], [yb - 2.54, yb - 2.26], color=K, lw=LW, zorder=5)
    ax.text(hx[0] - 0.15, yb + 0.18, name, fontsize=5.2, va="bottom")


layer(9.8, {1, 5}, "layer 1: 8 heads")
layer(6.2, {3}, "layer 2: 8 heads")
ax.add_patch(Rectangle((xr - 0.6, 1.6), 1.2, 0.85, fc="white", ec=K, lw=LW, zorder=3))
arrow(ax, xr, 1.6, xr, 0.9)
ax.text(xr, 0.6, "next token", ha="center", va="center", fontsize=5.4)
title(ax, "a", "Induction")

# ---------------------------------------------------------------- b  induction runs
RUNS = [r for r in (pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl")))
        if isinstance(r, dict) and "cfg" in r]
SAME = dict(probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0, start_frac=0.25,
            rule="collateral")
base = ICfg()
rs = [r for r in RUNS if all(r["cfg"].get(k, getattr(base, k)) == v for k, v in SAME.items()) and tuple(r["sizes"]) == (8, 8)]
ax = fig.add_subplot(gs[1])
for r in rs:
    led = np.asarray(r["hist"]["ledger"]) > 0
    for cols, color, label in ((slice(0, 8), RED, "layer 1"), (slice(8, None), K, "layer 2")):
        ax.plot(led[:, cols].sum(1), color=color, lw=0.8, alpha=0.6, label=label if r["cfg"]["seed"] == 0 else None)
ax.axhline(1, color=G2, lw=0.7, ls="--", zorder=0)
ax.set(xlim=(0, 6000), ylim=(0, 8.6), yticks=[0, 1, 2, 4, 8], xticks=[0, 3000, 6000], xticklabels=["0", "3k", "6k"],
       xlabel="training step", ylabel="open heads")
ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.4, borderaxespad=0, fontsize=5.6)
title(ax, "b", f"{len(rs)} runs")

# ---------------------------------------------------------------- c  GPT-2 small circuit
WANG = {
    "name movers": [(9, 6), (9, 9), (10, 0)],
    "backup name movers": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "negative name movers": [(10, 7), (11, 10)],
    "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
role = {h: r for r, hs in WANG.items() for h in hs}
R = [pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi_wang", f"ioiwang_pf0.01_s{s}.pkl"), "rb")) for s in range(5)]
kept = set.intersection(*(set(map(tuple, r["kept"])) for r in R))
outside = sorted(h for h in kept if h not in role)
CR = [pickle.load(open(os.path.join(ROOT, "results", "gpt2_cross", f"cross_s{s}.pkl"), "rb")) for s in range(5)]

ax = fig.add_subplot(gs[2]); ax.axis("off"); ax.set(xlim=(-0.05, 11.0), ylim=(-0.4, 6.25))
CW, CH, GAP = 0.8, 0.5, 0.1


def box(x, y, w, name, heads, per_row):
    rows = int(np.ceil(len(heads) / per_row))
    h = 0.55 + rows * (CH + GAP) + 0.12
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc="#f7f7f7", ec=G2, lw=0.6))
    ax.text(x + w / 2, y + h - 0.28, name, ha="center", va="center", fontsize=5.6)
    for i, hd in enumerate(sorted(heads)):
        r, c = divmod(i, per_row)
        n = min(per_row, len(heads) - r * per_row)
        x0 = x + (w - n * CW - (n - 1) * GAP) / 2 + c * (CW + GAP)
        y0 = y + h - 0.6 - (r + 1) * (CH + GAP) + GAP
        on = hd in kept
        ax.add_patch(FancyBboxPatch((x0, y0), CW, CH, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=RED if on else "white", ec="none" if on else G2, lw=0.6))
        ax.text(x0 + CW / 2, y0 + CH / 2, f"{hd[0]}.{hd[1]}", ha="center", va="center", fontsize=5.8,
                color="white" if on else G1)
    return dict(l=x, r=x + w, b=y, t=y + h, m=y + h / 2)


P = box(0.0, 3.55, 2.1, "previous token", WANG["previous token"], 2)
D = box(0.0, 1.05, 2.1, "duplicate token", WANG["duplicate token"], 2)
I = box(2.65, 3.0, 2.0, "induction", WANG["induction"], 2)
S = box(5.3, 2.15, 2.0, "S-inhibition", WANG["S-inhibition"], 2)
N = box(7.95, 4.6, 3.0, "name movers", WANG["name movers"], 3)
B = box(7.95, 1.65, 3.0, "backup name movers", WANG["backup name movers"], 3)
M = box(7.95, 0.0, 3.0, "negative name movers", WANG["negative name movers"], 3)
arrow(ax, P["r"], P["m"], I["l"], P["m"])
arrow(ax, I["r"], I["m"], S["l"], S["m"] + 0.3)
arrow(ax, D["r"], D["m"], S["l"], S["m"] - 0.3)
for T in (N, B, M):
    arrow(ax, S["r"], S["m"], T["l"], T["m"])
ax.text(0.0, 6.12, "Then, Mary and John went to the store. John gave a drink to → Mary",
        fontsize=5.8, color=G1, style="italic", va="center")
ax.add_patch(Rectangle((0.0, 0.53), 0.26, 0.2, fc=RED, ec="none"))
ax.text(0.36, 0.63, "kept in all 5 seeds", fontsize=5.6, va="center")
ax.add_patch(Rectangle((2.75, 0.53), 0.26, 0.2, fc="white", ec=G2, lw=0.6))
ax.text(3.11, 0.63, "not kept" + (f"    also kept: {', '.join(f'{l}.{h}' for l, h in outside)}" if outside else ""),
        fontsize=5.6, va="center")


def pr(v):
    lo, hi = round(100 * min(v)), round(100 * max(v))
    return f"{lo}%" if lo == hi else f"{lo}\u2013{hi}%"


nk = [len(r["kept"]) for r in R]
l1 = f"Correct name, held out:  ours ({min(nk)}\u2013{max(nk)} heads) {pr([c['acc']['rule']['0.01'] for c in CR])}"
l2 = (f"published (26) {pr([c['acc']['published'] for c in CR])}"
      f"  \u00b7  full (144) {pr([c['acc']['full'] for c in CR])}"
      f"  \u00b7  ranked once {pr([c['acc']['once']['0.01'] for c in CR])}")
ax.text(0.0, 0.15, l1, fontsize=5.6, va="center", color=K)
ax.text(0.0, -0.2, l2, fontsize=5.6, va="center", color=K)
line = l1 + " | " + l2
title(ax, "c", "GPT-2 small: the published circuit, heads our rule keeps filled")

out = os.path.join(ROOT, "paper", "neuroai", "fig_circuits_v8")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out, "|", line)
