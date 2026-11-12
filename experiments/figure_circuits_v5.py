"""Known circuits (v4: published prompts of Wang et al., 5 seeds), simplified 4-page version: (a) induction, open heads per layer, 10 runs;
(b) GPT-2 small IOI: the published circuit as a flow of roles, each head shaded by the highest upkeep at which it is kept; (c) performance vs heads kept.
  python experiments/figure_circuits_v3.py  ->  paper/neuroai/fig_circuits_v2.pdf (and .png)
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
RUNS = [r for r in (pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))) if isinstance(r, dict) and "cfg" in r]
SAME = dict(probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0, start_frac=0.25)


def pick(heads=(8, 8), **kw):
    base = ICfg()
    match = lambda r: all(r["cfg"].get(k, getattr(base, k)) == v for k, v in kw.items())
    return sorted([r for r in RUNS if match(r) and tuple(r["sizes"]) == heads], key=lambda r: r["cfg"]["seed"])


from matplotlib.patches import Patch
RED, K, G1, G2, G3 = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
fig = plt.figure(figsize=(7.0, 1.95))
gs = fig.add_gridspec(1, 3, width_ratios=[0.7, 2.05, 1.1], wspace=0.2)
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
SEEDS = range(5)
R = {f: [pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi_wang", f"ioiwang_pf{f}_s{s}.pkl"), "rb"))
         for s in SEEDS] for f in FR}
kept = {f: set.intersection(*(set(map(tuple, r["kept"])) for r in R[f])) for f in FR}   # kept in all 5 seeds
assert kept["0.05"] <= kept["0.02"] <= kept["0.01"], "not nested"


# ---------------------------------------------------------------- b  circuit as a flow diagram
from matplotlib.patches import FancyBboxPatch
ax = fig.add_subplot(gs[1]); ax.axis("off")
ax.set(xlim=(-0.12, 10.68), ylim=(0, 6.2))
tier = lambda h: "0.05" if h in kept["0.01"] else None
TXT = {"0.05": "white", "0.02": K, "0.01": K, None: G2}
CW, CH = 0.78, 0.42                                             # chip size


def box(x, y, w, h, name, heads, per_row=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#fafafa", ec=G2, lw=0.6))
    ax.text(x + w / 2, y + h - 0.2, name, ha="center", va="center", fontsize=4.7, color=K)
    for i, hd in enumerate(sorted(heads)):
        r, c = divmod(i, per_row)
        n_row = min(per_row, len(heads) - r * per_row)
        x0 = x + (w - n_row * CW - (n_row - 1) * 0.08) / 2 + c * (CW + 0.08)
        y0 = y + h - 0.48 - (r + 1) * (CH + 0.08)
        t = tier(hd)
        ax.add_patch(FancyBboxPatch((x0, y0), CW, CH, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=SHADE[t] if t else "white", ec=G2 if not t else "none", lw=0.5))
        ax.text(x0 + CW / 2, y0 + CH / 2, f"{hd[0]}.{hd[1]}", ha="center", va="center", fontsize=4.3, color=TXT[t])
    return (x, y, w, h)


def arrow(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=G1, lw=0.6, mutation_scale=5))


P = box(0.0, 3.6, 2.15, 1.55, "previous token", WANG["previous token"], 2)
D = box(0.0, 1.75, 2.15, 1.55, "duplicate token", WANG["duplicate token"], 2)
I = box(2.55, 3.6, 2.0, 1.55, "induction", WANG["induction"], 2)
S = box(4.95, 2.55, 2.0, 1.55, "S-inhibition", WANG["S-inhibition"], 2)
N = box(7.35, 4.25, 3.2, 1.05, "name movers", WANG["name mover"], 3)
B = box(7.35, 1.95, 3.2, 2.05, "backup name movers", WANG["backup name mover"], 3)
M = box(7.35, 0.6, 3.2, 1.05, "negative name movers", WANG["negative name mover"], 3)
arrow((2.15, 4.4), (2.55, 4.4)); arrow((4.55, 4.2), (4.95, 3.6)); arrow((2.15, 2.5), (4.95, 3.0))
arrow((6.95, 3.6), (7.35, 4.7)); arrow((6.95, 3.2), (7.35, 3.0)); arrow((6.95, 2.8), (7.35, 1.2))
ax.text(0.0, 5.75, "Then, Mary and John went to the store. John gave a drink to → Mary",
        fontsize=4.9, color=G1, style="italic", va="center")
outside = sorted(h for h in kept["0.01"] if h not in role)
ax.text(0.0, 1.15, "kept, not in circuit:\n" + "  ".join(f"{l}.{h}" for l, h in outside), fontsize=4.6, color=G1, va="center")
ax.text(10.55, 0.15, "early layers → late layers", fontsize=4.6, color=G2, va="center", ha="right")
title(ax, "b", "GPT-2 small: published circuit, kept heads filled")

# ---------------------------------------------------------------- key, inside panel b
n = {f: len(kept[f]) for f in FR}
ins = {f: sum(k in role for k in kept[f]) for f in FR}
hand = [Patch(facecolor=SHADE["0.05"], label="kept by our rule in all 5 seeds"),
        Patch(facecolor="white", edgecolor=G2, label="not kept")]
ax.legend(handles=hand, loc="lower left", fontsize=4.9, handlelength=0.9, handleheight=0.9, labelspacing=0.3,
          borderaxespad=0, bbox_to_anchor=(0.23, 0.02))

# ---------------------------------------------------------------- c  bars: accuracy and logit difference, 5 seeds
CR = [pickle.load(open(os.path.join(ROOT, "results", "gpt2_cross", f"cross_s{s_}.pkl"), "rb")) for s_ in SEEDS]
ax = fig.add_subplot(gs[2])
rows = [("full\n(144)", lambda c: c["acc"]["full"], lambda c: 1.0, K),
        ("published\n(26)", lambda c: c["acc"]["published"], lambda c: c["ld"]["published"] / c["ld"]["full"], G2),
        ("ours\n(20-21)", lambda c: c["acc"]["rule"]["0.01"], lambda c: c["ld"]["rule"]["0.01"] / c["ld"]["full"], RED),
        ("ranked\nonce", lambda c: c["acc"]["once"]["0.01"], lambda c: c["ld"]["once"]["0.01"] / c["ld"]["full"], G1)]
w = 0.38
for i, (lab, fa, fl, col) in enumerate(rows):
    for j, (f, alpha) in enumerate(((fa, 1.0), (fl, 0.45))):
        v = np.array([f(c) for c in CR])
        x = i + (j - 0.5) * w
        ax.bar(x, v.mean(), width=w * 0.92, color=col, alpha=alpha, lw=0)
        ax.vlines(x, v.min(), v.max(), color=K, lw=0.6)
        ax.text(x, max(v.max(), 0) + 0.04, f"{v.mean() * 100:.0f}", ha="center", va="bottom", fontsize=4.2)
ax.axhline(0, color=K, lw=0.5)
ax.set(xlim=(-0.6, 3.6), ylim=(-0.35, 1.8), xticks=range(4), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100"])
ax.set_xticklabels([r[0] for r in rows], fontsize=5)
ax.tick_params(axis="x", length=0)
from matplotlib.patches import Patch as _P
ax.legend(handles=[_P(facecolor=G1, label="correct name (%)"), _P(facecolor=G1, alpha=0.45, label="logit difference (% of full)")],
          loc="upper left", fontsize=4.8, handlelength=0.9, borderaxespad=0, ncol=1)
title(ax, "c", "Held out, 5 seeds")

out = os.path.join(ROOT, "paper", "neuroai", "fig_circuits_v5")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
