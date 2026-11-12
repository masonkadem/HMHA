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
fig = plt.figure(figsize=(7.0, 2.45))
gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1], wspace=0.16)

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
ax = fig.add_subplot(gs[0]); ax.axis("off")
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
title(ax, "a", "GPT-2 small: the published circuit; heads our rule keeps are filled")

# ---------------------------------------------------------------- key, inside panel b
n = {f: len(kept[f]) for f in FR}
ins = {f: sum(k in role for k in kept[f]) for f in FR}
hand = [Patch(facecolor=SHADE["0.05"], label="kept by our rule in all 5 seeds"),
        Patch(facecolor="white", edgecolor=G2, label="not kept")]
ax.legend(handles=hand, loc="lower left", fontsize=4.9, handlelength=0.9, handleheight=0.9, labelspacing=0.3,
          borderaxespad=0, bbox_to_anchor=(0.23, 0.02))



# ---------------------------------------------------------------- b  heads kept vs performance, 5 seeds
CR = [pickle.load(open(os.path.join(ROOT, "results", "gpt2_cross", f"cross_s{s_}.pkl"), "rb")) for s_ in SEEDS]
ax = fig.add_subplot(gs[1])
nk = [np.mean([len(r["kept"]) for r in R[f]]) for f in FR]
def line(get, color, mk, ls, lab):
    v = np.array([[get(c, f) for f in FR] for c in CR])
    ax.plot(nk, v.mean(0), ls=ls, marker=mk, color=color, ms=3, lw=1.0, label=lab)
    ax.vlines(nk, v.min(0), v.max(0), color=color, lw=1.4)
    return v
acc = line(lambda c, f: c["acc"]["rule"][f], RED, "o", "-", "ours: correct name")
ld = line(lambda c, f: c["ld"]["rule"][f] / c["ld"]["full"], RED, "s", "--", "ours: logit difference")
once = line(lambda c, f: c["acc"]["once"][f], G1, "o", "-", "ranked once: correct name")
pa = np.mean([c["acc"]["published"] for c in CR]); pl = np.mean([c["ld"]["published"] / c["ld"]["full"] for c in CR])
ax.plot(26, pa, "o", mfc="white", mec=K, ms=3.6, mew=0.9)
ax.plot(26, pl, "s", mfc="white", mec=K, ms=3.3, mew=0.9)
ax.text(26, pa + 0.07, "published", ha="center", fontsize=5.2, color=K)
ax.text(nk[2] - 0.6, acc.mean(0)[2] - 0.13, f"{acc.min(0)[2]:.0%}-{acc.max(0)[2]:.0%}", ha="center", fontsize=5.2, color=RED)
ax.axhline(1, color=G2, lw=0.5, ls=":")
ax.text(9, 1.02, "full model", fontsize=5, color=G2, va="bottom")
ax.axhline(0.5, color=G3, lw=0.5)
ax.set(xlim=(8, 28.5), ylim=(0, 1.45), xticks=[10, 17, 20, 26], yticks=[0, 0.5, 1], yticklabels=["0", "50", "100"],
       xlabel="heads kept (of 144)")
ax.legend(loc="upper left", fontsize=4.8, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.0, 1.0), ncol=1,
          labelspacing=0.25)
ax.set_ylim(0, 1.75)
title(ax, "b", "Held-out performance (%), 5 seeds")

out = os.path.join(ROOT, "paper", "neuroai", "fig_circuits_v6")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
