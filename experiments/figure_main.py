"""Main figure (publication style). Every number is read from the result pickles.

  Figure 2 (results). (a) loss and (b) heads during training for k* = 2, 4, 6, 8, collateral
  rule against the dense model  (c) count and stability  (d) final loss  (e) duplicated
  heads  (f) backup heads under damage. The task and model are in figure_setup.py.

  python experiments/figure_main.py   ->  figures/fig_main.png and figures/fig_main.pdf
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED = "#b2182b"                       # the collateral rule (blood)
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})

RUNS = []
for d in ("results", "results/proposal", "results/confirm"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
            RUNS.append(r)
DENSE = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "dense", "*_H32_*.pkl"))]
DEFAULT = Cfg()
PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", kappa_end=1.5,
              target_frac=0.02, delay=0, pool_beta=0.0, leak=0.0, n_territories=8,
              stall_gate=0.0, autoreg_gain=0.003, precondition=0.0, flow_exponent=4.0,
              price_frac=0.01, probe_every=25, taper=0, budget_hold_frac=0.25, prune_stop=1,
              conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, probe_masks=32)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]
dense = lambda R: [d for d in DENSE if d["cfg"]["n_rel"] == R]
title = lambda ax, letter, text="": ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


def stayed_solved(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


def compute(r):
    L = r["hist"]["ledger"]
    probes = r["hist"].get("n_probes", 0) * get(r, "probe_batch") / (3 * get(r, "batch_size"))
    return (L > 0).sum(1).mean() / L.shape[1] + probes / len(L)


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
SIZES = (2, 3, 4, 6, 8)
BAR = 0.02
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 4, height_ratios=[0.85, 0.7, 1.1], hspace=0.45, wspace=0.55)

# ---------------------------------------------------------------- c, d  training, small multiples
show = (2, 4, 6, 8)
cd = gs[0:2, :].subgridspec(2, 4, height_ratios=[0.85, 0.7], hspace=0.16, wspace=0.55)
for i, k in enumerate(show):
    r = sorted(runs(**RULE, n_rel=k), key=lambda r: get(r, "seed"))[0]
    d0 = next(d for d in dense(k) if d["cfg"]["seed"] == get(r, "seed"))
    axl = fig.add_subplot(cd[0, i])
    axl.plot(d0["hist"]["val_step"], d0["hist"]["val_loss"], color=K, lw=0.9, label="dense")
    axl.plot(r["hist"]["val_step"], r["hist"]["val_loss"], color=RED, lw=0.9, label="collateral")
    axl.axhline(BAR, color=G2, lw=0.6, ls=":")
    axl.set(yscale="log", ylim=(3e-6, 2), xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=[])
    axl.minorticks_off()
    axh = fig.add_subplot(cd[1, i])
    on = (r["hist"]["ledger"] > 0).sum(1)
    axh.plot([0, 4000], [32, 32], color=K, lw=0.9)
    axh.plot(np.arange(len(on)), on, color=RED, lw=0.9, drawstyle="steps-post")
    axh.axhline(k, color=G2, lw=0.6, ls=":")
    axh.set(yscale="log", ylim=(1.5, 45), yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"],
            xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=["0", "2k", "4k"])
    axh.minorticks_off()
    axh.text(3950, 22, f"compute {compute(r):.2f}", ha="right", fontsize=6, color=RED)
    if i == 0:
        title(axl, "a", f"$k^*={k}$")
        axh.text(-0.42, 1.0, r"$\mathbf{b}$", transform=axh.transAxes, fontsize=7, va="bottom")
        axl.set_ylabel("loss")
        axh.set_ylabel("heads")
        axl.legend(loc="upper right", handlelength=1.2, borderaxespad=0)
    else:
        axl.set_title(f"$k^*={k}$", pad=4)
        axl.set_yticklabels([])
        axh.set_yticklabels([])
    if i == 1:
        axh.set_xlabel("training step", x=1.15)

# ---------------------------------------------------------------- e  count and stability
ax = fig.add_subplot(gs[2, 0])
arms = [("collateral", RULE, SIZES, RED, "o"),
        ("pruning", dict(supply="prune"), SIZES, K, "s"),
        ("importance", {**RULE, "local_value": "ablate"}, (2, 4, 8), K, "D"),
        ("random", {**RULE, "local_value": "random"}, (2, 4, 8), K, "v")]
for name, kw, sizes, color, m in arms:
    rs = [x for k in sizes for x in runs(**kw, n_rel=k)]
    ax.plot(100 * np.mean([kept(x) == get(x, "n_rel") for x in rs]), 100 * np.mean([stayed_solved(x) for x in rs]),
            m, color=color, ms=4.5, mfc=color if color == RED else "white", mew=0.8, ls="none", label=name)
ax.set(xlim=(-5, 106), ylim=(-6, 108), xticks=[0, 50, 100], yticks=[0, 50, 100],
       xlabel="exact count (% runs)", ylabel="stays solved (% runs)")
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), handletextpad=0.2, borderaxespad=0)
title(ax, "c", "Count and stability")

# ---------------------------------------------------------------- f  final loss
ax = fig.add_subplot(gs[2, 1])
rng = np.random.default_rng(1)
for i, k in enumerate(SIZES):
    dl = [d["hist"]["val_loss"][-1] for d in dense(k)]
    cl = [x["final_loss"] for x in runs(**RULE, n_rel=k)]
    ax.plot(i - 0.18 + rng.uniform(-0.06, 0.06, len(dl)), dl, "o", ms=2.3, mfc="white", mec=K, mew=0.5,
            label="dense" if i == 0 else None)
    ax.plot(i + 0.18 + rng.uniform(-0.06, 0.06, len(cl)), cl, "o", ms=2.3, color=RED,
            label="collateral" if i == 0 else None)
ax.axhline(BAR, color=G2, lw=0.6, ls=":")
ax.set(yscale="log", ylim=(3e-6, 0.1), xticks=range(len(SIZES)), xticklabels=[str(k) for k in SIZES],
       xlabel="$k^*$", ylabel="final loss")
ax.minorticks_off()
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), handletextpad=0.1, borderaxespad=0)
title(ax, "d", "Final loss")

# ---------------------------------------------------------------- g  duplicated heads
ax = fig.add_subplot(gs[2, 2])
res = {}
for value in ("refit", "ablate"):
    res[value] = np.array([sum(bool(o[h] and o[h + 16]) for h in range(16))
                           for o in (x["hist"]["ledger"][-1] > 0
                                     for x in runs(**RULE, n_rel=4, plant_copies=1, local_value=value))])
for i, (value, face, label) in enumerate((("refit", RED, "collateral"), ("ablate", "white", "importance"))):
    v = res[value]
    ax.bar(i, v.mean(), 0.6, color=face, edgecolor=K if face == "white" else face, lw=0.6)
    ax.plot(i + rng.uniform(-0.15, 0.15, len(v)), v, ".", color=G1, ms=2.5, zorder=3)
ax.set(xticks=[0, 1], ylim=(0, 8), xlim=(-0.6, 1.6), ylabel="copy pairs both kept")
ax.set_xticklabels(["collateral", "importance"])
title(ax, "e", "Start with every head copied")

# ---------------------------------------------------------------- h  damage
ax = fig.add_subplot(gs[2, 3])


def predict(R, p, price=0.03):
    q = 1 - p
    val = lambda B: q * sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1)) / R
    return max(B for B in range(1, 33) if val(B) >= price)


ps = [0.0, 0.05, 0.1, 0.2, 0.3]
for R, m in ((4, "o"), (2, "s")):
    ax.plot(ps, [predict(R, p) for p in ps], color=G2, lw=0.9, ls="--", zorder=1,
            label="predicted" if R == 4 else None)
    means = [np.mean([kept(x) for x in runs(**RULE, n_rel=R, head_dropout=p)][:10]) for p in ps]
    ax.plot(ps, means, m, color=RED, ms=3.5, mfc=RED if R == 4 else "white", mew=0.8, ls="none",
            label="measured" if R == 4 else None)
    ax.text(0.325, predict(R, 0.3), f"$k^*={R}$", va="center", fontsize=6, color=G1)
ax.set(xlabel="chance a head fails", ylabel="heads kept", ylim=(1, 8), xlim=(-0.02, 0.385),
       xticks=[0, 0.1, 0.2, 0.3], xticklabels=["0", "0.1", "0.2", "0.3"], yticks=[2, 4, 6, 8])
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0)
title(ax, "f", "Backup heads")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_main.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_main.png and figures/fig_main.pdf")
