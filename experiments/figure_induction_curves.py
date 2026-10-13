"""Induction loss curves: the phase transition. Every number is read from results/induction/*.pkl.

  Top row: dense models trained from scratch, 10 seeds each, one panel per architecture. Two layers
  sit on a plateau and then drop suddenly (the induction heads form); one layer never drops.
  Bottom row: hemodynamic attenuation (8 + 8 heads, the paper's settings): loss, and open heads per
  layer, so the closures can be read against the transition.

  python experiments/figure_induction_curves.py   ->  figures/fig_induction_curves.png and .pdf
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RED = "#b2182b"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})


def load(pattern):
    return [pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(ROOT, "results", "induction", pattern)))]


def transition(r, acc=0.5):
    """First validation step at which half the predictable tokens are right: the induction circuit
    has formed. None if it never happens."""
    a = np.array(r["hist"]["val_acc"])
    return r["hist"]["val_step"][int(np.argmax(a >= acc))] if (a >= acc).any() else None


dense = [("1 layer, 1 head", "ind_dense_h1-0_*st6000_s*.pkl"),
         ("1 layer, 8 heads", "ind_dense_h8-0_*st6000_s*.pkl"),
         ("2 layers, 1 + 1", "ind_dense_h1-1_*st6000_s*.pkl"),
         ("2 layers, 2 + 1", "ind_dense_h2-1_*st6000_s*.pkl"),
         ("2 layers, 8 + 8", "ind_dense_h8-8_*st6000_s*.pkl")]
rule = load("ind_collateral_bar_logit_h8-8_pf0.05_*st6000_s*.pkl")

fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.55, wspace=0.35)
YLIM = (2e-4, 8)
ZOOM = 1500                                        # the top row shows the first 1500 of 6000 steps
for i, (name, pat) in enumerate(dense):
    ax = fig.add_subplot(gs[0, i])
    runs = load(pat)
    for r in runs:
        ax.plot(r["hist"]["val_step"], r["hist"]["val_loss"], color=K, lw=0.6, alpha=0.55)
    bar, triv = runs[0]["bar"], runs[0]["trivial"]
    ax.axhline(bar, color=G2, lw=0.6, ls=":")
    ax.axhline(triv, color=G3, lw=0.6)
    t = [transition(r) for r in runs]
    formed = [x for x in t if x is not None]
    if formed:
        ax.axvline(np.median(formed), color=RED, lw=0.7, ls="--")
        note = f"forms at step {int(np.median(formed))}\n{len(formed)}/{len(runs)} runs (median)"
    else:
        note = f"never forms\n(0/{len(runs)} runs)"
    if formed:
        ax.text(0.28, 0.04, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color=RED)
    else:
        ax.text(0.97, 0.55, note, transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=G1)
    ax.set(yscale="log", ylim=YLIM, xlim=(0, ZOOM), xticks=[0, 500, 1000, 1500], xticklabels=["0", "500", "1k", "1.5k"])
    ax.minorticks_off()
    ax.set_title(name, pad=3)
    if i == 0:
        ax.set_ylabel("validation loss (nats)")
        ax.text(ZOOM * 0.97, triv * 1.12, "chance", fontsize=5.5, color=G1, ha="right", va="bottom")
        ax.text(ZOOM * 0.97, bar * 1.15, "solved", fontsize=5.5, color=G1, ha="right", va="bottom")
    else:
        ax.set_yticklabels([])
    if i == 2:
        ax.set_xlabel("training step")
fig.text(0.075, 0.955, "a", fontsize=8, fontweight="bold")
fig.text(0.105, 0.955, "Dense models trained from scratch, 10 seeds each (first 1,500 of 6,000 steps)", fontsize=7)

# ---------------------------------------------------------------- b  the rule: loss, and heads per layer
ax = fig.add_subplot(gs[1, 0:2])
for r in rule:
    ax.plot(r["hist"]["val_step"], r["hist"]["val_loss"], color=RED, lw=0.6, alpha=0.6)
ax.axhline(rule[0]["bar"], color=G2, lw=0.6, ls=":")
ax.axhline(rule[0]["trivial"], color=G3, lw=0.6)
start = rule[0]["cfg"]["start_frac"] * rule[0]["cfg"]["steps"]
ax.axvline(start, color=G2, lw=0.6, ls="--")
ax.text(start + 80, 2.5, "valves start\nmoving", fontsize=5.6, color=G1, va="top")
tr = [transition(r) for r in rule]
ax.axvline(np.median(tr), color=RED, lw=0.7, ls="--")
ax.text(np.median(tr) + 80, 0.35, f"forms at\nstep {int(np.median(tr))}", fontsize=5.6, color=RED, ha="left", va="center")
ax.text(2300, 0.25, "heads closing", fontsize=5.6, color=G1, ha="left", va="center")
ax.set(yscale="log", ylim=YLIM, xlim=(0, 6000), xticks=[0, 3000, 6000], xticklabels=["0", "3k", "6k"],
       xlabel="training step", ylabel="validation loss (nats)")
ax.minorticks_off()
ax.set_title("Hemodynamic attenuation, 8 + 8 heads: loss", pad=3)

ax = fig.add_subplot(gs[1, 2:4])
h1 = rule[0]["sizes"][0]
for r in rule:
    L = np.asarray(r["hist"]["ledger"]) > 0
    ax.plot(L[:, :h1].sum(1), color=RED, lw=0.7, alpha=0.5)
    ax.plot(L[:, h1:].sum(1), color=K, lw=0.7, alpha=0.5)
ax.axvline(np.median(tr), color=RED, lw=0.7, ls="--")
ax.text(5900, 2.3, "layer 1", color=RED, fontsize=6, ha="right", va="bottom")
ax.text(5900, 1.3, "layer 2", color=K, fontsize=6, ha="right", va="bottom")
ax.set(xlim=(0, 6000), ylim=(0, 8.6), yticks=[0, 1, 2, 4, 8], xticks=[0, 3000, 6000], xticklabels=["0", "3k", "6k"],
       xlabel="training step", ylabel="open heads")
ax.set_title("Hemodynamic attenuation: open heads per layer", pad=3)
fig.text(0.075, 0.47, "b", fontsize=8, fontweight="bold")

# summary of the transition, computed from the runs
ax = fig.add_subplot(gs[1, 4])
ax.axis("off")
lines = []
for name, pat in dense[2:]:
    t = [transition(r) for r in load(pat)]
    lines.append(f"{name.split(', ')[1]:<6} {int(np.median(t)):>5}   {min(t)}-{max(t)}")
lines.append(f"{'ours':<6} {int(np.median(tr)):>5}   {min(tr)}-{max(tr)}")
first_close = [int(np.argmax((np.asarray(r['hist']['ledger']) > 0).sum(1) < r['hist']['ledger'].shape[1])) for r in rule]
ax.text(0, 1.0, "step the circuit forms\n(half of answers right)\n", fontsize=6, va="top", color=K)
ax.text(0, 0.78, "model   median   range\n" + "\n".join(lines), fontsize=5.8, va="top", family="monospace", color=K)
ax.text(0, 0.30, f"first head closed by the\nrule: step {int(np.median(first_close))} (median)", fontsize=5.8, va="top", color=G1)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_induction_curves.{ext}"), bbox_inches="tight", facecolor="white")
print("transition steps, dense:", {n: [transition(r) for r in load(p)] for n, p in dense})
print("transition steps, rule:", tr, " first closure:", first_close)
print("wrote figures/fig_induction_curves.png and .pdf")
