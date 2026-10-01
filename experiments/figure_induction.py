"""Induction figure. Every number is read from results/induction/*.pkl, except panels a and b,
which use one collateral-rule model (seed 0, the batch settings), trained here once and
cached in figures/induction_model.pt.

  python experiments/figure_induction.py   ->  figures/fig_induction.png and .pdf
"""
import glob, os, pickle, sys
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import ICfg, InductionNet, train, make_batch, predict_mask, role_scores

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
title = lambda ax, letter, text="": ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)

RUNS = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))]


def pick(rule, price=0.03, heads=(8, 8)):
    return sorted([r for r in RUNS if r["cfg"]["rule"] == rule and tuple(r["sizes"]) == heads
                   and (rule not in ("collateral", "ablate", "random") or r["cfg"]["price_frac"] == price)],
                  key=lambda r: r["cfg"]["seed"])


def stayed_solved(r):
    h, start = r["hist"], r["cfg"]["start_frac"] * r["cfg"]["steps"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= start]
    first = next((i for i, l in enumerate(ls) if l <= r["bar"]), None)
    return first is not None and max(ls[first:]) <= r["bar"]


# ---------------------------------------------------------------- one solved model
cfg = replace(ICfg(), rule="collateral", seed=0)
CACHE = os.path.join(ROOT, "figures", "induction_model.pt")
if os.path.exists(CACHE):
    blob = torch.load(CACHE, map_location="cpu")
    model = InductionNet(cfg)
    model.load_state_dict(blob["state"])
    kept = blob["kept"]
else:
    model, out = train(cfg)
    model = model.cpu()
    kept = torch.tensor(out["kept"])
    torch.save({"state": model.state_dict(), "kept": kept}, CACHE)
model.eval()
gate = kept.float()
h1 = model.sizes[0]
roles, chance = role_scores(model, cfg, "cpu")
off = gate.clone()
off[h1:] = 0                                              # every kept layer-2 head switched off
with torch.no_grad():
    gen = torch.Generator().manual_seed(7)
    for _ in range(200):                                  # an example with a short prefix and gap
        tok, starts = make_batch(1, cfg, gen)
        if int(starts[0, 0]) <= 2 and 3 <= int(starts[0, 1] - starts[0, 0]) - cfg.L <= 5:
            break
    pred_on = model(tok, gate).argmax(-1)[0]
    pred_off = model(tok, off).argmax(-1)[0]
mask = predict_mask(starts, cfg)[0]

fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1, 1], hspace=0.62, wspace=0.5)

# ---------------------------------------------------------------- a  task and predictions
ax = fig.add_subplot(gs[0, :2])
title(ax, "a", "Induction: copy what followed the earlier occurrence")
ax.axis("off")
f, s = int(starts[0, 0]), int(starts[0, 1])
end = s + cfg.L
ax.set(xlim=(-5.5, end + 0.2), ylim=(-3.6, 1.6))
ax.text(-5.4, 0, "sequence", va="center", color=G1)
ax.text(-5.4, -1.35, "prediction", va="center", color=G1)
ax.text(-5.4, -2.6, "induction\nheads off", va="center", color=G1, fontsize=6.5)
for t in range(end):
    in_copy = f <= t < f + cfg.L or s <= t < s + cfg.L
    ax.add_patch(Rectangle((t - 0.45, -0.4), 0.9, 0.8, facecolor=G3 if in_copy else "white",
                           edgecolor=K if in_copy else G2, lw=0.5))
    ax.text(t, 0, str(int(tok[0, t])), ha="center", va="center", fontsize=5.3, color=K if in_copy else G1)
for t in range(end - 1):
    if not mask[t]:
        continue
    for row, pred in ((-1.35, pred_on), (-2.6, pred_off)):
        ok = int(pred[t]) == int(tok[0, t + 1])
        ax.add_patch(Rectangle((t + 1 - 0.45, row - 0.4), 0.9, 0.8, facecolor="white",
                               edgecolor=K if ok else RED, lw=0.5 if ok else 1.0))
        ax.text(t + 1, row, str(int(pred[t])), ha="center", va="center", fontsize=5.3, color=K if ok else RED)
ax.annotate("", xy=(f + 3, 0.45), xytext=(s + 2, 0.45),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.7, mutation_scale=6,
                            connectionstyle="bar,fraction=-0.12"))
ax.text((f + s) / 2 + 2.5, 1.35, "find the earlier copy, predict what came next", ha="center", fontsize=6.3, color=RED)
nr = int(mask.sum())
ok_on = sum(int(pred_on[t]) == int(tok[0, t + 1]) for t in range(end - 1) if mask[t])
ok_off = sum(int(pred_off[t]) == int(tok[0, t + 1]) for t in range(end - 1) if mask[t])
ax.text(end + 0.1, -1.35, f"{ok_on}/{nr}", va="center", fontsize=6.3, color=K)
ax.text(end + 0.1, -2.6, f"{ok_off}/{nr}", va="center", fontsize=6.3, color=RED)

# ---------------------------------------------------------------- b  kept heads and their roles
ax = fig.add_subplot(gs[0, 2])
title(ax, "b", "Heads kept (one run)")
for layer in (0, 1):
    score = roles["prev"][0] if layer == 0 else roles["ind"][1]
    ch = chance["prev"] if layer == 0 else chance["ind"]
    for h in range(8):
        on = bool(kept[layer * h1 + h]) if layer else bool(kept[h])
        ax.add_patch(Rectangle((h, 1 - layer), 0.86, 0.72, facecolor=RED if on else "white",
                               edgecolor=RED if on else G2, lw=0.6))
        if on:
            ax.text(h + 0.43, 1 - layer + 0.36, f"{score[h] / ch:.0f}", ha="center", va="center",
                    fontsize=5.5, color="white")
ax.set(xlim=(-0.2, 8.1), ylim=(-0.25, 1.9), xticks=[], yticks=[0.36, 1.36])
ax.set_yticklabels(["layer 2", "layer 1"])
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
ax.tick_params(left=False)
ax.text(0, -0.2, "number = role score / chance\n(layer 1 previous token, layer 2 induction)", fontsize=5.8,
        color=G1, va="top")

# ---------------------------------------------------------------- c  heads kept per layer, all runs
ax = fig.add_subplot(gs[1, 0])
rng = np.random.default_rng(0)
arms = [("collateral", "o", RED), ("importance", "D", K), ("random", "v", K), ("pruning", "s", K)]
rule_of = {"collateral": "collateral", "importance": "ablate", "random": "random", "pruning": "prune"}
for name, m, color in arms:
    rs = pick(rule_of[name])
    x = [int(r["kept"][:r["sizes"][0]].sum()) for r in rs]
    y = [int(r["kept"][r["sizes"][0]:].sum()) for r in rs]
    ax.plot(np.array(x) + rng.uniform(-0.18, 0.18, len(x)), np.array(y) + rng.uniform(-0.18, 0.18, len(y)),
            m, color=color, ms=3.5, mfc=color if color == RED else "white", mew=0.7, ls="none", label=name)
ax.plot(1, 1, "*", color=K, ms=9, mfc="none", mew=0.8)
ax.annotate("minimal (1, 1)", xy=(1, 1), xytext=(3.2, 0.2), fontsize=6.2, color=G1,
            arrowprops=dict(arrowstyle="-", color=G2, lw=0.5))
ax.set(xlim=(-0.5, 8.5), ylim=(-0.5, 8.5), xticks=range(0, 9, 2), yticks=range(0, 9, 2),
       xlabel="heads kept, layer 1", ylabel="heads kept, layer 2")
ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0)
title(ax, "c", "Heads kept (10 seeds)")

# ---------------------------------------------------------------- d  loss during training
ax = fig.add_subplot(gs[1, 1:])
for rule, color, ls, label in (("dense", K, "-", "dense (8 + 8)"), ("prune", G1, "-", "standard pruning"),
                               ("collateral", RED, "-", "collateral")):
    r = pick(rule)[0]
    ax.plot(r["hist"]["val_step"], r["hist"]["val_loss"], color=color, ls=ls, lw=0.9, label=label)
r = pick("collateral")[0]
ax.axhline(r["bar"], color=G2, lw=0.6, ls=":")
ax.text(2990, r["bar"] * 1.3, "solved bar", ha="right", fontsize=6, color=G1)
ax.axvspan(0, r["cfg"]["start_frac"] * r["cfg"]["steps"], color=G3, lw=0, zorder=0)
ax.text(40, 2.2, "all heads on\n(all three lines identical)", fontsize=6, color=G1, va="top")
ax.set(yscale="log", ylim=(1e-3, 6), xlim=(0, 3000), xlabel="training step", ylabel="loss")
ax.legend(loc="upper right", borderaxespad=0)
title(ax, "d", "Loss during training (one seed)")

# ---------------------------------------------------------------- e  stays solved
ax = fig.add_subplot(gs[2, 0])
names = ["collateral", "importance", "random", "pruning"]
vals, ns = [], []
for name in names:
    rs = pick(rule_of[name])
    vals.append(100 * np.mean([stayed_solved(r) for r in rs]))
    ns.append(len(rs))
ax.bar(range(4), vals, 0.6, color=[RED, "white", "white", "white"], edgecolor=[RED, K, K, K], lw=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=6.2)
ax.set(ylim=(0, 115), yticks=[0, 50, 100], ylabel="runs never above the bar\nafter solving (%)")
ax.set_xticks(range(4), names, rotation=25, ha="right")
title(ax, "e", "Stays solved while pruning")

# ---------------------------------------------------------------- f  price
ax = fig.add_subplot(gs[2, 1])
prices = [0.01, 0.03, 0.1, 0.3]
for i, p in enumerate(prices):
    rs = pick("collateral", price=p)
    tot = [int(r["kept"].sum()) for r in rs]
    ok = [r["final_loss"] <= r["bar"] for r in rs]
    for t, good in zip(tot, ok):
        ax.plot(i + rng.uniform(-0.12, 0.12), t, "o", ms=3.3, color=RED, mfc=RED if good else "white", mew=0.7)
    ax.text(i, 13.2, f"{sum(ok)}/{len(ok)}", ha="center", fontsize=6.2, color=G1)
ax.axhline(2, color=G2, lw=0.6, ls=":")
ax.text(3.35, 2.4, "minimal", ha="right", fontsize=6, color=G1)
ax.text(-0.4, 14.6, "solved (open circle = not solved):", fontsize=6.2, color=G1)
ax.set(xticks=range(4), xticklabels=[str(p) for p in prices], ylim=(0, 15.5), yticks=[0, 4, 8, 12],
       xlabel="price of a head", ylabel="heads kept (total)")
title(ax, "f", "The price sets the reserve")

# ---------------------------------------------------------------- g  spare heads lower the loss
ax = fig.add_subplot(gs[2, 2])
for rule, heads, m, color, label in (("dense", (1, 1), "*", K, "dense 1 + 1"), ("dense", (8, 8), "s", K, "dense 8 + 8"),
                                     ("prune", (8, 8), "s", G1, None), ("collateral", (8, 8), "o", RED, "collateral")):
    rs = pick(rule, heads=heads) if rule == "dense" else pick(rule)
    x = [int(r["kept"].sum()) for r in rs]
    y = [r["final_loss"] for r in rs]
    ax.plot(np.array(x) + rng.uniform(-0.2, 0.2, len(x)), y, m, color=color, ms=3.5 if m != "*" else 6,
            mfc=color if color == RED else "white", mew=0.7, ls="none", label=label or "pruning")
ax.set(yscale="log", ylim=(1e-3, 3e-2), xlabel="heads kept (total)", ylabel="final loss", xticks=[2, 4, 8, 16])
ax.minorticks_off()
ax.legend(loc="upper right", handletextpad=0.2, borderaxespad=0)
title(ax, "g", "Final loss against heads kept")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_induction.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_induction.png and .pdf")
