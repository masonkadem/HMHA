"""Induction figure. Every number is read from results/induction/*.pkl, except panels a and b,
which use one hemodynamic-attenuation model (seed 0), trained here once and cached in
figures/induction_model.pt. Every method in c and d uses the same settings as the paper's text:
loss-based head values, price 0.05 of the know-nothing loss, 6000 steps, 10 seeds.

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
title = lambda ax, letter, text="", pad=4: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=pad)

RUNS = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))]
SAME = dict(probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0)
METHODS = [  # (label, settings, colour, fill)
    ("hemodynamic\nattenuation", dict(rule="collateral", **SAME), RED, (0.698, 0.094, 0.169, 0.25)),
    ("Michel et al.\n2019", dict(rule="prune", steps=6000), K, (1, 1, 1, 0.6)),
    ("no re-fit", dict(rule="ablate", **SAME), K, (1, 1, 1, 0.6)),
    ("random\nchoice", dict(rule="random", **SAME), K, (1, 1, 1, 0.6)),
]


def pick(heads=(8, 8), **kw):
    """Runs whose settings match kw (missing settings count as the defaults)."""
    base = ICfg()
    match = lambda r: all(r["cfg"].get(k, getattr(base, k)) == v for k, v in kw.items())
    return sorted([r for r in RUNS if match(r) and tuple(r["sizes"]) == heads], key=lambda r: r["cfg"]["seed"])


def never_broke(r):
    """Once below the solved bar, the loss never went back above it."""
    ls = r["hist"]["val_loss"]
    first = next((i for i, l in enumerate(ls) if l <= r["bar"]), None)
    return first is not None and max(ls[first:]) <= r["bar"]


# ---------------------------------------------------------------- one solved model
cfg = replace(ICfg(), rule="collateral", seed=0, **SAME)
CACHE = os.path.join(ROOT, "figures", "induction_model.pt")
blob = torch.load(CACHE, map_location="cpu") if os.path.exists(CACHE) else None
if blob is None or blob.get("cfg") != SAME:
    model, out = train(cfg)
    blob = {"state": model.cpu().state_dict(), "kept": torch.tensor(out["kept"]), "cfg": SAME}
    torch.save(blob, CACHE)
model = InductionNet(cfg)
model.load_state_dict(blob["state"])
model.eval()
kept = blob["kept"]
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

fig = plt.figure(figsize=(7.2, 4.6))
outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 1], hspace=0.45)
top = outer[0].subgridspec(1, 2, width_ratios=[2.6, 1], wspace=0.28)
bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1.2], wspace=0.55)

# ---------------------------------------------------------------- a  task and predictions
ax = fig.add_subplot(top[0])
title(ax, "a", "Induction: find the earlier copy, predict what came next")
ax.axis("off")
f, s = int(starts[0, 0]), int(starts[0, 1])
end = s + cfg.L
ax.set(xlim=(-5.5, end + 1.2), ylim=(-3.2, 2.1))
ax.text(-5.4, 0, "sequence", va="center", color=G1)
ax.text(-5.4, -1.35, "prediction", va="center", color=G1)
ax.text(-5.4, -2.5, "layer-2 heads off", va="center", color=G1)
for t in range(end):
    in_copy = f <= t < f + cfg.L or s <= t < s + cfg.L
    ax.add_patch(Rectangle((t - 0.45, -0.4), 0.9, 0.8, facecolor=G3 if in_copy else "white",
                           edgecolor=K if in_copy else G2, lw=0.5))
    ax.text(t, 0, str(int(tok[0, t])), ha="center", va="center", fontsize=5.3, color=K if in_copy else G1)
for t in range(end - 1):
    if not mask[t]:
        continue
    for row, pred in ((-1.35, pred_on), (-2.5, pred_off)):
        ok = int(pred[t]) == int(tok[0, t + 1])
        ax.add_patch(Rectangle((t + 1 - 0.45, row - 0.4), 0.9, 0.8, facecolor="white",
                               edgecolor=K if ok else RED, lw=0.5 if ok else 1.0))
        ax.text(t + 1, row, str(int(pred[t])), ha="center", va="center", fontsize=5.3, color=K if ok else RED)
# the bracket runs above the boxes: from a token in the repeat back to what followed it the first time
x0, x1, yb = s + 2, f + 3, 1.05
ax.plot([x0, x0, x1], [0.45, yb, yb], color=RED, lw=0.7)
ax.annotate("", xy=(x1, 0.45), xytext=(x1, yb + 0.01),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
ax.text((x0 + x1) / 2, yb + 0.15, f"after {int(tok[0, x0])} came {int(tok[0, x1])} last time, "
        f"so predict {int(tok[0, x1])}", ha="center", va="bottom", fontsize=6, color=RED)
nr = int(mask.sum())
ok_on = sum(int(pred_on[t]) == int(tok[0, t + 1]) for t in range(end - 1) if mask[t])
ok_off = sum(int(pred_off[t]) == int(tok[0, t + 1]) for t in range(end - 1) if mask[t])
ax.text(end + 0.1, -1.35, f"{ok_on}/{nr}", va="center", fontsize=6.3, color=K)
ax.text(end + 0.1, -2.5, f"{ok_off}/{nr}", va="center", fontsize=6.3, color=RED)

# ---------------------------------------------------------------- b  kept heads and their jobs
ax = fig.add_subplot(top[1])
title(ax, "b", "Heads kept (one run)")
for layer in (0, 1):
    score = roles["prev"][0] if layer == 0 else roles["ind"][1]
    ch = chance["prev"] if layer == 0 else chance["ind"]
    for h in range(8):
        on = bool(kept[layer * h1 + h])
        ax.add_patch(Rectangle((h, 1 - layer), 0.86, 0.72, facecolor=RED if on else "white",
                               edgecolor=RED if on else G2, lw=0.6))
        if on:
            ax.text(h + 0.43, 1 - layer + 0.36, f"{score[h] / ch:.0f}×", ha="center", va="center",
                    fontsize=5.3, color="white")
ax.set(xlim=(-0.2, 8.1), ylim=(-0.2, 1.8), xticks=[], yticks=[0.36, 1.36])
ax.set_yticklabels(["layer 2", "layer 1"])
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
ax.tick_params(left=False)
ax.text(0, -0.15, "number: how much more than chance the head\nlooks where its job says (layer 1: the token\n"
        "before; layer 2: the token after the earlier copy)", fontsize=5.6, color=G1, va="top", linespacing=1.2)

# ---------------------------------------------------------------- c  never broke, every run
# every method kept the same heads here (2 in layer 1, 1 in layer 2, in every run), so the panel
# shows the one thing that differs: whether the model stayed solved while heads closed
ax = fig.add_subplot(bot[0])
kept_all = set()
for i, (name, kw, color, fill) in enumerate(METHODS):
    rs = pick(**kw)
    kept_all |= {(int(r["kept"][:h1].sum()), int(r["kept"][h1:].sum())) for r in rs}
    n_ok = sum(never_broke(r) for r in rs)
    ax.barh(i, n_ok, 0.6, color=RED if color == RED else "white", edgecolor=RED if color == RED else K, lw=0.6)
    ax.text(n_ok + 0.2, i, f"{n_ok}/{len(rs)}", va="center", fontsize=6.3, color=RED if color == RED else K)
ax.set(yticks=range(len(METHODS)), ylim=(len(METHODS) - 0.5, -0.5), xlim=(0, 11.5), xticks=[0, 5, 10],
       xlabel="runs that never broke (of 10)")
ax.set_yticklabels([m[0].replace("\n", " ") for m in METHODS], fontsize=6.5)
assert kept_all == {(2, 1)}, kept_all
ax.text(0, len(METHODS) + 0.35, "every run of every method kept 2 + 1 heads (minimal: 1 + 1)",
        fontsize=6, color=G1, va="top", transform=ax.transData)
title(ax, "c", "Stays solved while closing heads")

# ---------------------------------------------------------------- d  loss during training, one seed
ax = fig.add_subplot(bot[1])
for label, kw, color in (("dense (8 + 8)", dict(rule="dense", steps=6000), G2),
                         ("Michel et al. 2019", METHODS[1][1], K),
                         ("ours", METHODS[0][1], RED)):
    r = pick(**kw)[0]
    ax.plot(r["hist"]["val_step"], r["hist"]["val_loss"], color=color, lw=0.9, label=label)
r = pick(**METHODS[0][1])[0]
ax.axhline(r["bar"], color=G2, lw=0.6, ls=":")
ax.text(5950, r["bar"] * 1.25, "solved", ha="right", va="bottom", fontsize=5.5, color=G1)
ax.set(yscale="log", ylim=(3e-4, 8), xlim=(0, 6000), xticks=[0, 2000, 4000, 6000],
       xticklabels=["0", "2k", "4k", "6k"], xlabel="training step", ylabel="loss")
ax.minorticks_off()
ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0, fontsize=6)
title(ax, "d", "What breaking looks like (seed 0)")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_induction.{ext}"), bbox_inches="tight", facecolor="white")
for name, kw, _, _ in METHODS:
    rs = pick(**kw)
    print(name.replace("\n", " "), len(rs), "runs; kept", sorted((int(r["kept"][:8].sum()), int(r["kept"][8:].sum()))
          for r in rs), "never broke", sum(never_broke(r) for r in rs), "ended solved",
          sum(r["final_loss"] <= r["bar"] for r in rs))
print("wrote figures/fig_induction.png and .pdf")
