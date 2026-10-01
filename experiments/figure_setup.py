"""Figure 1 (setup): the task, the model, one head one equation, and what a solved model
looks like (attention of the heads the collateral rule kept, and predicted against true
outputs on held-out data).

Trains one collateral-rule model (k* = 4, seed 0, the settings of the confirmation runs)
and caches its weights in figures/solved_model.pt; later runs reuse the cache.

  python experiments/figure_setup.py   ->  figures/fig_setup.png and figures/fig_setup.pdf
"""
import os, pickle, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataclasses import replace
from hemo.config import Cfg
from hemo.model import HemoAttn
from hemo.tasks import make_val, trivial_loss, offsets_for
from hemo.train import train, pick_device, evaluate

RED = "#b2182b"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text="": ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)

# ---------------------------------------------------------------- the solved model
cfg = replace(Cfg(), supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0,
              n_rel=4, seed=0)
dev = pick_device(cfg)
val = make_val(cfg, dev)
CACHE = os.path.join(ROOT, "figures", "solved_model.pt")
if os.path.exists(CACHE):
    model = HemoAttn(cfg).to(dev)
    model.load_state_dict(torch.load(CACHE, map_location=dev))
else:
    model, hist = train(cfg, val, dev, hemo=True)
    torch.save(model.state_dict(), CACHE)
model.eval()
gate = model.gate().detach()
kept = torch.nonzero(gate > 0).flatten().tolist()
N, offs = cfg.seq_len, offsets_for(cfg)
with torch.no_grad():
    X, Y, T, aux = val[0], val[1], val[2], val[3]
    _, attn, _ = model.heads(X, Y)                                   # (batch, head, query, slot)
    p = aux["p"]
    maps = torch.stack([torch.stack([attn[:, h][p == q].mean(0) for q in range(N)]) for h in kept]).cpu().numpy()
    pred = model(X, Y, gate=gate)[0]
    off = gate.clone()
    off[kept[0]] = 0
    pred_off = model(X, Y, gate=off)[0]
loss_on = evaluate(model, val, gate=gate)
loss_off = evaluate(model, val, gate=off)
print(f"heads kept {kept}; loss {loss_on:.2e}; with head {kept[0]} switched off {loss_off:.3f}; "
      f"trivial {trivial_loss(val):.3f}")

fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 4, height_ratios=[0.85, 1], hspace=0.35, wspace=0.55)

# ---------------------------------------------------------------- a  task
ax = fig.add_subplot(gs[0, :2])
title(ax, "a", "Task")
ax.axis("off")
pq = 8
ax.set(xlim=(-4.2, 16), ylim=(-2.9, 2.4))
ax.add_patch(Rectangle((pq - 0.42, 1.45), 0.84, 0.72, facecolor="white", edgecolor=K, lw=0.7))
ax.text(pq, 1.81, "$p$", ha="center", va="center")
ax.text(-4.1, 1.81, "query", color=G1, va="center")
ax.text(-4.1, 0, "memory", color=G1, va="center")
targets = [(pq + d) % N for d in offs]
for j in range(N):
    hit = j in targets
    ax.add_patch(Rectangle((j - 0.42, -0.36), 0.84, 0.72, facecolor=G3 if hit else "white",
                           edgecolor=K if hit else G2, lw=0.7))
    ax.text(j, 0, str(j), ha="center", va="center", fontsize=5.5, color=K if hit else G1)
for j in targets:
    ax.annotate("", xy=(j, 0.42), xytext=(pq, 1.42),
                arrowprops=dict(arrowstyle="-|>", color=K, lw=0.6, mutation_scale=6, shrinkA=0, shrinkB=0))
ax.text(-4.1, -1.5, "target", color=G1, va="center")
ax.text(7.5, -1.5, r"$y=(c_{p+1},\ c_{p+5},\ c_{p+9},\ c_{p+13})$", ha="center", va="center", fontsize=7.5)
ax.text(7.5, -2.5, r"needs $k^*=4$ heads", ha="center", va="center", color=G1)

# ---------------------------------------------------------------- b  model
ax = fig.add_subplot(gs[0, 2:])
title(ax, "b", "Model: one attention layer, 32 heads, each with a supply")
ax.axis("off")
ax.set(xlim=(-1.6, 10), ylim=(0, 6.3))


def box(x, y, w, h, text, face="white", edge=K, lw=0.7, ls="-", color=K, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=K, lw=0.6, mutation_scale=6, shrinkA=0, shrinkB=0))


box(0.9, 0.2, 2.6, 0.7, "query $X$")
box(6.5, 0.2, 2.6, 0.7, "memory $Y$")
xs = [0.55 + 1.05 * i for i in range(9)]
alive = {0, 2, 3, 7}                                  # drawn pattern: most heads starved, a few fed
for i, x in enumerate(xs):
    if i == 5:
        ax.text(x + 0.35, 2.55, r"$\cdots$", ha="center", va="center", fontsize=8, color=G1)
        continue
    on = i in alive
    box(x, 2.15, 0.7, 0.8, "", edge=K if on else G2, ls="-" if on else (0, (2, 1.5)))
    ax.text(x + 0.35, 2.55, "head", ha="center", va="center", fontsize=5, color=K if on else G2)
    ax.add_patch(Rectangle((x + 0.2, 3.3), 0.3, 0.9, facecolor=RED if on else "white",
                           edgecolor=RED if on else G2, lw=0.5))
    ax.plot([x + 0.35, x + 0.35], [2.95, 3.3], color=K if on else G2, lw=0.5)
    ax.plot([x + 0.35, x + 0.35], [4.2, 4.8], color=K if on else G2, lw=0.5)
    ax.plot([x + 0.35, x + 0.35], [1.75, 2.15], color=G1, lw=0.5)
ax.plot([xs[0] + 0.35, xs[-1] + 0.35], [1.75, 1.75], color=G1, lw=0.5)
arrow(2.2, 0.95, 2.2, 1.75)
arrow(7.8, 0.95, 7.8, 1.75)
ax.text(-1.55, 3.75, "supply\n$g_h$", ha="left", va="center", fontsize=6.5, color=RED)
box(0.45, 4.8, 9.05, 0.65, r"$y=W_O\ \sum_h\, g_h\,\mathrm{softmax}(Q_hK_h^{\top})\,V_h$")
arrow(5.0, 5.5, 5.0, 6.1)
ax.text(5.15, 5.9, "4 answers", ha="left", va="center", fontsize=6.5, color=G1)

# ---------------------------------------------------------------- c  one head, one equation
ax = fig.add_subplot(gs[1, 0])
E1 = pickle.load(open(os.path.join(ROOT, "results", "proposal", "equations.pkl"), "rb"))
triv = E1["trivial"]
ax.plot([0, 4], [triv, 0], color=G2, lw=0.8, ls="--", zorder=1, label="predicted")
plain = [r for r in E1["rows"] if len(set(r["heads"])) == len(r["heads"])]
copies = [r for r in E1["rows"] if len(set(r["heads"])) < len(r["heads"])]
ax.plot([r["different"] for r in plain], [r["loss"] for r in plain], "o", color=K, ms=3, label="heads removed")
ax.plot([r["different"] for r in copies], [r["loss"] for r in copies], "o", mfc="none", mec=RED,
        ms=6.5, mew=0.8, label="one head copied")
ax.set(xlabel="different heads", ylabel="loss", xticks=range(5), xlim=(-0.2, 4.3), ylim=(-0.05, 1.08))
ax.legend(loc="upper right", handletextpad=0.3, borderaxespad=0)
title(ax, "c", "One head, one equation")

# ---------------------------------------------------------------- d  attention of the kept heads
sub = gs[1, 1:3].subgridspec(1, len(kept), wspace=0.12)
vmax = maps.max()
for i, h in enumerate(kept):
    a = fig.add_subplot(sub[i])
    a.imshow(maps[i], cmap="Reds", vmin=0, vmax=vmax, interpolation="nearest")
    a.set_xticks([0, 15]); a.set_yticks([0, 15])
    a.tick_params(length=1.5, labelsize=5.5)
    for s in a.spines.values():
        s.set_visible(True); s.set_color(G2); s.set_linewidth(0.5)
    if i == 0:
        title(a, "d", "Where the kept heads look")
        a.set_ylabel("query position $p$", fontsize=6.5)
    else:
        a.set_yticklabels([])
    a.set_xlabel(f"slot (head {h})", fontsize=6)

# ---------------------------------------------------------------- e  predicted against true
ax = fig.add_subplot(gs[1, 3])
rng = np.random.default_rng(0)
idx = rng.choice(T.numel(), 2500, replace=False)
t = T.flatten()[idx].cpu().numpy()
ax.plot(t, pred_off.flatten()[idx].cpu().numpy(), ".", color=G2, ms=1.2, alpha=0.6)
ax.plot(t, pred.flatten()[idx].cpu().numpy(), ".", color=RED, ms=1.2, alpha=0.8)
ax.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[-2, 0, 2], yticks=[-2, 0, 2], xlabel="true output",
       ylabel="model output")
ax.text(-2.85, 2.75, "4 kept heads", color=RED, fontsize=6.5, va="top")
ax.text(2.85, -2.75, "one switched off", color=G1, fontsize=6.5, ha="right", va="bottom")
title(ax, "e", "Solved")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_setup.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_setup.png and figures/fig_setup.pdf")
