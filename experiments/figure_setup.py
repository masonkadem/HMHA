"""Figure 1 (setup): the task, the model, one head one equation, and what a solved model
looks like (attention of the heads hemodynamic attenuation kept, and which memory slot each of
the model's answers points to, with all kept heads and with one switched off).

Trains one hemodynamic-attenuation model (k* = 4, seed 0, the settings of the confirmation runs)
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
from hemo.tasks import make_val, make_batch, trivial_loss, offsets_for
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
title = lambda ax, letter, text="", pad=4: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=pad)

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
M, R = cfg.m_content, cfg.n_rel


def decode(X, Y, g):
    """For every query and every answer block, the memory slot whose item is nearest."""
    with torch.no_grad():
        pr = model(X, Y, gate=g)[0].view(X.size(0), N, R, M)              # (batch, query, block, m)
    items = Y[:, :, N:N + M]                                              # (batch, slot, m)
    dist = ((pr[:, :, :, None, :] - items[:, None, None, :, :]) ** 2).sum(-1)
    return dist.argmin(-1)                                                # (batch, query, block)


true_val = torch.stack([(p + o) % N for o in offs], -1)
acc_on = (decode(X, Y, gate) == true_val).float().mean().item()
acc_off = (decode(X, Y, off) == true_val).float().mean().item()
# one fresh example whose 16 query tokens ask for p = 0, 1, ..., 15 in order
Xg, Yg, _, _ = make_batch(1, cfg, "cpu", gen=torch.Generator().manual_seed(1))
Xg[:, :, :N] = torch.eye(N)
Xg, Yg = Xg.to(dev), Yg.to(dev)
grid_true = [[(q + o) % N for o in offs] for q in range(N)]
grid_on = decode(Xg, Yg, gate)[0].cpu().numpy()
grid_off = decode(Xg, Yg, off)[0].cpu().numpy()
print(f"answers pointing to the right slot: {acc_on:.4f} with all kept heads, {acc_off:.4f} "
      f"with head {kept[0]} off, over {true_val.numel()} held-out answers")
loss_on = evaluate(model, val, gate=gate)
loss_off = evaluate(model, val, gate=off)
print(f"heads kept {kept}; loss {loss_on:.2e}; with head {kept[0]} switched off {loss_off:.3f}; "
      f"trivial {trivial_loss(val):.3f}")

fig = plt.figure(figsize=(7.2, 4.7))
outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.15], hspace=0.32)
gs = outer[0].subgridspec(1, 4, wspace=0.55)
bot = outer[1].subgridspec(1, 3, width_ratios=[1.25, 1.75, 3.2], wspace=0.38)

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
ax.set(xlim=(-1.6, 10), ylim=(0, 6.6))


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
box(0.45, 4.8, 9.05, 0.9, r"$y = W_O\,\Sigma_h\; g_h\, \mathrm{softmax}(Q_h K_h^{\mathsf{T}})\, V_h$", fs=7)
arrow(5.0, 5.75, 5.0, 6.35)
ax.text(5.15, 6.1, "4 answers", ha="left", va="center", fontsize=6.5, color=G1)

# ---------------------------------------------------------------- c  one head, one equation
ax = fig.add_subplot(bot[0])
E1 = pickle.load(open(os.path.join(ROOT, "results", "proposal", "equations.pkl"), "rb"))
triv = E1["trivial"]
ax.plot([0, 4], [triv, 0], color=G2, lw=0.8, ls="--", zorder=1, label="predicted")
plain = [r for r in E1["rows"] if len(set(r["heads"])) == len(r["heads"])]
copies = [r for r in E1["rows"] if len(set(r["heads"])) < len(r["heads"])]
ax.plot([r["different"] for r in plain], [r["loss"] for r in plain], "o", color=K, ms=3, label="heads removed")
ax.plot([r["different"] for r in copies], [r["loss"] for r in copies], "o", mfc="none", mec=RED,
        ms=6.5, mew=0.8, label="one head copied")
ax.set(xlabel="different heads", ylabel="loss", xticks=range(5), xlim=(-0.2, 4.3), ylim=(-0.05, 1.45),
       yticks=[0, 0.5, 1.0])
ax.legend(loc="upper right", handletextpad=0.3, borderaxespad=0, labelspacing=0.3)
title(ax, "c", "One head, one equation")

# ---------------------------------------------------------------- d  attention of the kept heads, 2 x 2
sub = bot[1].subgridspec(2, 2, wspace=0.12, hspace=0.32)
vmax = maps.max()
for i, h in enumerate(kept):
    a = fig.add_subplot(sub[i // 2, i % 2])
    a.imshow(maps[i], cmap="Reds", vmin=0, vmax=vmax, interpolation="nearest")
    a.set_xticks([0, 15]); a.set_yticks([0, 15])
    a.tick_params(length=1.5, labelsize=5.5, pad=1)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color(G2); sp.set_linewidth(0.5)
    a.text(0.5, 1.03, f"head {h}", transform=a.transAxes, ha="center", va="bottom", fontsize=6, color=G1)
    if i == 0:
        title(a, "d", "Where the kept heads look", pad=11)
    if i % 2 == 0:
        a.set_ylabel("query $p$", fontsize=6.5, labelpad=1)
    else:
        a.set_yticklabels([])
    if i // 2 == 1:
        a.set_xlabel("slot", fontsize=6.5, labelpad=1)
    else:
        a.set_xticklabels([])

# ---------------------------------------------------------------- e  which slot each answer points to
sub = bot[2].subgridspec(1, 2, wspace=0.14)
GRID = "#c8c8c8"
for i, (grid, name, acc) in enumerate(((grid_on, "4 kept heads", acc_on),
                                       (grid_off, f"head {kept[0]} switched off", acc_off))):
    a = fig.add_subplot(sub[i])
    a.set(xlim=(-0.5, N - 0.5), ylim=(N - 0.5, -0.5), aspect="equal")
    for q in range(N):
        for j in grid_true[q]:                                       # the right slots: grey squares
            a.add_patch(Rectangle((j - 0.5, q - 0.5), 1, 1, facecolor=GRID, edgecolor="none"))
        for r in range(R):                                           # the model's answers: dots or crosses
            j = grid[q, r]
            if j == grid_true[q][r]:
                a.plot(j, q, "o", color=RED, ms=2.6, mew=0)
            else:
                a.plot(j, q, "x", color=K, ms=3.2, mew=0.7)
    a.set_xticks([0, 5, 10, 15]); a.set_yticks([0, 5, 10, 15])
    a.tick_params(length=1.5, labelsize=5.5, pad=1)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color(G2); sp.set_linewidth(0.5)
    a.text(0.5, 1.03, name, transform=a.transAxes, ha="center", va="bottom", fontsize=6.5)
    a.set_xlabel(f"slot\n{100 * acc:.0f}% of answers right", fontsize=6.5, labelpad=1)
    if i == 0:
        title(a, "e", "Which slot each answer points to", pad=13)
        a.set_ylabel("query position $p$", fontsize=6.5, labelpad=1)
    else:
        a.set_yticklabels([])
a.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=GRID, edgecolor="none"),
                  plt.Line2D([], [], ls="none", marker="o", color=RED, ms=3, mew=0),
                  plt.Line2D([], [], ls="none", marker="x", color=K, ms=3.5, mew=0.7)],
         labels=["right slot", "answer, right", "answer, wrong"], loc="upper left",
         bbox_to_anchor=(1.03, 1.0), handlelength=1, handletextpad=0.4, borderaxespad=0, fontsize=6)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_setup.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_setup.png and figures/fig_setup.pdf")
