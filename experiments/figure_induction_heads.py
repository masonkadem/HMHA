"""The three heads hemodynamic attenuation kept on induction, and why it keeps two in layer 1.

  a  What each surviving head does, on one sequence, with the real tokens on the axes. The outlined
     cell is where the circuit says the head should look: the previous token (layer 1), or the token
     that followed the earlier copy of the current token (layer 2).
  b  The rule's own decision: the loss it predicts without each layer-1 head (other heads re-fit),
     against the limit a closure must stay under. Both are far above it, so neither can close.
  c  Why: how often the previous token is missed (attention to it below 0.4) over the first copy,
     with only one of the two heads, and with both.

Uses the cached seed-0 model of figure_induction.py (figures/induction_model.pt); trains it if absent.

  python experiments/figure_induction_heads.py   ->  figures/fig_induction_heads.png and .pdf
"""
import os, sys
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import ICfg, InductionNet, train, make_batch, probe_values_logit

RED = "#b2182b"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2, "ytick.major.size": 2,
    "savefig.dpi": 400, "pdf.fonttype": 42})
CMAP = LinearSegmentedColormap.from_list("attn", ["#ffffff", RED])

SAME = dict(probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0)
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
kept = blob["kept"].bool()
gate = kept.float()
h1 = model.sizes[0]
L1 = [i for i in range(h1) if kept[i]]
L2 = [i - h1 for i in range(h1, model.H) if kept[i]]

# ---------------------------------------------------------------- one example
gen = torch.Generator().manual_seed(7)
for _ in range(500):
    tok, starts = make_batch(1, cfg, gen)
    p0, p1 = int(starts[0, 0]), int(starts[0, 1])
    if p0 >= 1 and 3 <= p1 - p0 - cfg.L <= 5:
        break
with torch.no_grad():
    _, att = model.residual(tok, gate, keep_attn=True)
seq = tok[0].tolist()
c1 = list(range(p0, p0 + cfg.L))                      # first copy
c2 = list(range(p1, p1 + cfg.L - 1))                  # second copy, where induction predicts

# ---------------------------------------------------------------- b  the rule's own numbers
ptok, psec = make_batch(1024, cfg, torch.Generator().manual_seed(11))
value, _, e_triv, e_now = probe_values_logit(model, ptok, psec, cfg, kept)
limit = cfg.price_frac * e_triv                       # the closing limit in "bar" mode
without = {h: e_now + value[h] for h in L1}

# ---------------------------------------------------------------- c  misses with one head and with both
big, st = make_batch(2048, cfg, torch.Generator().manual_seed(5))
with torch.no_grad():
    _, A_ = model.residual(big, gate, keep_attn=True)
b = torch.arange(len(big)).repeat_interleave(cfg.L - 1)
t = (st[:, :1] + torch.arange(1, cfg.L)).reshape(-1)
weak = {h: (A_[0][b, h, t, t - 1] < 0.4).numpy() for h in L1}
miss = [weak[L1[0]].mean(), weak[L1[1]].mean(), (weak[L1[0]] & weak[L1[1]]).mean()]
indep = weak[L1[0]].mean() * weak[L1[1]].mean()
print(f"kept L1 {L1} L2 {L2}; loss now {e_now:.4f}, without heads {[round(v, 3) for v in without.values()]}, "
      f"limit {limit:.3f}; misses {[round(m, 4) for m in miss]}, if independent {indep:.4f}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.45, wspace=0.42)


def grid(ax, rows, cols, A, target, title, ylabel, xlabel):
    sub = A[np.ix_(rows, cols)]
    ax.imshow(sub, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    for i, r in enumerate(rows):
        j = cols.index(target(r)) if target(r) in cols else None
        if j is not None:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=K, lw=0.9))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([seq[c] for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([seq[r] for r in rows])
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(G2); s.set_linewidth(0.5)
    ax.set_title(title, pad=3)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xlabel(xlabel, labelpad=2)


rows1 = c1[1:]
cols1 = c1
for i, h in enumerate(L1):
    grid(fig.add_subplot(gs[0, i]), rows1, cols1, att[0][0, h].numpy(), lambda r: r - 1,
         f"layer 1, head {h}: previous-token", "token in copy 1 (looks from)" if i == 0 else "",
         "token it looks at")
gap = p1 - p0
grid(fig.add_subplot(gs[0, 2]), c2, c1, att[1][0, L2[0]].numpy(), lambda r: r - gap + 1,
     f"layer 2, head {L2[0]}: induction", "token in copy 2 (looks from)", "token in copy 1 it looks at")
fig.text(0.07, 0.965, "a", fontsize=8, fontweight="bold")
fig.text(0.095, 0.965, "What the three surviving heads do (one sequence; outlined: where the circuit should look)",
         fontsize=7)

# b
ax = fig.add_subplot(gs[1, 0:2])
names = ["all three\nheads", f"without\nhead {L1[0]}", f"without\nhead {L1[1]}"]
vals = [e_now] + [without[h] for h in L1]
ax.bar(range(3), vals, width=0.55, color=[G2, RED, RED], edgecolor="white", linewidth=1.0)
for x, v in enumerate(vals):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.axhline(limit, color=K, lw=0.8, ls="--")
ax.text(-0.45, limit + 0.02, f"limit: a head may close only if the loss\nwithout it stays below {limit:.2f}",
        fontsize=6, ha="left", va="bottom", color=K)
ax.set(xticks=range(3), ylabel="loss after the read-out re-fits (nats)", xlim=(-0.5, 2.5),
       ylim=(0, max(vals) * 1.25))
ax.set_xticklabels(names)
ax.tick_params(axis="x", length=0)
ax.set_title("The rule's decision: removing either layer-1 head costs too much", pad=3)
fig.text(0.07, 0.49, "b", fontsize=8, fontweight="bold")

# c
ax = fig.add_subplot(gs[1, 2])
names = [f"head {L1[0]}\nalone", f"head {L1[1]}\nalone", "both", "both, if\nindep."]
ax.bar(range(3), [100 * m for m in miss], width=0.6, color=[G1, G1, RED], edgecolor="white", linewidth=1.0)
ax.bar([3], [100 * indep], width=0.6, facecolor="white", edgecolor=K, linewidth=0.7, ls="--")
for x, m in enumerate(miss + [indep]):
    ax.text(x, 100 * m + 0.4, f"{100 * m:.1f}%", ha="center", va="bottom", fontsize=6.5)
ax.set(xticks=range(4), ylabel="positions where the previous\ntoken is missed (%)", ylim=(0, 100 * max(miss) * 1.3))
ax.set_xticklabels(names)
ax.tick_params(axis="x", length=0)
ax.set_title("Why: two heads back each other up", pad=3)
fig.text(0.665, 0.49, "c", fontsize=8, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_induction_heads.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_induction_heads.png and .pdf")
