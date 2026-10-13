"""Two previous-token heads back each other up: a simple picture.

  a  Miss map: 300 positions from the first copy of many sequences, in a row. A tick marks each
     position where a layer-1 head misses the previous token (attention to it below 0.4). Top row:
     one head; middle: the other; bottom: both at once. The two heads rarely miss the same position.
  b  The same in numbers, for every seed checked: how often one head alone misses, how often both
     would miss together if their misses were unrelated, and how often both actually miss together.

Panel a uses the cached seed-0 model of figure_induction.py; panel b reads results/induction/team_s*.pkl
(experiments/induction_team_check.py). One head's miss rate is taken as only_one / 2 + both, which
assumes the two heads miss about equally often (their mean attention to the previous token agrees to
within 0.01 in every seed).

  python experiments/figure_induction_backup.py   ->  figures/fig_induction_backup.png and .pdf
                                                      figures/fig_induction_backup_map.png and .pdf (panel a alone)
"""
import glob, os, pickle, sys
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import ICfg, InductionNet, make_batch

RED = "#b2182b"
K, G1, G2 = "#111111", "#6b6b6b", "#a8a8a8"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.dpi": 400, "pdf.fonttype": 42})

# ---------------------------------------------------------------- a  the miss map, cached seed-0 model
SAME = dict(probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0)
cfg = replace(ICfg(), rule="collateral", seed=0, **SAME)
blob = torch.load(os.path.join(ROOT, "figures", "induction_model.pt"), map_location="cpu")
model = InductionNet(cfg)
model.load_state_dict(blob["state"])
model.eval()
kept = blob["kept"].bool()
L1 = [i for i in range(model.sizes[0]) if kept[i]]
tok, st = make_batch(2048, cfg, torch.Generator().manual_seed(5))
with torch.no_grad():
    _, att = model.residual(tok, kept.float(), keep_attn=True)
b = torch.arange(len(tok)).repeat_interleave(cfg.L - 1)
t = (st[:, :1] + torch.arange(1, cfg.L)).reshape(-1)
miss_a = (att[0][b, L1[0], t, t - 1] < 0.4).numpy()
miss_b = (att[0][b, L1[1], t, t - 1] < 0.4).numpy()
N = 300
ma, mb = miss_a[:N], miss_b[:N]

# ---------------------------------------------------------------- b  every seed checked
seeds = []
for p in sorted(glob.glob(os.path.join(ROOT, "results", "induction", "team_s*.pkl"))):
    r = pickle.load(open(p, "rb"))
    if "both" in r:
        one = r["only_one"] / 2 + r["both"]
        seeds.append((r["seed"], one, one * one, r["both"]))
seeds.sort()

fig = plt.figure(figsize=(7.2, 2.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.25)

def miss_map(ax):
    shared = ma & mb                                   # positions both heads miss: red in every row
    rows = [(2, ma, f"head {L1[0]} misses"), (1, mb, f"head {L1[1]} misses"), (0, shared, "both miss")]
    for y, m, name in rows:
        alone, both = np.nonzero(m & ~shared)[0], np.nonzero(m & shared)[0]
        ax.vlines(alone, y - 0.32, y + 0.32, color=G2, lw=1.4)
        ax.vlines(both, y - 0.32, y + 0.32, color=RED, lw=1.8)
        ax.text(-6, y, name, ha="right", va="center", color=RED if y == 0 else K)
        ax.text(N + 6, y, f"{int(m.sum())}", ha="left", va="center", color=RED if y == 0 else K)
    ax.axhline(0.5, color=G2, lw=0.5, ls=":")
    ax.set(xlim=(0, N), ylim=(-0.6, 2.6), yticks=[], xticks=[])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f"first {N} of {len(miss_a):,} positions checked (seed 0)", color=G1)
    ax.set_title("Where each head misses the previous token", pad=4)


# panel a on its own, for a slide
solo = plt.figure(figsize=(5.0, 2.3))
miss_map(solo.add_subplot(1, 1, 1))
for ext in ("png", "pdf"):
    solo.savefig(os.path.join(ROOT, "figures", f"fig_induction_backup_map.{ext}"), bbox_inches="tight",
                 facecolor="white")
plt.close(solo)

miss_map(fig.add_subplot(gs[0, 0]))
fig.text(0.04, 0.96, "a", fontsize=8.5, fontweight="bold")

ax = fig.add_subplot(gs[0, 1])
w = 0.26
for i, (s, one, indep, both) in enumerate(seeds):
    ax.bar(i - w, 100 * one, w, color=G1, edgecolor="white", lw=0.6, label="one head alone" if i == 0 else None)
    ax.bar(i, 100 * indep, w, facecolor="white", edgecolor=K, lw=0.7, ls="--",
           label="both, if unrelated" if i == 0 else None)
    ax.bar(i + w, 100 * both, w, color=RED, edgecolor="white", lw=0.6, label="both, measured" if i == 0 else None)
    ax.text(i + w, 100 * both + 0.4, f"{100 * both:.1f}", ha="center", va="bottom", fontsize=6.5, color=RED)
    ax.text(i, 100 * indep + 0.4, f"{100 * indep:.1f}", ha="center", va="bottom", fontsize=6.5, color=K)
ax.set(xticks=range(len(seeds)), ylabel="positions missed (%)", ylim=(0, 23), yticks=[0, 5, 10, 15])
ax.set_xticklabels([f"seed {s}" for s, *_ in seeds])
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.2)
ax.set_title("Every run", pad=4)
fig.text(0.635, 0.96, "b", fontsize=8.5, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_induction_backup.{ext}"), bbox_inches="tight", facecolor="white")
print("seeds:", [(s, round(100 * o, 1), round(100 * i, 2), round(100 * b_, 2)) for s, o, i, b_ in seeds])
print("wrote figures/fig_induction_backup.png and .pdf")
