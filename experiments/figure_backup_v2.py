"""Backup heads, draft v2: (a) failure tree, (b) prediction vs all damaged runs (results/confirm and
the pre-registered results/backup), (c) payoff under test-time damage (backup_robust.py).

Old docstring of v1 follows.

  (a) Is one more head worth its upkeep? Each head fails with probability p. An extra head
      earns anything only on one branch of the tree: it works itself, and fewer than R of
      the other heads work, so it supplies the missing answer, worth L_triv / R. Its expected
      value is the product along that branch, and the rule keeps it if that beats the upkeep.
      Worked for k* = 4, p = 0.2, upkeep 0.03 L_triv: head 6 is kept, head 7 is not.
  (b) That value for heads 1 to 9 against the upkeep, which gives the predicted count.
  (c) Heads kept in every damaged run against the count predicted from the tree before the
      runs (results/confirm/predictions.md).

  python experiments/figure_backup.py  ->  paper/neuroai/fig_backup.pdf (and .png)
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED, PINK = "#b2182b", "#e8a5a5"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text="", pad=4: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=pad)


def p_short(n, R, p):
    """P(fewer than R of n other heads work)."""
    q = 1 - p
    return sum(comb(n, a) * q ** a * p ** (n - a) for a in range(0, min(R - 1, n) + 1))


def value(B, R, p):
    return (1 - p) * p_short(B - 1, R, p) / R


def predict(R, p, price):
    return max(B for B in range(1, 33) if value(B, R, p) >= price)


# ---------------------------------------------------------------- runs
def load(d):
    out = []
    for f in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(f, "rb"))
        if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
            out.append(r)
    return out


DEFAULT = Cfg()
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
RULE = dict(task="multi_relation", supply="local", taper=100, budget_hold_frac=0.1, conserve=0,
            local_value="refit", oneshot=0, trial=0, plant_copies=0, probe_masks=32, steps=4000,
            phases="", sprout=0)


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


RUNS = [r for r in load("results/confirm") + load("results/backup") if all(get(r, k) == v for k, v in RULE.items())
        and get(r, "head_dropout") > 0]
pts = [(predict(get(r, "n_rel"), get(r, "head_dropout"), get(r, "price_frac")), kept(r),
        get(r, "n_rel"), get(r, "price_frac")) for r in RUNS]
exact = sum(int(m == pr) for pr, m, _, _ in pts)
print(f"{len(pts)} damaged runs, {exact} exact, misses:",
      sorted(m - pr for pr, m, _, _ in pts if m != pr))

fig = plt.figure(figsize=(7.0, 1.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 0.95, 1.05], wspace=0.42)

# ---------------------------------------------------------------- a  the tree
R, p, price = 4, 0.2, 0.03
ax = fig.add_subplot(gs[0])
ax.axis("off")
ax.set(xlim=(0, 10), ylim=(-1.6, 6))
title(ax, "a", "Is one more head worth its upkeep?")
ax.text(0.0, 5.75, f"task needs 4 heads; each head fails with probability {p:g}", fontsize=5.8, color=G1, va="center")


def node(x, y, text, color=K, edge=G2):
    ax.text(x, y, text, ha="center", va="center", fontsize=6, color=color,
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.2", fc="white", ec=edge, lw=0.6))


def line(x0, y0, x1, y1, color=G2, lw=0.7):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=0)


def prob(x, y, t, color=G1):
    ax.text(x, y, t, ha="center", va="center", fontsize=5.8, color=color,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


node(0.9, 3.0, "head $B$")
node(3.6, 4.9, "fails", color=G1)
node(3.6, 1.6, "works", color=RED, edge=RED)
node(6.5, 3.0, "$\\geq 4$ of the other\n$B{-}1$ work", color=G1)
node(6.5, 0.2, "$< 4$ of the other\n$B{-}1$ work", color=RED, edge=RED)
line(1.45, 3.25, 3.15, 4.8); prob(2.2, 4.25, f"{p:g}")
line(1.45, 2.75, 3.1, 1.7, RED, 1.1); prob(2.2, 2.0, f"{1 - p:g}", RED)
line(4.1, 1.8, 5.45, 2.85)
line(4.1, 1.4, 5.45, 0.35, RED, 1.1)
ax.text(4.2, 4.9, "worth 0", va="center", fontsize=5.8, color=G1)
ax.text(8.2, 3.0, "not needed:\nworth 0", va="center", fontsize=5.8, color=G1)
ax.text(8.2, 0.2, "fills the gap:\nworth $L_{\\mathrm{triv}}/4$", va="center", fontsize=5.8, color=RED)
ax.text(0.0, -1.35, "value of head $B$ = 0.8 × P(fewer than 4 of the other $B{-}1$ work) × $L_{\\mathrm{triv}}/4$",
        fontsize=5.8, color=K, va="center")

# ---------------------------------------------------------------- b  predicted vs measured, bubbles
from collections import Counter
ax = fig.add_subplot(gs[1])
lo, hi = 1.5, 13.5
ax.plot([lo, hi], [lo, hi], color=G2, lw=0.6, ls="--", zorder=0)
cnt = Counter((pr, m) for pr, m, _, _ in pts)
for (pr, m), n in cnt.items():
    ax.scatter(pr, m, s=4 + 3.2 * n, color=RED if pr == m else K, alpha=0.85 if pr == m else 0.6, lw=0)
ax.set(xlim=(lo, hi), ylim=(lo, hi), xticks=[3, 6, 9, 12], yticks=[3, 6, 9, 12],
       xlabel="predicted heads", ylabel="heads kept")
ax.set_aspect("equal")
within = sum(abs(m - pr) <= 1 for pr, m, _, _ in pts)
ax.text(0.03, 0.97, f"{exact}/{len(pts)} exact\n{within}/{len(pts)} within 1", transform=ax.transAxes,
        va="top", fontsize=5.6, color=RED)
ax.text(0.97, 0.03, "k* = 2, 4, 8", transform=ax.transAxes, ha="right", fontsize=5.2, color=G1)
title(ax, "b", "Prediction vs. runs")

# ---------------------------------------------------------------- c  do the backups pay off
ax = fig.add_subplot(gs[2])
arms = (("aware", RED, "keeps backups (6)"), ("blind", G1, "no backups (4), trained w/ failure"),
        ("lean", G2, "no backups (4), never failed"))
q, w = 0.2, 0.26
for i, (arm, c, lab) in enumerate(arms):
    rs = [pickle.load(open(f, "rb")) for f in glob.glob(os.path.join(ROOT, "results", "backup", f"robust_{arm}_s[0-9].pkl"))]
    for j, kind in enumerate(("refit", "fixed")):
        v = [r["damage"][q][kind].mean() for r in rs]
        x = j + (i - 1) * w
        ax.bar(x, np.mean(v), width=w * 0.9, color=c, lw=0, label=lab if j == 0 else None)
        ax.errorbar(x, np.mean(v), yerr=np.std(v), color=K, lw=0.6, capsize=1.2)
        ax.text(x, np.mean(v) + np.std(v) + 0.01, f"{np.mean(v):.2f}".lstrip("0"), ha="center", va="bottom", fontsize=5)
ax.set(xlim=(-0.5, 1.5), ylim=(0, 0.62), xticks=[0, 1], yticks=[0, 0.2, 0.4],
       ylabel="loss when 20% fail ($L_{\\mathrm{triv}}$)")
ax.set_xticklabels(["re-routes to\nsurvivors", "as trained"], fontsize=5.8)
ax.legend(loc="upper left", fontsize=5.0, handlelength=0.8, labelspacing=0.2, borderaxespad=0)
title(ax, "c", "Do backups pay off?")

out = os.path.join(ROOT, "paper", "neuroai", "fig_backup_v2")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out + ".pdf")
