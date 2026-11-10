"""Overview figure, 2 x 2 (for discussion, and a candidate Figure 1).

  a  The rule: each head's value against a fixed upkeep (schematic)
  b  One objective predicts the head count: every run, predicted heads vs heads kept
     (exact count, collapse above the breakpoint, backups under failure)
  c  GPT-2 small: the published IOI circuit by stage, heads kept at upkeep 0.02 filled
  d  The same value used two ways on GPT-2: our rule vs one-shot vs random, same number of heads

  python experiments/figure_overview.py  ->  paper/neuroai/fig_overview.pdf (and .png)
"""
import glob, os, pickle, sys
from collections import Counter
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED = "#a50f15"
K, G1, G2, G3 = "#1a1a1a", "#666666", "#a6a6a6", "#e6e6e6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.titlelocation": "left", "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})
title = lambda ax, letter, text: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=6)

# ---------------------------------------------------------------- runs (same selection as the paper figures)
DEFAULT = Cfg()
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))


def load(dirs):
    out, seen = [], set()
    for d in dirs:
        for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
            if os.path.basename(p) in seen:
                continue
            seen.add(os.path.basename(p))
            r = pickle.load(open(p, "rb"))
            if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
                out.append(r)
    return out


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", kappa_end=1.5,
              target_frac=0.02, delay=0, pool_beta=0.0, leak=0.0, n_territories=8,
              stall_gate=0.0, autoreg_gain=0.003, precondition=0.0, flow_exponent=4.0,
              price_frac=0.01, probe_every=25, taper=0, budget_hold_frac=0.25, prune_stop=1,
              conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, probe_masks=32,
              copy_noise=0.0, oneshot=0, trial=0, l0_lambda=0.01, phases="", sprout=0)
RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
RUNS = load(("results", "results/proposal", "results/confirm", "results/l0", "results/review",
             "results/equal", "results/instant", "results/breakpoint"))
same = {a: v for a, v in {**PINNED, **RULE}.items() if a != "price_frac"}
clean = [r for r in RUNS if all(get(r, a) == v for a, v in same.items()) and get(r, "n_rel") in (2, 4, 8)
         and (get(r, "price_frac") != 0.03 or get(r, "seed") < 10)]


def p_short(n, R, p):
    q = 1 - p
    return sum(comb(n, a) * q ** a * p ** (n - a) for a in range(0, min(R - 1, n) + 1))


def predict_backup(R, p, price):
    return max(B for B in range(1, 33) if (1 - p) * p_short(B - 1, R, p) / R >= price)


BRULE = dict(task="multi_relation", supply="local", taper=100, budget_hold_frac=0.1, conserve=0,
             local_value="refit", oneshot=0, trial=0, plant_copies=0, probe_masks=32, steps=4000,
             phases="", sprout=0)
damaged = [r for r in load(("results/confirm", "results/backup")) if all(get(r, k) == v for k, v in BRULE.items())
           and get(r, "head_dropout") > 0]

pts = []                                                   # (kind, setting, predicted, kept)
for r in clean:
    R, c = get(r, "n_rel"), get(r, "price_frac")
    pred = R if c * R < 1 else 1
    pts.append(("collapse" if pred == 1 else "count", R, pred, kept(r)))
for r in damaged:
    R, p, c = get(r, "n_rel"), get(r, "head_dropout"), get(r, "price_frac")
    pts.append(("backup", (R, p, c), predict_backup(R, p, c), kept(r)))
n_all = len(pts)
exact = sum(int(p == k) for _, _, p, k in pts)
within = sum(int(abs(p - k) <= 1) for _, _, p, k in pts)
print(f"panel b: {n_all} runs, {exact} exact, {within} within one;",
      {kind: (sum(1 for x in pts if x[0] == kind), sum(1 for x in pts if x[0] == kind and x[2] == x[3])) for kind in ("count", "collapse", "backup")})

fig = plt.figure(figsize=(7.0, 4.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1.08], wspace=0.28, hspace=0.5)

# ---------------------------------------------------------------- a  the rule (schematic)
ax = fig.add_subplot(gs[0, 0])
vals = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.004, 0.0]          # four needed heads; copies are worth ~0
names = ["1", "2", "3", "4", "1′", "2′", "5", "3′"]
upkeep = 0.06
x = np.arange(len(vals))
ax.bar(x, vals, width=0.62, color=[RED if v > upkeep else G2 for v in vals], lw=0)
for xi, v in zip(x, vals):
    if v < upkeep:
        ax.plot([xi - 0.31, xi + 0.31], [0.006, 0.006], color=G2, lw=1.2)
ax.axhline(upkeep, color=K, lw=0.7, ls=(0, (3, 2)))
ax.text(7.55, upkeep + 0.01, "upkeep", ha="right", va="bottom", fontsize=6.5)
ax.text(1.5, 0.262, "kept", ha="center", va="bottom", color=RED, fontsize=6.5)
ax.text(5.5, 0.02, "closed", ha="center", va="bottom", color=G1, fontsize=6.5)
ax.text(7.55, 0.30, r"minimises  loss $+$ upkeep $\times$ open heads", ha="right", va="top", fontsize=6.5, color=K)
ax.set(xticks=x, ylim=(0, 0.32), yticks=[0, 0.25], yticklabels=["0", r"$L_{\mathrm{triv}}/4$"],
       xlim=(-0.6, 7.6), ylabel="value of head", xlabel="head  (′ = copy of another head)")
ax.set_xticklabels(names)
ax.tick_params(axis="x", length=0)
title(ax, "a", "When is a head worth keeping?")

# ---------------------------------------------------------------- b  predicted vs kept, every run
ax = fig.add_subplot(gs[0, 1])
ax.plot([0, 13.5], [0, 13.5], color=G2, lw=0.6, zorder=0)
style = {"count": ("o", K, "task size"), "collapse": ("o", G1, "upkeep above breakpoint"),
         "backup": ("o", RED, "heads that can fail")}
for kind, (mk, c, lab) in style.items():
    groups = {}
    for kd, s, p, k in pts:
        if kd == kind:
            groups.setdefault(s, (p, []))[1].append(k)
    first = True
    for s, (p, ks) in groups.items():
        ks = np.array(ks)
        ax.plot([p, p], [ks.min(), ks.max()], color=c, lw=0.8, zorder=2)
        ax.plot(p, ks.mean(), mk, color=c, ms=3.6, mec="white", mew=0.4, zorder=3, label=lab if first else None)
        first = False
ax.set(xlim=(0, 13.5), ylim=(0, 13.5), xticks=[1, 4, 8, 12], yticks=[1, 4, 8, 12],
       xlabel="heads predicted before training", ylabel="heads kept")
ax.legend(loc="upper left", fontsize=6.3, handletextpad=0.2, borderaxespad=0, handlelength=1)
ax.text(13.3, 0.6, f"{n_all} runs: {exact} exact,\nall within one head", ha="right", va="bottom", fontsize=6.3,
        color=K, linespacing=1.35)
title(ax, "b", "Does the upkeep predict how many heads stay?")

# ---------------------------------------------------------------- c  GPT-2 circuit, kept heads at upkeep 0.02
WANG = {
    "name mover": [(9, 9), (9, 6), (10, 0)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}
role = {h: r for r, hs in WANG.items() for h in hs}
g2 = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", "ioi_pf0.02_s0.pkl"), "rb"))
keep = set(map(tuple, g2["kept"]))
ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
ax.set(xlim=(0, 11.4), ylim=(0, 6.0))
CW, CH = 0.78, 0.46


def stage(x, y, w, name, heads, per_row):
    rows = int(np.ceil(len(heads) / per_row))
    h = 0.55 + rows * (CH + 0.1)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=G2, lw=0.5))
    ax.text(x + 0.08, y + h - 0.12, name, ha="left", va="top", fontsize=5.8, color=G1)
    for i, hd in enumerate(sorted(heads)):
        r, c = divmod(i, per_row)
        x0 = x + 0.12 + c * (CW + 0.08)
        y0 = y + h - 0.5 - (r + 1) * (CH + 0.1) + 0.05
        on = hd in keep
        ax.add_patch(Rectangle((x0, y0), CW, CH, fc=RED if on else "white", ec=RED if on else G2, lw=0.5))
        ax.text(x0 + CW / 2, y0 + CH / 2, f"{hd[0]}.{hd[1]}", ha="center", va="center", fontsize=4.9,
                color="white" if on else G2)
    return h


def arrow(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=G1, lw=0.5, mutation_scale=4))


stage(0.0, 3.85, 1.95, "prev. token", WANG["previous token"], 2)
stage(0.0, 1.95, 1.95, "duplicate", WANG["duplicate token"], 2)
stage(2.55, 3.25, 1.95, "induction", WANG["induction"], 2)
stage(5.1, 2.55, 1.95, "S-inhibition", WANG["S-inhibition"], 2)
stage(7.75, 4.6, 2.85, "name movers", WANG["name mover"], 3)
stage(7.75, 1.65, 2.85, "backup movers", WANG["backup name mover"], 3)
stage(7.75, 0.2, 2.85, "negative movers", WANG["negative name mover"], 3)
arrow((1.95, 4.4), (2.55, 4.2)); arrow((1.95, 2.5), (5.1, 3.0)); arrow((4.5, 3.9), (5.1, 3.6))
arrow((7.05, 3.75), (7.75, 4.9)); arrow((7.05, 3.25), (7.75, 3.1)); arrow((7.05, 2.85), (7.75, 0.75))
outside = sorted(h for h in keep if h not in role)
ins = len(keep) - len(outside)
ax.text(0.0, 0.85, f"kept {len(keep)} of 144 heads, {ins} in the circuit\n(also kept: "
        + ", ".join(f"{l}.{h}" for l, h in outside) + ")", fontsize=6.0, color=K, va="center", linespacing=1.4)
title(ax, "c", "Does it find a known circuit in GPT-2?")

# ---------------------------------------------------------------- d  ours vs one-shot vs random
b = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", "baselines_s0.pkl"), "rb"))["sizes"]
ax = fig.add_subplot(gs[1, 1])
FR = ("0.05", "0.02", "0.01")
xs = np.arange(3); w = 0.25
series = (("our rule", [b[f]["ours"][2] for f in FR], RED),
          ("same value, used once", [b[f]["oneshot"][2] for f in FR], K),
          ("random heads", [b[f]["random"][:, 2].mean() for f in FR], G2))
for i, (lab, v, c) in enumerate(series):
    ax.bar(xs + (i - 1) * w, v, width=w * 0.88, color=c, lw=0, label=lab)
    for xi, vi in zip(xs + (i - 1) * w, v):
        ax.text(xi, vi + 0.015, f"{vi * 100:.0f}", ha="center", va="bottom", fontsize=5.8, color=K)
ax.axhline(0.5, color=G2, lw=0.5, ls=(0, (1, 2)))
ax.text(2.5, 0.51, "chance", ha="left", va="center", fontsize=5.8, color=G1, clip_on=False,
        transform=ax.transData).set_position((2.52, 0.5))
ax.set(xticks=xs, ylim=(0, 1.12), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100"],
       ylabel="correct name (%)", xlim=(-0.5, 2.5))
ax.set_xticklabels([f"{b[f]['n']} heads" for f in FR])
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper left", fontsize=6.3, ncol=3, handlelength=0.8, columnspacing=1.0, borderaxespad=0,
          bbox_to_anchor=(0, 1.04))
title(ax, "d", "Is it more than ranking heads once?")

out = os.path.join(ROOT, "paper", "neuroai", "fig_overview")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
