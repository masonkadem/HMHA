"""Main figure (publication style): task, rank argument, open- vs closed-loop supply, supply
during training, count against stability, planted copies, backups under damage. Every
number is read from the result pickles.

  python experiments/figure_main.py   ->  figures/fig_main.png and figures/fig_main.pdf
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

ACC = "#1f5fa8"                       # the one accent: the collateral rule
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1,
    "legend.frameon": False, "savefig.dpi": 400, "pdf.fonttype": 42})

RUNS = []
for d in ("results", "results/proposal", "results/confirm"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r:
            RUNS.append(r)
DEFAULT = Cfg()
PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", kappa_end=1.5,
              target_frac=0.02, delay=0, pool_beta=0.0, leak=0.0, n_territories=8,
              stall_gate=0.0, autoreg_gain=0.003, precondition=0.0, flow_exponent=4.0,
              price_frac=0.01, probe_every=25, taper=0, budget_hold_frac=0.25, prune_stop=1,
              conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, probe_masks=32)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


def stayed_solved(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


THRESH = dict(supply="threshold")
LOOP = dict(supply="autoreg", stall_gate=0.02)
RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
SIZES = (2, 3, 4, 6, 8)

fig = plt.figure(figsize=(7.2, 8.4))
gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1, 1], hspace=0.62, wspace=0.48)

# ---------------------------------------------------------------- (a) task
ax = fig.add_subplot(gs[0, :2])
ax.set_title("(a)  Task with a known circuit size", pad=4)
ax.axis("off")
N, p, offs = 16, 8, (1, 5, 9, 13)
ax.set(xlim=(-4.6, 16), ylim=(-4.3, 2.4))
ax.add_patch(Rectangle((p - 0.42, 1.3), 0.84, 0.72, facecolor="white", edgecolor=K, lw=0.8))
ax.text(p, 1.66, "$p$", ha="center", va="center", fontsize=8)
ax.text(-4.5, 1.66, "query", fontsize=7.5, color=G1, va="center")
ax.text(-4.5, 0, "memory $Y$", fontsize=7.5, color=G1, va="center")
targets = {(p + d) % N: d for d in offs}
for j in range(N):
    hit = j in targets
    ax.add_patch(Rectangle((j - 0.42, -0.36), 0.84, 0.72, facecolor=G3 if hit else "white",
                           edgecolor=K if hit else G2, lw=0.8))
    ax.text(j, 0, str(j), ha="center", va="center", fontsize=6.5, color=K if hit else G1)
for j, d in targets.items():
    ax.add_patch(FancyArrowPatch((p, 1.28), (j, 0.4), connectionstyle=f"arc3,rad={0.3 if j < p else -0.3}",
                                 arrowstyle="-|>", mutation_scale=6, color=K, lw=0.6))
ax.text(-4.5, -1.55, "target", fontsize=7.5, color=G1, va="center")
ax.text(5.8, -1.55, r"$y=\left(c_{p+\delta_1},\,\dots,\,c_{p+\delta_R}\right),\quad \delta=(1,5,9,13),\quad N=16$",
        ha="center", va="center", fontsize=8)
ax.text(5.8, -2.75, r"head $h$ returns one blend $u_h=\sum_r A_{hr}\,c_{p+\delta_r}$;  "
        r"all $c$ recovered iff $\mathrm{rank}\,A=R$", ha="center", va="center", fontsize=7, color=G1)
ax.text(5.8, -3.75, r"$\Rightarrow\ k^\star=R$ heads (here $k^\star=4$ of $H=32$)", ha="center", va="center", fontsize=8)

# ---------------------------------------------------------------- (b) rank
ax = fig.add_subplot(gs[0, 2])
E1 = pickle.load(open(os.path.join(ROOT, "results", "proposal", "equations.pkl"), "rb"))
triv = E1["trivial"]
ax.plot([0, 4], [1, 0], color=G2, lw=0.9, ls="--", zorder=1)
plain = [r for r in E1["rows"] if len(set(r["heads"])) == len(r["heads"])]
copies = [r for r in E1["rows"] if len(set(r["heads"])) < len(r["heads"])]
ax.plot([r["different"] for r in plain], [r["loss"] / triv for r in plain], "o", color=K, ms=3.5,
        label="subset")
ax.plot([r["different"] for r in copies], [r["loss"] / triv for r in copies], "o", mfc="white", mec=K,
        ms=6.5, mew=0.8, label="with a duplicate")
ax.text(4.2, 0.93, r"$L=\left(1-\frac{\mathrm{rank}\,A}{R}\right)L_{\mathrm{triv}}$", fontsize=8, ha="right")
ax.set(xlabel=r"distinct heads, $\mathrm{rank}\,A$", ylabel=r"$L\,/\,L_{\mathrm{triv}}$",
       xticks=range(5), xlim=(-0.2, 4.3), ylim=(-0.05, 1.05))
ax.legend(loc="lower left", handletextpad=0.3)
ax.set_title("(b)  One head, one equation")

# ---------------------------------------------------------------- (c) open vs closed loop
ax = fig.add_subplot(gs[1, 0])
ax.plot([1.6, 8.4], [1.6, 8.4], color=G2, lw=0.8, ls=":", zorder=1)
for kw, style, label in ((THRESH, dict(color=G1, marker="s", mfc="white", ls="-"), "fixed threshold"),
                         (LOOP, dict(color=K, marker="^", mfc="white", ls="--"), "autoregulated loop"),
                         (RULE, dict(color=ACC, marker="o", mfc=ACC, ls="-"), "collateral rule")):
    B = {k: [kept(r) for r in runs(**kw, n_rel=k)] for k in SIZES}
    B = {k: v for k, v in B.items() if v}
    ax.errorbar(list(B), [np.mean(v) for v in B.values()], yerr=[np.std(v) for v in B.values()],
                ms=3.8, mew=0.8, capsize=1.5, elinewidth=0.6, label=label, **style)
ax.set(xlabel=r"true circuit size $k^\star$", ylabel=r"heads kept $B$", xticks=SIZES, yticks=SIZES,
       xlim=(1.5, 8.5), ylim=(-0.6, 11.2))
ax.legend(loc="upper left", handlelength=2.2)
ax.text(8.4, -0.3, r"open loop: $B=\#\{h: z_h>\kappa\}$" "\n"
        r"closed loop: $\kappa\leftarrow\kappa+\eta\ln(L^\star/\hat L)$", fontsize=6.3, ha="right", color=G1)
ax.set_title("(c)  Supply that follows the deficit\n       finds the circuit size")

# ---------------------------------------------------------------- (d) during training
ax = fig.add_subplot(gs[1, 1:])
for kw, color, ls, label in ((LOOP, K, "--", "autoregulated loop"), (RULE, ACC, "-", "collateral rule")):
    r = sorted(runs(**kw, n_rel=4), key=lambda r: get(r, "seed"))[0]
    on = (r["hist"]["ledger"] > 0).sum(1)
    ax.plot(np.arange(len(on)), on, color=color, ls=ls, lw=0.9, label=label, drawstyle="steps-post")
ax.axhline(4, color=G2, lw=0.8, ls=":")
ax.text(3990, 4.35, r"$k^\star=4$", ha="right", fontsize=7, color=G1)
ax.set(yscale="log", yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"],
       xlabel="training step", ylabel="heads receiving supply", xlim=(0, 4000), ylim=(1.8, 40))
ax.minorticks_off()
ax.legend(loc="upper right")
ax.text(2900, 2.35, "loop alternates between 3 and 4 heads", fontsize=6.5, color=G1, ha="center")
ax.set_title("(d)  Supply during training ($k^\\star=4$, one run): heads are starved until the circuit remains")

# ---------------------------------------------------------------- (e) count vs stability
ax = fig.add_subplot(gs[2, 0])
arms = [("collateral rule", RULE, SIZES, ACC, "o"),
        ("autoregulated loop", LOOP, SIZES, K, "^"),
        ("standard pruning", dict(supply="prune"), SIZES, G1, "s"),
        ("ordinary importance", {**RULE, "local_value": "ablate"}, (2, 4, 8), G1, "D"),
        ("random choice", {**RULE, "local_value": "random"}, (2, 4, 8), G1, "v")]
place = {"collateral rule": (-3, -24, "right"), "autoregulated loop": (-5, 11, "right"),
         "standard pruning": (-5, -1, "right"), "ordinary importance": (1, -14, "left"),
         "random choice": (-5, 5, "right")}
for name, kw, sizes, color, m in arms:
    rs = [r for k in sizes for r in runs(**kw, n_rel=k)]
    x = 100 * np.mean([kept(r) == get(r, "n_rel") for r in rs])
    y = 100 * np.mean([stayed_solved(r) for r in rs])
    ax.plot(x, y, m, color=color, ms=5, mfc=color if color == ACC else "white", mew=0.9,
            ls="none", label=f"{name} ($n={len(rs)}$)")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.52), fontsize=6.3, handletextpad=0.3)
ax.set(xlim=(-5, 106), ylim=(-6, 108), xticks=[0, 25, 50, 75, 100], xlabel="runs with $B=k^\\star$ (%)",
       ylabel="runs never above the\nsolved bar after solving (%)")
ax.set_title("(e)  Count and stability")

# ---------------------------------------------------------------- (f) copies
ax = fig.add_subplot(gs[2, 1])
res = {}
for value in ("refit", "ablate"):
    out = []
    for r in runs(**RULE, n_rel=4, plant_copies=1, local_value=value):
        on = r["hist"]["ledger"][-1] > 0
        out.append((int(on.sum()), sum(bool(on[h] and on[h + 16]) for h in range(16))))
    res[value] = np.array(out)
rng = np.random.default_rng(1)
for i, (value, face, label) in enumerate((("refit", ACC, "collateral value"), ("ablate", "white", "ordinary importance"))):
    for j in range(2):
        v = res[value][:, j]
        x0 = j * 2.6 + i * 0.95
        ax.bar(x0, v.mean(), 0.8, color=face, edgecolor=K if face == "white" else face, lw=0.7,
               label=label if j == 0 else None)
        ax.plot(x0 + rng.uniform(-0.2, 0.2, len(v)), v, ".", color=G1, ms=3, zorder=3)
ax.plot([-0.5, 1.45], [4, 4], color=K, lw=0.7, ls=":")
ax.set_xticks([0.475, 3.075], ["heads kept", "duplicate pairs\nboth kept"])
ax.set(ylim=(0, 22), yticks=[0, 4, 8, 12, 16], ylabel=f"count ($n={len(res['refit'])}$ each)")
ax.legend(loc="upper right")
ax.text(1.75, 11.5, r"$v_h=E(S\setminus h)-E(S)$," "\n" r"read-out refit", fontsize=6.5, color=G1)
ax.set_title("(f)  Every head duplicated at start")

# ---------------------------------------------------------------- (g) damage
ax = fig.add_subplot(gs[2, 2])


def predict(R, p, price=0.03):
    q = 1 - p
    val = lambda B: q * sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1)) / R
    return max(B for B in range(1, 33) if val(B) >= price)


ps = [0.0, 0.05, 0.1, 0.2, 0.3]
hit = n = 0
for R, m in ((4, "o"), (2, "s")):
    ax.plot(ps, [predict(R, p) for p in ps], color=G2, lw=0.9, ls="--", zorder=1)
    for p in ps:
        B = [kept(r) for r in runs(**RULE, n_rel=R, head_dropout=p)][:10]
        ax.plot(p + rng.uniform(-0.007, 0.007, len(B)), B, m, color=ACC, ms=3.2, mfc=ACC if R == 4 else "white",
                mew=0.8, label=f"$k^\\star={R}$" if p == 0 else None)
        if p > 0:
            hit += sum(b == predict(R, p) for b in B)
            n += len(B)
ax.set_xlim(-0.025, 0.335)
ax.text(0.33, 11.8, r"$v(B)=\frac{1-p}{R}\,P[\mathrm{Bin}(B{-}1,1{-}p)<R]$" "\n"
        r"$B^\star=\max\{B: v(B)\geq\pi\}$" "\n"
        f"dashed: prediction, {hit}/{n} exact", fontsize=6, ha="right", va="top", color=G1)
ax.set(xlabel="head failure probability $p$", ylabel="heads kept $B$", ylim=(1, 12), xticks=ps,
       yticks=[2, 4, 6, 8],
       xticklabels=["0", ".05", ".1", ".2", ".3"])
ax.legend(loc="center left", bbox_to_anchor=(0, 0.52))
ax.set_title("(g)  Backups under damage")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_main.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_main.png and figures/fig_main.pdf")
