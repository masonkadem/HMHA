"""Main figure (publication style). Every number is read from the result pickles.

  (a) task   (b) one head, one equation   (c, d) loss and heads during training, collateral
  rule against the dense model   (e) count and stability against the baselines   (f) final
  loss against dense   (g) duplicated heads   (h) backup heads under damage

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
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial", "mathtext.sf": "Arial",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})

RUNS = []
for d in ("results", "results/proposal", "results/confirm"):
    for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
        r = pickle.load(open(p, "rb"))
        if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
            RUNS.append(r)
DENSE = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "dense", "*_H32_*.pkl"))]
DEFAULT = Cfg()
PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", kappa_end=1.5,
              target_frac=0.02, delay=0, pool_beta=0.0, leak=0.0, n_territories=8,
              stall_gate=0.0, autoreg_gain=0.003, precondition=0.0, flow_exponent=4.0,
              price_frac=0.01, probe_every=25, taper=0, budget_hold_frac=0.25, prune_stop=1,
              conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, probe_masks=32)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]
dense = lambda R: [d for d in DENSE if d["cfg"]["n_rel"] == R]


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


def stayed_solved(r):
    h, bar = r["hist"], 0.02 * r["trivial"]
    ls = [l for s, l in zip(h["val_step"], h["val_loss"]) if s >= get(r, "budget_hold_frac") * get(r, "steps")]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


def compute(r):
    L = r["hist"]["ledger"]
    probes = r["hist"].get("n_probes", 0) * get(r, "probe_batch") / (3 * get(r, "batch_size"))
    return (L > 0).sum(1).mean() / L.shape[1] + probes / len(L)


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
SIZES = (2, 3, 4, 6, 8)
fig = plt.figure(figsize=(7.2, 8.0))
gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.15, 1], hspace=0.55, wspace=0.5)

# ---------------------------------------------------------------- (a) task
ax = fig.add_subplot(gs[0, :2])
ax.set_title("a   Task with a known circuit size", pad=4)
ax.axis("off")
N, p, offs = 16, 8, (1, 5, 9, 13)
ax.set(xlim=(-4.6, 16), ylim=(-4.3, 2.4))
ax.add_patch(Rectangle((p - 0.42, 1.3), 0.84, 0.72, facecolor="white", edgecolor=K, lw=0.7))
ax.text(p, 1.66, "$p$", ha="center", va="center")
ax.text(-4.5, 1.66, "query", color=G1, va="center")
ax.text(-4.5, 0, "memory", color=G1, va="center")
targets = {(p + d) % N: d for d in offs}
for j in range(N):
    hit = j in targets
    ax.add_patch(Rectangle((j - 0.42, -0.36), 0.84, 0.72, facecolor=G3 if hit else "white",
                           edgecolor=K if hit else G2, lw=0.7))
    ax.text(j, 0, str(j), ha="center", va="center", fontsize=6, color=K if hit else G1)
for j, d in targets.items():
    ax.add_patch(FancyArrowPatch((p, 1.28), (j, 0.4), connectionstyle=f"arc3,rad={0.3 if j < p else -0.3}",
                                 arrowstyle="-|>", mutation_scale=6, color=K, lw=0.6))
ax.text(-4.5, -1.55, "target", color=G1, va="center")
ax.text(5.8, -1.55, r"$y=(c_{p+\delta_1},\ \dots,\ c_{p+\delta_R}),\quad \delta=(1,5,9,13),\quad N=16$",
        ha="center", va="center", fontsize=7.5)
ax.text(5.8, -2.75, r"head $h$ returns one blend $u_h=\Sigma_r\,A_{hr}\,c_{p+\delta_r}$;  "
        r"all $c$ recovered only if rank $A=R$", ha="center", va="center", fontsize=6.8, color=G1)
ax.text(5.8, -3.8, r"so the task needs $k^*=R$ heads (here 4 of $H=32$)", ha="center", va="center", fontsize=7.5)

# ---------------------------------------------------------------- (b) rank
ax = fig.add_subplot(gs[0, 2])
E1 = pickle.load(open(os.path.join(ROOT, "results", "proposal", "equations.pkl"), "rb"))
triv = E1["trivial"]
ax.plot([0, 4], [1, 0], color=G2, lw=0.8, ls="--", zorder=1)
plain = [r for r in E1["rows"] if len(set(r["heads"])) == len(r["heads"])]
copies = [r for r in E1["rows"] if len(set(r["heads"])) < len(r["heads"])]
ax.plot([r["different"] for r in plain], [r["loss"] / triv for r in plain], "o", color=K, ms=3.2, label="subset of heads")
ax.plot([r["different"] for r in copies], [r["loss"] / triv for r in copies], "o", mfc="none", mec=K,
        ms=6.5, mew=0.7, label="with a duplicate")
ax.text(4.2, 0.93, r"$L=(1-\mathrm{rank}\,A\,/\,R)\,L_{\mathrm{triv}}$", ha="right")
ax.set(xlabel="distinct heads (rank $A$)", ylabel=r"loss / $L_{\mathrm{triv}}$",
       xticks=range(5), xlim=(-0.2, 4.3), ylim=(-0.05, 1.05))
ax.legend(loc="lower left", handletextpad=0.3)
ax.set_title("b   One head, one equation")

# ---------------------------------------------------------------- (c, d) during training
sub = gs[1, :2].subgridspec(2, 1, height_ratios=[1.35, 1], hspace=0.35)
axl, axh = fig.add_subplot(sub[0]), fig.add_subplot(sub[1], sharex=None)
r = sorted(runs(**RULE, n_rel=4), key=lambda r: get(r, "seed"))[0]
d0 = next(d for d in dense(4) if d["cfg"]["seed"] == get(r, "seed"))
bar = 0.02 * r["trivial"]
axl.plot(d0["hist"]["val_step"], np.array(d0["hist"]["val_loss"]) / d0["trivial"], color=K, lw=1,
         label="dense, 32 heads")
axl.plot(r["hist"]["val_step"], np.array(r["hist"]["val_loss"]) / r["trivial"], color=ACC, lw=1,
         label="collateral rule")
axl.axhline(0.02, color=G2, lw=0.6, ls=":")
axl.text(3990, 0.011, "solved bar", ha="right", va="top", color=G1, fontsize=6.2)
axl.set(yscale="log", ylim=(3e-6, 2), xlim=(0, 4000), ylabel=r"loss / $L_{\mathrm{triv}}$", xticklabels=[])
axl.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
axl.set_title("c   Loss during training (4-head task, one seed)")
on = (r["hist"]["ledger"] > 0).sum(1)
axh.plot([0, 4000], [32, 32], color=K, lw=1)
axh.plot(np.arange(len(on)), on, color=ACC, lw=1, drawstyle="steps-post")
axh.axhline(4, color=G2, lw=0.6, ls=":")
axh.text(4000, 4.6, "$k^*=4$", ha="right", color=G1, fontsize=6.2)
axh.set(yscale="log", yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"], ylim=(2.5, 45),
        xlim=(0, 4000), xlabel="training step", ylabel="heads")
axh.minorticks_off()
axh.set_title(f"d   Heads receiving supply (collateral rule used {compute(r):.2f} of the dense compute)")

# ---------------------------------------------------------------- (e) count and stability
ax = fig.add_subplot(gs[1, 2])
arms = [("collateral rule", RULE, SIZES, ACC, "o"),
        ("standard pruning", dict(supply="prune"), SIZES, K, "s"),
        ("ordinary importance", {**RULE, "local_value": "ablate"}, (2, 4, 8), K, "D"),
        ("random choice", {**RULE, "local_value": "random"}, (2, 4, 8), K, "v")]
for name, kw, sizes, color, m in arms:
    rs = [x for k in sizes for x in runs(**kw, n_rel=k)]
    ax.plot(100 * np.mean([kept(x) == get(x, "n_rel") for x in rs]), 100 * np.mean([stayed_solved(x) for x in rs]),
            m, color=color, ms=5, mfc=color if color == ACC else "white", mew=0.8, ls="none",
            label=f"{name} ({len(rs)})")
ax.set(xlim=(-5, 106), ylim=(-6, 108), xticks=[0, 25, 50, 75, 100], xlabel="runs with exact count (%)",
       ylabel="runs that stay solved (%)")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), handletextpad=0.3, title="rule (runs)",
          title_fontsize=6.5, alignment="left")
ax.set_title("e   Count and stability")

# ---------------------------------------------------------------- (f) final loss against dense
ax = fig.add_subplot(gs[2, 0])
rng = np.random.default_rng(1)
for i, k in enumerate(SIZES):
    dl = [d["hist"]["val_loss"][-1] / d["trivial"] for d in dense(k)]
    cl = [x["final_loss"] / x["trivial"] for x in runs(**RULE, n_rel=k)]
    ax.plot(i - 0.17 + rng.uniform(-0.06, 0.06, len(dl)), dl, "o", ms=2.6, mfc="white", mec=K, mew=0.6,
            label="dense, 32 heads" if i == 0 else None)
    ax.plot(i + 0.17 + rng.uniform(-0.06, 0.06, len(cl)), cl, "o", ms=2.6, color=ACC,
            label="collateral rule" if i == 0 else None)
    ax.text(i, 2.2e-3, f"{np.mean([compute(x) for x in runs(**RULE, n_rel=k)]):.2f}", ha="center",
            fontsize=6, color=ACC)
ax.text(-0.45, 5.5e-3, "compute of the collateral rule (dense = 1)", fontsize=6, color=G1)
ax.axhline(0.02, color=G2, lw=0.6, ls=":")
ax.text(4.45, 0.026, "solved bar", ha="right", fontsize=6, color=G1)
ax.set(yscale="log", ylim=(1e-6, 1e-1), xticks=range(len(SIZES)), xticklabels=[str(k) for k in SIZES],
       xlabel="true circuit size $k^*$", ylabel=r"final loss / $L_{\mathrm{triv}}$")
ax.minorticks_off()
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.7), handletextpad=0.2)
ax.set_title("f   Final loss, 10 seeds each")

# ---------------------------------------------------------------- (g) duplicates
ax = fig.add_subplot(gs[2, 1])
res = {}
for value in ("refit", "ablate"):
    out = []
    for x in runs(**RULE, n_rel=4, plant_copies=1, local_value=value):
        o = x["hist"]["ledger"][-1] > 0
        out.append((int(o.sum()), sum(bool(o[h] and o[h + 16]) for h in range(16))))
    res[value] = np.array(out)
for i, (value, face, label) in enumerate((("refit", ACC, "collateral rule"), ("ablate", "white", "ordinary importance"))):
    for j in range(2):
        v = res[value][:, j]
        x0 = j * 2.6 + i * 0.95
        ax.bar(x0, v.mean(), 0.8, color=face, edgecolor=K if face == "white" else face, lw=0.6,
               label=label if j == 0 else None)
        ax.plot(x0 + rng.uniform(-0.2, 0.2, len(v)), v, ".", color=G1, ms=2.5, zorder=3)
ax.plot([-0.5, 1.45], [4, 4], color=K, lw=0.6, ls=":")
ax.set_xticks([0.475, 3.075], ["heads kept", "duplicate pairs\nboth kept"])
ax.set(ylim=(0, 22), yticks=[0, 4, 8, 12, 16], ylabel=f"count ({len(res['refit'])} seeds)")
ax.legend(loc="upper right")
ax.set_title("g   Every head duplicated at start")

# ---------------------------------------------------------------- (h) damage
ax = fig.add_subplot(gs[2, 2])


def predict(R, p, price=0.03):
    q = 1 - p
    val = lambda B: q * sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1)) / R
    return max(B for B in range(1, 33) if val(B) >= price)


ps = [0.0, 0.05, 0.1, 0.2, 0.3]
hit = n = 0
for R, m in ((4, "o"), (2, "s")):
    ax.plot(ps, [predict(R, p) for p in ps], color=G2, lw=0.8, ls="--", zorder=1)
    for p in ps:
        B = [kept(x) for x in runs(**RULE, n_rel=R, head_dropout=p)][:10]
        ax.plot(p + rng.uniform(-0.007, 0.007, len(B)), B, m, color=ACC, ms=3, mfc=ACC if R == 4 else "white",
                mew=0.7, label=f"$k^*={R}$" if p == 0 else None)
        if p > 0:
            hit += sum(b == predict(R, p) for b in B)
            n += len(B)
ax.set_xlim(-0.025, 0.335)
ax.text(0.33, 11.8, r"$v(B)=\frac{1-p}{R}\,P[\mathrm{Bin}(B-1,1-p)<R]$" "\n"
        r"$B^*=\max\{B:\,v(B)\geq\pi\}$" "\n" f"dashed: prediction, {hit}/{n} exact",
        fontsize=6, ha="right", va="top", color=G1)
ax.set(xlabel="head failure probability $p$", ylabel="heads kept", ylim=(1, 12), xticks=ps,
       yticks=[2, 4, 6, 8], xticklabels=["0", ".05", ".1", ".2", ".3"])
ax.legend(loc="center left", bbox_to_anchor=(0, 0.52))
ax.set_title("h   Backups under damage")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_main.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_main.png and figures/fig_main.pdf")
