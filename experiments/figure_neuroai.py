"""Results figure for the NeuroAI workshop paper (AAAI-27 two-column, full width).

  (a) open heads during training, k* = 2, 4, 8 (seed 0)
  (b) every method on the same 30 runs: exact count and never broke
  (c) heads kept against the upkeep, in units of the value of one needed head (L_triv / R)
  (d) demand rises from 2 to 4 offsets and falls back to 2

Every number is read from the result pickles.

  python experiments/figure_neuroai.py  ->  paper/neuroai/fig_neuroai.pdf (and .png)
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED, PINK = "#b2182b", "#e8a5a5"
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.dpi": 450, "pdf.fonttype": 42})


def load(dirs):
    out, seen = [], set()
    for d in dirs:
        for p in glob.glob(os.path.join(ROOT, d, "*.pkl")):
            if os.path.basename(p) in seen:          # the same run saved in two folders counts once
                continue
            seen.add(os.path.basename(p))
            r = pickle.load(open(p, "rb"))
            if isinstance(r, dict) and "hist" in r and "cfg" in r and "ledger" in r["hist"]:
                out.append(r)
    return out


RUNS = load(("results", "results/proposal", "results/confirm", "results/l0", "results/review",
             "results/equal", "results/instant", "results/breakpoint"))
SWITCH = load(("results/switch",))
DEFAULT = Cfg()
PINNED = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", kappa_end=1.5,
              target_frac=0.02, delay=0, pool_beta=0.0, leak=0.0, n_territories=8,
              stall_gate=0.0, autoreg_gain=0.003, precondition=0.0, flow_exponent=4.0,
              price_frac=0.01, probe_every=25, taper=0, budget_hold_frac=0.25, prune_stop=1,
              conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, probe_masks=32,
              copy_noise=0.0, oneshot=0, trial=0, l0_lambda=0.01, phases="", sprout=0)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


def stayed_solved(r):
    ls, bar = r["hist"]["val_loss"], 0.02 * r["trivial"]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
EQUAL = (2, 4, 8)


def score(kw):
    rs = [x for k in EQUAL for x in runs(**kw, n_rel=k) if get(x, "seed") < 10]
    return sum(kept(x) == get(x, "n_rel") for x in rs), sum(stayed_solved(x) for x in rs), len(rs)


fig = plt.figure(figsize=(7.0, 1.95))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.1, 1, 1.1], wspace=0.85)

# ---------------------------------------------------------------- a  heads closing
ax = fig.add_subplot(gs[0])
for k, shade in zip(EQUAL, (PINK, "#d6604d", RED)):
    r = sorted(runs(**RULE, n_rel=k), key=lambda r: get(r, "seed"))[0]
    on = (r["hist"]["ledger"] > 0).sum(1)
    ax.plot(np.arange(len(on)), on, color=shade, lw=1.0, drawstyle="steps-post", label=f"$k^*={k}$")
    ax.axhline(k, color=G2, lw=0.6, ls=":")
ax.set(yscale="log", ylim=(1.5, 45), yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"],
       xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=["0", "2k", "4k"],
       xlabel="training step", ylabel="open heads")
ax.minorticks_off()
ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0)
ax.set_title(r"$\mathbf{a}$   Heads closing", pad=4)

# ---------------------------------------------------------------- b  methods
gates = {lam: score(dict(supply="l0", budget_hold_frac=0.1, l0_lambda=lam)) for lam in (0.05, 0.2, 1.0)}
best = max(gates, key=lambda lam: gates[lam][0] + gates[lam][1])
rows = [("Ours", score(RULE)),
        ("ZipLM-style", score({**RULE, "oneshot": 1, "budget_hold_frac": 0.3})),
        ("Michel et al.", score(dict(supply="prune"))),
        ("Voita et al.", gates[best]),
        ("no fade", score({**RULE, "taper": 0})),
        ("no re-fit", score({**RULE, "local_value": "ablate"})),
        ("random head", score({**RULE, "local_value": "random"}))]
print("panel b:", rows)
ax = fig.add_subplot(gs[1])
y = np.arange(len(rows))
for i, (name, (ex, ok, n)) in enumerate(rows):
    ours = i == 0
    ax.barh(i - 0.19, ex / n, height=0.36, color=RED if ours else K, lw=0)
    ax.barh(i + 0.19, ok / n, height=0.36, color=PINK if ours else G2, lw=0)
    ax.text(1.03, i - 0.19, f"{ex}", va="center", fontsize=5.3, color=RED if ours else K)
    ax.text(1.03, i + 0.19, f"{ok}", va="center", fontsize=5.3, color=RED if ours else G1)
ax.axhline(3.5, color=G3, lw=0.5)
ax.set(yticks=y, xlim=(0, 1), xticks=[0, 0.5, 1], xticklabels=["0", "15", "30"], ylim=(len(rows) - 0.4, -0.6))
ax.set_yticklabels([r[0] for r in rows])
ax.get_yticklabels()[0].set_color(RED)
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=K, label="exact count"), Patch(color=G2, label="never broke")],
          loc="lower left", bbox_to_anchor=(0.12, 1.0), ncol=2, handlelength=0.8, columnspacing=0.8,
          borderaxespad=0.1)
ax.set_title(r"$\mathbf{b}$", pad=4, x=-0.5)
ax.set_xlabel("runs (of 30)")

# ---------------------------------------------------------------- c  upkeep breakpoint
ax = fig.add_subplot(gs[2])
for k, shade, mk in zip(EQUAL, (PINK, "#d6604d", RED), ("o", "s", "D")):
    pts = {}
    same = {a: v for a, v in {**PINNED, **RULE, "n_rel": k}.items() if a != "price_frac"}
    for r in [r for r in RUNS if all(get(r, a) == v for a, v in same.items())]:
        x = get(r, "price_frac") * k
        pts.setdefault(round(x, 4), []).append(kept(r))
    xs = sorted(pts)
    ax.plot(xs, [np.mean(pts[x]) for x in xs], color=shade, lw=0.8, marker=mk, ms=2.8, label=f"$k^*={k}$")
    print(f"panel c k*={k}:", {x: pts[x] for x in xs})
ax.axvline(1.0, color=G2, lw=0.6, ls="--")
ax.set(xscale="log", yscale="log", ylim=(0.8, 12), yticks=[1, 2, 4, 8], yticklabels=["1", "2", "4", "8"],
       xlabel="upkeep / value of\none needed head", ylabel="heads kept")
ax.minorticks_off()
ax.text(0.85, 1.25, "predicted\ncollapse", fontsize=5.5, color=G1, va="center", ha="right")
ax.set_title(r"$\mathbf{c}$   Upkeep", pad=4)

# ---------------------------------------------------------------- d  changing demand
ax = fig.add_subplot(gs[3])
total = None
for r in sorted(SWITCH, key=lambda r: (get(r, "sprout"), get(r, "seed"))):
    on = (r["hist"]["ledger"] > 0).sum(1)
    total = len(on)
    sp = get(r, "sprout") > 0
    ax.plot(np.arange(len(on)), on, color=RED if sp else K, lw=0.8 if sp else 0.7, alpha=0.8,
            drawstyle="steps-post")
if total:
    ph = [int(c) for c in get(SWITCH[0], "phases")]
    edges = np.linspace(0, total, len(ph) + 1)
    for a, b, need in zip(edges[:-1], edges[1:], ph):
        ax.plot([a, b], [need, need], color=G2, lw=0.6, ls=":")
    ax.set(xlim=(0, total), xticks=edges, xticklabels=[f"{e / 1000:g}k" for e in edges])
ax.plot([], [], color=K, label="reopen only")
ax.plot([], [], color=RED, label="with sprouting")
ax.set(yscale="log", ylim=(1.5, 45), yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"],
       xlabel="training step", ylabel="open heads")
ax.minorticks_off()
ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0)
ax.set_title(r"$\mathbf{d}$   Demand 2$\to$4$\to$2", pad=4)

out = os.path.join(ROOT, "paper", "neuroai")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig_neuroai.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(out, "fig_neuroai.pdf"))
