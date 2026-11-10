"""The whole study in one figure, one row per question: the experiment (drawn) next to what we found.

  1  Does the rule keep exactly the heads a task needs?          task + valves | heads closing | collapse
  2  Does it predict how many spare heads to keep?               heads failing | predicted vs kept | payoff
  3  Does it find the induction circuit?                         repeated sequence | heads per layer
  4  Does it find a known circuit in a pretrained model?         GPT-2 prompt + 144 valves | circuit | vs one-shot

  python experiments/figure_story.py  ->  paper/neuroai/fig_story.pdf (and .png)
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg
from hemo.induction import ICfg

SIMPLE = "--simple" in sys.argv          # drops panels m and o
RED = "#a50f15"
K, G1, G2, G3 = "#1a1a1a", "#666666", "#a6a6a6", "#e6e6e6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": 400,
    "pdf.fonttype": 42})
title = lambda ax, letter, text: ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4, color=K)

# ================================================================ data
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
clean = [r for r in RUNS if all(get(r, a) == v for a, v in same.items()) and get(r, "n_rel") in (2, 4, 8)]


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

IND = [pickle.load(open(p, "rb")) for p in glob.glob(os.path.join(ROOT, "results", "induction", "*.pkl"))]


def pick_ind(heads=(8, 8), **kw):
    base = ICfg()
    match = lambda r: all(r["cfg"].get(k, getattr(base, k)) == v for k, v in kw.items())
    return sorted([r for r in IND if match(r) and tuple(r["sizes"]) == heads], key=lambda r: r["cfg"]["seed"])


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
G2R = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", "ioi_pf0.01_s0.pkl"), "rb"))
KEEP = set(map(tuple, G2R["kept"]))
BASE = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", "baselines_s0.pkl"), "rb"))["sizes"]

# ================================================================ methods table data (as figure_neuroai.py, panel b)
def stayed_solved(r):
    ls, bar = r["hist"]["val_loss"], 0.02 * r["trivial"]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]


def score(kw):
    rs = [x for k in (2, 4, 8) for x in runs(**kw, n_rel=k) if get(x, "seed") < 10]
    return sum(kept(x) == get(x, "n_rel") for x in rs), sum(stayed_solved(x) for x in rs), len(rs)


gates = {lam: score(dict(supply="l0", budget_hold_frac=0.1, l0_lambda=lam)) for lam in (0.05, 0.2, 1.0)}
METHODS = [("Ours", "upkeep", score(RULE)),
           ("ZipLM", "target", score({**RULE, "oneshot": 1, "budget_hold_frac": 0.3})),
           ("Michel et al.", "target", score(dict(supply="prune"))),
           ("L0 gates", "penalty", gates[max(gates, key=lambda lam: sum(gates[lam][:2]))])]
print("methods:", METHODS)

# ================================================================ layout
fig = plt.figure(figsize=(7.0, 11.7))
rows = fig.add_gridspec(5, 1, height_ratios=[0.85, 1, 1.05, 0.95, 1.15], hspace=0.85)


def row(i, ratios, wspace=0.5):
    return rows[i].subgridspec(1, len(ratios), width_ratios=ratios, wspace=wspace)


def header(ax, n, text):
    fig.text(0.035, ax.get_position().y1 + 0.03, f"{n}   {text}", fontsize=8.2, color=RED, weight="bold", va="bottom")


def box(ax, x, y, w, h, text="", fc="white", ec=G2, tc=K, fs=6, lw=0.6):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, a, b, color=G1, rad=0.0, lw=0.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


SHADES = (G2, G1, K)


# ================================================================ row 0: where the rule comes from
g = row(0, [1.0, 1.0, 1.0], wspace=0.45)
ax = fig.add_subplot(g[0]); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 6.2))
title(ax, "a", "Vessels are kept while they carry flow")
V = "#7f0000"
for (x0, y0, x1, y1, lw) in ((0.3, 3.1, 2.8, 3.1, 4.5), (2.8, 3.1, 5.2, 4.7, 3.2), (2.8, 3.1, 5.2, 1.5, 3.2),
                             (5.2, 4.7, 8.0, 5.5, 2.2), (5.2, 4.7, 8.0, 3.9, 2.2), (5.2, 1.5, 8.0, 0.7, 2.2)):
    ax.plot([x0, x1], [y0, y1], color=V, lw=lw, solid_capstyle="round")
ax.plot([5.2, 8.0], [1.5, 2.3], color=G2, lw=1.0, ls=(0, (2, 1.5)))
ax.text(8.2, 2.3, "little flow:\nregresses", fontsize=5.8, color=G1, va="center")

ax = fig.add_subplot(g[1])
r = np.linspace(0.55, 2.4, 200)
work, keep_ = 1 / r ** 4, 0.5 * r ** 2
ax.plot(r, work, color=G1, lw=0.9); ax.plot(r, keep_, color=G1, lw=0.9, ls=(0, (3, 2)))
ax.plot(r, work + keep_, color=RED, lw=1.3)
i = np.argmin(work + keep_)
ax.plot(r[i], (work + keep_)[i], "o", color=RED, ms=3.5)
ax.text(1.0, 3.7, "pumping work", fontsize=5.8, color=G1, va="top")
ax.text(2.4, 0.9, "upkeep of blood", fontsize=5.8, color=G1, ha="right", va="center")
ax.text(r[i] + 0.05, (work + keep_)[i] + 0.45, "best size", fontsize=5.8, color=RED, ha="left", va="bottom")
ax.set(xlim=(0.5, 2.45), ylim=(0, 4), xticks=[], yticks=[], xlabel="vessel radius", ylabel="cost")
title(ax, "b", "Murray's law: work + upkeep")
header(ax, 0, "Where the rule comes from")

ax = fig.add_subplot(g[2])
k = np.arange(0, 9)
loss, up = np.maximum(0, 1 - k / 4), 0.06 * k
ax.plot(k, loss, color=G1, lw=0.9, marker="o", ms=2)
ax.plot(k, up, color=G1, lw=0.9, ls=(0, (3, 2)))
ax.plot(k, loss + up, color=RED, lw=1.3, marker="o", ms=2.4)
ax.plot(4, (loss + up)[4], "o", color=RED, ms=4)
ax.text(1.4, 0.33, "loss", fontsize=5.8, color=G1, va="center")
ax.text(8, 0.36, "upkeep", fontsize=5.8, color=G1, ha="right", va="top")
ax.text(4.3, 0.62, "heads needed", fontsize=5.8, color=RED, ha="left", va="center")
ax.set(xlim=(-0.3, 8.3), ylim=(0, 1.15), xticks=[0, 4, 8], yticks=[], xlabel="open heads", ylabel="cost")
title(ax, "c", "Heads: loss + upkeep")

# ================================================================ row 1: retrieval task
g = row(1, [1.05, 0.85, 1.3, 0.85], wspace=0.45)
ax = fig.add_subplot(g[0]); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 6.4))
title(ax, "d", "Retrieval task")
cw = 0.55
for j in range(16):
    hit = j in (1, 5, 9, 13)
    box(ax, 1.0 + j * cw, 3.4, cw, 0.6, "", fc=G3 if hit else "white", ec=K if hit else G2)
box(ax, 1.0, 5.2, cw, 0.6, "p", ec=K, fs=6)
for j in (1, 5, 9, 13):
    arrow(ax, (1.0 + cw / 2, 5.2), (1.0 + j * cw + cw / 2, 4.02), color=K, lw=0.5)
ax.text(0.85, 5.5, "query", ha="right", va="center", fontsize=5.8, color=G1)
ax.text(0.85, 3.7, "memory", ha="right", va="center", fontsize=5.8, color=G1)
for i in range(10):                                            # heads with valves
    x, on = 1.0 + i * 0.85, i in (0, 3, 4, 8)
    box(ax, x, 0.6, 0.6, 0.5, "", ec=K if on else G2)
    ax.add_patch(Rectangle((x + 0.18, 1.25), 0.24, 0.45, fc=RED if on else "white", ec=RED if on else G2, lw=0.6))
ax.text(0.85, 0.85, "heads", ha="right", va="center", fontsize=5.8, color=G1)
ax.text(0.85, 1.48, "valves", ha="right", va="center", fontsize=5.8, color=RED)

ax = fig.add_subplot(g[1])
for k, shade in zip((2, 4, 8), SHADES):
    r = sorted([r for r in clean if get(r, "n_rel") == k and get(r, "price_frac") == 0.03], key=lambda r: get(r, "seed"))[0]
    on = (r["hist"]["ledger"] > 0).sum(1)
    ax.plot(np.arange(len(on)), on, color=shade, lw=1.0, drawstyle="steps-post")
    ax.text(4000, k * 1.08, f"{k}", ha="right", va="bottom", fontsize=6, color=shade)
ax.set(yscale="log", ylim=(1.5, 45), yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"],
       xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=["0", "2k", "4k"], xlabel="training step",
       ylabel="open heads")
ax.minorticks_off()
title(ax, "e", "Heads closing")
header(ax, 1, "Does the rule keep exactly the heads a task needs?")

ax = fig.add_subplot(g[2]); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 6.4))
title(ax, "f", "Same 30 runs, every method")
cols = (0.0, 4.4, 7.6, 9.4)
for x, t, ha in zip(cols, ("method", "set by", "exact", "intact"), ("left", "left", "center", "center")):
    ax.text(x, 5.6, t, fontsize=5.6, color=G1, ha=ha, va="center")
ax.plot([0, 10], [5.15, 5.15], color=K, lw=0.6)
for i, (name, by, (ex, ok, n)) in enumerate(METHODS):
    y, c = 4.4 - i * 1.0, RED if i == 0 else K
    w_ = "bold" if i == 0 else "normal"
    ax.text(cols[0], y, name, fontsize=6.2, color=c, va="center", weight=w_)
    ax.text(cols[1], y, by, fontsize=6.2, color=c, va="center")
    ax.text(cols[2], y, f"{ex}", fontsize=6.2, color=c, va="center", ha="center", weight=w_)
    ax.text(cols[3], y, f"{ok}", fontsize=6.2, color=c, va="center", ha="center", weight=w_)
ax.plot([0, 10], [0.85, 0.85], color=K, lw=0.6)
ax.text(0, 0.25, "ZipLM uses our score, once", fontsize=5.5, color=G1, va="center")

ax = fig.add_subplot(g[3])
for k, shade in zip((2, 4, 8), SHADES):
    pts = {}
    for r in [r for r in clean if get(r, "n_rel") == k]:
        pts.setdefault(round(get(r, "price_frac") * k, 4), []).append(kept(r))
    xs = sorted(pts)
    ax.plot(xs, [np.mean(pts[x]) for x in xs], color=shade, lw=0.9, marker="o", ms=2.2)
ax.axvspan(1.0, 3.0, color=G3, lw=0, zorder=0)
ax.text(0.03, 0.97, "rent < value\nof a head", transform=ax.transAxes, fontsize=5.6, color=G1, va="top")
ax.text(0.97, 0.97, "rent >\nvalue", transform=ax.transAxes, fontsize=5.6, color=K, va="top", ha="right")
ax.set(xscale="log", yscale="log", xlim=(0.015, 3.0), ylim=(0.8, 14), yticks=[1, 2, 4, 8], yticklabels=["1", "2", "4", "8"],
       xticks=[0.1, 1], xticklabels=["0.1", "1"], xlabel="rent ÷ value of one head", ylabel="heads kept")
ax.minorticks_off()
title(ax, "g", "Too much rent: collapse")

# ================================================================ row 2: induction
g = row(2, [1.5, 0.9, 0.9])
ax = fig.add_subplot(g[0]); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0.5, 6.4))
title(ax, "h", "Repeated random sequence, 2 layers × 8 heads")
toks = ["7", "3", "9", "4", "1", "…", "7", "3", "9", "?"]
xs = [0.3 + i * 0.85 for i in range(len(toks))]
for i, (x, t) in enumerate(zip(xs, toks)):
    box(ax, x, 3.0, 0.65, 0.65, t, ec=RED if t == "?" else (K if i in (2, 3) else G2), fs=6.5,
        tc=RED if t == "?" else K)
arrow(ax, (xs[9] + 0.32, 3.7), (xs[3] + 0.32, 3.7), color=RED, rad=0.35)
ax.text((xs[3] + xs[9]) / 2 + 0.3, 5.45, "induction head (layer 2)", ha="center", fontsize=5.8, color=RED)
arrow(ax, (xs[3] + 0.32, 2.95), (xs[2] + 0.32, 2.95), color=K, rad=-0.6)
ax.text(xs[2] - 0.1, 1.9, "previous-token head (layer 1)", fontsize=5.8, color=K)

ax = fig.add_subplot(g[1])
rs = pick_ind(rule="collateral", probe="logit", price_mode="bar", price_frac=0.05, steps=6000, squeeze_at=0.0, trial=0)
for r in rs:
    led = np.asarray(r["hist"]["ledger"]) > 0
    for cols_, color in ((slice(0, 8), G2), (slice(8, None), K)):
        ax.plot(led[:, cols_].sum(1), color=color, lw=0.7, alpha=0.7)
ax.text(5900, 2.6, "layer 1", ha="right", va="bottom", fontsize=5.8, color=G1)
ax.text(5900, 1.15, "layer 2", ha="right", va="bottom", fontsize=5.8, color=K)
ax.set(xlim=(0, 6000), ylim=(0, 8.6), yticks=[0, 1, 2, 4, 8], xticks=[0, 3000, 6000],
       xticklabels=["0", "3k", "6k"], xlabel="training step", ylabel="open heads")
title(ax, "i", f"Heads closing ({len(rs)} runs)")
header(ax, 2, "Does it find the known induction circuit?")

ax = fig.add_subplot(g[2])
rng = np.random.default_rng(2)
for i, (heads, name) in enumerate((((1, 0), "1"), ((8, 0), "8"), ((1, 1), "1+1"), ((2, 1), "2+1"))):
    rr = pick_ind(heads=heads, rule="dense", steps=6000)
    loss = np.array([r["final_loss"] for r in rr])
    ax.plot(i + rng.uniform(-0.15, 0.15, len(loss)), loss, "o", ms=2.6, mec=K if heads[1] else G2, mew=0.6,
            mfc="white", ls="none")
ax.axhline(rr[0]["bar"], color=G2, lw=0.6, ls=(0, (1, 2)))
ax.set(yscale="log", ylim=(1e-4, 150), xlim=(-0.5, 3.5), xticks=range(4), ylabel="final loss",
       yticks=[1e-4, 1e-2, 1e0], xlabel="heads per layer, trained alone")
ax.set_xticklabels(["1", "8", "1+1", "2+1"])
ax.minorticks_off()
title(ax, "j", "1+1 suffices: the 2nd layer-1 head is a backup")

# ================================================================ row 3: backup heads, failure tree
g = row(3, [1.5, 0.9, 0.9])
ax = fig.add_subplot(g[0]); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(-1.2, 6))
title(ax, "k", "Is one more head worth its upkeep?")


def node(x, y, text, color=K, edge=G2):
    ax.text(x, y, text, ha="center", va="center", fontsize=6, color=color,
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=edge, lw=0.6))


def line(x0, y0, x1, y1, color=G2, lw=0.7):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=0)


def prob(x, y, t, color=G1):
    ax.text(x, y, t, ha="center", va="center", fontsize=5.8, color=color,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


node(0.7, 3.0, "head B")
node(2.9, 4.9, "fails", color=G1)
node(2.9, 1.4, "works", color=RED, edge=RED)
node(5.1, 2.9, "≥ 4 others work", color=G1)
node(5.1, 0.0, "< 4 others work", color=RED, edge=RED)
line(1.3, 3.25, 2.5, 4.75); prob(1.8, 4.2, "p")
line(1.3, 2.75, 2.45, 1.55, RED, 1.1); prob(1.75, 1.95, "1 − p", RED)
line(3.35, 1.6, 3.7, 2.7)
line(3.35, 1.2, 3.7, 0.2, RED, 1.1)
ax.text(3.75, 4.9, "worth 0", va="center", fontsize=5.8, color=G1)
ax.text(7.25, 2.9, "worth 0", va="center", fontsize=5.8, color=G1)
ax.text(7.25, 0.0, r"worth $L_{\mathrm{triv}}/4$", va="center", fontsize=5.8, color=RED)
ax.text(0.0, -1.1, "task needs 4 heads; keep head B while its value ≥ upkeep", fontsize=5.8, color=K)

ax = fig.add_subplot(g[1])
groups = {}
for r in damaged:
    R, p, c = get(r, "n_rel"), get(r, "head_dropout"), get(r, "price_frac")
    groups.setdefault((R, p, c), (predict_backup(R, p, c), []))[1].append(kept(r))
n_runs = sum(len(v[1]) for v in groups.values())
n_exact = sum(sum(int(k == pr) for k in ks) for pr, ks in groups.values())
ax.plot([2, 13], [2, 13], color=G2, lw=0.6, zorder=0)
for pr, ks in groups.values():
    ks = np.array(ks)
    ax.plot([pr, pr], [ks.min(), ks.max()], color=K, lw=0.8)
    ax.plot(pr, ks.mean(), "o", color=K, ms=3.0, mec="white", mew=0.4)
ax.set(xlim=(2, 13), ylim=(2, 13), xticks=[3, 6, 9, 12], yticks=[3, 6, 9, 12], xlabel="predicted",
       ylabel="heads kept")
n_within = sum(sum(int(abs(k - pr) <= 1) for k in ks) for pr, ks in groups.values())
ax.text(12.8, 2.4, f"exact: {n_exact}/{n_runs}\nwithin one: {n_within}/{n_runs}", ha="right", va="bottom", fontsize=6,
        linespacing=1.3)
title(ax, "l", "Prediction vs. runs")
header(ax, 3, "When is a backup head worth keeping?")

if not SIMPLE:
    ax = fig.add_subplot(g[2])
    P = (0.0, 0.1, 0.2, 0.3)
    for arm, c, lab in (("aware", RED, "backups"), ("blind", K, "no backups")):
        rs = [pickle.load(open(f, "rb")) for f in glob.glob(os.path.join(ROOT, "results", "backup", f"robust_{arm}_s[0-9].pkl"))]
        ref = [np.mean([r["damage"][q]["refit"].mean() for r in rs]) for q in P]
        fix = [np.mean([r["damage"][q]["fixed"].mean() for r in rs]) for q in P]
        ax.plot(P, ref, "-o", color=c, lw=1.1, ms=2.4)
        ax.plot(P, fix, "--", color=c, lw=0.8)
        ax.text(0.31, ref[-1], lab, fontsize=5.8, color=c, va="center")
    ax.text(0.01, 0.47, "solid: re-routed to the survivors\ndashed: as trained", fontsize=5.4, color=G1, va="top")
    ax.set(xlim=(0, 0.3), ylim=(0, 0.5), xticks=list(P), yticks=[0, 0.2, 0.4], xlabel="share of heads knocked out",
           ylabel="loss")
    title(ax, "m", "Backups pay off when heads fail")

# ================================================================ row 4: GPT-2
g = row(4, [1.7, 0.85], wspace=0.25)
ax = fig.add_subplot(g[0]); ax.axis("off"); ax.set(xlim=(0, 11.4), ylim=(0, 6.6))
ax.text(0.0, 6.3, "“… Mary and John went to the store. John gave a drink to”  →  Mary   (frozen GPT-2 small, 144 heads)",
        fontsize=5.9, color=G1, va="center")
CW, CH = 0.78, 0.46


def stage(x, y, w, name, heads, per_row):
    nrow = int(np.ceil(len(heads) / per_row))
    h = 0.55 + nrow * (CH + 0.1)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=G2, lw=0.5))
    ax.text(x + 0.08, y + h - 0.12, name, ha="left", va="top", fontsize=5.2, color=G1)
    for i, hd in enumerate(sorted(heads)):
        r, c = divmod(i, per_row)
        x0 = x + 0.12 + c * (CW + 0.08)
        y0 = y + h - 0.5 - (r + 1) * (CH + 0.1) + 0.05
        on = hd in KEEP
        ax.add_patch(Rectangle((x0, y0), CW, CH, fc=RED if on else "white", ec=RED if on else G2, lw=0.5))
        ax.text(x0 + CW / 2, y0 + CH / 2, f"{hd[0]}.{hd[1]}", ha="center", va="center", fontsize=4.4,
                color="white" if on else G2)


stage(0.0, 3.85, 1.95, "prev. token", WANG["previous token"], 2)
stage(0.0, 1.95, 1.95, "duplicate", WANG["duplicate token"], 2)
stage(2.55, 3.25, 1.95, "induction", WANG["induction"], 2)
stage(5.1, 2.55, 1.95, "S-inhibition", WANG["S-inhibition"], 2)
stage(7.75, 4.6, 2.85, "name movers", WANG["name mover"], 3)
stage(7.75, 1.65, 2.85, "backup movers", WANG["backup name mover"], 3)
stage(7.75, 0.2, 2.85, "negative movers", WANG["negative name mover"], 3)
for a, b in (((1.95, 4.4), (2.55, 4.2)), ((1.95, 2.5), (5.1, 3.0)), ((4.5, 3.9), (5.1, 3.6)),
             ((7.05, 3.75), (7.75, 4.9)), ((7.05, 3.25), (7.75, 3.1)), ((7.05, 2.85), (7.75, 0.75))):
    arrow(ax, a, b, lw=0.5)
outside = sorted(h for h in KEEP if h not in role)
ax.text(0.0, 0.75, f"{len(KEEP)} heads kept, {len(KEEP) - len(outside)} in the circuit\n"
        f"correct name: {G2R['test'][2]:.0%}\n(published 26 heads: {G2R['wang_test'][2]:.0%})",
        fontsize=5.9, color=K, va="center", linespacing=1.3)
title(ax, "n", "Published circuit; heads kept by the rule filled")
header(ax, 4, "Does it find a known circuit in a pretrained model?")

if not SIMPLE:
    ax = fig.add_subplot(g[1])
    FR = ("0.05", "0.02", "0.01")
    x = np.arange(3); w = 0.25
    for i, (lab, v, c) in enumerate((("our rule", [BASE[f]["ours"][2] for f in FR], RED),
                                     ("ranked once", [BASE[f]["oneshot"][2] for f in FR], K),
                                     ("random", [BASE[f]["random"][:, 2].mean() for f in FR], G2))):
        ax.bar(x + (i - 1) * w, v, width=w * 0.88, color=c, lw=0, label=lab)
    ax.axhline(0.5, color=G2, lw=0.5, ls=(0, (1, 2)))
    ax.set(xticks=x, ylim=(0, 1.3), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100"], ylabel="correct name (%)",
           xlim=(-0.5, 2.5), xlabel="heads kept")
    ax.set_xticklabels([str(BASE[f]["n"]) for f in FR])
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", fontsize=5.5, handlelength=0.8, borderaxespad=0)
    title(ax, "o", "Same number of heads")

out = os.path.join(ROOT, "paper", "neuroai", "fig_story_simple" if SIMPLE else "fig_story")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", out)
