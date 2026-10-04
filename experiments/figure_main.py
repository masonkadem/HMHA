"""Main figure (publication style). Every number is read from the result pickles.

  Figure 2 (results). (a) loss and (b) heads during training for k* = 2, 4, 6, 8, collateral
  rule against the dense model  (c) table: every method on the same 30 runs (k* = 2, 4, 8, 10 seeds),
  score, use, exact count, never broke  (d) duplicated heads  (e) spare heads under failure  (f) one run per method.
  The task and model are in figure_setup.py; the full comparison table in figure_compare.py.

  python experiments/figure_main.py   ->  figures/fig_main.png and figures/fig_main.pdf
"""
import glob, os, pickle, sys
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory, offset_copy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg

RED = "#b2182b"                       # the collateral rule (blood)
K, G1, G2, G3 = "#111111", "#6b6b6b", "#a8a8a8", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "legend.frameon": False, "savefig.dpi": 450, "pdf.fonttype": 42})

RUNS = []
for d in ("results", "results/proposal", "results/confirm", "results/l0", "results/review", "results/equal",
          "results/instant"):
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
              conserve=1, local_value="refit", plant_copies=0, head_dropout=0.0, probe_masks=32,
              copy_noise=0.0, oneshot=0, trial=0, l0_lambda=0.01)
get = lambda r, k: r["cfg"].get(k, getattr(DEFAULT, k))
runs = lambda **w: [r for r in RUNS if all(get(r, k) == v for k, v in {**PINNED, **w}.items())]
dense = lambda R: [d for d in DENSE if d["cfg"]["n_rel"] == R]
title = lambda ax, letter, text="": ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", pad=4)


def kept(r):
    on = (r["hist"]["ledger"] > 0).sum(1)
    return float(np.median(on[int((get(r, "budget_hold_frac") + get(r, "budget_anneal_frac")) * len(on)):]))


def stayed_solved(r):              # once first solved, the loss never goes back above the solved bar
    ls, bar = r["hist"]["val_loss"], 0.02 * r["trivial"]
    first = next((i for i, l in enumerate(ls) if l < bar), None)
    return first is not None and max(ls[first:]) <= bar


def compute(r):
    L = r["hist"]["ledger"]
    probes = r["hist"].get("n_probes", 0) * get(r, "probe_batch") / (3 * get(r, "batch_size"))
    return (L > 0).sum(1).mean() / L.shape[1] + probes / len(L)


RULE = dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1, conserve=0)
SIZES = (2, 3, 4, 6, 8)
BAR = 0.02
fig = plt.figure(figsize=(7.2, 9.6))
outer = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.45, 1.1], hspace=0.3)
bottom = outer[2].subgridspec(1, 3, wspace=0.5)

# ---------------------------------------------------------------- a, b  training, small multiples
show = (2, 4, 6, 8)
cd = outer[0].subgridspec(2, 4, height_ratios=[0.85, 0.7], hspace=0.16, wspace=0.55)
for i, k in enumerate(show):
    r = sorted(runs(**RULE, n_rel=k), key=lambda r: get(r, "seed"))[0]
    d0 = next(d for d in dense(k) if d["cfg"]["seed"] == get(r, "seed"))
    axl = fig.add_subplot(cd[0, i])
    axl.plot(d0["hist"]["val_step"], d0["hist"]["val_loss"], color=K, lw=0.9, label="dense")
    axl.plot(r["hist"]["val_step"], r["hist"]["val_loss"], color=RED, lw=0.9, label="collateral")
    axl.axhline(BAR, color=G2, lw=0.6, ls=":")
    axl.set(yscale="log", ylim=(3e-6, 2), xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=[])
    axl.minorticks_off()
    axh = fig.add_subplot(cd[1, i])
    on = (r["hist"]["ledger"] > 0).sum(1)
    axh.plot([0, 4000], [32, 32], color=K, lw=0.9)
    axh.plot(np.arange(len(on)), on, color=RED, lw=0.9, drawstyle="steps-post")
    axh.axhline(k, color=G2, lw=0.6, ls=":")
    axh.set(yscale="log", ylim=(1.5, 45), yticks=[2, 4, 8, 16, 32], yticklabels=["2", "4", "8", "16", "32"],
            xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=["0", "2k", "4k"])
    axh.minorticks_off()
    axh.text(3950, 22, f"compute {compute(r):.2f}", ha="right", fontsize=6, color=RED)
    if i == 0:
        axl.set_title(f"$k^*={k}$", pad=4)
        AX_A, AX_B = axl, axh
        axl.set_ylabel("loss")
        axh.set_ylabel("heads")
        axl.legend(loc="upper right", handlelength=1.2, borderaxespad=0)
        axl.text(1450, 2.2e-3, "heads\nclosing", fontsize=5.5, color=G1, va="bottom")
        axl.text(3950, BAR * 1.3, "solved", fontsize=5.5, color=G1, ha="right", va="bottom")
    else:
        axl.set_title(f"$k^*={k}$", pad=4)
        axl.set_yticklabels([])
        axh.set_yticklabels([])
    if i == 1:
        axh.set_xlabel("training step", x=1.15)

# ---------------------------------------------------------------- c  the comparison, as a table with bars
EQUAL = (2, 4, 8)                                  # tasks needing 2, 4 and 8 heads, 10 seeds each


def score(kw):
    rs = [x for k in EQUAL for x in runs(**kw, n_rel=k) if get(x, "seed") < 10]
    return sum(kept(x) == get(x, "n_rel") for x in rs), sum(stayed_solved(x) for x in rs), len(rs)


gates = {lam: score(dict(supply="l0", budget_hold_frac=0.1, l0_lambda=lam)) for lam in (0.05, 0.2, 1.0)}
best_lam = max(gates, key=lambda lam: gates[lam][0] + gates[lam][1])
ONESHOT = {**RULE, "oneshot": 1, "budget_hold_frac": 0.3}
OBD = {**RULE, "local_value": "ablate"}
RANDOM = {**RULE, "local_value": "random"}
INSTANT = {**RULE, "taper": 0}
OBS = r"$\mathrm{min}_W\, L_{-h} - \mathrm{min}_W\, L$"
SURGEON, DAMAGE = "Optimal Brain Surgeon (Hassibi & Stork 1993)", "Optimal Brain Damage (LeCun et al. 1990)"
rows = [  # (group, method, (score, where it comes from), during training, gradual fade, reopens, gate learned, result)
    (None, "Collateral rule (ours)", (OBS, SURGEON), "yes", "yes", "yes", "no", score(RULE)),
    ("Published methods", "ZipLM-style\n(Kurtic et al. 2023)", (OBS, SURGEON), "no", "no", "no", "no", score(ONESHOT)),
    (None, "Michel et al. 2019", (r"$|\partial L / \partial g_h|$", "gradient of the head gate"), "yes", "no", "no", "no",
     score(dict(supply="prune"))),
    (None, "Learned gates\n(Voita et al. 2019)", (r"$L + \lambda\, \Sigma_h\, P(g_h \neq 0)$", "L0 penalty, best of 3"),
     "yes", "no", "no", "yes", gates[best_lam]),
    ("Ablations of our rule (one part removed)", "no fade\n(instant closing)", (OBS, SURGEON), "yes", "no", "yes", "no",
     score(INSTANT)),
    (None, "no re-fit", (r"$L_{-h}(W) - L(W)$", DAMAGE), "yes", "yes", "yes", "no", score(OBD)),
    (None, "no score\n(random choice)", ("random head", ""), "yes", "yes", "yes", "no", score(RANDOM)),
]
ax = fig.add_subplot(outer[1])
ax.axis("off")
COLS = dict(method=0.0, score=0.165, during=0.455, fade=0.522, reopen=0.589, learned=0.656, bars=0.735)
BW = 0.15                                          # bar length for 100% of runs
y = 0.0
ax.text(0, y - 1.25, "Every method, same 30 runs", fontsize=7, va="top")
AX_C, C_TITLE_Y = ax, y - 1.25
for key, name in (("method", "Method"), ("score", "Head score"), ("during", "During\ntraining"), ("fade", "Gradual\nfade"),
                  ("reopen", "Reopens"), ("learned", "Gate\nlearned")):
    ax.text(COLS[key], y - 0.5, name, fontsize=6.3, fontweight="bold", va="center", linespacing=1.15)
ax.add_patch(Rectangle((COLS["bars"], y - 0.82), 0.018, 0.2, color=K, lw=0))
ax.text(COLS["bars"] + 0.024, y - 0.72, "exact count", fontsize=6.3, fontweight="bold", va="center")
ax.add_patch(Rectangle((COLS["bars"], y - 0.40), 0.018, 0.2, color=G2, lw=0))
ax.text(COLS["bars"] + 0.024, y - 0.30, "never broke", fontsize=6.3, fontweight="bold", va="center")
ax.plot([0, 1], [y - 0.05, y - 0.05], color=K, lw=0.6)
y = 0.45
for group, method, sc, during, fade, reopen, learned, (exact, safe, n) in rows:
    if group:
        y += 0.25
        ax.text(0, y, group, fontsize=6, color=G1, style="italic", va="center")
        ax.plot([0, 1], [y - 0.28, y - 0.28], color=G3, lw=0.5)
        y += 0.72
    ours = method.startswith("Collateral")
    c = RED if ours else K
    ax.text(COLS["method"], y, method, color=c, fontsize=6.2, va="center", linespacing=1.15)
    formula, source = sc
    ax.text(COLS["score"], y - (0.16 if source else 0), formula, color=c, fontsize=6.0, va="center")
    if source:
        ax.text(COLS["score"], y + 0.24, source, color=c if ours else G1, fontsize=5.2, va="center")
    for key, v in (("during", during), ("fade", fade), ("reopen", reopen), ("learned", learned)):
        ax.text(COLS[key] + 0.02, y, v, color=c if v == "yes" else G2, fontsize=6.2, va="center", ha="center")
    for dy, v, shade in ((-0.17, exact, RED if ours else K), (0.17, safe, "#e8a5a5" if ours else G2)):
        ax.add_patch(Rectangle((COLS["bars"], y + dy - 0.13), BW * v / n, 0.26, color=shade, lw=0))
        ax.add_patch(Rectangle((COLS["bars"], y + dy - 0.13), BW, 0.26, fill=False, edgecolor=G3, lw=0.4))
        ax.text(COLS["bars"] + BW + 0.008, y + dy, f"{v}/{n}", color=c, fontsize=5.6, va="center")
    y += 0.95
ax.plot([0, 1], [y - 0.45, y - 0.45], color=K, lw=0.6)
ax.text(0, y - 0.3, "Tasks needing 2, 4 and 8 heads, 10 seeds each.", fontsize=5.6, color=G1, va="top")
ax.set(xlim=(0, 1), ylim=(y + 0.2, -1.3))
print("panel c:", [(m.replace(chr(10), " "), r) for _, m, _, _, _, _, _, r in rows], "learned-gate penalty", best_lam)

# ---------------------------------------------------------------- d  duplicated heads
# every run starts with head h + 16 an exact twin of head h; a twin is fully covered by its
# copy, so it should be removed. Bars: heads kept, split into distinct heads and extra twins.
ax = fig.add_subplot(bottom[0])
rng = np.random.default_rng(1)
for i, (value, face, label) in enumerate((("refit", RED, "collateral"), ("ablate", "white", "importance"))):
    L = [x["hist"]["ledger"][-1] > 0 for x in runs(**RULE, n_rel=4, plant_copies=1, local_value=value)]
    n_kept = np.array([o.sum() for o in L])
    twins = np.array([sum(bool(o[h] and o[h + 16]) for h in range(16)) for o in L])
    distinct = (n_kept - twins).mean()
    ax.bar(i, distinct, 0.6, color=face, edgecolor=RED if face == RED else K, lw=0.6)
    ax.bar(i, twins.mean(), 0.6, bottom=distinct, color="white", edgecolor=K, lw=0.6, hatch="//////")
    ax.plot(i + rng.uniform(-0.15, 0.15, len(n_kept)), n_kept, ".", color=G1, ms=2.5, zorder=3)
ax.axhline(4, color=G2, lw=0.6, ls=":", zorder=0)
ax.text(-0.57, 4.3, "needed", color=G1, fontsize=6, va="bottom", ha="left")
ax.bar(0, 0, color="white", edgecolor=K, lw=0.6, hatch="//////", label="twin also kept")
ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.4, borderaxespad=0)
ax.set(xticks=[0, 1], ylim=(0, 18), yticks=[0, 4, 8, 12, 16], xlim=(-0.6, 1.6), ylabel="heads kept")
ax.set_xticklabels(["collateral\nrule", "OBD-style"])
ax.set_title("Every head given a twin", pad=4)
AX_D = ax

# ---------------------------------------------------------------- e  spare heads under failure
ax = fig.add_subplot(bottom[1])


def predict(R, p, price=0.03):
    q = 1 - p
    val = lambda B: q * sum(comb(B - 1, a) * q ** a * p ** (B - 1 - a) for a in range(min(R - 1, B - 1) + 1)) / R
    return max(B for B in range(1, 33) if val(B) >= price)


ps = [0.0, 0.05, 0.1, 0.2, 0.3]
for R, m in ((4, "o"), (2, "s")):
    pred = [predict(R, p) for p in ps]
    ax.fill_between(ps, R, pred, color=RED, alpha=0.10, lw=0, step=None)
    ax.plot(ps, [R] * len(ps), color=G2, lw=0.6, ls=":")
    ax.plot(ps, pred, color=G2, lw=0.9, ls="--", zorder=1, label="predicted before the runs" if R == 4 else None)
    means = [np.mean([kept(x) for x in runs(**RULE, n_rel=R, head_dropout=p)][:10]) for p in ps]
    ax.plot(ps, means, m, color=RED, ms=3.5, mfc=RED if R == 4 else "white", mew=0.8, ls="none",
            label="measured" if R == 4 else None)
    ax.text(0.31, R - 0.35 if R == 4 else R, f"needs {R}", va="center", fontsize=5.5, color=G1)
ax.text(0.16, 4.45, "spares", fontsize=5.5, color=RED, ha="center")
ax.set(xlabel="chance a head fails", ylabel="heads kept", ylim=(1, 8.6), xlim=(-0.02, 0.38),
       xticks=[0, 0.1, 0.2, 0.3], xticklabels=["0", "0.1", "0.2", "0.3"], yticks=[2, 4, 6, 8])
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0, fontsize=6)
title(ax, "e", "Spare heads when heads can fail")

# ---------------------------------------------------------------- f  what breaking looks like
ax = fig.add_subplot(bottom[2])
first = lambda kw: next(x for x in runs(**kw, n_rel=4) if get(x, "seed") == 0)
for kw, color, ls, label in ((dict(supply="prune"), K, "-", "Michel et al."),
                             (ONESHOT, G1, "-", "ZipLM-style"),
                             (RANDOM, G2, "--", "random"),
                             (RULE, RED, "-", "collateral")):
    x = first(kw)
    ax.plot(x["hist"]["val_step"], x["hist"]["val_loss"], color=color, ls=ls, lw=0.9, label=label)
ax.axhline(BAR, color=G2, lw=0.6, ls=":")
ax.text(3950, BAR * 1.3, "solved", fontsize=5.5, color=G1, ha="right", va="bottom")
ax.set(yscale="log", ylim=(3e-6, 2), xlim=(0, 4000), xticks=[0, 2000, 4000], xticklabels=["0", "2k", "4k"],
       xlabel="training step", ylabel="loss")
ax.minorticks_off()
ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0, fontsize=5.8)
title(ax, "f", "What breaking looks like ($k^*=4$)")

X_LETTER = AX_A.get_position().x0 - 0.085           # one column for the left-hand panel letters
for letter, axis in (("a", AX_A), ("b", AX_B), ("d", AX_D)):
    tr = offset_copy(blended_transform_factory(fig.transFigure, axis.transAxes), fig=fig, y=4, units="points")
    fig.text(X_LETTER, 1.0, letter, transform=tr, fontsize=7, fontweight="bold", va="baseline")
fig.text(X_LETTER, C_TITLE_Y, "c", transform=blended_transform_factory(fig.transFigure, AX_C.transData),
         fontsize=7, fontweight="bold", va="top")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_main.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/fig_main.png and figures/fig_main.pdf")
