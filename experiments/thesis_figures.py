"""Figures, tables and prose numbers for the thesis chapter (paper/thesis_chapter.tex).

Every number is read from results/*.pkl, results/dense/*.pkl, or from one dense 4-head
model trained here (cached in figures/thesis_heads.pkl) to measure what each head attends
to. Nothing is typed in. Run selection follows notebooks/walkthrough.ipynb: pick() pins
every swept knob, so one arm's runs cannot leak into another's.

  python experiments/thesis_figures.py            # uses the cache if present
  python experiments/thesis_figures.py --retrain  # retrain the dense 4-head model
"""
import argparse, glob, os, pickle, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from hemo.config import Cfg
from hemo.tasks import offsets_for, make_val, trivial_loss

FIG, PAPER = os.path.join(ROOT, "figures"), os.path.join(ROOT, "paper")
CACHE = os.path.join(FIG, "thesis_heads.pkl")

# categorical slots 1-3 of the reference palette, grey for the failing reference rule
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY, MUTED, INK, DARK, SURF = "#9c9a93", "#c9c7c0", "#52514e", "#0b0b0b", "#ffffff"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlecolor": DARK,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "lines.linewidth": 2, "lines.markersize": 6})

# ---------------------------------------------------------------- run selection
DEFAULT = Cfg()
RUNS = []
for path in glob.glob(os.path.join(ROOT, "results", "*.pkl")):
    with open(path, "rb") as f:
        RUNS.append(pickle.load(f))

MAIN = dict(task="multi_relation", steps=4000, d_k=32, demand="outnorm_ema", kappa_end=1.5,
            target_frac=0.02, delay=0, pool_beta=0.0, leak=0.0, n_territories=8,
            stall_gate=0.0, autoreg_gain=0.003, precondition=0.0, flow_exponent=4.0,
            price_frac=0.01, probe_every=25, taper=0, budget_hold_frac=0.25, prune_stop=1)
SIZES = [2, 3, 4, 6, 8]


def setting(run, key):
    return run["cfg"].get(key, getattr(DEFAULT, key))


def pick(**want):
    want = {**MAIN, **want}
    return [r for r in RUNS if all(setting(r, k) == v for k, v in want.items())]


def held_counts(run):
    ledger = run["hist"]["ledger"]
    start = int((setting(run, "budget_hold_frac") + setting(run, "budget_anneal_frac")) * len(ledger))
    return (ledger[start:] > 1e-9).sum(axis=1)


def heads_kept(run):
    return float(np.median(held_counts(run)))


def failed(run):
    return run["final_loss"] > 0.02 * run["trivial"]


def compute_used(run):
    ledger = run["hist"]["ledger"]
    heads = (ledger > 1e-9).sum(1).mean() / ledger.shape[1]
    probes = (run["hist"].get("n_probes", 0) * setting(run, "probe_batch")
              / (3 * setting(run, "batch_size")) / len(ledger))
    return heads + probes


def by_size(runs, key="n_rel"):
    g = defaultdict(list)
    for r in runs:
        g[setting(r, key)].append(heads_kept(r))
    return {k: float(np.mean(v)) for k, v in sorted(g.items())}


def score(runs):
    B = by_size(runs)
    ks = list(B)
    slope = np.polyfit(ks, [B[k] for k in ks], 1)[0] if len(ks) > 1 else float("nan")
    return dict(B=B, slope=slope, err=float(np.mean([abs(B[k] - k) for k in ks])),
                bad=sum(failed(r) for r in runs), n=len(runs),
                roles=float(np.mean([r["recovery"]["role_coverage"] for r in runs])),
                compute=float(np.mean([compute_used(r) for r in runs])))


def worst_after_solved(run):
    h, hold = run["hist"], int(setting(run, "budget_hold_frac") * setting(run, "steps"))
    bar = 0.02 * run["trivial"]
    solved = next((s for s, l in zip(h["val_step"], h["val_loss"]) if s >= hold and l < bar), None)
    after = [l for s, l in zip(h["val_step"], h["val_loss"]) if solved is not None and s >= solved]
    return max(after) / bar if after else float("nan")


def dense(R, H, seed=0):
    with open(os.path.join(ROOT, "results", "dense", f"dense_R{R}_H{H}_s{seed}.pkl"), "rb") as f:
        dd = pickle.load(f)
    steps = np.array(dd["hist"]["val_step"])
    return steps * H / 32 / 4000, np.array(dd["hist"]["val_loss"])


def curves(run):
    h = run["hist"]
    on = (h["ledger"] > 1e-9).sum(1)
    cost = on / h["ledger"].shape[1]
    if setting(run, "supply") == "local":
        hold = int(setting(run, "budget_hold_frac") * len(on))
        cost = cost.copy()
        cost[hold::setting(run, "probe_every")] += setting(run, "probe_batch") / (3 * setting(run, "batch_size"))
    spent = np.cumsum(cost) / len(on)
    return spent, spent[np.array(h["val_step"])], np.array(h["val_loss"]), on


ARM = {  # the arms the chapter reports, in presentation order
    "fixed threshold": dict(supply="threshold"),
    "feedback loop": dict(supply="autoreg"),
    "loop + stall gate": dict(supply="autoreg", stall_gate=0.02),
    "pruning baseline": dict(supply="prune"),
    "pruning, told k*": dict(supply="prune", prune_stop=0),
    "local rule": dict(supply="local", price_frac=0.03, taper=100, budget_hold_frac=0.1),
}


# ---------------------------------------------------------------- the dense 4-head model
def measure_heads(retrain):
    if os.path.exists(CACHE) and not retrain:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from hemo.train import train, pick_device
    cfg = Cfg(device="auto", seed=0)
    dev = pick_device(cfg)
    val = make_val(cfg, dev)
    R, N = cfg.n_rel, cfg.seq_len
    model, hist = train(cfg, val, dev, hemo=False, num_heads=R,
                        steps=int(cfg.steps * cfg.scratch_mult))
    with torch.no_grad():
        X, Y, aux = val[0][:1024], val[1][:1024], val[3]
        _, attn, _ = model.heads(X, Y)                   # (batch, head, query, slot)
        p = aux["p"][:1024]
        # M[h, p, j]: mean attention from a query carrying position p to memory slot j
        M = torch.zeros(R, N, N, device=dev)
        for q in range(N):
            M[:, q] = attn.permute(1, 0, 2, 3)[:, p == q].mean(1)
    out = dict(M=M.cpu().numpy(), offsets=offsets_for(cfg), N=N, R=R,
               final_loss=hist["val_loss"][-1], trivial=trivial_loss(val))
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def mixing(M, offsets):
    # A[h, r]: weight head h puts on offset r, averaged over query positions
    N = M.shape[1]
    return np.array([[np.mean([M[h, q, (q + d) % N] for q in range(N)]) for d in offsets]
                     for h in range(M.shape[0])])


# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("wrote figures/" + name)


def fig_task(stairs):
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    N, p, offs = 16, 8, offsets_for(Cfg())
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    ax.set(xlim=(-3.2, N), ylim=(-2.9, 2.4))
    ax.axis("off")
    ax.set_title("a   One query, four answers", pad=2)
    ax.text(-3.1, 1.6, "query\ncarries p", fontsize=8, color=INK, va="center")
    ax.add_patch(Rectangle((p - 0.4, 1.25), 0.8, 0.7, color=DARK))
    ax.text(p, 1.6, f"{p}", color="white", ha="center", va="center", fontweight="bold")
    ax.text(-3.1, 0, "memory\nslot j", fontsize=8, color=INK, va="center")
    for j in range(N):
        r = next((i for i, d in enumerate(offs) if (p + d) % N == j), None)
        ax.add_patch(Rectangle((j - 0.42, -0.35), 0.84, 0.7, facecolor="#f1f0ec",
                               edgecolor=cols[r] if r is not None else MUTED,
                               lw=2 if r is not None else 0.8))
        ax.text(j, 0, str(j), ha="center", va="center", fontsize=7.5, color=INK)
    for r, d in enumerate(offs):
        j = (p + d) % N
        ax.add_patch(FancyArrowPatch((p, 1.22), (j, 0.4), connectionstyle=f"arc3,rad={0.25 if j < p else -0.25}",
                                     arrowstyle="-|>", mutation_scale=9, color=cols[r], lw=1.4))
        x0 = -0.5 + r * 4.0
        ax.add_patch(Rectangle((x0 + 0.1, -2.6), 3.7, 1.1, color=cols[r]))
        ax.text(x0 + 1.95, -2.05, f"block {r + 1} = c[{j}]\n({p} + {d}) mod {N} = {j}",
                color="white", ha="center", va="center", fontsize=7.5)
    ax.text(-3.1, -2.05, "target:\nR = 4 blocks", fontsize=8, color=INK, va="center")

    ax = fig.add_subplot(gs[1])
    ax.set_title("b   Each missing head costs 1/R of the loss", pad=4)
    x = np.linspace(0, 1.5, 50)
    ax.plot(x, np.maximum(0, 1 - x), color=GREY, lw=1.5, ls="--", zorder=1)
    ax.text(0.72, 0.40, "dashed: predicted loss = (1 - k/R) x trivial", color=INK, fontsize=8)
    marks = {2: "o", 3: "s", 4: "D", 6: "^", 8: "v"}
    for R, pts in sorted(stairs.items()):
        k, l = np.array(pts).T
        keep = k / R <= 1.5
        ax.plot(k[keep] / R, l[keep], marks[R], ls="none", color=BLUE, mfc=SURF, mew=1.6,
                ms=7, label=f"R = {R}")
    ax.set(xlabel="heads k as a fraction of offsets R", ylabel="loss / loss of a model\nthat learned nothing",
           xlim=(0, 1.55), ylim=(-0.05, 1.05), xticks=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=7.5, loc="upper right", handletextpad=0.2, columnspacing=0.9)
    save(fig, "thesis_fig1_task.png")


def fig_heads(H):
    M, offs, N = H["M"], H["offsets"], H["N"]
    A = mixing(M, offs)
    fig = plt.figure(figsize=(7.4, 2.2))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.55, 1.25], wspace=0.3)
    vmax = M.max()
    for h in range(M.shape[0]):
        ax = fig.add_subplot(gs[h])
        ax.imshow(M[h], cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"head {h + 1}", fontsize=8.5, loc="center")
        ax.set_xticks([0, 8, 15]); ax.set_yticks([0, 8, 15])
        ax.tick_params(labelsize=7, length=2)
        ax.set_xlabel("memory slot j", fontsize=7.5)
        if h == 0:
            ax.set_ylabel("query position p", fontsize=7.5)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(MUTED)
    fig.text(0.125, 0.93, "a   Where each head looks: stripes at j = p + 1, 5, 9, 13",
             fontweight="bold", fontsize=9.5, color=DARK)
    ax = fig.add_subplot(gs[5])
    ax.imshow(A, cmap="Blues", vmin=0, vmax=A.max() * 1.15)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if A[i, j] > 0.55 * A.max() else DARK)
    ax.set_xticks(range(4), [f"+{d}" for d in offs], fontsize=7.5)
    ax.set_yticks(range(4), [f"{h + 1}" for h in range(4)], fontsize=7.5)
    ax.set_ylabel("head", fontsize=7.5)
    ax.set_xlabel("offset", fontsize=7.5)
    sv = np.linalg.svd(A, compute_uv=False)
    ax.set_title(f"b   Mixing matrix A, rank {np.linalg.matrix_rank(A, tol=1e-3)}", fontsize=9.5, pad=14)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "thesis_fig2_heads.png")
    return A, sv


def fig_supply():
    run = pick(n_rel=4, seed=0, **ARM["local rule"])[0]
    g = run["hist"]["ledger"]                         # (steps, heads)
    fig = plt.figure(figsize=(7.2, 3.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
    left = outer[0].subgridspec(3, 1, hspace=0.55)
    on = (g > 1e-9).sum(1)
    first = int(np.argmax(on < 32))
    moments = [0, first + int(np.argmax(on[first:] <= 16)), len(g) - 1]
    order = np.argsort(-g[-1], kind="stable")
    for i, t in enumerate(moments):
        ax = fig.add_subplot(left[i])
        ax.bar(np.arange(32), g[t, order], width=0.8, color=BLUE if i == 2 else GREY)
        ax.set(ylim=(0, 9.5), yticks=[0, 3, 6], xticks=[0, 8, 16, 24, 31] if i == 2 else [])
        ax.tick_params(labelsize=7)
        ax.text(31, 9.4, f"step {t}: {int(on[t])} heads on, sum of g = {g[t].sum():.0f}",
                ha="right", va="top", fontsize=7, color=INK)
        if i == 0:
            ax.set_title("a   Shares at three moments")
        if i == 1:
            ax.set_ylabel("supply g$_h$")
    ax.set_xlabel("head (sorted by final share)")
    ax = fig.add_subplot(outer[1])
    im = ax.imshow(g[:, order].T, aspect="auto", cmap="Blues", interpolation="nearest",
                   extent=(0, len(g), 31.5, -0.5))
    ax.set(xlabel="training step", ylabel="head (sorted)")
    ax.set_title("b   Supply over training, local rule, k* = 4")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("g$_h$", fontsize=8); cb.outline.set_visible(False)
    save(fig, "thesis_fig3_supply.png")
    return dict(step_mid=moments[1], final_on=int(on[-1]), final_share=float(g[-1].max()))


def fig_count():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(wspace=0.32, width_ratios=[1.3, 1]))
    ax = axes[0]
    ax.plot([1.5, 8.5], [1.5, 8.5], ls=":", color=GREY, lw=1.2)
    ax.text(7.0, 8.05, "B = k*", color=GREY, fontsize=8, rotation=33)
    show = [("fixed threshold", GREY, -0.18), ("loop + stall gate", BLUE, -0.06),
            ("pruning baseline", ORANGE, 0.06), ("local rule", AQUA, 0.18)]
    rng = np.random.default_rng(0)
    for name, c, dx in show:
        rs = pick(**ARM[name])
        for r in rs:
            ax.plot(setting(r, "n_rel") + dx + rng.uniform(-0.03, 0.03), heads_kept(r), "o",
                    color=c, ms=3.2, alpha=0.45, mew=0)
        B = by_size(rs)
        ks = list(B)
        ax.plot(np.array(ks) + dx, [B[k] for k in ks], "-o", color=c, lw=1.8, ms=5.5,
                mfc=SURF, mew=1.6, label=name)
    ax.set(xlabel="heads the task needs, k*", ylabel="heads kept, B", xticks=SIZES,
           xlim=(1.4, 8.7), ylim=(0, 11))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("a   Heads kept against heads needed")
    ax = axes[1]
    rows = []
    for R in SIZES:
        for r in pick(supply="threshold", n_rel=R):
            dm = np.asarray(r["ablation"]["score_demand"])
            z = (dm - dm.mean()) / dm.std(ddof=1)
            rows.append((R, int((z > setting(r, "kappa_end")).sum()), r["recovery"]["n_perfused"]))
    rows = np.array(rows)
    ax.plot([0, 7], [0, 7], ls=":", color=GREY, lw=1.2)
    sc = ax.scatter(rows[:, 1] + rng.uniform(-0.1, 0.1, len(rows)), rows[:, 2] + rng.uniform(-0.1, 0.1, len(rows)),
                    c=rows[:, 0], cmap="Blues", vmin=0, vmax=8, s=34, edgecolor=DARK, lw=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, ticks=SIZES)
    cb.set_label("k*", fontsize=8); cb.outline.set_visible(False)
    ax.set(xlabel="heads above mean + 1.5 std\n(counted from demand alone)", ylabel="heads kept",
           xlim=(0, 7), ylim=(0, 7))
    ax.set_title("b   Fixed threshold: count set by\nthe demand shape, not by k*")
    save(fig, "thesis_fig4_count.png")
    return rows


def fig_target():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.2))
    targets = [0.005, 0.02, 0.05]
    err = {0.0: [], 0.02: []}
    for ax, R in zip(axes, [2, 4, 8]):
        for gate, c, name in [(0.0, ORANGE, "plain loop"), (0.02, BLUE, "loop + stall gate")]:
            B = [np.mean([heads_kept(r) for r in pick(supply="autoreg", stall_gate=gate, target_frac=t, n_rel=R)])
                 for t in targets]
            err[gate] += [abs(b - R) for b in B]
            ax.plot(targets, B, "-o", color=c, mfc=SURF, mew=1.6, label=name)
        ax.axhline(R, color=GREY, ls=":", lw=1.2)
        ax.set(xscale="log", xticks=targets, xticklabels=["0.005", "0.02", "0.05"], ylim=(0, 17),
               xlim=(0.0035, 0.07))
        ax.minorticks_off()
        ax.set_title(f"k* = {R}", loc="center", fontweight="normal")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("heads kept")
    axes[1].set_xlabel("loss target (fraction of the trivial loss)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("The stall gate makes the count insensitive to the loss target", x=0.02, ha="left",
                 fontweight="bold", fontsize=9.5, y=1.02)
    save(fig, "thesis_fig5_target.png")
    return float(np.mean(err[0.0])), float(np.mean(err[0.02]))


def fig_compute():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(wspace=0.3, width_ratios=[1.3, 1]))
    ax = axes[0]
    x, y = dense(4, 32)
    ax.plot(x, y, color=DARK, lw=1.6, label="dense, 32 heads")
    x, y = dense(4, 4)
    ax.plot(x, y, color=GREY, lw=1.6, ls="--", label="4 heads from scratch (oracle)")
    three = [("pruning baseline", ORANGE), ("loop + stall gate", BLUE), ("local rule", AQUA)]
    for name, c in three:
        _, xs, loss, _ = curves(pick(n_rel=4, seed=0, **ARM[name])[0])
        ax.plot(xs, loss, color=c, lw=1.4, label=name)
    bar = 0.02 * pick(supply="prune", n_rel=4)[0]["trivial"]
    ax.axhline(bar, color=GREY, lw=0.8, ls=":")
    ax.text(1.0, bar * 1.4, "solved", ha="right", fontsize=7.5, color=GREY)
    ax.set(yscale="log", ylim=(5e-6, 2), xlim=(0, 1.02), xlabel="compute spent (1 = one dense run)",
           ylabel="validation loss")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.36))
    ax.set_title("a   Loss against compute, k* = 4, seed 0")
    ax = axes[1]
    for name, c in three:
        rs = [r for r in pick(**ARM[name]) if setting(r, "n_rel") in (2, 4, 8)]
        ax.scatter([compute_used(r) for r in rs], [worst_after_solved(r) for r in rs],
                   color=c, s=30, alpha=0.8, edgecolor=SURF, lw=0.6, label=name)
    ax.axhline(1, color=GREY, ls=":", lw=1)
    ax.text(0.305, 1.25, "solved bar", fontsize=7.5, color=GREY)
    ax.set(yscale="log", xlabel="compute used (1 = one dense run)",
           ylabel="worst loss after first solved\n(multiple of the bar)", xlim=(0.3, 0.56))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right", handletextpad=0.2)
    ax.set_title("b   Every run, k* = 2, 4, 8")
    save(fig, "thesis_fig6_compute.png")


# ---------------------------------------------------------------- tables and numbers
def fmt(x, d=2):
    return f"{x:.{d}f}"


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return m + r" \times 10^{" + str(int(e)) + "}"


def write_tables(S):
    lines = [r"\begin{tabular}{l ccccc c c c}", r"\toprule",
             r"& \multicolumn{5}{c}{heads kept $B$ when the task needs $k^\star$} & & & \\",
             r"\cmidrule(lr){2-6}",
             r"rule & 2 & 3 & 4 & 6 & 8 & miss (heads) & failed runs & compute \\", r"\midrule"]
    for name, s in S.items():
        cells = " & ".join(fmt(s["B"][k], 1) if k in s["B"] else "--" for k in SIZES)
        label = name.replace("k*", r"$k^\star$")
        lines.append(f"{label} & {cells} & {fmt(s['err'])} & {s['bad']}/{s['n']} & {fmt(s['compute'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(PAPER, "thesis_count_table.tex"), "w") as f:
        f.write("% generated by experiments/thesis_figures.py, do not edit\n" + "\n".join(lines) + "\n")
    print("wrote paper/thesis_count_table.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    print(f"{len(RUNS)} runs loaded")

    stairs, triv4 = defaultdict(list), None
    seen = set()
    for r in RUNS:
        red = r.get("redundancy")
        if setting(r, "task") != "multi_relation" or not red:
            continue
        R = setting(r, "n_rel")
        key = (R, setting(r, "seed"), tuple(red["ks"]))
        if key in seen:
            continue
        seen.add(key)
        stairs[R] += [(k, l / red["trivial"]) for k, l in zip(red["ks"], red["curve"])]
        if R == 4:
            triv4, curve4 = red["trivial"], dict(zip(red["ks"], red["curve"]))
    pred_err = max(abs(l - max(0, 1 - k / R)) for R, pts in stairs.items() for k, l in pts)

    heads = measure_heads(args.retrain)
    fig_task(stairs)
    A, sv = fig_heads(heads)
    sup = fig_supply()
    rows = fig_count()
    e_plain, e_gate = fig_target()
    fig_compute()

    S = {name: score(pick(**kw)) for name, kw in ARM.items()}
    write_tables(S)

    slow = score([r for r in pick(supply="autoreg", autoreg_gain=0.0003)])
    eps = {e: score([r for r in pick(supply="autoreg", stall_gate=e) if setting(r, "n_rel") in (2, 4, 8)])
           for e in (0.005, 0.02, 0.08)}
    local = [r for r in pick(**ARM["local rule"]) if setting(r, "n_rel") in (2, 4, 8)]
    wa = {n: float(np.median([worst_after_solved(r) for r in pick(**ARM[n]) if setting(r, "n_rel") in (2, 4, 8)]))
          for n in ("loop + stall gate", "pruning baseline", "local rule")}
    same, full = 0, 0
    for r in local:
        spent, _, loss, _ = curves(r)
        dx, dl = dense(setting(r, "n_rel"), 32, setting(r, "seed"))
        same += loss[-1] < np.exp(np.interp(spent[-1], dx, np.log(dl)))
        full += loss[-1] <= dl[-1]
    mr = [r for r in RUNS if setting(r, "task") == "multi_relation" and "ablation" in r]
    rho = {s: float(np.nanmean([r["ablation"][f"rho_{s}"] for r in mr])) for s in ("demand", "qnorm", "neg_entropy")}
    qsa = [r for r in RUNS if setting(r, "task") == "qsa_cross" and setting(r, "supply") == "autoreg"]

    N = {
        "LossOne": fmt(curve4[1], 3), "LossTwo": fmt(curve4[2], 3), "LossFour": sci(curve4[4]),
        "Triv": fmt(triv4, 3), "PredOne": fmt(0.75 * triv4, 3), "PredTwo": fmt(0.5 * triv4, 3),
        "StairMaxErr": fmt(pred_err, 3), "StairRuns": str(len(seen)),
        "MixMin": fmt(sv[-1]), "MixOnTarget": fmt(A.sum(1).min()), "MixRank": str(np.linalg.matrix_rank(A, tol=1e-3)),
        "DenseFourLoss": sci(heads['final_loss']),
        "FinalShare": fmt(sup["final_share"], 1), "FinalOn": str(sup["final_on"]),
        "ThrSlope": fmt(S["fixed threshold"]["slope"]), "ThrErr": fmt(S["fixed threshold"]["err"]),
        "ThrMatch": f"{int((rows[:, 1] == rows[:, 2]).sum())}/{len(rows)}",
        "LoopErr": fmt(S["feedback loop"]["err"]), "LoopBad": f"{S['feedback loop']['bad']}/{S['feedback loop']['n']}",
        "GateBad": f"{S['loop + stall gate']['bad']}/{S['loop + stall gate']['n']}",
        "PruneErr": fmt(S["pruning baseline"]["err"]), "PruneBad": f"{S['pruning baseline']['bad']}/{S['pruning baseline']['n']}",
        "LocalErr": fmt(S["local rule"]["err"]), "LocalBad": f"{S['local rule']['bad']}/{S['local rule']['n']}",
        "LocalTwo": fmt(S["local rule"]["B"][2], 0), "LocalCompute": fmt(S["local rule"]["compute"]),
        "GateCompute": fmt(S["loop + stall gate"]["compute"]), "PruneCompute": fmt(S["pruning baseline"]["compute"]),
        "TargetPlain": fmt(e_plain), "TargetGate": fmt(e_gate),
        "SlowErr": fmt(slow["err"]), "EpsErrMax": fmt(max(s["err"] for s in eps.values())),
        "WorstGate": fmt(wa["loop + stall gate"], 1), "WorstPrune": fmt(wa["pruning baseline"], 1),
        "WorstLocal": fmt(wa["local rule"], 1),
        "BelowSame": f"{same}/{len(local)}", "BelowFull": f"{full}/{len(local)}",
        "RhoDemand": f"{rho['demand']:+.2f}", "RhoQnorm": f"{rho['qnorm']:+.2f}", "RhoEntropy": f"{rho['neg_entropy']:+.2f}",
        "QsaBad": f"{sum(failed(r) for r in qsa)}/{len(qsa)}",
        "NRuns": str(len(RUNS)),
    }
    with open(os.path.join(PAPER, "thesis_numbers.tex"), "w") as f:
        f.write("% generated by experiments/thesis_figures.py, do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\th{k}}}{{{v}}}\n")
    print("wrote paper/thesis_numbers.tex")
    for k, v in N.items():
        print(f"  {k:<14} {v}")


if __name__ == "__main__":
    main()
