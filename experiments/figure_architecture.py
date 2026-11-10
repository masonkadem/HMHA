"""Architecture diagrams of the two models, in block-diagram style.

  a  Retrieval model: one cross-attention layer, 32 heads, each with a valve, read out by W_O.
  b  Induction model: two attention-only layers, 8 + 8 heads, each with a valve, on a residual stream.

Valves are drawn red and filled when open at the end of training, white when closed: the counts
(4 of 32; 2 + 1 of 8 + 8) are the rule's results; which drawn head is open is illustrative.

Shapes: a widening trapezoid is a linear map to a wider vector, a narrowing one to a narrower vector,
and a box keeps the width. (A triangle would mean a map down to one number; neither model has one.)

  python experiments/figure_architecture.py            ->  figures/fig_architecture.png and .pdf
  python experiments/figure_architecture.py --dims     ->  figures/fig_architecture_dims.png and .pdf
                                                           (shapes plus the size of what goes in and out)
  python experiments/figure_architecture.py --notext   ->  figures/fig_architecture_notext.png and .pdf
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle

NOTEXT = "--notext" in sys.argv                      # shapes only
DIMS = "--dims" in sys.argv                          # shapes plus sizes

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RED, K, G1, G2 = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "savefig.dpi": 300, "pdf.fonttype": 42})
LW, STREAM = 1.0, 2.6


def say(ax, x, y, s, **kw):
    """Descriptive text: only in the full version."""
    if not (NOTEXT or DIMS):
        ax.text(x, y, s, **kw)


def size(ax, x, y, full, short, **kw):
    """A size label: the full wording, just the size with --dims, nothing with --notext."""
    if NOTEXT:
        return
    if DIMS:
        kw = {**kw, "fontsize": 7.5, "color": K}
    ax.text(x, y, short if DIMS else full, **kw)


def arrow(ax, x0, y0, x1, y1, color=K):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=LW, mutation_scale=7, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, lw=LW, color=K):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=1)


def plus(ax, x, y, r=0.24):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=K, lw=LW, zorder=4))
    ax.plot([x - r * 0.65, x + r * 0.65], [y, y], color=K, lw=LW, zorder=5)
    ax.plot([x, x], [y - r * 0.65, y + r * 0.65], color=K, lw=LW, zorder=5)


def trapezoid(ax, cx, y_top, h, w_top, w_bot):
    ax.add_patch(Polygon([(cx - w_top / 2, y_top), (cx + w_top / 2, y_top), (cx + w_bot / 2, y_top - h),
                          (cx - w_bot / 2, y_top - h)], closed=True, facecolor="white", edgecolor=K, lw=LW, zorder=3))


def heads(ax, xs, y_top, open_set, h=0.95, w=0.34):
    """Head bars with a valve under each; returns the y where the outputs leave the valves."""
    for i, x in enumerate(xs):
        on = i in open_set
        ax.add_patch(Rectangle((x - w / 2, y_top - h), w, h, facecolor="white", edgecolor=K if on else G2,
                               lw=LW, ls="-" if on else (0, (2, 1.5)), zorder=3))
        vy = y_top - h - 0.42
        line(ax, [x, x], [y_top - h, vy + 0.3], color=K if on else G2)
        ax.add_patch(Rectangle((x - 0.1, vy), 0.2, 0.3, facecolor=RED if on else "white",
                               edgecolor=RED if on else G2, lw=0.8, zorder=3))
    return y_top - h - 0.42


# ==================================================================================== figure
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 6.4), gridspec_kw=dict(wspace=0.08))
for ax in (axa, axb):
    ax.set(xlim=(0, 10), ylim=(0.0, 15.4), aspect="equal")
    ax.axis("off")


def panel(ax, letter, name):
    if not NOTEXT:
        ax.text(0.1, 15.3, rf"$\mathbf{{{letter}}}$" + ("" if DIMS else f"   {name}"), fontsize=9, va="top")


# ---------------------------------------------------------------- a  retrieval model
ax = axa
panel(ax, "a", "Retrieval model")
xq, xm = 2.0, 8.0
size(ax, xq, 13.35, "query $X$\n16 × 128", "16 × 128", ha="center", va="bottom")
size(ax, xm, 13.35, "memory $Y$\n16 × 128", "16 × 128", ha="center", va="bottom")
arrow(ax, xq, 13.3, xq, 11.25)
arrow(ax, xm, 13.3, xm, 11.25)
line(ax, [xq, xq], [11.25, 10.95], lw=STREAM)
line(ax, [xm, xm], [11.25, 10.95], lw=STREAM)
line(ax, [xq, xm], [10.95, 10.95])
say(ax, 5.0, 11.15, "cross-attention", ha="center", va="bottom")
say(ax, 5.0, 11.75, "$Q$ from $X$;  $K$, $V$ from $Y$", ha="center", va="bottom", fontsize=7, color=G1)
xs = [2.5, 3.1, 3.7, 4.3, 5.7, 6.3, 6.9, 7.5]
for x in xs:
    arrow(ax, x, 10.95, x, 10.45)
ax.plot([4.75, 5.0, 5.25], [9.95] * 3, "o", ms=1.8, color=G1, zorder=5)   # the 24 heads not drawn
y_out = heads(ax, xs, 10.45, open_set={0, 3, 4, 7})
ycol = y_out - 0.35
for x in xs:
    line(ax, [x, x], [y_out, ycol], color=G2)
line(ax, [xs[0], xs[-1]], [ycol, ycol])
size(ax, 8.05, 9.95, "32 heads\n$d_k$ = 32", "32 × 32", ha="left", va="center", fontsize=7)
say(ax, 8.05, 8.85, "valve $g_h$", ha="left", va="center", fontsize=7, color=RED)
arrow(ax, 5.0, ycol, 5.0, 7.15)
trapezoid(ax, 5.0, 7.15, 1.05, 5.0, 1.4)               # W_O: 1024 -> 64, narrowing
size(ax, 7.75, 6.6, "$W_O$\n1024 → 64", "1024 → 64", ha="left", va="center", fontsize=7)
arrow(ax, 5.0, 6.1, 5.0, 5.25)
size(ax, 5.0, 5.15, "4 answers\n4 × 16 = 64 numbers", "16 × 64", ha="center", va="top")
say(ax, 5.0, 3.35, "one layer · no residual, MLP or LayerNorm\n461,888 parameters\n"
    "after training: 4 of 32 valves open", ha="center", va="top", fontsize=7, color=G1, linespacing=1.4)

# ---------------------------------------------------------------- b  induction model
ax = axb
panel(ax, "b", "Induction model")
xs_ = 2.4                                            # the residual stream
size(ax, xs_, 13.35, "tokens\n40 positions, vocabulary 64", "40 tokens", ha="center", va="bottom")
arrow(ax, xs_, 13.3, xs_, 12.85)
trapezoid(ax, xs_, 12.85, 0.85, 1.0, 2.0)
size(ax, xs_ + 1.2, 12.4, "token + position\nembedding, $d$ = 64", "40 × 64", ha="left", va="center", fontsize=7)
line(ax, [xs_, xs_], [12.0, 2.75], lw=STREAM)
say(ax, xs_ - 0.35, 7.0, "residual stream", rotation=90, ha="right", va="center", fontsize=7, color=G1)

hx = [4.0 + 0.56 * i for i in range(8)]


def layer(y_branch, open_set, title, note):
    line(ax, [xs_, hx[-1]], [y_branch, y_branch])
    for x in hx:
        arrow(ax, x, y_branch, x, y_branch - 0.4)
    y_out = heads(ax, hx, y_branch - 0.4, open_set)
    ycol = y_out - 0.3
    for x in hx:
        line(ax, [x, x], [y_out, ycol], color=G2)
    line(ax, [hx[-1], hx[0]], [ycol, ycol])
    line(ax, [hx[0], xs_ + 0.24], [ycol, ycol])
    arrow(ax, hx[0], ycol, xs_ + 0.24, ycol)
    plus(ax, xs_, ycol)
    size(ax, hx[0] - 0.2, y_branch + 0.2, title, "8 × 16", ha="left", va="bottom", fontsize=7)
    say(ax, hx[-1] + 0.25, y_branch - 0.9, note, ha="left", va="center", fontsize=6.5, color=RED)
    return ycol


layer(11.25, {1, 5}, "layer 1: 8 heads, $d_{head}$ = 16", "2 kept\nprevious-\ntoken")
layer(7.55, {3}, "layer 2: 8 heads, $d_{head}$ = 16", "1 kept\ninduction")
trapezoid(ax, xs_, 3.6, 1.0, 1.6, 1.6)                 # unembedding: 64 -> 64, a box
size(ax, xs_ + 1.05, 3.1, "unembedding\n64 → 64", "64 → 64", ha="left", va="center", fontsize=7)
arrow(ax, xs_, 2.55, xs_, 1.95)
size(ax, xs_, 1.85, "next token", "40 × 64", ha="center", va="top")
say(ax, 6.9, 2.15, "attention only · no MLP\nor LayerNorm\n76,352 parameters\n"
    "after training: 2 + 1\nof 8 + 8 valves open", ha="center", va="top", fontsize=7, color=G1, linespacing=1.4)

name = "fig_architecture_notext" if NOTEXT else "fig_architecture_dims" if DIMS else "fig_architecture"
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
print(f"wrote figures/{name}.png and .pdf")
