"""Spreadsheet-style "L-shape" picture of the whole computation on a tiny version of the task:
4 memory slots, 4 queries (one per position), 2 heads (look 1 ahead, look 3 ahead), every
matrix written out. In each product the right factor sits above, the left factor to the
left, and the result where they meet.

  python experiments/figure_matrices.py   ->  figures/fig_matrices.png and .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N, d = 4, 4
content = np.array([3.0, 1.0, 4.0, 2.0])                 # what each slot stores
SCALE = 12.0                                             # sharpness of the query weights


def shift(k):
    """P[i, p] = 1 when i = p + k (mod N): moves position p to p + k."""
    P = np.zeros((N, N))
    for p in range(N):
        P[(p + k) % N, p] = 1
    return P


X = np.eye(N)                                            # queries: column t = one-hot of position t
Y = np.vstack([np.eye(N), content])                      # memory: column j = [one-hot j ; content j]
heads = {}
for h, k in ((1, 1), (2, 3)):
    Wq = SCALE * shift(k)                                # d x N
    Wk = np.hstack([np.eye(N), np.zeros((N, 1))])        # d x (N + 1): reads the position part
    Wv = np.array([[0, 0, 0, 0, 1.0]])                   # 1 x (N + 1): reads the content
    Q, K, V = Wq @ X, Wk @ Y, Wv @ Y                     # d x N, d x N, 1 x N
    S = K.T @ Q / np.sqrt(d)                             # slots x queries
    A = np.exp(S) / np.exp(S).sum(0, keepdims=True)      # softmax over slots, per query
    O = V @ A                                            # 1 x queries: the head's blend
    heads[h] = dict(k=k, Wq=Wq, Wk=Wk, Wv=Wv, Q=Q, K=K, V=V, S=S, A=A, O=O)
target = np.vstack([content[(np.arange(N) + 1) % N], content[(np.arange(N) + 3) % N]])

YEL, PINK, OUT, GRID = "#fbe7b5", "#f6d3cc", "#fff176", "#9a9a9a"


def mat(ax, M, x, y, title=None, color="white", rows=None, cols=None, fmt="{:g}", fs=6.2, digits=2):
    """Draw M with its top-left corner at (x, y); one cell = 1 x 0.7 units."""
    r, c = M.shape
    for i in range(r):
        for j in range(c):
            ax.add_patch(Rectangle((x + j, y + 0.7 * i), 1, 0.7, facecolor=color, edgecolor=GRID, lw=0.4))
            v = M[i, j]
            ax.text(x + j + 0.5, y + 0.7 * i + 0.36, fmt.format(round(v, digits) if abs(v - round(v)) > 1e-9 else int(round(v))),
                    ha="center", va="center", fontsize=fs)
    if title:
        ax.text(x, y - (0.45 if cols else 0.12), title, fontsize=6.8, va="bottom")
    if rows:
        for i, lab in enumerate(rows):
            ax.text(x - 0.15, y + 0.7 * i + 0.36, lab, ha="right", va="center", fontsize=5.5, color="#555555")
    if cols:
        for j, lab in enumerate(cols):
            ax.text(x + j + 0.5, y - 0.05, lab, ha="center", va="bottom", fontsize=5.5, color="#555555")
    return x + c, y + 0.7 * r


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333333",
                                                                  mutation_scale=6))


fig, ax = plt.subplots(figsize=(11, 13.5))
ax.set_xlim(-1, 40)
ax.set_ylim(42.5, -1.5)
ax.axis("off")
pos = [f"p{i}" for i in range(N)]
slots = [f"s{j}" for j in range(N)]
feat = [f"pos {i}" for i in range(N)] + ["content"]

# ---------------------------------------------------------------- inputs
ax.text(0, -1.0, "Inputs", fontsize=9, fontweight="bold")
mat(ax, X, 3, 0.6, "X: queries", YEL, rows=[f"pos {i}" for i in range(N)], cols=pos)
mat(ax, Y, 12, 0.6, "Y: memory", YEL, rows=feat, cols=slots)
ax.text(19.5, 1.6, "task: query at p must return\n  content of slot p+1 (head 1's job)\n  content of slot p+3 (head 2's job)",
        fontsize=6.5, va="top")
mat(ax, target, 31, 0.6, "target (correct answers)", OUT, rows=["slot p+1", "slot p+3"], cols=pos)

# ---------------------------------------------------------------- heads
for row, h in enumerate((1, 2)):
    H = heads[h]
    top = 5.6 + row * 16.2
    ax.add_patch(FancyBboxPatch((-0.6, top - 0.9), 40.2, 15.6, boxstyle="square,pad=0", fill=False, lw=1.4))
    ax.text(-0.3, top - 0.45, f"Head {h}  (looks {H['k']} slot{'s' if H['k'] > 1 else ''} ahead)", fontsize=9, fontweight="bold", va="center")
    # linear part
    ax.add_patch(FancyBboxPatch((0.2, top + 0.4), 16.6, 13.6, boxstyle="square,pad=0", fill=False, lw=0.6, ls="--"))
    ax.text(0.5, top + 0.85, "Linear: weights x inputs", fontsize=7)
    y0 = top + 2.2
    mat(ax, H["Wq"], 2, y0, "Wq (d x 4)", PINK)
    arrow(ax, 6.3, y0 + 1.4, 9.7, y0 + 1.4); ax.text(7.0, y0 + 1.1, "x X", fontsize=6)
    mat(ax, H["Q"], 10, y0, "Q = Wq X (d x 4)", cols=pos)
    y1 = y0 + 3.9
    mat(ax, H["Wk"], 1, y1, "Wk (d x 5)", PINK)
    arrow(ax, 6.3, y1 + 1.4, 9.7, y1 + 1.4); ax.text(7.0, y1 + 1.1, "x Y", fontsize=6)
    mat(ax, H["K"], 10, y1, "K = Wk Y (d x 4)", cols=slots)
    y2 = y1 + 4.1
    mat(ax, H["Wv"], 1, y2, "Wv (1 x 5)", PINK)
    arrow(ax, 6.3, y2 + 0.35, 9.7, y2 + 0.35); ax.text(7.0, y2 + 0.05, "x Y", fontsize=6)
    mat(ax, H["V"], 10, y2, "V = Wv Y (1 x 4)", cols=slots)
    ax.text(1, y2 + 1.6, "Wq moves position p to p+%d,\nso Q for query p points at slot p+%d" % (H["k"], H["k"]),
            fontsize=6, color="#555555", va="top")

    # scaled dot product, L-shaped
    ax.add_patch(FancyBboxPatch((17.4, top + 0.4), 22.0, 13.6, boxstyle="square,pad=0", fill=False, lw=0.6,
                                ls="--", edgecolor="#7b52ab"))
    ax.text(17.7, top + 0.85, "Scaled dot product (L-shape: right factor above, left factor to the left)", fontsize=7)
    qx, qy = 24, top + 1.9
    mat(ax, H["Q"], qx, qy, "Q", cols=pos)
    sy = qy + 3.4
    mat(ax, H["K"].T, 19, sy, "K^T (4 x d)", rows=slots)
    mat(ax, H["S"], qx, sy, "S = K^T Q / sqrt(d)", rows=None)
    mat(ax, H["A"], 30.5, sy, "softmax (columns sum to 1)", cols=pos, digits=3, fs=5.6)
    arrow(ax, 28.2, sy + 1.4, 30.3, sy + 1.4)
    oy = sy + 3.9
    mat(ax, H["V"], 25.3, oy, "V (1 x 4)")
    mat(ax, H["O"], 30.5, oy, f"head {h} output O = V x softmax", OUT)
    ax.text(19, oy + 1.4, "the stripe: each query puts almost all its\nweight on one slot, so O = the content there",
            fontsize=6, color="#555555", va="top")

# ---------------------------------------------------------------- combine
top = 38.2
ax.add_patch(FancyBboxPatch((-0.6, top - 0.9), 40.2, 5.2, boxstyle="square,pad=0", fill=False, lw=1.4))
ax.text(-0.3, top - 0.45, "Combine: each head's output x its valve g_h (supply) x its output weight", fontsize=9,
        fontweight="bold", va="center")
O = np.vstack([heads[1]["O"], heads[2]["O"]])
for i, (g, name) in enumerate((((1, 1), "both valves open, g = (1, 1)"), ((1, 0), "valve 2 closed, g = (1, 0)"))):
    out = np.diag(g) @ O
    mat(ax, out, 2 + i * 13, top + 0.9, name, OUT, rows=["head 1", "head 2"] if i == 0 else None, cols=pos)
mat(ax, target, 28, top + 0.9, "target", OUT, cols=pos)
ax.text(28, top + 3.0, "with both valves open the output matches the target;\nclosing valve 2 removes head 2's row exactly",
        fontsize=6.2, color="#555555", va="top")

out = os.path.join(ROOT, "figures", "fig_matrices")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white", dpi=220)
print("wrote figures/fig_matrices.png and .pdf")
print("head outputs:\n", O.round(2), "\ntarget:\n", target)
