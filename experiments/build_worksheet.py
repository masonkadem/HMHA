"""By-hand worksheets in grid style: blank boxes to fill in, and an answer key.

  Sheet 1  One attention head on the tiny retrieval task (4 slots, 4 queries): Q, K, scores, a sharp
           softmax, V, and the output, which should equal the target.
  Sheet 2  Valuing a head: two heads, three tokens. Overlap matrix A = Z^T Z, its inverse, the best
           weights, removing head 2 with and without re-fitting, the one-inverse shortcut, the price
           decision, and the copy case.

Every answer is computed here (exact fractions for sheet 2), so the key always matches the math.

  python experiments/build_worksheet.py
      ->  figures/worksheet_1_attention.png / _answers.png
      ->  figures/worksheet_2_value.png / _answers.png
      ->  figures/worksheet_3_exact.png / _answers.png   (Taylor, Hessian, OBD and OBS)
"""
import os
from fractions import Fraction as Fr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAPER, INK, MUTED = "#fbf9f4", "#1b1b1b", "#6b6b6b"
FILL = {"data": "#fbe7a8", "weight": "#ffffff", "answer": "#ffffff", "result": "#fff7c2"}
EDGE = {"data": "#9a8a50", "weight": "#c0262d", "answer": "#555555", "result": "#9a8a50"}
HAND = "#8a1a7a"                                        # answers look handwritten, in purple
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "savefig.dpi": 220})


def fmt(v):
    if isinstance(v, Fr):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    return str(v)


def grid(ax, x, y, vals, kind, title=None, rows=None, cols=None, blank=False, cell=1.0, fs=11):
    """A matrix drawn as boxes, top-left corner at (x, y). blank=True leaves the boxes empty."""
    vals = np.atleast_2d(np.asarray(vals, dtype=object))
    R, C = vals.shape
    lw = 2.0 if kind == "weight" else 1.0
    for i in range(R):
        for j in range(C):
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                                   facecolor=FILL[kind], edgecolor=EDGE[kind], lw=lw if kind != "weight" else 0.8))
            if not blank:
                ax.text(x + (j + 0.5) * cell, y - (i + 0.5) * cell, fmt(vals[i, j]), ha="center", va="center",
                        fontsize=fs if len(fmt(vals[i, j])) < 4 else fs - 3,
                        color=HAND if kind == "answer" else INK)
    if kind == "weight":
        ax.add_patch(Rectangle((x, y - R * cell), C * cell, R * cell, fill=False, edgecolor=EDGE[kind], lw=2.2))
    if title:
        ax.text(x, y + 0.35, title, ha="left", va="bottom", fontsize=11, color=INK)
    if rows:
        for i, r in enumerate(rows):
            ax.text(x - 0.15, y - (i + 0.5) * cell, r, ha="right", va="center", fontsize=8.5, color=MUTED)
    if cols:
        for j, c in enumerate(cols):
            ax.text(x + (j + 0.5) * cell, y + 0.05, c, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    return x + C * cell, y - R * cell


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=10))


def note(ax, x, y, s, fs=9.5, color=MUTED):
    ax.text(x, y, s, ha="left", va="top", fontsize=fs, color=color, linespacing=1.35)


def page(w, h, title):
    fig = plt.figure(figsize=(w * 0.42, h * 0.42))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis("off")
    ax.text(w / 2, h - 0.6, title, ha="center", va="top", fontsize=20, color=INK)
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(ROOT, "figures", name), facecolor=PAPER)
    plt.close(fig)
    print("wrote figures/" + name)


# =====================================================================================================
# Sheet 1: one attention head
N = 4
content = np.array([3, 1, 4, 2])
Y = np.vstack([np.eye(N, dtype=int), content])          # memory: one-hot slot position + content row
X = np.eye(N, dtype=int)                                 # queries at positions 0..3, one per column
WQ = np.roll(np.eye(N, dtype=int), 1, axis=0)            # moves position p to p + 1
WK = np.hstack([np.eye(N, dtype=int), np.zeros((N, 1), dtype=int)])   # reads the slot position
WV = np.array([[0, 0, 0, 0, 1]])                         # reads the content
Q, K, V = WQ @ X, WK @ Y, WV @ Y
S = K.T @ Q
A = (S == S.max(axis=0, keepdims=True)).astype(int)      # sharp softmax: 1 on each column's largest
Z = V @ A
target = content[(np.arange(N) + 1) % N]


def sheet1(answers):
    fig, ax = page(30, 30, "One attention head, by hand" + ("  (answers)" if answers else ""))
    ans = dict(kind="answer", blank=not answers)
    q, s_ = [f"q{i}" for i in range(N)], [f"s{i}" for i in range(N)]
    grid(ax, 2.0, 27.0, Y, "data", "Y: memory", rows=["pos 0", "pos 1", "pos 2", "pos 3", "content"], cols=s_)
    grid(ax, 9.5, 27.0, X, "data", "X: queries", rows=[f"pos {i}" for i in range(N)], cols=q)
    grid(ax, 17.0, 27.0, target, "result", "target: content at p + 1", cols=q)
    note(ax, 17.0, 25.3, "query p must fetch the content\nstored one slot ahead (wrapping round)")

    grid(ax, 2.0, 20.0, WQ, "weight", "W_Q")
    arrow(ax, 6.3, 18.0, 8.7, 18.0)
    grid(ax, 9.5, 20.0, Q, title="1   Q = W_Q X", cols=q, **ans)
    grid(ax, 17.0, 20.0, WK, "weight", "W_K")
    arrow(ax, 22.3, 18.0, 23.2, 18.0)
    grid(ax, 23.8, 20.0, K, title="2   K = W_K Y", cols=s_, **ans)

    grid(ax, 2.0, 13.0, S, title="3   scores = K^T Q", rows=s_, cols=q, **ans)
    arrow(ax, 6.3, 11.0, 8.7, 11.0)
    grid(ax, 9.5, 13.0, A, title="4   softmax (sharp)", rows=s_, cols=q, **ans)
    note(ax, 15.0, 12.6, "Sharp softmax, by hand: in each column\nput 1 on the largest score, 0 elsewhere.\n"
         "(A real softmax gives about 0.99 there\nwhen the scores are scaled up.)")

    grid(ax, 2.0, 5.5, WV, "weight", "W_V")
    arrow(ax, 7.3, 5.0, 8.7, 5.0)
    grid(ax, 9.5, 5.5, V, title="5   V = W_V Y", cols=s_, **ans)
    grid(ax, 17.0, 5.5, Z, title="6   output = V × softmax", cols=q, **ans)
    note(ax, 17.0, 4.0, "Check: does the output equal the target?" + ("   Yes." if answers else ""),
         color=HAND if answers else MUTED)
    save(fig, "worksheet_1_attention" + ("_answers" if answers else "") + ".png")


# =====================================================================================================
# Sheet 2: valuing a head (exact fractions)
h1, h2, y = [Fr(1), Fr(1), Fr(0)], [Fr(1), Fr(0), Fr(1)], [Fr(3), Fr(2), Fr(1)]
dot = lambda a, b: sum(i * j for i, j in zip(a, b))
A2 = [[dot(h1, h1), dot(h1, h2)], [dot(h2, h1), dot(h2, h2)]]
b2 = [dot(h1, y), dot(h2, y)]
det = A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0]
M2 = [[A2[1][1] / det, -A2[0][1] / det], [-A2[1][0] / det, A2[0][0] / det]]
W2 = [M2[0][0] * b2[0] + M2[0][1] * b2[1], M2[1][0] * b2[0] + M2[1][1] * b2[1]]
pred_no = [W2[0] * v for v in h1]
res_no = [a - p for a, p in zip(y, pred_no)]
loss_no = dot(res_no, res_no)
w1 = dot(h1, y) / dot(h1, h1)
pred_re = [w1 * v for v in h1]
res_re = [a - p for a, p in zip(y, pred_re)]
loss_re = dot(res_re, res_re)
short = W2[1] ** 2 / M2[1][1]
PRICE = Fr(1)
# the copy case: head 2 = head 1, target = head1 + head2
c2, yc = h1, [2 * v for v in h1]
noref_copy = dot(h1, h1)                                  # weights (1, 1); removing the copy leaves head1 * 1
w1c = dot(h1, yc) / dot(h1, h1)
ref_copy = dot([a - w1c * v for a, v in zip(yc, h1)], [a - w1c * v for a, v in zip(yc, h1)])
assert short == loss_re


def sheet2(answers):
    fig, ax = page(30, 34, "Valuing a head, by hand" + ("  (answers)" if answers else ""))
    ans = dict(kind="answer", blank=not answers)
    tk = ["token 1", "token 2", "token 3"]
    grid(ax, 3.0, 31.0, [[a, b] for a, b in zip(h1, h2)], "data", "Z: head outputs", rows=tk, cols=["head 1", "head 2"])
    grid(ax, 9.0, 31.0, [[v] for v in y], "data", "y: target", cols=["y"])
    note(ax, 13.0, 31.2, "Each column is one head's output, each row one token.\n"
         "The model predicts y as  W1 · head 1 + W2 · head 2.\nPrice of a head: 1.   Loss = sum of squared errors.")

    grid(ax, 3.0, 25.0, A2, title="1   A = Z^T Z  (overlap)", rows=["h1", "h2"], cols=["h1", "h2"], **ans)
    grid(ax, 11.5, 25.0, [[v] for v in b2], title="2   b = Z^T y", **ans)
    grid(ax, 17.0, 25.0, [[det]], title="3   det A", **ans)
    grid(ax, 22.0, 25.0, M2, title="4   M = A^-1", **ans)

    grid(ax, 3.0, 19.0, [[v] for v in W2], title="5   W = M b", rows=["W1", "W2"], **ans)
    note(ax, 6.0, 18.6, "Check: W1 · head 1 + W2 · head 2 = y ?")

    note(ax, 3.0, 15.4, "Remove head 2.", fs=11, color=INK)
    grid(ax, 3.0, 13.5, [[v] for v in pred_no], title="6a  W1 · head 1", **ans)
    grid(ax, 8.5, 13.5, [[v] for v in res_no], title="6b  y − that", **ans)
    grid(ax, 14.0, 13.5, [[loss_no]], title="6c  loss, no re-fit", **ans)
    note(ax, 19.5, 13.5, "Nobody adjusts:\nloss = W2² · A22")

    grid(ax, 3.0, 8.0, [[w1]], title="7a  best weight, head 1 alone", **ans)
    note(ax, 3.0, 6.6, "(head1 · y) / (head1 · head1)")
    grid(ax, 11.0, 8.0, [[v] for v in res_re], title="7b  y − w · head 1", **ans)
    grid(ax, 16.5, 8.0, [[loss_re]], title="7c  loss, re-fit", **ans)
    grid(ax, 23.0, 8.0, [[short]], title="8   W2² / M22", **ans)
    note(ax, 23.0, 6.6, "the shortcut:\nsame as 7c?")

    keep = "keep" if loss_re > PRICE else "close"
    note(ax, 3.0, 3.3, "9   Head 2's value (7c) against the price 1:   keep or close?   "
         + (f"{keep}  ({fmt(loss_re)} > 1)" if answers else "______"), fs=10.5, color=HAND if answers else INK)
    note(ax, 3.0, 2.0, "10  Now let head 2 be a copy of head 1, (1, 1, 0), and y = (2, 2, 0).  "
         "Loss without the copy, no re-fit: " + (fmt(noref_copy) if answers else "____")
         + ".   With re-fit: " + (fmt(ref_copy) if answers else "____") + ".   Keep or close?  "
         + ("close" if answers else "______"), fs=10.5, color=HAND if answers else INK)
    save(fig, "worksheet_2_value" + ("_answers" if answers else "") + ".png")


# =====================================================================================================
# Sheet 3: why the head value is exact (Taylor, Hessian, OBD and OBS), same two heads as sheet 2
# A. one head: loss(w) = sum (y - w * head1)^2 = c0 + c1 w + c2 w^2
c0, c1, c2 = dot(y, y), -2 * dot(h1, y), dot(h1, h1)
loss1 = lambda w: c0 + c1 * w + c2 * w * w
slope0, curv = c1, 2 * c2                                  # slope at w = 0, curvature (everywhere)
w_star = -c1 / (2 * c2)                                    # the bottom: slope = 0
taylor = loss1(w_star) + Fr(1, 2) * curv * w_star ** 2     # from the bottom, step to w = 0
assert taylor == loss1(0)
# B. the Hessian of the two-head loss: H = 2 Z^T Z, and its inverse
H = [[2 * A2[i][j] for j in range(2)] for i in range(2)]
detH = H[0][0] * H[1][1] - H[0][1] * H[1][0]
Hi = [[H[1][1] / detH, -H[0][1] / detH], [-H[1][0] / detH, H[0][0] / detH]]
# C. the two scores for head 2 and head 1
obd = [Fr(1, 2) * H[k][k] * W2[k] ** 2 for k in range(2)]
obs = [W2[k] ** 2 / (2 * Hi[k][k]) for k in range(2)]
assert obd[1] == loss_no and obs[1] == loss_re
# D. the copy case: two identical heads, target head1 + head2, weights (1, 1)
Hc = [[2 * dot(h1, h1)] * 2] * 2
detHc = Hc[0][0] * Hc[1][1] - Hc[0][1] * Hc[1][0]
obd_c = Fr(1, 2) * Hc[1][1] * 1 ** 2


def sheet3(answers):
    fig, ax = page(30, 40, "Why the head value is exact, by hand" + ("  (answers)" if answers else ""))
    ans = dict(kind="answer", blank=not answers)
    say_ = lambda x, y_, text, key: note(ax, x, y_, text + ("   " + key if answers else "   ______"),
                                         fs=10.5, color=HAND if answers else INK)
    note(ax, 2.0, 37.6, "Same data as sheet 2:  head 1 = (1, 1, 0),  head 2 = (1, 0, 1),  y = (3, 2, 1).  "
         "Loss = sum of squared errors.", fs=10.5, color=INK)

    note(ax, 2.0, 36.0, "A   One head only.  loss(w) = sum of (y − w · head 1)²   multiplies out to   "
         "loss(w) = c0 + c1·w + c2·w²", fs=11, color=INK)
    grid(ax, 2.0, 33.6, [[c0, c1, c2]], title="A1  c0, c1, c2", cols=["c0", "c1", "c2"], **ans)
    note(ax, 5.5, 33.4, "c0 = y·y\nc1 = −2 · head1·y\nc2 = head1·head1")
    grid(ax, 11.0, 33.6, [[loss1(0)]], title="A2  loss(0)", **ans)
    grid(ax, 15.0, 33.6, [[curv]], title="A3  curvature = 2·c2", **ans)
    grid(ax, 22.0, 33.6, [[w_star]], title="A4  bottom w*", **ans)
    note(ax, 22.0, 32.2, "slope = c1 + 2·c2·w = 0\n(same as Tool 2)")
    grid(ax, 2.0, 29.4, [[loss1(w_star)]], title="A5  loss(w*)", **ans)
    grid(ax, 8.0, 29.4, [[taylor]], title="A6  Taylor from w*:  loss(w*) + ½ · curvature · (w*)²", **ans)
    say_(2.0, 27.4, "A7  Is A6 equal to A2, the loss when the weight is removed?  Why?",
         "Yes. The loss is a parabola, so the Taylor formula has no higher terms. It is exact.")

    note(ax, 2.0, 25.2, "B   Both heads.  The Hessian (curvature table) of the loss is  H = 2 · ZᵀZ.", fs=11, color=INK)
    grid(ax, 2.0, 22.8, H, title="B1  H = 2·ZᵀZ", rows=["w1", "w2"], cols=["w1", "w2"], **ans)
    grid(ax, 9.0, 22.8, [[detH]], title="B2  det H", **ans)
    grid(ax, 14.0, 22.8, Hi, title="B3  H^-1", **ans)
    note(ax, 20.5, 22.4, "diagonal: each head on its own\noff-diagonal: how the two heads overlap")

    note(ax, 2.0, 18.4, "C   The lecture's two scores, with W = (2, 1) from sheet 2.", fs=11, color=INK)
    grid(ax, 2.0, 16.6, [obd], title="C1  OBD = ½ · H_kk · W_k²", cols=["head 1", "head 2"], **ans)
    grid(ax, 11.0, 16.6, [obs], title="C2  OBS = W_k² / (2 · [H^-1]_kk)", cols=["head 1", "head 2"], **ans)
    say_(2.0, 14.6, "C3  Compare head 2 with sheet 2.  OBD = step 6c?  OBS = step 7c?",
         "Yes: OBD = 2 = no re-fit,  OBS = 3/2 = re-fit.")

    note(ax, 2.0, 12.4, "D   A copy: head 2 = head 1 = (1, 1, 0), weights (1, 1).", fs=11, color=INK)
    grid(ax, 2.0, 10.6, Hc, title="D1  H", **ans)
    grid(ax, 7.5, 10.6, [[detHc]], title="D2  det H", **ans)
    grid(ax, 12.0, 10.6, [[obd_c]], title="D3  OBD, head 2", **ans)
    say_(17.0, 10.2, "D4  OBS (re-fit) for the copy?", "0: the twin takes weight 2.")
    say_(17.0, 9.2, "D5  det = 0, so H has no inverse.  Fix?", "add a tiny ridge, λ·I")

    note(ax, 2.0, 6.6, "E   Say it in words.", fs=11, color=INK)
    say_(2.0, 5.6, "E1  Is our score exact or approximate?", "Exact for the output weights (A7, C3).")
    say_(2.0, 4.6, "E2  What stays fixed while scoring?", "Attention patterns and earlier layers.")
    say_(2.0, 3.6, "E3  Why OBS, not OBD?", "OBD ignores overlap, so a copy looks important (D3 vs D4).")
    say_(2.0, 2.6, "E4  Why is the weight squared?", "Squared error: the ½ · curvature · step² term.")
    save(fig, "worksheet_3_exact" + ("_answers" if answers else "") + ".png")


for answers in (False, True):
    sheet1(answers)
    sheet2(answers)
    sheet3(answers)
