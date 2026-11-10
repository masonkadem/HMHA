"""One-page summary for discussion (not a paper figure): the paper's figures stacked, with a headline per row,
plus a small chart of GPT-2 heads chosen by our rule vs one-shot vs random (results/gpt2_ioi/baselines_s0.pkl).

  python experiments/figure_summary.py  ->  paper/neuroai/summary_for_meeting.png
Run the figure scripts first (fig_task, fig_neuroai_col, fig_backup_v2, fig_circuits_v3 in paper/neuroai).
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "paper", "neuroai")
RED, PINK, K, G1, G2 = "#b2182b", "#e8a5a5", "#111111", "#6b6b6b", "#a8a8a8"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.titlesize": 7, "axes.titlelocation": "left", "legend.frameon": False})

# ---------------------------------------------------------------- GPT-2: ours vs one-shot vs random
b = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", "baselines_s0.pkl"), "rb"))["sizes"]
fig, ax = plt.subplots(figsize=(3.35, 1.55))
FR = ("0.05", "0.02", "0.01")
x = np.arange(3); w = 0.26
ours = [b[f]["ours"][2] for f in FR]
one = [b[f]["oneshot"][2] for f in FR]
rnd = [b[f]["random"][:, 2].mean() for f in FR]
for dx, v, c, lab in ((-w, ours, RED, "our rule (re-values every step)"), (0, one, K, "one-shot (same value, used once)"),
                      (w, rnd, G2, "random (mean of 200)")):
    ax.bar(x + dx, v, width=w * 0.9, color=c, lw=0, label=lab)
    for xi, vi in zip(x + dx, v):
        ax.text(xi, vi + 0.02, f"{vi:.0%}", ha="center", va="bottom", fontsize=5.2)
ax.set(xticks=x, ylim=(0, 1.45), yticks=[0, 0.5, 1], yticklabels=["0", "50%", "100%"], ylabel="right name picked")
ax.set_xticklabels([f"{b[f]['n']} heads" for f in FR])
ax.legend(loc="upper left", fontsize=5.4, handlelength=0.9, borderaxespad=0)
ax.set_title(r"$\mathbf{e}$   GPT-2: same number of heads, chosen three ways", pad=4)
fig.savefig(os.path.join(P, "fig_gpt2_compare.png"), dpi=450, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- stack
W = 3000
try:
    big, small = ImageFont.truetype("arialbd.ttf", 58), ImageFont.truetype("arial.ttf", 40)
except OSError:
    big = small = ImageFont.load_default()


def fit(path, width):
    im = Image.open(path).convert("RGB")
    return im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)


def side(a, b_, gap=60):
    h = max(a.height, b_.height)
    out = Image.new("RGB", (a.width + b_.width + gap, h), "white")
    out.paste(a, (0, (h - a.height) // 2)); out.paste(b_, (a.width + gap, (h - b_.height) // 2))
    return out


half = (W - 60) // 2
rows = [
    ("1  What it is: every head has a valve and pays rent; close it when it's worth less than its rent",
     "The rule is the step-by-step minimiser of  loss + rent × open heads  (Murray's law for vessels).",
     fit(os.path.join(P, "fig_task.png"), W)),
    ("2  Keeps exactly the heads needed, collapses where theory says; beats one-shot on GPT-2",
     "29/30 exact, never broke the model (3 published methods broke it every time); 36/36 at the predicted collapse.  "
     "e: same value used once picks 9%; ours 89%.",
     side(fit(os.path.join(P, "fig_neuroai_col.png"), half), fit(os.path.join(P, "fig_gpt2_compare.png"), half))),
    ("3  It predicts how many spare heads to keep when heads can fail, before training",
     "Predictions written down first: 79/100 exact, 100/100 within one head. Spares cut the loss 5x when heads fail.",
     fit(os.path.join(P, "fig_backup_v2.png"), W)),
    ("4  It recovers circuits found by hand",
     "Induction 10/10 runs. GPT-2: core of 11 heads all in the published circuit; one head kept at every stage; nested.",
     fit(os.path.join(P, "fig_circuits_v3.png"), W)),
]
blocks = []
for head, sub, im in rows:
    band = Image.new("RGB", (W, 150), "white")
    d = ImageDraw.Draw(band)
    d.text((0, 10), head, fill=(178, 24, 43), font=big)
    d.text((0, 88), sub, fill=(60, 60, 60), font=small)
    blocks += [band, im, Image.new("RGB", (W, 70), "white")]
out = Image.new("RGB", (W + 160, sum(x.height for x in blocks) + 160), "white")
y = 80
for x_ in blocks:
    out.paste(x_, (80, y)); y += x_.height
out.save(os.path.join(P, "summary_for_meeting.png"))
out.save(os.path.join(P, "summary_for_meeting.pdf"), resolution=300)
print("wrote", os.path.join(P, "summary_for_meeting.png"), out.size)
