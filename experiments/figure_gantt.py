"""Gantt chart for the hemodynamic attenuation project, October 2026 to December 2027, in the
same style as the thesis proposal's chart (serif text, boxed year and month header, grey bars,
dotted grid).

  python experiments/figure_gantt.py   ->  figures/gantt_hmha.png and .pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "savefig.dpi": 300, "pdf.fonttype": 42})

MONTHS = "ONDJFMAMJJASOND"                      # October 2026 ... December 2027
TASKS = [  # (label, first month, last month), months counted from 0 = October 2026
    ("Proposal, comprehensive exam, preliminary results", 0, 1),
    ("qSA-Cross: compare with Chapter 4 heads", 2, 4),
    ("Cross-Site Transformer on WildPPG", 4, 6),
    ("Attenuate, then audit a blood pressure model", 5, 7),
    ("Scaling to large language models", 7, 9),
    ("Paper writing and submission", 9, 10),
    ("Thesis writing and defence", 11, 14),
]

n, rows = len(MONTHS), len(TASKS)
fig, ax = plt.subplots(figsize=(10.5, 0.42 * rows + 1.0))
ax.set(xlim=(-0.02, n + 0.02), ylim=(rows + 0.05, -2.05))
ax.axis("off")
line = dict(color="black", lw=0.9)
# header: years, then months
for x0, x1, year in ((0, 3, "2026"), (3, n, "2027")):
    ax.add_patch(Rectangle((x0, -2), x1 - x0, 1, fill=False, **line))
    ax.text((x0 + x1) / 2, -1.5, year, ha="center", va="center", fontsize=13)
for i, m in enumerate(MONTHS):
    ax.add_patch(Rectangle((i, -1), 1, 1, fill=False, **line))
    ax.text(i + 0.5, -0.5, m, ha="center", va="center", fontsize=13)
# body: dotted grid, grey bars, labels on the left
ax.add_patch(Rectangle((0, 0), n, rows, fill=False, color="#bbbbbb", lw=0.8))
for i in range(1, n):
    ax.plot([i, i], [0, rows], color="black", lw=0.6, ls=(0, (1, 3)))
for r in range(1, rows):
    ax.plot([0, n], [r, r], color="black", lw=0.6, ls=(0, (1, 3)))
for r, (label, a, b) in enumerate(TASKS):
    ax.add_patch(Rectangle((a, r + 0.25), b - a + 1, 0.5, facecolor="#bfbfbf", edgecolor="#7f7f7f", lw=0.8))
    ax.text(-0.25, r + 0.5, label, ha="right", va="center", fontsize=12.5)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "figures", f"gantt_hmha.{ext}"), bbox_inches="tight", facecolor="white")
print("wrote figures/gantt_hmha.png and .pdf")
