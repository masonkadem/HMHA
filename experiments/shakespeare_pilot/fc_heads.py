"""Functional connectivity of attention heads, in the style of resting-state fMRI.

Signal: for every token of a long stretch of held-out text, the size (L2 norm) of what each
head writes into the residual stream at that token, with the full 64-token context behind it.
That gives one time series per head. Heads that rise and fall together are 'in sync'.
  scales        the series as it is, and averaged over windows of 8 and 32 tokens
  connectivity  Pearson correlation between every pair of heads, at each scale
  reliability   the text is cut into 4 segments; ICC(3,1) of the pairwise correlations
                (Fisher z) across segments = does the same network show up again
  layer norm    the same for a model trained WITH layer norm and one trained WITHOUT
  sync vs cover do heads in sync back each other up? Against the pair-removal data of round 1.

  python fc_heads.py --seed 0 --out out --fig <dir>"""
import sys, os, json, time
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes.py")).read().split("t0 = time.time()")[0]
src = src.replace("a = ap.parse_args()", 'ap.add_argument("--fig", default="."); ap.add_argument("--ntok", type=int, default=4096); a = ap.parse_args()')
exec(src)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr

class NetNoLN(Net):
    def __init__(s):
        super().__init__()
        for b in s.blocks: b.ln1, b.ln2 = nn.Identity(), nn.Identity()
        s.lnf = nn.Identity()

def get(kind):
    ck = os.path.join(a.out, f"ckpt_none_p0.0_s{a.seed}.pt" if kind == "ln" else f"ckpt_noln_s{a.seed}.pt")
    torch.manual_seed(a.seed); m = Net() if kind == "ln" else NetNoLN()
    if os.path.exists(ck): m.load_state_dict(torch.load(ck))
    else:
        fit(m, a.steps, "none", 0.0); torch.save(m.state_dict(), ck)
    return m

@torch.no_grad()
def series(m):
    """(ntok, NH): size of each head's write at the last position of a 64-token window that
    slides one token at a time over held-out text."""
    m.eval(); idx = torch.arange(a.ntok); X = torch.stack([va[i:i + T] for i in idx]); out = []
    for i in range(0, len(X), 256):
        h = m.emb(X[i:i + 256]) + m.pos; row = []
        for b in m.blocks:
            B_, T_, _ = h.shape; n = b.ln1(h); sp = lambda t: t.view(B_, T_, H, DH).transpose(1, 2)
            q, k, v = sp(b.q(n)), sp(b.k(n)), sp(b.v(n))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
            att = att.masked_fill(~torch.ones(T_, T_, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
            z = att @ v; per = torch.einsum("bhd,hdm->bhm", z[:, :, -1], b.o); row.append(per.norm(dim=-1))
            h = h + torch.einsum("bhtd,hdm->btm", z, b.o); h = h + b.mlp(b.ln2(h))
        out.append(torch.cat(row, 1))
    return torch.cat(out).numpy(), vloss(m)

SCALES = (1, 8, 32)
def smooth(x, w):
    if w == 1: return x
    k = np.ones(w) / w; return np.stack([np.convolve(x[:, j], k, mode="valid") for j in range(x.shape[1])], 1)
def icc31(M):
    """ICC(3,1): rows = targets (head pairs), columns = sessions (text segments)."""
    n, k = M.shape; gm = M.mean(); msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse))
iu = np.triu_indices(NH, 1)

res, keep = {}, {}
for kind in ("ln", "noln"):
    x, loss = series(get(kind)); r = dict(val_loss=loss, fc={}, icc={})
    for w in SCALES:
        xs = smooth(x, w); r["fc"][w] = np.corrcoef(xs.T)
        segs = np.array_split(xs, 4); E = np.stack([np.arctanh(np.clip(np.corrcoef(s.T)[iu], -0.999, 0.999)) for s in segs], 1)
        r["icc"][w] = icc31(E)
    r["mean_abs_fc"] = {w: float(np.abs(r["fc"][w][iu]).mean()) for w in SCALES}
    same = (iu[0] // H) == (iu[1] // H)
    r["same_layer_fc"] = {w: float(r["fc"][w][iu][same].mean()) for w in SCALES}; r["cross_layer_fc"] = {w: float(r["fc"][w][iu][~same].mean()) for w in SCALES}
    res[kind], keep[kind] = r, x
    print(kind, f"loss {loss:.3f}", "icc", {w: round(v, 2) for w, v in r["icc"].items()}, "mean|r|", {w: round(v, 2) for w, v in r["mean_abs_fc"].items()}, flush=True)

# sync against cover: pair-removal data from round 1 (model with layer norm, this seed)
link = None; p1 = os.path.join(a.out, f"r1_none_p0.0_s{a.seed}.json")
if os.path.exists(p1):
    r1 = json.load(open(p1)); P_, ab = np.array(r1["pair"]), np.array(r1["ablate"]); inter = (P_ - ab[:, None] - ab[None, :])[iu] / r1["gap"]
    link = dict(inter=inter, rho={w: float(spearmanr(res["ln"]["fc"][w][iu], inter)[0]) for w in SCALES},
                rho_abs={w: float(spearmanr(np.abs(res["ln"]["fc"][w][iu]), inter)[0]) for w in SCALES})
    print("sync vs extra loss when both removed: spearman", link["rho"], "with |r|", link["rho_abs"])

# ---------------------------------------------------------------- figure
RED, K, G1, G2, BLUE = "#b2182b", "#111111", "#6b6b6b", "#a8a8a8", "#2166ac"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
NAME = dict(ln="with layer norm", noln="without layer norm"); lab = lambda h: f"L{h // H + 1}.{h % H + 1}"
fig = plt.figure(figsize=(13, 10.5)); gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1.25, 1.15], hspace=0.55, wspace=0.5, left=0.06, right=0.925, top=0.95, bottom=0.06)
z = lambda v: (v - v.mean()) / v.std()
for c, kind in enumerate(("ln", "noln")):
    x, fc = keep[kind], res[kind]["fc"][1]; f = fc.copy(); np.fill_diagonal(f, -2)
    i, j = np.unravel_index(f.argmax(), f.shape); f2 = fc.copy(); np.fill_diagonal(f2, 2); k_ = int(np.abs(f2[i]).argmin())
    for row, w in enumerate((1, 32)):
        ax = fig.add_subplot(gs[row, 3 * c:3 * c + 3]); xs = smooth(x, w)[:400]
        for h, col, lw in ((k_, G2, 1.0), (j, K, 1.2), (i, RED, 1.2)):
            ax.plot(z(xs[:, h]), color=col, lw=lw, label=f"head {lab(h)}")
        rr = np.corrcoef(smooth(x, w).T)
        ax.set_xlabel("token position in the text"); ax.set_ylabel("head activity (z-score)")
        scale = "every token" if w == 1 else f"averaged over {w} tokens"
        ax.set_title(f"{'ab'[c] if row == 0 else 'cd'[c]}   {NAME[kind].capitalize()}, {scale}", fontweight="bold")
        ax.text(0.01, 0.97, f"r(red, black) = {rr[i, j]:.2f}     r(red, grey) = {rr[i, k_]:.2f}", transform=ax.transAxes, va="top")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
        if row == 0: ax.legend(loc="upper right", ncol=3)
    order = leaves_list(linkage(res[kind]["fc"][1], "average"))
    for s_, w in enumerate(SCALES):
        ax = fig.add_subplot(gs[2, 3 * c + s_]); M = res[kind]["fc"][w][np.ix_(order, order)]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r"); ax.set_xticks(range(NH)); ax.set_yticks(range(NH))
        ax.set_xticklabels([lab(h) for h in order], rotation=90, fontsize=5); ax.set_yticklabels([lab(h) for h in order], fontsize=5)
        nm = "every token" if w == 1 else f"{w}-token avg"
        ax.set_title((f"{'ef'[c]}   {NAME[kind].capitalize()}\n" if s_ == 0 else "\n") + f"{nm}, |r| = {res[kind]['mean_abs_fc'][w]:.2f}", fontweight="bold" if s_ == 0 else "normal", fontsize=8)
cax = fig.add_axes([0.94, 0.33, 0.008, 0.16]); cb = fig.colorbar(im, cax=cax); cb.set_label("correlation between two heads")
ax = fig.add_subplot(gs[3, 0:2]); xs_ = np.arange(len(SCALES))
for kind, col, mk in (("ln", K, "o"), ("noln", RED, "D")):
    v = [res[kind]["icc"][w] for w in SCALES]; ax.plot(xs_, v, color=col, marker=mk, lw=1.6, ms=5, label=NAME[kind])
    for xx, vv in zip(xs_, v): ax.text(xx, vv + (0.04 if kind == "noln" else -0.09), f"{vv:.2f}", ha="center")
ax.set_xticks(xs_); ax.set_xticklabels(["every token", "8-token\naverage", "32-token\naverage"]); ax.set_ylim(0, 1.1); ax.set_ylabel("ICC of the network across\n4 segments of text"); ax.legend(loc="lower left")
ax.set_title("g   Does the same network reappear?", fontweight="bold")
ax = fig.add_subplot(gs[3, 2:4]); w_ = 0.36
for o, (kind, col) in enumerate((("ln", K), ("noln", RED))):
    ax.bar(np.arange(2) + (o - 0.5) * w_, [res[kind]["same_layer_fc"][1], res[kind]["cross_layer_fc"][1]], w_ * 0.92, color=col, label=NAME[kind])
    for xx, vv in zip(np.arange(2) + (o - 0.5) * w_, [res[kind]["same_layer_fc"][1], res[kind]["cross_layer_fc"][1]]): ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center")
ax.set_xticks(range(2)); ax.set_xticklabels(["same layer", "different layers"]); ax.set_ylabel("mean correlation (every token)"); ax.legend(loc="upper right"); ax.axhline(0, color=G2, lw=0.6); ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
ax.set_title("h   Where the sync is", fontweight="bold")
ax = fig.add_subplot(gs[3, 4:6])
if link is not None:
    ax.scatter(res["ln"]["fc"][1][iu], link["inter"], s=14, color=K, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=G2, lw=0.6); ax.axvline(0, color=G2, lw=0.6)
    ax.set_xlabel("correlation between the two heads (every token)"); ax.set_ylabel("extra loss when both are removed\n(beyond the two singly, fraction of gap)")
    ax.set_title("i   Do heads in sync back each other up?", fontweight="bold")
    ax.text(0.97, 0.6, f"rank correlation {link['rho'][1]:+.2f}", transform=ax.transAxes, va="top", ha="right")
os.makedirs(a.fig, exist_ok=True); p = os.path.join(a.fig, "fig_head_connectivity"); fig.savefig(p + ".png"); fig.savefig(p + ".pdf")
json.dump({k: dict(val_loss=v["val_loss"], icc=v["icc"], mean_abs_fc=v["mean_abs_fc"], same_layer_fc=v["same_layer_fc"], cross_layer_fc=v["cross_layer_fc"], fc={w: v["fc"][w].tolist() for w in SCALES}) for k, v in res.items()}
          | dict(link=None if link is None else dict(rho=link["rho"], rho_abs=link["rho_abs"])), open(os.path.join(a.out, f"fc_s{a.seed}.json"), "w"))
print("saved", p)
