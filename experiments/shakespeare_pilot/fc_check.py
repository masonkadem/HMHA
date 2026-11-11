"""Check on the head-connectivity result. Is the block of heads moving together in the model
without layer norm just every head scaling with the size of the residual stream it reads?
Per head trace, remove (a) the size of the residual stream entering that head's layer, by
linear regression, or (b) the global signal (mean of all z-scored head traces), then
recompute the correlations. One seed per process.  python fc_check.py --seed 0 --out out"""
import sys, os, json, time
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes.py")).read().split("t0 = time.time()")[0]
src = src.replace("a = ap.parse_args()", 'ap.add_argument("--ntok", type=int, default=4096); a = ap.parse_args()')
exec(src)
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
    m.eval(); X = torch.stack([va[i:i + T] for i in range(a.ntok)]); out, rn = [], []
    for i in range(0, len(X), 256):
        h = m.emb(X[i:i + 256]) + m.pos; row, rr = [], []
        for b in m.blocks:
            B_, T_, _ = h.shape; rr.append(h[:, -1].norm(dim=-1, keepdim=True)); n = b.ln1(h); sp = lambda t: t.view(B_, T_, H, DH).transpose(1, 2)
            q, k, v = sp(b.q(n)), sp(b.k(n)), sp(b.v(n))
            att = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
            att = att.masked_fill(~torch.ones(T_, T_, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
            z = att @ v; row.append(torch.einsum("bhd,hdm->bhm", z[:, :, -1], b.o).norm(dim=-1))
            h = h + torch.einsum("bhtd,hdm->btm", z, b.o); h = h + b.mlp(b.ln2(h))
        out.append(torch.cat(row, 1)); rn.append(torch.cat(rr, 1))
    return torch.cat(out).numpy(), torch.cat(rn).numpy(), vloss(m)
def icc31(M):
    n, k = M.shape; gm = M.mean(); msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse))
iu = np.triu_indices(NH, 1); lay = np.arange(NH) // H
same = lay[iu[0]] == lay[iu[1]]; late = (lay[iu[0]] >= 2) & (lay[iu[1]] >= 2)
zs = lambda x: (x - x.mean(0)) / x.std(0)
def resid(y, x):                                   # y minus its least-squares fit on x (with intercept)
    X_ = np.column_stack([x, np.ones(len(x))]); return y - X_ @ np.linalg.lstsq(X_, y, rcond=None)[0]
def stats(x):
    fc = np.corrcoef(x.T); e = fc[iu]
    E = np.stack([np.arctanh(np.clip(np.corrcoef(s.T)[iu], -0.999, 0.999)) for s in np.array_split(x, 4)], 1)
    return dict(mean_abs=float(np.abs(e).mean()), same_layer=float(e[same].mean()), cross_layer=float(e[~same].mean()),
                late_block=float(e[late].mean()), frac_negative=float((e < 0).mean()), icc=icc31(E), fc=fc.tolist())
out = dict(seed=a.seed)
for kind in ("ln", "noln"):
    x, rn, loss = series(get(kind)); x = zs(x)
    xr = np.column_stack([resid(x[:, h], rn[:, lay[h]]) for h in range(NH)])
    g = x.mean(1); xg = np.column_stack([resid(x[:, h], g) for h in range(NH)])
    r2 = [float(1 - xr[:, h].var() / x[:, h].var()) for h in range(NH)]
    out[kind] = dict(val_loss=loss, raw=stats(x), stream_removed=stats(xr), global_removed=stats(xg),
                     var_explained_by_stream=float(np.mean(r2)), var_by_layer=[float(np.mean(r2[l * H:(l + 1) * H])) for l in range(L)],
                     stream_norm_cv=[float(rn[:, l].std() / rn[:, l].mean()) for l in range(L)])
    print(kind, a.seed, f"loss {loss:.3f} var explained by stream size {np.mean(r2):.3f}",
          {k: (round(out[kind][k]["late_block"], 3), round(out[kind][k]["cross_layer"], 3), round(out[kind][k]["icc"], 2)) for k in ("raw", "stream_removed", "global_removed")}, flush=True)
json.dump(out, open(os.path.join(a.out, f"fccheck_s{a.seed}.json"), "w"))
