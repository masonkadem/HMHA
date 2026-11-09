"""Quick test: collateral value on a character-level Shakespeare transformer.
One seed per process.  python shakes.py --mode none|element|head --seed 0
mode = the kind of damage applied to head outputs during training (probability p):
  element: every activation of every head fails independently (standard dropout)
  head:    whole heads fail, per sequence
All measurements are made undamaged, with every head at full gain."""
import argparse, copy, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="none"); ap.add_argument("--p", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--rule_steps", type=int, default=1500); ap.add_argument("--price", type=float, default=0.02)
ap.add_argument("--out", default="."); a = ap.parse_args()
torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "data", "input.txt")).read()
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
L, H, D, DH, T, BS = 4, 4, 128, 32, 64, 64
NH = L * H

def batch(src, bs, gen=None):
    i = torch.randint(0, len(src) - T - 1, (bs,), generator=gen)
    x = torch.stack([src[j:j + T] for j in i]); y = torch.stack([src[j + 1:j + T + 1] for j in i])
    return x, y

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.q, s.k, s.v = (nn.Linear(D, H * DH, bias=False) for _ in range(3))
        s.o = nn.Parameter(torch.randn(H, DH, D) / math.sqrt(H * DH))
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x, g, mode, p, keep=None):
        B, T_, _ = x.shape; h = s.ln1(x)
        sp = lambda t: t.view(B, T_, H, DH).transpose(1, 2)
        q, k, v = sp(s.q(h)), sp(s.k(h)), sp(s.v(h))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
        att = att.masked_fill(~torch.ones(T_, T_, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
        z = att @ v                                              # (B, H, T, DH)
        if s.training and mode == "element": z = F.dropout(z, p)
        if s.training and mode == "head":
            z = z * (torch.rand(B, H, 1, 1) >= p).float() / (1 - p)
        if keep is not None: keep.append(z.detach())
        x = x + torch.einsum("bhtd,hdm->btm", z * g[None, :, None, None], s.o)
        return x + s.mlp(s.ln2(x))

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, D), nn.Parameter(torch.randn(T, D) * 0.02)
        s.blocks = nn.ModuleList([Block() for _ in range(L)]); s.lnf = nn.LayerNorm(D); s.un = nn.Linear(D, V)
    def resid(s, x, g=None, mode="none", p=0.0, keep=None):
        g = torch.ones(NH) if g is None else g
        h = s.emb(x) + s.pos[: x.size(1)]
        for i, b in enumerate(s.blocks): h = b(h, g[i * H:(i + 1) * H], mode, p, keep)
        return s.lnf(h)
    def forward(s, x, g=None, mode="none", p=0.0): return s.un(s.resid(x, g, mode, p))

gv = torch.Generator().manual_seed(123)
VX, VY = batch(va, 512, gv)                                      # held-out, fixed
PA, PB = slice(0, 96), slice(96, 192)                          # fit the read-out on A, score on B
cnt = torch.bincount(tr, minlength=V).float(); uni = cnt / cnt.sum()
TRIV = float(-(uni[VY.reshape(-1)]).log().mean())               # unigram loss: a model that learned nothing

@torch.no_grad()
def vloss(m, g=None):
    m.eval(); l = F.cross_entropy(m(VX, g).reshape(-1, V), VY.reshape(-1)).item(); m.train(); return l

@torch.no_grad()
def values(m, open_, base=None):
    """Per open head: ordinary importance (remove, keep the read-out) and collateral value
    (remove, re-fit the final linear read-out to the current logits by least squares on
    split A). Both are the rise in held-out loss on split B, in nats."""
    m.eval(); base = open_.float() if base is None else base
    def feats(g):
        z = m.resid(VX[:192], g).double(); return z[PA].reshape(-1, D), z[PB].reshape(-1, D)
    one = lambda z: torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)
    zA, zB = feats(base); tgtA = m.un(zA.float()).double(); yB = VY[PB].reshape(-1)
    E0 = F.cross_entropy(m.un(zB.float()), yB).item()
    def refit(zA_, zB_):
        ZA = one(zA_); G = ZA.T @ ZA / len(ZA); G += 1e-4 * torch.diagonal(G).mean() * torch.eye(D + 1, dtype=G.dtype)
        W = torch.linalg.solve(G, ZA.T @ tgtA / len(ZA)); return F.cross_entropy(one(zB_) @ W, yB).item()
    val, abl = np.full(NH, np.nan), np.full(NH, np.nan)
    for h in torch.nonzero(open_).flatten().tolist():
        g = base.clone(); g[h] = 0.0; a_, b_ = feats(g)
        abl[h] = F.cross_entropy(m.un(b_.float()), yB).item() - E0
        val[h] = refit(a_, b_) - E0
    m.train(); return val, abl, E0

def fit(m, steps, mode, p, rule=False, price=0.0, bar=None):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    open_, tone, order = torch.ones(NH, dtype=torch.bool), torch.ones(NH), []
    log = dict(step=[], val=[], heads=[])
    for st in range(steps):
        x, y = batch(tr, BS)
        loss = F.cross_entropy(m(x, tone, mode, p).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if rule and st % 25 == 0 and st < 0.8 * steps:
            v, _, e = values(m, open_, tone)
            if e > bar + 2 * price and order:                    # held-out loss left the bar: reopen the last
                open_[order.pop()] = True
            else:
                h = int(np.nanargmin(v))
                if v[h] < price and e + max(v[h], 0) <= bar + price and int(open_.sum()) > 1:
                    open_[h] = False; order.append(h)
        if rule: tone += torch.clamp(open_.float() - tone, -0.01, 0.01)   # fade over 100 steps
        if st % 100 == 0 or st == steps - 1:
            log["step"].append(st); log["val"].append(vloss(m, tone)); log["heads"].append(int(open_.sum()))
    return open_, tone, log

t0 = time.time()
m = Net(); _, _, log0 = fit(m, a.steps, a.mode, a.p); print('train', time.time()-t0, flush=True)
dense = vloss(m); gap = TRIV - dense
out = dict(args=vars(a), trivial=TRIV, dense=dense, gap=gap, train_log=log0)

# 1. two numbers per head, at full perfusion
allopen = torch.ones(NH, dtype=torch.bool)
val, abl, E0 = values(m, allopen)
out.update(value=val.tolist(), ablate=abl.tolist(), E0=E0)

# 2. within-head rank: participation ratio of each head's output covariance (max DH)
with torch.no_grad():
    m.eval(); keep = []; m.resid(VX[:128], keep=keep); m.train()
    pr = []
    for z in keep:
        for h in range(H):
            zz = z[:, h].reshape(-1, DH).double(); zz = zz - zz.mean(0); ev = torch.linalg.eigvalsh(zz.T @ zz / len(zz)).clamp_min(0)
            pr.append(float(ev.sum() ** 2 / (ev ** 2).sum()))
out["head_rank"] = pr

# 3. post-hoc greedy removal, no retraining: by collateral value (score = re-fitted loss)
#    and by ordinary importance (score = loss with the read-out kept)
for name in ("collateral", "ablate"):
    open_, curve = allopen.clone(), []
    for _ in range(NH - 1):
        v, ab, e = values(m, open_); s = v if name == "collateral" else ab
        h = int(np.nanargmin(s)); open_[h] = False; curve.append((h, float(e + s[h] - E0)))
    out["curve_" + name] = curve                                   # loss rise after removing 1..15 heads

# 4. the rule during training against a dense continuation of the same model
price = a.price * gap
md = copy.deepcopy(m); torch.manual_seed(a.seed + 1); _, _, logd = fit(md, a.rule_steps, a.mode, a.p)
mr = copy.deepcopy(m); torch.manual_seed(a.seed + 1)
open_r, tone_r, logr = fit(mr, a.rule_steps, a.mode, a.p, rule=True, price=price, bar=dense)
out.update(price_nats=price, dense_cont_log=logd, rule_log=logr, rule_open=open_r.tolist(),
           rule_final=vloss(mr, tone_r), dense_cont_final=vloss(md), seconds=time.time() - t0)
json.dump(out, open(os.path.join(a.out, f"shakes_{a.mode}_p{a.p}_s{a.seed}.json"), "w"))
print(a.mode, a.seed, f"dense {dense:.3f} triv {TRIV:.3f} rule kept {int(open_r.sum())}/{NH} "
      f"rule {out['rule_final']:.3f} dense-cont {out['dense_cont_final']:.3f} {time.time()-t0:.0f}s")
