"""Round 1. Train one char-level Shakespeare transformer under whole-head failure rate p,
save it, then measure at full perfusion (undamaged):
  ablate   loss rise, head removed, nothing re-fitted
  v1       final linear read-out re-fitted (the value used so far)
  v2       the removed head's OWN LAYER re-fitted (other heads of the layer get new output
           maps to reproduce what the layer wrote), everything downstream recomputed
  v2b      v2, then the final read-out re-fitted as well
  truth    loss rise after the whole network adapts for --ft gradient steps without the head,
           against the same model adapted for the same steps with the head
  pairs    loss rise with two heads removed, every pair (hot redundancy: superadditivity)
One seed per process."""
import sys, os, json, time, copy
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes.py")).read().split("t0 = time.time()")[0]
src = src.replace("a = ap.parse_args()", 'ap.add_argument("--ft", type=int, default=40); a = ap.parse_args()')
exec(src)
tag = f"{a.mode}_p{a.p}_s{a.seed}"; ck = os.path.join(a.out, f"ckpt_{tag}.pt"); t0 = time.time()
m = Net()
if os.path.exists(ck): m.load_state_dict(torch.load(ck))
else:
    fit(m, a.steps, a.mode, a.p); torch.save(m.state_dict(), ck)
print("trained", time.time() - t0, flush=True)
dense = vloss(m); gap = TRIV - dense
allopen = torch.ones(NH, dtype=torch.bool)
v1, abl, E0 = values(m, allopen)
XA, XB, yB = VX[:96], VX[96:192], VY[96:192].reshape(-1)
ce = lambda logits: F.cross_entropy(logits.reshape(-1, V), yB).item()
one = lambda z: torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)
def ridge(Z, Y, lam=1e-4):
    G = Z.T @ Z / len(Z); G += lam * torch.diagonal(G).mean() * torch.eye(G.size(0), dtype=G.dtype)
    return torch.linalg.solve(G, Z.T @ Y / len(Z))

@torch.no_grad()
def run_local(x, l, hh, W=None):
    """Forward with head hh of layer l removed. W None: return (features of the layer's other
    heads, what the full layer wrote) for the fit. W given: the layer writes features @ W."""
    h = m.emb(x) + m.pos[: x.size(1)]
    for i, b in enumerate(m.blocks):
        B_, T_, _ = h.shape; n = b.ln1(h); sp = lambda t: t.view(B_, T_, H, DH).transpose(1, 2)
        q, k, v = sp(b.q(n)), sp(b.k(n)), sp(b.v(n))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
        att = att.masked_fill(~torch.ones(T_, T_, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
        z = att @ v; write = torch.einsum("bhtd,hdm->btm", z, b.o)
        if i == l:
            oth = [j for j in range(H) if j != hh]
            Zr = one(z[:, oth].permute(0, 2, 1, 3).reshape(-1, len(oth) * DH).double())
            if W is None: return Zr, write.reshape(-1, D).double()
            write = (Zr @ W).float().view(B_, T_, D)
        h = h + write; h = h + b.mlp(b.ln2(h))
    return m.lnf(h)

m.eval()
with torch.no_grad():
    fullA = m.resid(XA).reshape(-1, D).double(); tgtA = m.un(fullA.float()).double()
    v2, v2b = np.zeros(NH), np.zeros(NH)
    for hd in range(NH):
        l, hh = divmod(hd, H)
        W = ridge(*run_local(XA, l, hh))
        zB = run_local(XB, l, hh, W); v2[hd] = ce(m.un(zB)) - E0
        zA = run_local(XA, l, hh, W).reshape(-1, D).double()
        Wr = ridge(one(zA), tgtA); v2b[hd] = F.cross_entropy(one(zB.reshape(-1, D).double()) @ Wr, yB).item() - E0
    pair = np.zeros((NH, NH))
    for i in range(NH):
        for j in range(i + 1, NH):
            g = torch.ones(NH); g[i] = g[j] = 0
            pair[i, j] = pair[j, i] = ce(m(XB, g)) - E0
    layer_off = []
    for l in range(L):
        g = torch.ones(NH); g[l * H:(l + 1) * H] = 0; layer_off.append(ce(m(XB, g)) - E0)
m.train()
print("probes", time.time() - t0, flush=True)

def adapt(hd):
    mm = copy.deepcopy(m); g = torch.ones(NH)
    if hd is not None: g[hd] = 0
    opt = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.0); gen = torch.Generator().manual_seed(777)
    for _ in range(a.ft):
        x, y = batch(tr, BS, gen)
        loss = F.cross_entropy(mm(x, g).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return vloss(mm, g)
ref = adapt(None); truth = np.array([adapt(h) - ref for h in range(NH)])
json.dump(dict(args=vars(a), dense=dense, trivial=TRIV, gap=gap, E0=E0, ablate=abl.tolist(), v1=v1.tolist(),
               v2=v2.tolist(), v2b=v2b.tolist(), truth=truth.tolist(), ref_adapt=ref, pair=pair.tolist(),
               layer_off=layer_off, seconds=time.time() - t0), open(os.path.join(a.out, f"r1_{tag}.json"), "w"))
print("done", tag, f"dense {dense:.3f}", time.time() - t0, flush=True)
