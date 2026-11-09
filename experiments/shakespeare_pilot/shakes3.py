"""Round 2 (undamaged model). (a) Which cheap score predicts the loss rise after the whole
network adapts without a head: ablation, or the own-layer re-fit? Truth at --ft and --ft2
gradient steps. (b) Removal with compensation: remove heads one at a time, choosing by
  ablate       smallest loss rise, nothing re-fitted
  local        smallest loss rise after the head's own layer is re-fitted; the re-fit is kept
  random       random order, nothing re-fitted
  random+fit   random order, own layer re-fitted (separates the choice from the compensation)
recording held-out loss after each removal, and after --ftp fine-tuning steps at 4, 8, 12 removed.
One seed per process."""
import sys, os, json, time, copy
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakes.py")).read().split("t0 = time.time()")[0]
src = src.replace("a = ap.parse_args()", 'ap.add_argument("--ft", type=int, default=40); ap.add_argument("--ft2", type=int, default=150); ap.add_argument("--ftp", type=int, default=150); a = ap.parse_args()')
exec(src)
tag = f"{a.mode}_p{a.p}_s{a.seed}"; ck = os.path.join(a.out, f"ckpt_{tag}.pt"); t0 = time.time()
m = Net()
if os.path.exists(ck): m.load_state_dict(torch.load(ck))
else:
    fit(m, a.steps, a.mode, a.p); torch.save(m.state_dict(), ck)
print("trained", time.time() - t0, flush=True)
XA, XB, yB = VX[:96], VX[96:192], VY[96:192].reshape(-1)
one = lambda z: torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)
def ridge(Z, Y, lam=1e-4):
    G = Z.T @ Z / len(Z); G += lam * torch.diagonal(G).mean() * torch.eye(G.size(0), dtype=G.dtype)
    return torch.linalg.solve(G, Z.T @ Y / len(Z))

def fwd(mm, x, g, bias, stop=None):
    """Logits with gains g (NH,) and a constant per-layer write bias. stop=l returns layer l's
    per-head outputs z and what the layer currently writes."""
    h = mm.emb(x) + mm.pos[: x.size(1)]
    for i, b in enumerate(mm.blocks):
        B_, T_, _ = h.shape; n = b.ln1(h); sp = lambda t: t.view(B_, T_, H, DH).transpose(1, 2)
        q, k, v = sp(b.q(n)), sp(b.k(n)), sp(b.v(n))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(DH)
        att = att.masked_fill(~torch.ones(T_, T_, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
        z = att @ v
        write = torch.einsum("bhtd,hdm->btm", z * g[i * H:(i + 1) * H][None, :, None, None], b.o) + bias[i]
        if stop == i: return z, write
        h = h + write; h = h + b.mlp(b.ln2(h))
    return mm.un(mm.lnf(h))

@torch.no_grad()
def hloss(mm, g, bias, x=XB, y=yB): return F.cross_entropy(fwd(mm, x, g, bias).reshape(-1, V), y).item()
@torch.no_grad()
def vfull(mm, g, bias):
    mm.eval(); l = hloss(mm, g, bias, VX, VY.reshape(-1)); mm.train(); return l

@torch.no_grad()
def local_fit(mm, g, bias, hd):
    """New output maps for the other open heads of hd's layer, and a new layer bias, so the
    layer writes as nearly as it can what it writes now with hd (least squares on split A)."""
    l, hh = divmod(hd, H); z, write = fwd(mm, XA, g, bias, stop=l)
    oth = [j for j in range(H) if j != hh and g[l * H + j] > 0]
    Zr = one(z[:, oth].permute(0, 2, 1, 3).reshape(-1, len(oth) * DH).double()) if oth else torch.ones(z.size(0) * z.size(2), 1, dtype=torch.float64)
    W = ridge(Zr, write.reshape(-1, D).double()).float()
    return l, oth, W[:-1].view(len(oth), DH, D), W[-1]

def apply_fit(mm, g, bias, hd, fitd):
    l, oth, o_new, b_new = fitd
    old = (mm.blocks[l].o.data.clone(), bias[l].clone(), g[hd].item())
    for j, o in zip(oth, o_new): mm.blocks[l].o.data[j] = o
    bias[l] = b_new.clone(); g[hd] = 0.0
    return old
def undo(mm, g, bias, hd, old):
    l = hd // H; mm.blocks[l].o.data.copy_(old[0]); bias[l] = old[1]; g[hd] = old[2]

def cand(mm, g, bias, hd, comp):
    if not comp:
        g[hd] = 0.0; e = hloss(mm, g, bias); g[hd] = 1.0; return e
    old = apply_fit(mm, g, bias, hd, local_fit(mm, g, bias, hd)); e = hloss(mm, g, bias); undo(mm, g, bias, hd, old); return e

def tune(mm, g, bias, steps):
    opt = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.0); gen = torch.Generator().manual_seed(777)
    for _ in range(steps):
        x, y = batch(tr, BS, gen)
        loss = F.cross_entropy(fwd(mm, x, g, bias).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return vfull(mm, g, bias)

zero = lambda: [torch.zeros(D) for _ in range(L)]
m.eval(); g0 = torch.ones(NH); E0 = hloss(m, g0, zero()); dense = vfull(m, g0, zero()); m.eval(); gap = TRIV - dense
out = dict(args=vars(a), dense=dense, gap=gap, trivial=TRIV)

# (a) singles
abl = [cand(m, torch.ones(NH), zero(), h, False) - E0 for h in range(NH)]
loc = [cand(m, torch.ones(NH), zero(), h, True) - E0 for h in range(NH)]
out.update(ablate=abl, local=loc)
for ft in (a.ft, a.ft2):
    ref = tune(copy.deepcopy(m), torch.ones(NH), zero(), ft); tr_ = []
    for h in range(NH):
        g = torch.ones(NH); g[h] = 0; tr_.append(tune(copy.deepcopy(m), g, zero(), ft) - ref)
    out[f"truth{ft}"] = tr_; out[f"ref{ft}"] = ref
    print("truth", ft, time.time() - t0, flush=True)

# (b) removal curves
perm = torch.randperm(NH, generator=torch.Generator().manual_seed(a.seed + 5)).tolist()
ref_p = tune(copy.deepcopy(m), torch.ones(NH), zero(), a.ftp); out["ref_ftp"] = ref_p
for arm, comp, rnd in (("ablate", False, False), ("local", True, False), ("random", False, True), ("random+fit", True, True)):
    mm = copy.deepcopy(m); mm.eval(); g, bias = torch.ones(NH), zero(); curve, order, tuned = [], [], {}
    for k in range(1, 13):
        opens = [h for h in range(NH) if g[h] > 0]
        if rnd: h = [x for x in perm if g[x] > 0][0]
        else:
            sc = [cand(mm, g, bias, x, comp) for x in opens]; h = opens[int(np.argmin(sc))]
        if comp: apply_fit(mm, g, bias, h, local_fit(mm, g, bias, h))
        else: g[h] = 0.0
        order.append(h); curve.append(vfull(mm, g, bias) - dense); mm.eval()
        if k in (4, 8, 12):
            tuned[k] = tune(copy.deepcopy(mm), g.clone(), [b.clone() for b in bias], a.ftp) - ref_p
    out["arm_" + arm] = dict(order=order, curve=curve, tuned=tuned)
    print(arm, [round(c / gap, 3) for c in curve], {k: round(v / gap, 3) for k, v in tuned.items()}, time.time() - t0, flush=True)
out["seconds"] = time.time() - t0
json.dump(out, open(os.path.join(a.out, f"r2_{tag}.json"), "w"))
