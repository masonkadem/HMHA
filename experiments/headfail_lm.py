"""Does whole-head failure during training make a real language model keep backup heads?
Predictions written first: results/headfail_lm/predictions.md

Character-level transformer on Tiny Shakespeare. During training each head is switched off independently
with probability p (per sequence). Then, frozen, the hemodynamic attenuation rule decides which heads stay,
valued under the same failures ("aware") and with nothing failing ("blind").

  python experiments/headfail_lm.py --p 0 0.1 0.2 0.3 --seeds 0 1 2  ->  results/headfail_lm/*.pkl
"""
import argparse, math, os, pickle, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "headfail_lm")
L_, H_, D, T = 2, 16, 128, 128
DH = D // H_


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, gate):                                    # gate: (H,) or (B, H)
        B, T_, _ = x.shape
        q, k, v = (t.view(B, T_, H_, DH).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(D, -1))
        att = (q @ k.transpose(-1, -2) / math.sqrt(DH)).masked_fill(
            torch.ones(T_, T_, dtype=torch.bool, device=x.device).triu(1), float("-inf")).softmax(-1)
        z = (att @ v) * gate.view(-1, H_, 1, 1)
        x = x + self.proj(z.transpose(1, 2).reshape(B, T_, D))
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, D), nn.Parameter(torch.randn(T, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(L_)])
        self.ln, self.out = nn.LayerNorm(D), nn.Linear(D, vocab)

    def forward(self, idx, gates):                                 # gates: (L, H) or (L, B, H)
        x = self.emb(idx) + self.pos[: idx.size(1)]
        for l, b in enumerate(self.blocks):
            x = b(x, gates[l])
        return self.out(self.ln(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--abs_price", type=float, default=None, help="same upkeep (nats) at every p")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = open(os.path.join(ROOT, "data", "tinyshakespeare.txt"), encoding="utf-8").read()
    chars = sorted(set(text))
    data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_d, val_d = data[:n], data[n:]
    g = torch.Generator().manual_seed(123)
    NV = 512
    vstart = torch.randint(0, len(val_d) - T - 1, (NV,), generator=g)
    VX = torch.stack([val_d[s:s + T] for s in vstart]).to(dev)
    VY = torch.stack([val_d[s + 1:s + T + 1] for s in vstart]).to(dev)

    for p in a.p:
        # one fixed failure pattern per validation sequence, shared by every candidate gate
        alive = (torch.rand(L_, NV, H_, generator=torch.Generator().manual_seed(999)) >= p).float().to(dev)

        @torch.no_grad()
        def val_loss(model, gate, aware):
            gg = gate[:, None, :] * alive if aware else gate
            return float(F.cross_entropy(model(VX, gg).reshape(-1, len(chars)), VY.reshape(-1)))

        for seed in a.seeds:
            torch.manual_seed(seed)
            model = LM(len(chars)).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
            t0 = time.time()
            for step in range(a.steps):
                s = torch.randint(0, len(train_d) - T - 1, (64,))
                x = torch.stack([train_d[i:i + T] for i in s]).to(dev)
                y = torch.stack([train_d[i + 1:i + T + 1] for i in s]).to(dev)
                lr = 2e-3 * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
                for gp in opt.param_groups:
                    gp["lr"] = lr
                gates = (torch.rand(L_, 64, H_, device=dev) >= p).float()      # whole heads fail
                loss = F.cross_entropy(model(x, gates).reshape(-1, len(chars)), y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            ones, zeros = torch.ones(L_, H_, device=dev), torch.zeros(L_, H_, device=dev)
            res = dict(p=p, seed=seed, train_seconds=time.time() - t0)
            for mode, aware in (("aware", True), ("blind", False)):
                full, empty = val_loss(model, ones, aware), val_loss(model, zeros, aware)
                price = a.abs_price if a.abs_price is not None else a.frac * (empty - full)
                gate, log = ones.clone(), []
                for _ in range(300):                               # the rule, on the frozen model
                    L0 = val_loss(model, gate, aware)
                    vals = torch.zeros(L_, H_)
                    for l in range(L_):
                        for h in range(H_):
                            gg = gate.clone(); gg[l, h] = 1 - gg[l, h]
                            vals[l, h] = val_loss(model, gg, aware) - L0
                    op = gate.cpu() > 0
                    v_open = torch.where(op, vals, torch.tensor(float("inf")))
                    g_closed = torch.where(~op, -vals, torch.tensor(float("-inf")))
                    if g_closed.max() > 2 * price:
                        l, h = divmod(int(g_closed.argmax()), H_); gate[l, h] = 1; log.append(("reopen", l, h))
                    elif v_open.min() < price:
                        l, h = divmod(int(v_open.argmin()), H_); gate[l, h] = 0; log.append(("close", l, h))
                    else:
                        break
                kept = gate.sum(1).int().tolist()
                res[mode] = dict(full=full, empty=empty, price=price, kept=kept, gate=gate.cpu().numpy(),
                                 vals=vals.numpy(), kept_loss=val_loss(model, gate, aware), log=log)
                print(f"p {p:g} seed {seed} [{mode}]: loss all open {full:.3f}, closed {empty:.3f}, "
                      f"kept {kept[0]} + {kept[1]} = {sum(kept)}, loss with kept {res[mode]['kept_loss']:.3f}", flush=True)
            res["seconds"] = time.time() - t0
            pickle.dump(res, open(os.path.join(OUT, f"hf_p{p:g}_s{seed}{'_abs' if a.abs_price is not None else ''}.pkl"), "wb"))


if __name__ == "__main__":
    main()
