"""Pilot: does ordinary dropout make a small language model keep more heads?
Prediction written first: results/dropout_lm/predictions.md

Character-level transformer on Tiny Shakespeare (data/tinyshakespeare.txt), 2 layers x 8 heads,
trained with dropout p. Then, frozen, the hemodynamic attenuation rule decides how many heads stay.

  python experiments/dropout_lm.py --p 0 0.1 0.2 0.3 --seeds 0 1  ->  results/dropout_lm/*.pkl
"""
import argparse, math, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "dropout_lm")
L_, H_, D, T = 2, 8, 128, 128
DH = D // H_


class Block(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.drop, self.p = nn.Dropout(p), p

    def forward(self, x, gate):
        B, T_, _ = x.shape
        q, k, v = (t.view(B, T_, H_, DH).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(D, -1))
        att = (q @ k.transpose(-1, -2) / math.sqrt(DH)).masked_fill(
            torch.ones(T_, T_, dtype=torch.bool, device=x.device).triu(1), float("-inf")).softmax(-1)
        z = self.drop(att) @ v                                     # (B, H, T, DH)
        z = z * gate.view(1, H_, 1, 1)
        x = x + self.drop(self.proj(z.transpose(1, 2).reshape(B, T_, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class LM(nn.Module):
    def __init__(self, vocab, p):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, D), nn.Parameter(torch.randn(T, D) * 0.02)
        self.blocks = nn.ModuleList([Block(p) for _ in range(L_)])
        self.ln, self.out = nn.LayerNorm(D), nn.Linear(D, vocab)

    def forward(self, idx, gates=None):
        gates = torch.ones(L_, H_, device=idx.device) if gates is None else gates
        x = self.emb(idx) + self.pos[: idx.size(1)]
        for l, b in enumerate(self.blocks):
            x = b(x, gates[l])
        return self.out(self.ln(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--under_dropout", type=int, default=0, help="value heads with dropout on")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = open(os.path.join(ROOT, "data", "tinyshakespeare.txt"), encoding="utf-8").read()
    chars = sorted(set(text))
    data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_d, val_d = data[:n], data[n:]
    g = torch.Generator().manual_seed(123)
    vstart = torch.randint(0, len(val_d) - T - 1, (256,), generator=g)
    VX = torch.stack([val_d[s:s + T] for s in vstart]).to(dev)
    VY = torch.stack([val_d[s + 1:s + T + 1] for s in vstart]).to(dev)

    @torch.no_grad()
    def val_loss(model, gates):
        if not a.under_dropout:
            return float(F.cross_entropy(model(VX, gates).reshape(-1, len(chars)), VY.reshape(-1)))
        # value under the failures the model faces: dropout on, same random patterns for every gate
        model.train()
        out = 0.0
        for k in range(4):
            torch.manual_seed(777 + k)
            out += float(F.cross_entropy(model(VX, gates).reshape(-1, len(chars)), VY.reshape(-1))) / 4
        model.eval()
        return out

    for p in a.p:
        for seed in a.seeds:
            torch.manual_seed(seed)
            model = LM(len(chars), p).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
            t0 = time.time()
            model.train()
            for step in range(a.steps):
                s = torch.randint(0, len(train_d) - T - 1, (64,))
                x = torch.stack([train_d[i:i + T] for i in s]).to(dev)
                y = torch.stack([train_d[i + 1:i + T + 1] for i in s]).to(dev)
                lr = 2e-3 * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
                for gp in opt.param_groups:
                    gp["lr"] = lr
                loss = F.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            ones, zeros = torch.ones(L_, H_, device=dev), torch.zeros(L_, H_, device=dev)
            full, empty = val_loss(model, ones), val_loss(model, zeros)
            price = a.frac * (empty - full)
            gate, log = ones.clone(), []
            for _ in range(200):                                   # the rule, on the frozen model
                L0 = val_loss(model, gate)
                vals = torch.zeros(L_, H_)
                for l in range(L_):
                    for h in range(H_):
                        gg = gate.clone(); gg[l, h] = 1 - gg[l, h]
                        vals[l, h] = val_loss(model, gg) - L0
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
            res = dict(p=p, seed=seed, full=full, empty=empty, kept_loss=val_loss(model, gate), kept=kept,
                       gate=gate.cpu().numpy(), log=log, price=price, seconds=time.time() - t0)
            pickle.dump(res, open(os.path.join(OUT, f"lm_p{p:g}_s{seed}{'_ud' if a.under_dropout else ''}.pkl"), "wb"))
            print(f"[{'under dropout' if a.under_dropout else 'dropout off'}] dropout {p:g} seed {seed}: val loss {full:.3f}, heads kept {kept[0]} + {kept[1]} = {sum(kept)}, "
                  f"loss with kept heads {res['kept_loss']:.3f}, {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
