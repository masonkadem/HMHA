"""A shared core of GPT-2 small: run hemodynamic attenuation on three tasks and compare the heads kept.

  ioi      indirect object identification ("... John gave a drink to" -> Mary); inhibition-like
  repeat   a random token sequence repeated; predict the repeat (induction); working-memory-like
  greater  "The war lasted from the year 17YY to the year 17" -> a year after YY (Hanna et al. 2023)

Same rule as gpt2_ioi.py (frozen weights, mean ablation over corrupted prompts, greedy minimisation of
loss + upkeep x open heads, upkeep = frac x gap between all-closed and all-open loss). Faster: all 144
single-head flips are evaluated in one batched pass, with a separate valve setting per sequence.

  python experiments/gpt2_core.py --tasks ioi repeat greater --price 0.02   ->  results/gpt2_core/*.pkl
"""
import argparse, os, pickle, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gpt2_ioi import GPT2, make_prompts, ROLE, L_, H_, D, DH

OUT = os.path.join(ROOT, "results", "gpt2_core")
MAX_ROWS = 6000                                           # sequences x scored positions per chunk


class BatchedGPT2(GPT2):
    """Forward pass with one valve setting per sequence: gate (B, L, H). Returns the final hidden states
    (after ln_f) at the positions asked for, so the loss never builds a full (B, T, vocab) tensor."""

    @torch.no_grad()
    def hidden(self, ids, gate, zmean, pos):
        w, (B, T) = self.w, ids.shape
        x = w["wte.weight"][ids] + w["wpe.weight"][:T]
        mask = torch.full((T, T), float("-inf"), device=self.dev).triu(1)
        for l in range(L_):
            p = f"h.{l}."
            a = self.ln(x, p + "ln_1") @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
            q, k, v = (t.view(B, T, H_, DH).transpose(1, 2) for t in a.split(D, -1))
            z = ((q @ k.transpose(-1, -2) / DH ** 0.5 + mask).softmax(-1) @ v).transpose(1, 2)
            g = gate[:, l].view(B, 1, H_, 1)
            z = g * z + (1 - g) * zmean[l]
            x = x + z.reshape(B, T, D) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
            m = F.gelu(self.ln(x, p + "ln_2") @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"], approximate="tanh")
            x = x + m @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
        return self.ln(x[:, pos], "ln_f")                    # (B, P, D)

    def mean_z(self, ids):
        zs = GPT2.__call__(self, ids, record=True)[1]
        return torch.stack([z.mean(0, keepdim=True) for z in zs])   # (L, 1, T, H, DH)


# ---------------------------------------------------------------- tasks
# Each task: list of groups (ids (n, T), pos (positions), zmean, target info), a loss and an accuracy
# computed from the final hidden states h (n, P, D) and the unembedding E (vocab, D).

def task_ioi(tok, model, dev, seed):
    fit, test = make_prompts(tok, 50, 1000 + seed), make_prompts(tok, 50, 2000 + seed)
    abc = make_prompts(tok, 50, 3000 + seed, abc=True)
    zm = [model.mean_z(i.to(dev)) for i, _, _ in abc]
    mk = lambda data: [dict(ids=i.to(dev), pos=[-1], zmean=zm[t], io=io.to(dev), s=s.to(dev)) for t, (i, io, s) in enumerate(data)]

    def stats(h, E, g):
        lg = h[:, 0] @ E.T
        ce = F.cross_entropy(lg, g["io"], reduction="none")
        d = lg.gather(1, g["io"][:, None])[:, 0] - lg.gather(1, g["s"][:, None])[:, 0]
        return ce, (d > 0).float()
    return mk(fit), mk(test), stats


def task_repeat(tok, model, dev, seed, n=100, half=10):
    def make(s, repeat=True):
        rng = np.random.default_rng(s)
        a = rng.integers(1000, 20000, (n, half))
        b = a if repeat else rng.integers(1000, 20000, (n, half))
        return torch.tensor(np.concatenate([np.full((n, 1), 50256), a, b], 1), device=dev)
    zm = model.mean_z(make(3000 + seed, repeat=False))
    pos = list(range(half, 2 * half))                     # predicting tokens of the second half
    mk = lambda ids: [dict(ids=ids, pos=pos, zmean=zm, tgt=ids[:, half + 1:2 * half + 1])]

    def stats(h, E, g):
        lg = h @ E.T                                      # (n, P, vocab), P = 10
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), g["tgt"].reshape(-1), reduction="none").view(len(h), -1)
        return ce.mean(1), (lg.argmax(-1) == g["tgt"]).float().mean(1)
    return mk(make(1000 + seed)), mk(make(2000 + seed)), stats


NOUNS = ["war", "expedition", "siege", "drought", "pilgrimage", "voyage", "dynasty", "conflict",
         "reign", "famine", "occupation", "migration", "rebellion", "plague", "crusade", "blockade"]


def task_greater(tok, model, dev, seed, n=100):
    yy = [tok.encode(f"{y:02d}").ids[0] for y in range(100)]
    YY = torch.tensor(yy, device=dev)

    def make(s, corrupt=False):
        rng = random.Random(s)
        ids, ys = [], []
        for _ in range(n):
            y = rng.randint(2, 98)
            text = f"The {rng.choice(NOUNS)} lasted from the year 17{1 if corrupt else y:02d} to the year 17"
            ids.append(tok.encode(text).ids); ys.append(y)
        lens = {len(i) for i in ids}
        assert len(lens) == 1, lens
        return torch.tensor(ids, device=dev), torch.tensor(ys, device=dev)
    ids_c, _ = make(3000 + seed, corrupt=True)
    zm = model.mean_z(ids_c)
    mk = lambda d: [dict(ids=d[0], pos=[-1], zmean=zm, y=d[1])]

    def stats(h, E, g):
        p = (h[:, 0] @ E[YY].T).softmax(-1)               # over the 100 two-digit years
        later = torch.arange(100, device=dev)[None, :] > g["y"][:, None]
        pl = (p * later).sum(1)
        return -torch.log(pl.clamp_min(1e-9)), (pl > 0.5).float()
    return mk(make(1000 + seed)), mk(make(2000 + seed)), stats


def task_ioi_wang(tok, model, dev, seed, n_fit=10, n_test=50, n_abc=50):
    """IOI on the prompt distribution of Wang et al. (2023): their 15 templates in both name orders,
    their names, places and objects (experiments/ioi_wang_data.py). Words that are not one GPT-2 token
    are dropped. Closed heads take their mean output on the same template with a third name as subject."""
    import ioi_wang_data as W
    one = lambda ws: [x for x in ws if len(tok.encode(" " + x).ids) == 1]
    names, places, objects = one(W.NAMES), one(W.PLACES), one(W.OBJECTS)
    swap = lambda t: t.replace("[B]", "[X]", 1).replace("[A]", "[B]", 1).replace("[X]", "[A]", 1)
    templates = W.BABA_TEMPLATES + [swap(t) for t in W.BABA_TEMPLATES]
    print(f"[ioiwang] {len(templates)} templates, {len(names)}/{len(W.NAMES)} names, {len(places)} places, "
          f"{len(objects)} objects; fit {n_fit}, test {n_test} per template", flush=True)

    def make(n_per, s, abc=False):
        rng, out = random.Random(s), []
        for tpl in templates:
            assert tpl.endswith(" [A]")
            body = tpl[:-4]
            i2 = body.index("[B]", body.index("[B]") + 1)             # second mention of the subject
            ids, io, sj = [], [], []
            for _ in range(n_per):
                A, B_, C = rng.sample(names, 3)
                t = body[:i2] + ("[C]" if abc else "[B]") + body[i2 + 3:]
                t = t.replace("[A]", A).replace("[B]", B_).replace("[C]", C)
                t = t.replace("[PLACE]", rng.choice(places)).replace("[OBJECT]", rng.choice(objects))
                ids.append(tok.encode(t).ids)
                io.append(tok.encode(" " + A).ids[0]); sj.append(tok.encode(" " + B_).ids[0])
            assert len({len(i) for i in ids}) == 1, tpl
            out.append((torch.tensor(ids), torch.tensor(io), torch.tensor(sj)))
        return out

    zm = [model.mean_z(i.to(dev)) for i, _, _ in make(n_abc, 3000 + seed, abc=True)]
    mk = lambda data: [dict(ids=i.to(dev), pos=[-1], zmean=zm[t], io=io.to(dev), s=s.to(dev)) for t, (i, io, s) in enumerate(data)]

    def stats(h, E, g):                                    # in chunks: full logits are 50k wide
        ce, ok = [], []
        for c in range(0, h.size(0), 1500):
            lg = h[c:c + 1500, 0] @ E.T
            io, s = g["io"][c:c + 1500], g["s"][c:c + 1500]
            ce.append(F.cross_entropy(lg, io, reduction="none"))
            ok.append((lg.gather(1, io[:, None])[:, 0] > lg.gather(1, s[:, None])[:, 0]).float())
        return torch.cat(ce), torch.cat(ok)
    return mk(make(n_fit, 1000 + seed)), mk(make(n_test, 2000 + seed)), stats


TASKS = dict(ioi=task_ioi, repeat=task_repeat, greater=task_greater, ioiwang=task_ioi_wang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["ioi", "repeat", "greater"])
    ap.add_argument("--price", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_it", type=int, default=1000, help="smoke tests only")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    tok = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    model = BatchedGPT2(dev)
    E = model.w["wte.weight"]

    for name in a.tasks:
        fit, test, stats = TASKS[name](tok, model, dev, a.seed)

        def evaluate(gates, data):
            """gates (G, L, H) -> mean loss and accuracy per gate setting, (G,), (G,)."""
            G = gates.size(0)
            loss, acc, cnt = torch.zeros(G, device=dev), torch.zeros(G, device=dev), 0
            for g in data:
                n = g["ids"].size(0)
                step = max(1, MAX_ROWS // (n * len(g["pos"])))
                for s in range(0, G, step):
                    gg = gates[s:s + step]
                    ids = g["ids"].repeat(gg.size(0), 1)
                    gate = gg.repeat_interleave(n, 0)
                    h = model.hidden(ids, gate, g["zmean"], g["pos"])
                    rep = {k: (v.repeat(gg.size(0), *[1] * (v.dim() - 1)) if torch.is_tensor(v) and k not in ("ids", "zmean") else v)
                           for k, v in g.items()}
                    ce, ok = stats(h, E, rep)
                    loss[s:s + gg.size(0)] += ce.view(gg.size(0), n).sum(1)
                    acc[s:s + gg.size(0)] += ok.view(gg.size(0), n).sum(1)
                cnt += n
            return loss / cnt, acc / cnt

        ones, zeros = torch.ones(1, L_, H_, device=dev), torch.zeros(1, L_, H_, device=dev)
        full, empty = evaluate(ones, fit), evaluate(zeros, fit)
        gap = float(empty[0] - full[0])
        price = a.price * gap
        print(f"\n[{name}] all open: loss {float(full[0]):.3f} acc {float(full[1]):.1%};  all closed: loss {float(empty[0]):.3f} "
              f"acc {float(empty[1]):.1%};  upkeep {price:.4f}", flush=True)
        gate, t0, log = torch.ones(L_, H_, device=dev), time.time(), []
        flips = torch.eye(L_ * H_, device=dev).view(-1, L_, H_)
        vals_first = None
        for it in range(a.max_it):
            L = float(evaluate(gate[None], fit)[0])
            trial = (gate[None] + flips) % 2                   # every single-head flip
            vals = (evaluate(trial, fit)[0] - L).view(L_, H_)  # closing: value; opening: minus gain
            if vals_first is None:
                vals_first = vals.cpu().numpy().copy()
            open_ = gate > 0
            v_open = torch.where(open_, vals, torch.full_like(vals, float("inf")))
            g_closed = torch.where(~open_, -vals, torch.full_like(vals, float("-inf")))
            if g_closed.max() > 2 * price:
                l, h = divmod(int(g_closed.argmax()), H_); gate[l, h] = 1; act = "reopen"
            elif v_open.min() < price:
                l, h = divmod(int(v_open.argmin()), H_); gate[l, h] = 0; act = "close"
            else:
                break
            log.append((act, l, h, int(gate.sum())))
        kept = [(l, h) for l in range(L_) for h in range(H_) if gate[l, h] > 0]
        te_full, te = evaluate(ones, test), evaluate(gate[None], test)
        print(f"[{name}] kept {len(kept)} after {len(log)} decisions, {time.time() - t0:.0f} s; held out: "
              f"loss {float(te[0]):.3f} (full {float(te_full[0]):.3f}), acc {float(te[1]):.1%} (full {float(te_full[1]):.1%})")
        print(f"[{name}] kept: {' '.join(f'{l}.{h}' for l, h in kept)}", flush=True)
        n_k = len(kept)
        order = np.argsort(-vals_first.ravel())[:n_k]
        g_one = torch.zeros(1, L_, H_, device=dev); g_one.view(-1)[torch.as_tensor(order, device=dev)] = 1
        one = evaluate(g_one, test)
        rng = np.random.default_rng(a.seed)
        g_rand = torch.zeros(50, L_ * H_, device=dev)
        for i in range(50):
            g_rand[i, torch.as_tensor(rng.choice(L_ * H_, n_k, replace=False), device=dev)] = 1
        rnd = evaluate(g_rand.view(50, L_, H_), test)
        masked = [(l, h) for (l, h) in kept if vals_first[l, h] < price]
        print(f"[{name}] same {n_k} heads: one-shot acc {float(one[1]):.1%}, random acc {float(rnd[1].mean()):.1%} "
              f"(best {float(rnd[1].max()):.1%}); kept heads a one-shot value would drop: {masked}", flush=True)
        g_pub = torch.zeros(1, L_, H_, device=dev)             # the published IOI circuit, same prompts
        for (l, h) in ROLE:
            g_pub[0, l, h] = 1
        pub = evaluate(g_pub, test)
        if name.startswith("ioi"):
            print(f"[{name}] published {len(ROLE)}-head circuit: loss {float(pub[0]):.3f}, acc {float(pub[1]):.1%}; "
                  f"{sum(k in ROLE for k in kept)} of {n_k} kept heads are in it", flush=True)
        pickle.dump(dict(task=name, price_frac=a.price, price=price, kept=kept, log=log, seed=a.seed,
                         published_test=(float(pub[0]), float(pub[1])),
                         full=(float(full[0]), float(full[1])), empty=(float(empty[0]), float(empty[1])),
                         test=(float(te[0]), float(te[1])), test_full=(float(te_full[0]), float(te_full[1])),
                         vals_first=vals_first, vals_last=vals.cpu().numpy(), oneshot_test=(float(one[0]), float(one[1])),
                         random_test=(rnd[0].cpu().numpy(), rnd[1].cpu().numpy()), masked=masked,
                         seconds=time.time() - t0),
                    open(os.path.join(a.out, f"{name}_pf{a.price:g}_s{a.seed}.pkl"), "wb"))


if __name__ == "__main__":
    main()
