"""GPT-2 small IOI: are the heads our rule keeps better than simpler choices of the same size?

For each upkeep already run (results/gpt2_ioi/ioi_pf*_s0.pkl), with n = number of heads kept:
  ours      the heads the rule kept
  one-shot  the n heads whose single removal from the full model raises the loss most
            (one round of the same value, no re-evaluation: a one-shot heuristic)
  random    n heads drawn uniformly, 200 draws
All evaluated on the same held-out prompts as gpt2_ioi.py (seed 2000), same mean ablation.

  python experiments/gpt2_baselines.py  ->  results/gpt2_ioi/baselines_s0.pkl
"""
import os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gpt2_ioi import GPT2, make_prompts, ROLE, L_, H_

SEED, N, DRAWS = 0, 50, 200


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    model = GPT2(dev)
    to = lambda d: [(i.to(dev), io.to(dev), s.to(dev)) for i, io, s in d]
    fit = to(make_prompts(tok, N, 1000 + SEED))
    test = to(make_prompts(tok, N, 2000 + SEED))
    abc = to(make_prompts(tok, N, 3000 + SEED, abc=True))
    zmean = []
    for ids, _, _ in abc:
        zs = model(ids, record=True)[1]
        zmean.append(torch.stack([z.mean(0, keepdim=True) for z in zs]))

    def evaluate(gate, data):
        ce, ld, acc, n = 0.0, 0.0, 0.0, 0
        for t, (ids, io, s) in enumerate(data):
            lg = model(ids, gate, zmean[t])
            ce += F.cross_entropy(lg, io, reduction="sum").item()
            d = lg.gather(1, io[:, None]) - lg.gather(1, s[:, None])
            ld += d.sum().item(); acc += (d > 0).float().sum().item(); n += len(ids)
        return ce / n, ld / n, acc / n

    def gate_of(heads):
        g = torch.zeros(L_, H_, device=dev)
        for l, h in heads:
            g[l, h] = 1
        return g

    t0 = time.time()
    ones = torch.ones(L_, H_, device=dev)
    full_fit = evaluate(ones, fit)[0]
    single = np.zeros((L_, H_))                       # loss rise when one head is removed from the full model
    for l in range(L_):
        for h in range(H_):
            g = ones.clone(); g[l, h] = 0
            single[l, h] = evaluate(g, fit)[0] - full_fit
    order = [divmod(int(i), H_) for i in np.argsort(-single.ravel())]
    allh = [(l, h) for l in range(L_) for h in range(H_)]
    rng = np.random.default_rng(SEED)
    out = dict(single=single, sizes={})
    for frac in ("0.05", "0.02", "0.01"):
        r = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", f"ioi_pf{frac}_s{SEED}.pkl"), "rb"))
        ours = [tuple(k) for k in r["kept"]]
        n = len(ours)
        one = order[:n]
        rand = [evaluate(gate_of([allh[i] for i in rng.choice(144, n, replace=False)]), test) for _ in range(DRAWS)]
        res = dict(n=n, ours=evaluate(gate_of(ours), test), oneshot=evaluate(gate_of(one), test),
                   random=np.array(rand), ours_in=sum(k in ROLE for k in ours), one_in=sum(k in ROLE for k in one),
                   oneshot_heads=one, full=r["full"], empty=r["empty"])
        out["sizes"][frac] = res
        rec = lambda x: (x[1] - r["empty"][1]) / (r["full"][1] - r["empty"][1])
        ra = res["random"]
        print(f"upkeep {frac}, n={n}:")
        print(f"  ours     loss {res['ours'][0]:.3f}  logit diff {rec(res['ours']):.0%}  acc {res['ours'][2]:.1%}  in circuit {res['ours_in']}/{n}")
        print(f"  one-shot loss {res['oneshot'][0]:.3f}  logit diff {rec(res['oneshot']):.0%}  acc {res['oneshot'][2]:.1%}  in circuit {res['one_in']}/{n}")
        print(f"  random   loss {ra[:, 0].mean():.3f} (best {ra[:, 0].min():.3f})  logit diff {np.mean([rec(x) for x in ra]):.0%}"
              f"  acc {ra[:, 2].mean():.1%} (best {ra[:, 2].max():.1%})", flush=True)
    out["seconds"] = time.time() - t0
    pickle.dump(out, open(os.path.join(ROOT, "results", "gpt2_ioi", f"baselines_s{SEED}.pkl"), "wb"))
    print(f"done in {out['seconds']:.0f} s")


if __name__ == "__main__":
    main()
