"""Logit difference on IOI and a cross-task test of the IOI heads. Predictions: results/gpt2_cross/predictions.md

  python experiments/gpt2_cross.py   ->  results/gpt2_cross/cross_s*.pkl   (CPU is enough)
"""
import argparse, os, pickle, sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gpt2_ioi import WANG, L_, H_
from gpt2_core import BatchedGPT2, TASKS

OUT = os.path.join(ROOT, "results", "gpt2_cross")
SRC = os.path.join(ROOT, "results", "gpt2_ioi_wang")
FR = ("0.05", "0.02", "0.01")
MAX_ROWS = 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dev", default="cpu")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(a.dev)
    tok = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    model = BatchedGPT2(dev)
    E = model.w["wte.weight"]
    flat = lambda hs: [l * H_ + h for l, h in hs]

    def gates(keep_sets):                                     # list of head-index lists that stay OPEN
        g = torch.zeros(len(keep_sets), L_ * H_, device=dev)
        for i, ks in enumerate(keep_sets):
            g[i, torch.as_tensor(sorted(ks), device=dev, dtype=torch.long)] = 1
        return g.view(-1, L_, H_)

    def run(gs, data, fn):
        """fn(h, group) -> (n*,) per-row score; returns mean per gate setting."""
        G, tot, cnt = gs.size(0), torch.zeros(gs.size(0)), 0
        for g in data:
            n = g["ids"].size(0)
            step = max(1, MAX_ROWS // (n * len(g["pos"])))
            for s in range(0, G, step):
                gg = gs[s:s + step]
                h = model.hidden(g["ids"].repeat(gg.size(0), 1), gg.repeat_interleave(n, 0), g["zmean"], g["pos"])
                rep = {k: (v.repeat(gg.size(0), *[1] * (v.dim() - 1)) if torch.is_tensor(v) and k not in ("ids", "zmean") else v)
                       for k, v in g.items()}
                tot[s:s + gg.size(0)] += fn(h, rep).view(gg.size(0), n).sum(1).cpu()
            cnt += n
        return (tot / cnt).numpy()

    def ld(h, g):                                             # correct-name logit minus subject logit
        out = []
        for c in range(0, h.size(0), 1500):
            lg = h[c:c + 1500, 0] @ E.T
            out.append(lg.gather(1, g["io"][c:c + 1500, None])[:, 0] - lg.gather(1, g["s"][c:c + 1500, None])[:, 0])
        return torch.cat(out)

    other = {}
    for name in ("greater", "repeat"):
        _, test, stats = TASKS[name](tok, model, dev, 0)
        other[name] = (test, stats)
    allh = list(range(L_ * H_))
    pub = flat([h for hs in WANG.values() for h in hs])

    for seed in a.seeds:
        R = {f: pickle.load(open(os.path.join(SRC, f"ioiwang_pf{f}_s{seed}.pkl"), "rb")) for f in FR}
        kept = {f: flat(R[f]["kept"]) for f in FR}
        order = np.argsort(-R["0.01"]["vals_first"].reshape(-1))        # ranked once against the full model
        once = {f: [int(i) for i in order[:len(kept[f])]] for f in FR}
        res = dict(seed=seed, n={f: len(kept[f]) for f in FR})

        # A. logit difference and accuracy on IOI
        _, test, _ = TASKS["ioiwang"](tok, model, dev, seed)
        sets = [allh, pub] + [kept[f] for f in FR] + [once[f] for f in FR]
        d = run(gates(sets), test, ld)
        acc = run(gates(sets), test, lambda h, g: (ld(h, g) > 0).float())
        res["ld"] = dict(full=d[0], published=d[1], rule={f: d[2 + i] for i, f in enumerate(FR)},
                         once={f: d[5 + i] for i, f in enumerate(FR)})
        res["acc"] = dict(full=acc[0], published=acc[1], rule={f: acc[2 + i] for i, f in enumerate(FR)},
                          once={f: acc[5 + i] for i, f in enumerate(FR)})
        print(f"\n[seed {seed}] IOI logit difference, full {d[0]:.2f}; as a fraction of full (accuracy):", flush=True)
        print(f"  published 26: {d[1] / d[0]:.2f} ({acc[1]:.1%})")
        for i, f in enumerate(FR):
            print(f"  upkeep {f}: rule {len(kept[f])} heads {d[2 + i] / d[0]:.2f} ({acc[2 + i]:.1%});  "
                  f"ranked once {d[5 + i] / d[0]:.2f} ({acc[5 + i]:.1%})")

        # B. other tasks: keep only the IOI heads; close only the IOI heads
        k = kept["0.01"]
        rng = np.random.default_rng(seed)
        rand = [[int(i) for i in rng.choice(L_ * H_, len(k), replace=False)] for _ in range(20)]
        comp = lambda ks: [i for i in allh if i not in set(ks)]
        sets = [allh, k, comp(k)] + rand + [comp(r) for r in rand]
        for name, (test, stats) in other.items():
            acc = run(gates(sets), test, lambda h, g: stats(h, E, g)[1])
            res[name] = dict(full=acc[0], keep_only=acc[1], close_only=acc[2],
                             keep_only_random=acc[3:23], close_only_random=acc[23:43])
            print(f"  {name}: full {acc[0]:.1%};  only the {len(k)} IOI heads open {acc[1]:.1%} "
                  f"(random {len(k)}: {acc[3:23].mean():.1%});  only the IOI heads closed {acc[2]:.1%} "
                  f"(random {len(k)} closed: {acc[23:43].mean():.1%}, min {acc[23:43].min():.1%})", flush=True)
        pickle.dump(res, open(os.path.join(OUT, f"cross_s{seed}.pkl"), "wb"))


if __name__ == "__main__":
    main()
