"""Removing an ability so that backup heads cannot restore it. Predictions: results/gpt2_remove/predictions.md

Frozen GPT-2 small on IOI (prompts of Wang et al.). Which heads must be removed (mean-ablated) to bring
held-out accuracy to chance? Compared: the heads that matter most when tested once against the full model
(one-shot), and heads chosen one at a time, each valued given those already removed (re-valued).

  python experiments/gpt2_remove.py --seeds 0 1 2 3 4   ->  results/gpt2_remove/remove_s*.pkl
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

OUT = os.path.join(ROOT, "results", "gpt2_remove")
MAX_ROWS = 3000
K = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target", choices=["loss", "margin"], default="loss",
                    help="what removal maximises: loss on the correct name, or minus (correct - subject logit)")
    a = ap.parse_args()
    tag = "" if a.target == "loss" else "margin_"
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    tok = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
    model = BatchedGPT2(dev)
    E = model.w["wte.weight"]
    name = lambda i: f"{i // H_}.{i % H_}"

    for seed in a.seeds:
        fit, test, stats = TASKS["ioiwang"](tok, model, dev, seed)
        if a.target == "margin":                              # "loss" below is then minus the logit margin
            def stats(h, E, g):
                d = []
                for c in range(0, h.size(0), 1500):
                    lg = h[c:c + 1500, 0] @ E.T
                    io, s = g["io"][c:c + 1500], g["s"][c:c + 1500]
                    d.append(lg.gather(1, io[:, None])[:, 0] - lg.gather(1, s[:, None])[:, 0])
                d = torch.cat(d)
                return -d, (d > 0).float()

        def evaluate(gates, data):
            """gates (G, L, H) -> mean loss and accuracy per gate setting."""
            G = gates.size(0)
            loss, acc, cnt = torch.zeros(G, device=dev), torch.zeros(G, device=dev), 0
            for g in data:
                n = g["ids"].size(0)
                step = max(1, MAX_ROWS // n)
                for s in range(0, G, step):
                    gg = gates[s:s + step]
                    h = model.hidden(g["ids"].repeat(gg.size(0), 1), gg.repeat_interleave(n, 0), g["zmean"], g["pos"])
                    rep = dict(io=g["io"].repeat(gg.size(0)), s=g["s"].repeat(gg.size(0)))
                    ce, ok = stats(h, E, rep)
                    loss[s:s + gg.size(0)] += ce.view(gg.size(0), n).sum(1)
                    acc[s:s + gg.size(0)] += ok.view(gg.size(0), n).sum(1)
                cnt += n
            return loss / cnt, acc / cnt

        def gates_of(order):                                  # prefixes of an order: remove 1, 2, ..., K heads
            g = torch.ones(len(order), L_ * H_, device=dev)
            for k in range(len(order)):
                g[k, torch.as_tensor(order[:k + 1], device=dev)] = 0
            return g.view(-1, L_, H_)

        ones = torch.ones(1, L_, H_, device=dev)
        flips = torch.eye(L_ * H_, device=dev)
        base = float(evaluate(ones, fit)[0])
        v0 = (evaluate((1 - flips).view(-1, L_, H_), fit)[0] - base).cpu().numpy()
        oneshot = [int(i) for i in np.argsort(-v0)[:K]]

        gate, greedy = torch.ones(L_ * H_, device=dev), []
        for _ in range(K):
            loss = evaluate((gate[None] * (1 - flips)).view(-1, L_, H_), fit)[0]
            loss[gate == 0] = -float("inf")
            i = int(loss.argmax()); gate[i] = 0; greedy.append(i)

        rng = np.random.default_rng(seed)
        res = dict(seed=seed, oneshot=oneshot, greedy=greedy, v0=v0)
        res["full"] = tuple(float(x) for x in evaluate(ones, test))
        for key, order in (("oneshot", oneshot), ("greedy", greedy)):
            lo, ac = evaluate(gates_of(order), test)
            res[key + "_test"] = (lo.cpu().numpy(), ac.cpu().numpy())
        rl, ra = [], []
        for _ in range(20):
            lo, ac = evaluate(gates_of([int(i) for i in rng.choice(L_ * H_, K, replace=False)]), test)
            rl.append(lo.cpu().numpy()); ra.append(ac.cpu().numpy())
        res["random_test"] = (np.mean(rl, 0), np.mean(ra, 0))
        idx = lambda hs: [l * H_ + h for l, h in hs]
        nm, bk = idx(WANG["name mover"]), idx(WANG["backup name mover"])
        g_pub = torch.ones(2, L_ * H_, device=dev)
        g_pub[0, torch.as_tensor(nm, device=dev)] = 0
        g_pub[1, torch.as_tensor(nm + bk, device=dev)] = 0
        lo, ac = evaluate(g_pub.view(-1, L_, H_), test)
        res["published_test"] = (lo.cpu().numpy(), ac.cpu().numpy())

        first = lambda acc: next((k + 1 for k, v in enumerate(acc) if v <= 0.5), None)
        print(f"\n[seed {seed}] full model: acc {res['full'][1]:.1%}", flush=True)
        print(f"  remove 3 name movers: acc {ac[0]:.1%};  name movers + 8 backups (11): acc {ac[1]:.1%}")
        for key in ("oneshot", "greedy", "random"):
            acc = res[key + "_test"][1]
            print(f"  {key:<8} heads to reach 50%: {first(acc)};  acc after 3/5/10/20 removed: "
                  + ", ".join(f"{acc[k - 1]:.1%}" for k in (3, 5, 10, 20))
                  + ";  loss after 5/10: " + ", ".join(f"{res[key + '_test'][0][k - 1]:.2f}" for k in (5, 10)))
        print("  one-shot order: " + " ".join(name(i) for i in oneshot[:10]))
        print("  re-valued order: " + " ".join(name(i) for i in greedy[:10]), flush=True)
        pickle.dump(res, open(os.path.join(OUT, f"remove_{tag}s{seed}.pkl"), "wb"))


if __name__ == "__main__":
    main()
