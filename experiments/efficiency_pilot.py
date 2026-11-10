"""Pilot: how much faster is a training step once closed heads are deleted?

Times forward + backward + optimiser step of attention layers with all heads against only the
heads the rule kept, at three sizes. Head counts come from our results:
  retrieval  1 layer, 32 heads -> 4 kept            (d_model 128, sequence 16)
  induction  2 layers, 8 + 8 heads -> 2 + 1 kept    (d_model 64, sequence 48)
  GPT-2-like 12 layers x 12 heads -> 17 kept heads in total, as on IOI, spread by layer as kept
             (d_model 768, sequence 128); attention only, MLPs are not pruned by the rule
CPU, one thread, median of repeated steps. A pilot: it measures the attention part only.

  python experiments/efficiency_pilot.py  ->  results/efficiency_pilot.txt
"""
import os, pickle, sys, time
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.induction import AttnLayer

torch.set_num_threads(1)


def step_time(heads_per_layer, d_model, d_head, T, batch, reps=15):
    layers = [AttnLayer(d_model, d_head, h) for h in heads_per_layer if h > 0]
    params = [p for l in layers for p in l.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3)
    x0 = torch.randn(batch, T, d_model)
    times = []
    for i in range(reps + 3):
        t0 = time.perf_counter()
        x = x0
        for l in layers:
            x = x + l(x)[0].sum(1)
        loss = (x ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if i >= 3:
            times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000, sum(p.numel() for p in params)


g2 = pickle.load(open(os.path.join(ROOT, "results", "gpt2_ioi", "ioi_pf0.02_s0.pkl"), "rb"))
per_layer = [sum(1 for l, h in g2["kept"] if l == L) for L in range(12)]
cases = [("retrieval", [32], [4], 128, 32, 16, 64),
         ("induction", [8, 8], [2, 1], 64, 16, 48, 64),
         ("GPT-2-like", [12] * 12, per_layer, 768, 64, 128, 4)]
lines = []
for name, full, kept, d, dh, T, B in cases:
    tf, pf = step_time(full, d, dh, T, B)
    tk, pk = step_time(kept, d, dh, T, B)
    lines.append(f"{name:<11} heads {sum(full):>3} -> {sum(kept):>3}   step {tf:8.1f} ms -> {tk:7.1f} ms "
                 f"({tf / tk:4.1f}x faster)   attention parameters {pf:>9,} -> {pk:>8,} ({pk / pf:.0%})")
    print(lines[-1], flush=True)
open(os.path.join(ROOT, "results", "efficiency_pilot.txt"), "w").write(
    "Attention layers only, CPU, one thread, median step (forward + backward + AdamW)\n" + "\n".join(lines) + "\n")
