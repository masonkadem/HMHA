"""Single-seed experiment runner. Writes one pickle per seed so long runs survive
interruption. Aggregate and plot with experiments/aggregate.py.

  python experiments/run.py --seed 0 --supply threshold --demand outnorm_ema
"""
import argparse, os, pickle, sys, time
from dataclasses import replace, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from hemo.config import Cfg
from hemo.tasks import make_val, trivial_loss, predicted_kstar
from hemo.train import train, pick_device
from hemo.analysis import (redundancy_check, circuit_recovery, head_ablation, k_list)


def main():
    ap = argparse.ArgumentParser()
    for f, t in [("seed", int), ("steps", int), ("num_heads", int), ("n_rel", int),
                 ("seq_len", int), ("d_k", int), ("n_territories", int), ("delay", int),
                 ("val_size", int)]:
        ap.add_argument(f"--{f}", type=t, default=None)
    for f in ["supply", "demand", "task", "device"]:
        ap.add_argument(f"--{f}", type=str, default=None)
    for f in ["pool_beta", "kappa_end", "leak", "lr"]:
        ap.add_argument(f"--{f}", type=float, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-redundancy", action="store_true")
    a = ap.parse_args()

    cfg = Cfg()
    for k, v in vars(a).items():
        if v is not None and hasattr(cfg, k):
            cfg = replace(cfg, **{k: v})
    device = pick_device(cfg)
    print(f"device {device}  task {cfg.task}  supply {cfg.supply}  demand {cfg.demand}  "
          f"seed {cfg.seed}  predicted k* {predicted_kstar(cfg)}")

    val = make_val(cfg, device)
    ks = k_list(cfg.num_heads, extra=(predicted_kstar(cfg),))
    out = {"cfg": asdict(cfg), "trivial": trivial_loss(val), "ks": ks}
    t0 = time.time()

    if not a.skip_redundancy:
        print("  ground-truth k* (dense, k heads from scratch)")
        out["redundancy"] = redundancy_check(cfg, val, device, ks)

    model, hist = train(cfg, val, device, hemo=True, desc="hemo", verbose=True)
    out["hist"] = hist
    out["final_loss"] = hist["val_loss"][-1]
    out["ablation"] = head_ablation(model, val)               # at FULL perfusion
    out["recovery"] = circuit_recovery(model, val, cfg, device)
    # circuit recovery as a function of perfusion, using the converged demand field
    out["recovery_vs_kappa"] = {}
    for kap in [-3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        g = model.gate(kappa=kap)
        out["recovery_vs_kappa"][kap] = {
            "loss": __import__("hemo.train", fromlist=["evaluate"]).evaluate(
                model, val, gate=g),
            **{k: v for k, v in circuit_recovery(model, val, cfg, device, gate=g).items()
               if k in ("n_perfused", "role_coverage", "role_purity",
                        "n_distinct_roles", "territory_span")}}

    os.makedirs(a.out, exist_ok=True)
    tag = f"{cfg.task}_{cfg.supply}_{cfg.demand}_T{cfg.n_territories}_tau{cfg.delay}_s{cfg.seed}"
    with open(os.path.join(a.out, tag + ".pkl"), "wb") as f:
        pickle.dump(out, f)
    print(f"  final loss {out['final_loss']:.5f}  perfused {out['recovery']['n_perfused']}"
          f"  role_coverage {out['recovery']['role_coverage']:.2f}"
          f"  territory_span {out['recovery']['territory_span']}")
    for n in ["qnorm", "neg_entropy", "outnorm", "demand"]:
        print(f"    rho({n:<12}, ablation) = {out['ablation'][f'rho_{n}']:+.3f}")
    print(f"  wrote {tag}.pkl in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
