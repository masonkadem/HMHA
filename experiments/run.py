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
    for f in ["pool_beta", "kappa_start", "kappa_end", "leak", "lr", "ema"]:
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
    final_gate = torch.tensor(hist["final_gate"], device=device)
    out["recovery"] = circuit_recovery(model, val, cfg, device, gate=final_gate)
    # loss at B = k*, read off DURING annealing (never by re-gating a converged model,
    # which is already adapted to its own final budget and so flatters small B)
    kstar = predicted_kstar(cfg)
    rec = list(zip(hist["n_perfused"], hist["val_loss"]))
    # B does not always land exactly on k* during annealing, so take the closest
    # recorded perfusion level and report which level it actually was.
    b_at, l_at = min(rec, key=lambda bl: (abs(bl[0] - kstar), -rec.index(bl)))
    out["loss_at_kstar"], out["B_at_kstar"], out["kstar"] = l_at, b_at, kstar
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
    # R, N and d_k belong in the tag for the same reason kappa_end and leak do: they are
    # swept, and a tag that omits a swept field silently overwrites the other arm.
    tag = (f"{cfg.task}_R{cfg.n_rel}_N{cfg.seq_len}_dk{cfg.d_k}"
           f"_{cfg.supply}_{cfg.demand}_T{cfg.n_territories}"
           f"_tau{cfg.delay}_k{cfg.kappa_end:g}_b{cfg.pool_beta:g}_lk{cfg.leak:g}"
           f"_s{cfg.seed}")
    with open(os.path.join(a.out, tag + ".pkl"), "wb") as f:
        pickle.dump(out, f)
    lk = out["loss_at_kstar"]
    print(f"  final loss {out['final_loss']:.5f}  loss@B={out['B_at_kstar']}"
          f"(k*={out['kstar']}) {lk:.5f}"
          f"  perfused {out['recovery']['n_perfused']}"
          f"  role_coverage {out['recovery']['role_coverage']:.2f}"
          f"  territory_span {out['recovery']['territory_span']}")
    for n in ["qnorm", "neg_entropy", "outnorm", "demand"]:
        print(f"    rho({n:<12}, ablation) = {out['ablation'][f'rho_{n}']:+.3f}")
    print(f"  wrote {tag}.pkl in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
