"""One seed of the two-layer induction benchmark. Writes one pickle per run.

  python experiments/induction_run.py --rule collateral --seed 0
  python experiments/induction_run.py --rule dense --heads1 1 --heads2 1 --seed 0   # ground truth
"""
import argparse, os, pickle, sys, time
from dataclasses import fields, replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from hemo.induction import ICfg, train

ap = argparse.ArgumentParser()
for f in fields(ICfg):
    ap.add_argument(f"--{f.name}", type=type(f.default), default=None)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "induction"))
a = ap.parse_args()
cfg = replace(ICfg(), **{k: v for k, v in vars(a).items() if v is not None and k != "out"})

t0 = time.time()
_, out = train(cfg)
h1, h2 = out["sizes"]
kept = out["kept"]
mode = ("" if cfg.price_mode == "absolute" else f"_{cfg.price_mode}") + ("" if cfg.probe == "mse" else f"_{cfg.probe}") + ("" if cfg.squeeze_at == 0 else f"_sq{cfg.squeeze_at:g}") + ("" if not cfg.trial else f"_trial{cfg.trial_taper}")
tag = (f"ind_{cfg.rule}{mode}_h{h1}-{h2}_pf{cfg.price_frac:g}_tp{cfg.taper}_sf{cfg.start_frac:g}"
       f"_L{cfg.L}_g{cfg.gap_max}_p{cfg.pre_max}_st{cfg.steps}_s{cfg.seed}")
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, tag + ".pkl"), "wb") as f:
    pickle.dump(out, f)
print(f"{tag}: loss {out['final_loss']:.4f} acc {out['final_acc']:.3f} kept layer1 {int(kept[:h1].sum())} "
      f"layer2 {int(kept[h1:].sum())} compute {out['compute']:.2f} [{time.time() - t0:.0f}s]")
