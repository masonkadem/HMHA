"""Caches what Figure 1's task panels need: a real trained model's per-head attention
offset profile, plus one concrete example of the task's input and target.

Run once; aggregate.py reads figures/task_panel.pkl. Nothing here is illustrative-only,
the head roles panel is measured from a trained model at its converged gate.

  python experiments/task_panel.py
"""
import os, pickle, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from hemo.config import Cfg
from hemo.tasks import make_val, make_batch, offsets_for, trivial_loss
from hemo.train import train, pick_device
from hemo.analysis import circuit_recovery, head_offsets

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "task_panel.pkl")


def main():
    cfg = Cfg(device="cpu", seed=0)
    dev = pick_device(cfg)
    val = make_val(cfg, dev)
    print(f"training seed {cfg.seed}, R={cfg.n_rel}, offsets {offsets_for(cfg)}")
    model, hist = train(cfg, val, dev, hemo=True, desc="taskfig", verbose=False)
    gate = torch.tensor(hist["final_gate"], device=dev)
    rec = circuit_recovery(model, val, cfg, dev, gate=gate)

    # one concrete example, for the schematic panel
    X, Y, T, aux = make_batch(1, cfg, dev, gen=torch.Generator().manual_seed(3))
    ex = {"p": int(aux["p"][0, 0]), "offsets": offsets_for(cfg), "N": cfg.seq_len,
          "m": cfg.m_content,
          "content": Y[0, :, cfg.seq_len:cfg.seq_len + cfg.m_content].cpu().numpy(),
          "target_row": T[0, 0].cpu().numpy()}

    # a dense 4-head model shows the roles without any starvation in the picture
    dense, _ = train(cfg, val, dev, hemo=False, num_heads=cfg.n_rel,
                     steps=int(cfg.steps * cfg.scratch_mult))
    dense_prof = head_offsets(dense, val, cfg, dev)

    out = {"cfg": {k: getattr(cfg, k) for k in
                   ["task", "seq_len", "n_rel", "m_content", "num_heads", "d_k",
                    "kappa_end", "steps"]},
           "offsets": offsets_for(cfg), "trivial": trivial_loss(val),
           "hemo_profile": rec["offset_profile"], "hemo_gate": gate.cpu().numpy(),
           "hemo_active": rec["active"], "hemo_purity": rec["head_purity"],
           "hemo_role": rec["head_role"], "n_perfused": rec["n_perfused"],
           "role_coverage": rec["role_coverage"],
           "dense_profile": dense_prof, "final_loss": hist["val_loss"][-1],
           "example": ex}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"perfused {rec['n_perfused']}, role_coverage {rec['role_coverage']:.2f}, "
          f"loss {hist['val_loss'][-1]:.5f}")
    print(f"dense {cfg.n_rel}-head roles: {sorted(dense_prof.argmax(1).tolist())} "
          f"vs true {offsets_for(cfg)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
