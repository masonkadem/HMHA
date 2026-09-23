"""Generates the job list for the ranked roadmap in README.md, one seed per process.

Every job is a full `experiments/run.py` invocation. Jobs whose result pickle already
exists are skipped, so an interrupted sweep resumes where it stopped.

  python experiments/sweep.py --list            # print the jobs
  python experiments/sweep.py --run --workers 4 # run them
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from hemo.config import Cfg


def tag(**kw):
    c = Cfg(**{k: v for k, v in kw.items() if k in Cfg.__dataclass_fields__})
    sys.path.insert(0, HERE)
    from run import result_tag
    return result_tag(c)


def jobs():
    """(experiment, kwargs) in roadmap order. Duplicates across experiments are emitted
    once; the aggregator reads the same pickle from both sweeps."""
    out, seen = [], set()

    def add(exp, **kw):
        t = tag(**kw)
        if t in seen:
            return
        seen.add(t)
        out.append((exp, kw, t))

    # 0. ground truth. Dense k-head models from scratch, the k* the whole repo rests on.
    for s in range(2):
        add("gt", seed=s, redundancy=True)

    # 1. emergent B vs true k*. threshold supply, sweep kappa_end, 5 seeds.
    for k in [0.5, 1.0, 1.5, 2.0, 2.5]:
        for s in range(5):
            add("emergent_B", seed=s, supply="threshold", demand="outnorm_ema",
                kappa_end=k)

    # 2. territory compaction. sweep n_territories at R=4.
    for T in [1, 2, 4, 8, 16, 32]:
        for s in range(3):
            add("territory", seed=s, supply="territory", demand="outnorm_ema",
                n_territories=T)

    # 3. delay. tau sweep. tau=0 is experiment 1's kappa_end=1.5 arm, reused.
    for tau in [0, 1, 5, 20, 100]:
        for s in range(3):
            add("delay", seed=s, supply="threshold", demand="outnorm_ema", delay=tau)

    # 4. pooled demand, with territory supply. beta=0 is experiment 2's T=8 arm, reused.
    for b in [0.0, 0.25, 0.5, 1.0]:
        for s in range(3):
            add("pool", seed=s, supply="territory", demand="outnorm_ema",
                n_territories=8, pool_beta=b)

    # 6. does emergent B TRACK k*, or did kappa_end merely happen to print 4? Sweep the
    #    task's true circuit size at FIXED kappa_end. Seed 0 of each R also runs the
    #    dense-from-scratch check, so B is compared against a MEASURED k*, not a
    #    predicted one. This is the experiment the headline claim rests on.
    for R in [2, 3, 4, 6, 8]:
        for s_ in range(3):
            add("kstar_track", seed=s_, supply="threshold", demand="outnorm_ema",
                n_rel=R, kappa_end=1.5, **({"redundancy": True} if s_ == 0 else {}))

    # 7. control for experiment 6. The hemo arm gets cfg.steps while the dense k* check
    #    gets scratch_mult x that, so B saturating at large R may be under-training
    #    rather than the mechanism. Re-run the two worst R at the dense budget.
    for R in [6, 8]:
        for s_ in range(3):
            add("kstar_control", seed=s_, supply="threshold", demand="outnorm_ema",
                n_rel=R, kappa_end=1.5, steps=6000)

    # 8. THE decisive test, applied to every candidate fix. Experiment 6 showed the
    #    fixed z-cut has dB/dk* = 0.29, because theta = mean + kappa*std admits a nearly
    #    fixed fraction of H whatever the task. Each arm below is a different way of
    #    letting the perfused count respond to the task instead.
    #      gap/otsu  read B off the SHAPE of the demand distribution
    #      autoreg   close the loop on a metabolic deficit, so supply answers to how
    #                hard the task is rather than to a fixed quantile
    #      q2norm    the user's question: can a different SENSOR rescue the z-cut?
    # the gain=0.05 autoreg runs are kept, relabelled, as the limit-cycle evidence
    for R in [2, 3]:
        for s_ in range(3):
            add("variants_oldgain", seed=s_, supply="autoreg", demand="outnorm_ema",
                n_rel=R, kappa_end=1.5, autoreg_gain=0.05)

    for supply, demand in [("autoreg", "outnorm_ema"), ("gap", "outnorm_ema"),
                           ("otsu", "outnorm_ema")]:
        for R in [2, 3, 4, 6, 8]:
            for s_ in range(3):
                add("variants", seed=s_, supply=supply, demand=demand, n_rel=R,
                    kappa_end=1.5)

    # Sensor arms. A different demand signal cannot fix a rule whose output is set by
    # (kappa, H), so these are deprioritised behind the supply rules and the control.
    for supply, demand in [("threshold", "q2norm"), ("gap", "q2norm"),
                           ("threshold", "entropy")]:
        for R in [2, 3, 4, 6, 8]:
            for s_ in range(3):
                add("sensors", seed=s_, supply=supply, demand=demand, n_rel=R,
                    kappa_end=1.5)

    # 9. The control that decides whether experiment 8 means anything. autoreg lands on
    #    B = k* at target_frac = 0.02. If that only holds at one target, it is
    #    kappa_end = 1.5 all over again: a hyperparameter tuned until it printed the
    #    right number. Sweep the target over a 20x range and see whether B still tracks.
    for tf in [0.005, 0.05, 0.2]:
        for R in [2, 4, 8]:
            for s_ in range(3):
                add("autoreg_target", seed=s_, supply="autoreg", demand="outnorm_ema",
                    n_rel=R, kappa_end=1.5, target_frac=tf)

    # 10. Step-budget calibration. Every arm below costs 15 runs at 4000 steps. If the
    #     k* result survives a shorter budget the whole programme gets cheaper, so this
    #     is measured rather than assumed. The controller gain is scaled with the run
    #     length so total control authority is held constant; without that a short run
    #     trivially fails for lack of time to move kappa.
    for steps, gain in [(400, 0.03), (1000, 0.012)]:
        for R in [2, 4, 8]:
            for s_ in range(2):
                add("steps_calib", seed=s_, supply="autoreg", demand="outnorm_ema",
                    n_rel=R, kappa_end=1.5, steps=steps, autoreg_gain=gain)

    # 11. Hemodynamically motivated arms, each aimed at a MEASURED failure.
    #     stall   the loop over-perfused at tight targets because it dilates through the
    #             early transient. Dilate only on a chronic deficit.
    #     poiseuille  flow ~ r^4 within the perfused set, the one real vascular law the
    #             equal-share family ignores. n = 1 is the linear control.
    #     watershed   territory compaction with a GLOBAL infarct floor, the only version
    #             of that experiment that can produce the predicted effect.
    for R in [2, 4, 8]:
        for s_ in range(3):
            add("arm_stall", seed=s_, supply="autoreg", demand="outnorm_ema", n_rel=R,
                kappa_end=1.5, stall_gate=0.02)
    # complete the stall arm's k* grid so it is compared against plain autoreg on the
    # same five circuit sizes rather than on a subset
    for R in [3, 6]:
        for s_ in range(3):
            add("arm_stall", seed=s_, supply="autoreg", demand="outnorm_ema", n_rel=R,
                kappa_end=1.5, stall_gate=0.02)
            for n in [1.0, 4.0]:
                add("arm_poiseuille", seed=s_, supply="poiseuille",
                    demand="outnorm_ema", n_rel=R, kappa_end=1.5, flow_exponent=n)
            add("arm_watershed", seed=s_, supply="watershed", demand="outnorm_ema",
                n_rel=R, kappa_end=1.5, n_territories=8)

    # the stall arm must also pass the target control that broke plain autoreg
    for tf in [0.005, 0.05]:
        for R in [2, 4, 8]:
            for s_ in range(2):
                add("arm_stall_target", seed=s_, supply="autoreg", demand="outnorm_ema",
                    n_rel=R, kappa_end=1.5, stall_gate=0.02, target_frac=tf)

    # 12. A SECOND BENCHMARK. Everything so far rests on multi_relation, where k* = R
    #     because one head cannot implement two offsets: k* is set by FUNCTION. In
    #     qsa_cross k* = ceil(q log2 N / d_k) is set by CAPACITY, a different reason for
    #     a circuit to have a minimum size. Sweeping d_k over 1, 2, 4, 8 gives k* of
    #     8, 4, 2, 1. Heads are interchangeable there, so it validates the COUNT and not
    #     the roles, which are reported as NaN rather than zero.
    for dk in [1, 2, 4, 8]:
        for s_ in range(3):
            for sup in ["autoreg", "threshold"]:
                add("qsa_kstar", seed=s_, task="qsa_cross", d_k=dk, supply=sup,
                    demand="outnorm_ema", kappa_end=1.5)

    # 13. Local deficit instead of a global loss target. -dL/dg_h is what a head would
    #     gain from more flow, which unlike the query norm depends on the gate and so
    #     measures shortfall rather than drive. The marginal rule perfuses while that
    #     gain exceeds a metabolic price, so there is no target loss to calibrate.
    for R in [2, 4, 8]:
        for s_ in range(3):
            # BASELINE, not a contribution: projected -dL/dg is gradient head
            # importance (Michel et al. 2019) under the conservation constraint.
            add("baseline_gradient", seed=s_, supply="threshold", demand="deficit",
                n_rel=R, kappa_end=1.5)
            # the contribution: reserve is addition under conservation, which no
            # head-importance score in that literature computes.
            add("arm_reserve", seed=s_, supply="threshold", demand="reserve",
                n_rel=R, kappa_end=1.5)
            add("arm_reserve_auto", seed=s_, supply="autoreg", demand="reserve",
                n_rel=R, kappa_end=1.5)
            for price in [1e-5, 1e-3]:
                add("arm_marginal", seed=s_, supply="marginal", demand="deficit",
                    n_rel=R, flow_price=price)

    # 14. The two controls the stall gate must survive, both modelled on the one that
    #     broke plain autoreg.
    #     (a) Is the stall THRESHOLD itself a tuned knob? If B only tracks k* at
    #         stall_gate = 0.02 we have moved the hyperparameter, not removed it.
    for sg in [0.005, 0.08]:
        for R in [2, 4, 8]:
            for s_ in range(2):
                add("stall_eps", seed=s_, supply="autoreg", demand="outnorm_ema",
                    n_rel=R, kappa_end=1.5, stall_gate=sg)
    #     (b) Is the stall gate just a SLOWER loop? A critic will say the event trigger
    #         only buys what a smaller gain would. Plain autoreg at a tenth the gain is
    #         the control: if it fails where stall gating succeeds, adaptivity is doing
    #         the work rather than sluggishness.
    for R in [2, 4, 8]:
        for s_ in range(2):
            add("slowgain_ctl", seed=s_, supply="autoreg", demand="outnorm_ema",
                n_rel=R, kappa_end=1.5, autoreg_gain=0.0003)

    # 5. demand arms at matched perfusion. topk supply so every arm anneals through the
    #    same budget ladder and the loss can be read at B = k*. leak 0 vs 0.05, since
    #    irreversible starvation may be penalising the EMA arm specifically.
    for dm in ["outnorm_ema", "outnorm_inst", "qnorm", "random"]:
        for lk in [0.0, 0.05]:
            for s in range(3):
                add("demand_arm", seed=s, supply="topk", demand=dm, leak=lk)
    return out


def cmd(kw, outdir):
    c = ["python", os.path.join(HERE, "run.py"), "--out", outdir, "--device", "cpu"]
    for k, v in kw.items():
        if k == "redundancy":
            continue
        c += [f"--{k}", str(v)]
    if not kw.get("redundancy"):
        c.append("--skip-redundancy")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated experiment names")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--logs", default=os.path.join(ROOT, "results", "logs"))
    a = ap.parse_args()

    js = jobs()
    if a.only:
        keep = set(a.only.split(","))
        js = [j for j in js if j[0] in keep]
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    todo = [j for j in js if not os.path.exists(os.path.join(a.out, j[2] + ".pkl"))]

    if a.list or not a.run:
        for exp, kw, t in js:
            done = "done" if (exp, kw, t) not in todo else "    "
            print(f"{done}  {exp:<12} {t}")
        print(f"\n{len(js)} jobs, {len(todo)} to run")
        return

    # one seed per process. Single torch thread per worker; the parallelism is across
    # processes, which is ~2x the throughput of one 4-thread process on this box.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTORCH_NUM_THREADS="1")
    t0 = time.time()
    n_done = [0]

    def one(j):
        exp, kw, t = j
        log = os.path.join(a.logs, t + ".log")
        with open(log, "w") as f:
            r = subprocess.run(cmd(kw, a.out), env=env, stdout=f, stderr=subprocess.STDOUT)
        n_done[0] += 1
        print(f"[{n_done[0]}/{len(todo)}  {(time.time()-t0)/60:.0f}m] "
              f"{'ok ' if r.returncode == 0 else 'FAIL'} {exp:<12} {t}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        codes = list(ex.map(one, todo))
    bad = sum(c != 0 for c in codes)
    print(f"finished {len(todo)} jobs in {(time.time()-t0)/60:.0f} min, {bad} failures")


if __name__ == "__main__":
    main()
