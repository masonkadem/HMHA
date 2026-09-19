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
