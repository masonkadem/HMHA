"""Run a list of experiments/run.py jobs on the GPU and the CPU cores at the same time.

Each line of the jobs file is one set of run.py arguments. Jobs whose result file already exists are
skipped, so the pool can be stopped and restarted safely.

  python experiments/run_pool.py results/equal/jobs.txt --out results/equal --gpu 8 --cpu 6 --cpu-threads 2
"""
import argparse, os, queue, shlex, subprocess, sys, threading, time
from dataclasses import replace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hemo.config import Cfg
from run import result_tag


def expected_file(args, out):
    """The .pkl that run.py would write for these arguments."""
    cfg, toks = Cfg(), shlex.split(args)
    for name, value in zip(toks[::2], toks[1::2]):
        field = name.lstrip("-")
        cfg = replace(cfg, **{field: type(getattr(cfg, field))(value)})
    return os.path.join(out, result_tag(cfg) + ".pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=8, help="jobs at once on the GPU")
    ap.add_argument("--cpu", type=int, default=6, help="jobs at once on the CPU")
    ap.add_argument("--cpu-threads", type=int, default=2, help="CPU threads per CPU job")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)

    jobs = [l.strip() for l in open(a.jobs) if l.strip()]
    todo = [j for j in jobs if not os.path.exists(expected_file(j, a.out))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"({a.gpu} on the GPU, {a.cpu} on the CPU)", flush=True)
    work = queue.Queue()
    for j in todo:
        work.put(j)
    lock, done, t0 = threading.Lock(), [0], time.time()

    def worker(device):
        env = dict(os.environ)
        if device == "cpu":
            env.update(OMP_NUM_THREADS=str(a.cpu_threads), MKL_NUM_THREADS=str(a.cpu_threads))
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            extra = " --device cpu" if device == "cpu" else ""
            log = os.path.join(a.out, "logs", job.replace(" ", "_").replace("-", "") + ".log")
            with open(log, "w") as f:
                rc = subprocess.call([sys.executable, "experiments/run.py", *shlex.split(job + extra),
                                      "--skip-redundancy", "--out", a.out], stdout=f, stderr=subprocess.STDOUT,
                                     cwd=ROOT, env=env)
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(todo)} {time.time() - t0:.0f}s {device}] exit {rc}: {job}", flush=True)

    threads = [threading.Thread(target=worker, args=("gpu",)) for _ in range(a.gpu)]
    threads += [threading.Thread(target=worker, args=("cpu",)) for _ in range(a.cpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
