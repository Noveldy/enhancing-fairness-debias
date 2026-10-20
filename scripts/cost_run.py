"""Phase 7.2-7.3 driver: run every timing task in its own pinned, single-threaded process.

    venv/bin/python scripts/cost_run.py --config configs/cost.yaml [--only-part cost|scaling] [--limit N]

Writes results/raw/cost/: tasks.jsonl (one line per finished task, appended, so an interrupted run
resumes and skips finished tasks), measurements.csv (one row per timed stage), run_cost.json (config,
environment, wall clock), run_log.txt; and results/hardware.json.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from experiments.cost import THREAD_ENV  # noqa: E402

os.environ.update(THREAD_ENV)          # inherited by every spawned task process (set before they start)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from experiments.cost import run_task  # noqa: E402
from experiments.memory_guard import GB, available_bytes, deprioritize_for_oom_killer  # noqa: E402

OUT = REPO_ROOT / "results" / "raw" / "cost"
log = logging.getLogger("cost")


# ----------------------------------------------------------------------------- task list
def build_tasks(cfg):
    tasks = {}

    def add(spec, deps=()):
        tid = "|".join(f"{k}={spec[k]}" for k in sorted(spec))
        tasks[tid] = {**spec, "task_id": tid, "deps": list(deps)}
        return tid

    c = cfg["cost"]
    fit_ids = {}
    for ds in c["datasets"]:
        for f in c["folds"]:
            base = {"dataset": ds, "repeat": c["repeat"], "fold": f, "frac": 1.0, "sub_seed": 0}
            fit_ids[(ds, 1.0, f, 0)] = add({"part": "cost", "task": "dbim_fit", **base})
            for m in c["search_methods"]:
                add({"part": "cost", "task": "search", "method": m, **base})
            for m in c["baselines"]:
                extra = {"mitigate": c["mitigate"][ds]} if m in ("reweighing", "dir", "exp_gradient",
                                                                 "threshold_opt") else {}
                add({"part": "cost", "task": "baseline_fit", "method": m, **base, **extra})
            for g in c["generators"]:
                add({"part": "cost", "task": "generator", "generator": g, **base})
    for ds in c["detection"]["datasets"]:
        for rep in range(c["detection"]["repeats"]):
            add({"part": "cost", "task": "detection", "dataset": ds, "head_seed": c["detection"]["head_seed"],
                 "rep": rep})
    ts = c["timeseries"]
    for f in ts["folds"]:
        for scen in ["reference", "mean", "linear", "spline", "proposed"]:
            add({"part": "cost", "task": "ts_cleaning", "fold": f, "scenario": scen})
    for scen in ["reference", "mean", "linear", "spline", "proposed"]:
        add({"part": "cost", "task": "ts_lstm", "fold": ts["lstm_fold"], "scenario": scen, "seed": ts["lstm_seed"]})

    s = cfg["scaling"]
    r0 = c["repeat"]
    for ds in s["datasets"]:
        points = [(fr, 0, sd) for fr in s["fracs"] for sd in s["sub_seeds"]] + [(1.0, f, 0) for f in c["folds"]]
        for fr, f, sd in points:
            base = {"dataset": ds, "repeat": r0, "fold": f, "frac": fr, "sub_seed": sd}
            if (ds, fr, f, sd) not in fit_ids:
                fit_ids[(ds, fr, f, sd)] = add({"part": "scaling", "task": "dbim_fit", **base,
                                               "config_from": (r0, 0)})
            if fr < 1.0:
                for g in s["generators"]:
                    add({"part": "scaling", "task": "generator", "generator": g, **base})
        # resamplers: each task waits for the first task of its own size and of the previous size, and is
        # skipped if one of them failed or timed out
        sizes = sorted({p[0] for p in points})
        for rs in s["resamplers"]:
            prev_first = None
            for fr in sizes:
                first = None
                for _, f, sd in [p for p in points if p[0] == fr]:
                    tid = add({"part": "scaling", "task": "resampler", "resampler": rs, "dataset": ds,
                               "repeat": r0, "fold": f, "frac": fr, "sub_seed": sd},
                              deps=[fit_ids[(ds, fr, f, sd)]] + [d for d in (prev_first, first) if d])
                    first = first or tid
                prev_first = first
        if s["search"]:
            prev = None
            for fr in s["fracs"]:
                prev = add({"part": "scaling", "task": "search", "method": "dbim", "dataset": ds, "repeat": r0,
                            "fold": 0, "frac": fr, "sub_seed": 0}, deps=[prev] if prev else [])
    return tasks


def task_memory(cfg, spec):
    m = cfg["memory"]["task_gb"]
    if spec.get("method") == "exp_gradient":
        return m["exp_gradient"]
    return m.get(spec["task"], m["default"])


# ----------------------------------------------------------------------------- environment
def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _stdout(cmd):
    """Output of a system tool, or '' where it is not installed (e.g. lspci in a slim container)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ""


def hardware():
    from run_experiments import environment
    env = environment()
    cpus = []
    for line in _stdout(["lscpu", "-e=CPU,CORE,MAXMHZ"]).splitlines()[1:]:
        cpu, core, mhz = line.split()
        cpus.append({"cpu": int(cpu), "core": int(core), "max_mhz": float(mhz.replace(",", "."))})
    lscpu = _stdout(["lscpu"])
    gpus = [l.split(": ", 1)[1] for l in _stdout(["lspci"]).splitlines()
            if any(k in l for k in ("VGA", "3D controller", "Display controller"))]
    pkgs = env["packages"]
    from importlib import metadata
    for p in ["transformers", "tokenizers", "threadpoolctl"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    swap = next((int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                 if l.startswith("SwapTotal")), 0)
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu_model": env["cpu"], "logical_cpus": env["logical_cpus"],
        "physical_cores": len({c["core"] for c in cpus}),
        "performance_cores_logical_cpus": _read("/sys/devices/cpu_core/cpus"),
        "efficiency_cores_logical_cpus": _read("/sys/devices/cpu_atom/cpus"),
        "per_cpu": cpus,
        "l3_cache": next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("L3 cache")), None),
        "cpufreq_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "energy_performance_preference": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "turbo_disabled": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "ram_gb": env["ram_gb"], "swap_gb": round(swap / 1024 ** 2, 1),
        "gpus": gpus,
        "gpu_used": False,
        "gpu_note": "No experiment uses a GPU: scikit-learn/imbalanced-learn are CPU-only, and the LSTM and BERT "
                    "runs use torch on the CPU (the author chose not to use the Arc A770).",
        "os": platform.platform(), "kernel": platform.release(),
        "distribution": next((l.split("=", 1)[1].strip('"') for l in (_read("/etc/os-release") or "").splitlines()
                              if l.startswith("PRETTY_NAME=")), None),
        "python": env["python"], "packages": pkgs,
    }


# ----------------------------------------------------------------------------- scheduler
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only-part", choices=["cost", "scaling"])
    ap.add_argument("--only-task", nargs="*", help="run only these task kinds (smoke tests)")
    ap.add_argument("--limit", type=int, help="run at most N tasks (smoke tests)")
    ap.add_argument("--match", nargs="*", help="run only tasks whose id contains all of these strings (pilots)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run_log.txt")])
    deprioritize_for_oom_killer()
    hw = hardware()
    # the main run records the machine in results/hardware.json; runs elsewhere (checks, pilots) keep it local
    hw_path = REPO_ROOT / "results" / "hardware.json" if out.resolve() == OUT.resolve() else out / "hardware.json"
    hw_path.write_text(json.dumps(hw, indent=2) + "\n")
    from run_experiments import git
    if git("status", "--porcelain", "--untracked-files=no"):
        log.warning("tracked files have uncommitted changes; results may not match the recorded commit")

    tasks = build_tasks(cfg)
    order = list(tasks)
    random.Random(cfg["shuffle_seed"]).shuffle(order)
    order = [t for t in order if (not args.only_part or tasks[t]["part"] == args.only_part)
             and (not args.only_task or tasks[t]["task"] in args.only_task)
             and (not args.match or all(m in t for m in args.match))]
    if args.limit:
        order = order[:args.limit]
    selected = set(order)
    stack = list(order)
    while stack:   # a task's dependencies always run with it (the resampler tasks need the cached data)
        for d in tasks[stack.pop()]["deps"]:
            if d not in selected:
                selected.add(d)
                order.insert(0, d)
                stack.append(d)
    done = {}
    jsonl = out / "tasks.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            rec = json.loads(line)
            if rec["status"] == "ok":
                done[rec["task_id"]] = rec
        log.info("resuming: %d tasks already finished", len(done))
    pending = [t for t in order if t not in done]
    log.info("%d tasks (%d to run), CPUs %s", len(order), len(pending), cfg["cpus"])

    ctx = mp.get_context("spawn")
    running = {}          # cpu -> (task_id, process, conn, t_start, timeout)
    free = list(cfg["cpus"])
    t_all = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reserve = cfg["memory"]["reserve_gb"]

    def finish(tid, status, rows, seconds, cpu, message=""):
        rec = {"task_id": tid, "spec": {k: v for k, v in tasks[tid].items() if k != "deps"}, "status": status,
               "seconds_total": round(seconds, 2), "cpu": cpu, "message": message, "rows": rows,
               "loadavg_end": os.getloadavg()}
        done[tid] = rec
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        log.info("%-8s %6.0fs cpu%-2s %s %s", status, seconds, cpu, tid, message.splitlines()[0] if message else "")

    while pending or running:
        # collect finished / timed-out tasks
        for cpu, (tid, proc, conn, t0, limit) in list(running.items()):
            elapsed = time.perf_counter() - t0
            if conn.poll():
                try:
                    status, payload = conn.recv()
                except EOFError:
                    status, payload = "error", f"process ended without a result (exit code {proc.exitcode})"
                proc.join()
                finish(tid, status, payload if status == "ok" else [], elapsed, cpu,
                       "" if status == "ok" else payload)
            elif not proc.is_alive():
                finish(tid, "error", [], elapsed, cpu, f"process died (exit code {proc.exitcode})")
            elif elapsed > limit:
                proc.kill()
                proc.join()
                finish(tid, "timeout", [], elapsed, cpu, f"killed after {limit} s")
            else:
                continue
            del running[cpu]
            free.append(cpu)
        # start eligible tasks
        for tid in list(pending):
            if not free:
                break
            spec = tasks[tid]
            deps = spec["deps"]
            failed = [d for d in deps if done.get(d, {}).get("status") not in (None, "ok")]
            if not failed and any(d not in done for d in deps):
                continue
            if failed:
                pending.remove(tid)
                finish(tid, "skipped", [], 0.0, None, f"dependency {done[failed[0]]['status']}: {failed[0]}")
                continue
            need = task_memory(cfg, spec)
            if available_bytes() / GB - need < reserve:
                break   # wait for memory
            cpu = free.pop(0)
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=run_task, args=(spec, str(cache_dir), cpu, child), daemon=True)
            proc.start()
            child.close()
            limit = cfg["timeouts_s"].get(spec["task"], cfg["timeouts_s"]["default"])
            running[cpu] = (tid, proc, parent, time.perf_counter(), limit)
            pending.remove(tid)
        time.sleep(0.2)

    rows = [dict(r, task_seconds_total=rec["seconds_total"]) for rec in done.values() if rec["task_id"] in selected
            for r in rec["rows"]]
    pd.DataFrame(rows).to_csv(out / "measurements.csv", index=False)
    statuses = pd.Series([done[t]["status"] for t in order if t in done]).value_counts().to_dict()
    record = {"script": "scripts/cost_run.py", "config_file": args.config, "config": cfg, "args": vars(args),
              "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "wall_clock_seconds": round(time.perf_counter() - t_all, 1), "n_tasks": len(order),
              "task_status": statuses, "thread_env": {k: os.environ[k] for k in sorted(os.environ)
                                                      if k.endswith("_THREADS")},
              "git_commit": git("rev-parse", "HEAD"),
              "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "hardware": hw}
    (out / "run_cost.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    log.info("done in %.0f s: %s", record["wall_clock_seconds"], statuses)


if __name__ == "__main__":
    main()
