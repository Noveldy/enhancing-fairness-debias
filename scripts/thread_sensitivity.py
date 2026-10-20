"""Phase 9.3 follow-up: do inner-CV scores depend on the number of BLAS/OpenMP threads?

    venv/bin/python scripts/thread_sensitivity.py            # ~4 min on an i5-14500

The reproduction check (results/reproduction_check/report.md) found that the inner-CV scores of
bank_dbim_with_duration, outer fold 0, differ from the saved ones, while the selection and the predictions are
identical. The saved run used 6 parallel workers and the check 4 (set by the memory guard), and joblib gives
each worker about cpu_count / n_workers threads. This script reruns that fold twice with the same number of
workers (2), once with the default threads and once with every thread pool limited to 1, and compares the
inner scores of the two runs with each other and with the saved ones.

Writes results/reproduction_check/thread_sensitivity.csv (one row per candidate) and thread_sensitivity.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "reproduction_check"
EXP = "bank_dbim_with_duration"
N_JOBS = 2
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
KEY = ["sampler", "clf_params"]


def run(work, name, env):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "main.yaml").read_text())
    cfg["tabular_experiments"] = [e for e in cfg["tabular_experiments"] if e["name"] == EXP]
    cfg["n_jobs"] = N_JOBS
    cfg["search_space"] = str(REPO_ROOT / cfg["search_space"])
    path = work / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "run_experiments.py", "--config", str(path), "--out", str(work / name),
                        "--only-folds", "0:0"], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    log = r.stdout + r.stderr
    (work / f"{name}.log").write_text(log)
    if r.returncode:
        sys.exit(f"{name} failed (exit {r.returncode}); see {work / f'{name}.log'}")
    used = [l for l in log.splitlines() if "r0 f0:" in l]
    return {"run": name, "seconds": round(time.perf_counter() - t, 1), "log_line": used[-1] if used else ""}


def scores(path):
    d = pd.read_csv(path, float_precision="round_trip")
    d = d[(d["repeat"] == 0) & (d["fold"] == 0)]
    return d.set_index(KEY)[["mean_inner_score", "rank"]]


def differ(a, b):
    a, b = a.reindex(b.index), b
    both_nan = a.isna() & b.isna()
    n = int(((a != b) & ~both_nan).sum())
    return n, float((a - b).abs().max())


def main():
    work = Path(os.environ.get("DEBIAS_THREAD_WORK", "/tmp/debias-thread-sensitivity"))
    work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    runs = [run(work, "default_threads", base),
            run(work, "one_thread", {**base, **{k: "1" for k in THREAD_VARS}})]
    f = Path(EXP) / "inner_scores.csv.gz"
    saved = scores(REPO_ROOT / "results" / "raw" / "main" / f)
    d = scores(work / "default_threads" / f)
    o = scores(work / "one_thread" / f)
    table = saved.join(d, rsuffix="_default_threads").join(o, rsuffix="_one_thread")
    table = table.rename(columns={"mean_inner_score": "mean_inner_score_saved", "rank": "rank_saved"})
    table.reset_index().to_csv(OUT / "thread_sensitivity.csv", index=False)
    pairs = {}
    for name, (a, b) in {"default_threads vs one_thread": (d, o), "default_threads vs saved": (d, saved),
                         "one_thread vs saved": (o, saved)}.items():
        n, mx = differ(a["mean_inner_score"], b["mean_inner_score"])
        top = (a["rank"] == 1).idxmax() == (b["rank"] == 1).idxmax()
        pairs[name] = {"candidates_differing": n, "max_abs_diff": mx, "same_rank_1_candidate": bool(top)}
    sel = {r: pd.read_csv(work / r / EXP / "selection.csv").iloc[0].to_dict() for r in ("default_threads", "one_thread")}
    rec = {"script": "scripts/thread_sensitivity.py", "started_utc": started, "experiment": EXP,
           "outer_fold": [0, 0], "n_jobs": N_JOBS, "cpu_count": os.cpu_count(), "candidates": len(saved),
           "thread_vars_one_thread_run": list(THREAD_VARS), "runs": runs, "comparisons": pairs,
           "selection": sel, "python": platform.python_version(),
           "git_commit": subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()}
    (OUT / "thread_sensitivity.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
