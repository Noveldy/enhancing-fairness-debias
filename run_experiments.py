"""Run the revised evaluation protocol end to end (revision Phase 2 acceptance).

    venv/bin/python run_experiments.py --config configs/main.yaml [--only NAME ...]

Writes results/raw/<config name>/<experiment>/ (predictions per fold and seed,
inner-CV selections and scores, per-fold metrics) and a run record
results/raw/<config name>/run_<experiment>.json with seeds, versions, data
checksums, hardware, git commit and wall-clock times.
"""
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent

from data import loaders  # noqa: E402
from experiments.memory_guard import MemoryBudgetExceeded, deprioritize_for_oom_killer  # noqa: E402
from experiments.protocol import run_tabular_experiment  # noqa: E402


def run_baseline_experiment(*args):
    # Imported lazily: it needs fairlearn/aif360, which the other runs do not.
    from experiments.baselines import run_baseline_experiment as run
    return run(*args)


def run_timeseries_experiment(*args):
    # Imported lazily: it needs darts/torch, which the tabular runs do not.
    from experiments.ts_protocol import run_timeseries_experiment as run
    return run(*args)

def run_synthetic_experiment(*args):
    from experiments.synthetic_protocol import run_synthetic_experiment as run
    return run(*args)

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "joblib", "pyyaml",
            "torch", "darts", "pytorch-lightning", "fairlearn", "aif360", "BlackBoxAuditing", "plotly", "rstr"]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def environment():
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else ""
    mem_kb = next((int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                   if l.startswith("MemTotal")), 0) if Path("/proc/meminfo").exists() else 0
    return {
        "python": platform.python_version(), "packages": versions, "platform": platform.platform(),
        "cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": round(mem_kb / 1024 ** 2, 1),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--out", help="output root (default results/raw/<config name>)")
    ap.add_argument("--only-folds", nargs="*", metavar="R:F",
                    help="run only these outer folds (repeat:fold), e.g. 0:0; for reproduction checks")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.only_folds:
        cfg.setdefault("outer", {})["only"] = [[int(x) for x in rf.split(":")] for rf in args.only_folds]
    space = yaml.safe_load((REPO_ROOT / cfg["search_space"]).read_text())
    out_root = Path(args.out) if args.out else REPO_ROOT / "results" / "raw" / cfg["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_root / "run_log.txt", mode="a")],
    )
    if deprioritize_for_oom_killer():
        logging.info("oom_score_adj=1000: if memory runs out, the kernel stops this run before other applications")
    env = environment()
    if env["git_dirty"]:
        logging.warning("tracked files have uncommitted changes; results may not match the recorded commit")

    jobs = [(e, run_tabular_experiment) for e in cfg.get("tabular_experiments", [])]
    jobs += [(e, run_baseline_experiment) for e in cfg.get("baseline_experiments", [])]
    jobs += [(e, run_timeseries_experiment) for e in cfg.get("timeseries_experiments", [])]
    jobs += [(e, run_synthetic_experiment) for e in cfg.get("synthetic_experiments", [])]
    for exp, runner in jobs:
        if args.only and exp["name"] not in args.only:
            continue
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.perf_counter()
        try:
            summary = runner(exp, cfg, space, out_root / exp["name"])
        except MemoryBudgetExceeded as e:
            logging.error("stopped %s by the memory guard: %s", exp["name"], e)
            raise SystemExit(3)
        record = {
            "experiment": exp, "config_file": args.config, "config": {k: v for k, v in cfg.items()
                                                                     if k not in ("tabular_experiments", "timeseries_experiments",
                                                                                  "baseline_experiments",
                                                                                  "synthetic_experiments")},
            "search_space": space, "seeds": exp.get("seeds", cfg.get("seeds")),
            "data": {exp["dataset"]: loaders.RAW_FILES[exp["dataset"]]},
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_clock_seconds": round(time.perf_counter() - t0, 1), "summary": summary, "environment": env,
        }
        (out_root / f"run_{exp['name']}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        logging.info("finished %s in %.0f s", exp["name"], record["wall_clock_seconds"])


if __name__ == "__main__":
    main()
