"""Peak memory of each pipeline step on one real inner-CV split (diagnostic, not a result).

    venv/bin/python scripts/profile_memory.py --config configs/main.yaml --experiment adult_dbim_protected_kept

Each step runs in a fresh subprocess. Before the measured step the kernel's
peak counter (VmHWM) is reset via /proc/self/clear_refs, so the reported peak
belongs to that step alone, not to the preparation before it. The split is outer repeat 0 / fold 0, inner fold 0,
i.e. exactly what one GridSearchCV worker fits first. Output: a CSV table on
stdout and in results/diagnostics/memory_profile_<experiment>.csv.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def child(args):
    from experiments.pipeline import make_pipeline, make_resampler, make_classifier
    from experiments.protocol import load_experiment_data, outer_splits, inner_splits

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    exp = next(e for e in cfg["tabular_experiments"] if e["name"] == args.experiment)
    X, y, _, strata = load_experiment_data(exp)
    r, f, seed, tr, _ = next(outer_splits(strata, 1, cfg["outer"]["n_splits"], cfg["seeds"]))
    itr, iva = inner_splits(strata[tr], cfg["inner"]["n_splits"], seed)[0]
    X_in, y_in, X_va = X.iloc[tr].iloc[itr], y[tr][itr], X.iloc[tr].iloc[iva]
    impute = exp["variant"] == "full" and bool(X.isna().any().any())
    base = reset_peak()
    pipe = make_pipeline(X_in, exp["classifier"], seed, impute)
    steps = dict(pipe.steps)
    t0 = time.perf_counter()
    out = {"step": args.step, "n_rows_in": len(X_in), "baseline_mb": base}
    if args.step == "impute_fit_transform":
        steps["impute"].fit_transform(X_in)
    elif args.step == "impute_transform_validation":
        steps["impute"].fit(X_in)
        mark = reset_peak()
        steps["impute"].transform(X_va)
        out["baseline_mb"] = mark
    else:
        Xt = steps["impute"].fit_transform(X_in) if impute else X_in
        Xe = steps["encode"].fit_transform(Xt)
        mark = reset_peak()
        out["baseline_mb"] = mark
        t0 = time.perf_counter()
        if args.step.startswith("resample:"):
            Xr, yr = make_resampler(args.step.split(":", 1)[1], seed).fit_resample(Xe, y_in)
            out["n_rows_out"] = len(Xr)
        elif args.step.startswith("clf:"):
            params = json.loads(args.step.split(":", 1)[1])
            sampler = params.pop("sampler", None)
            if sampler:
                Xe, y_in = make_resampler(sampler, seed).fit_resample(Xe, y_in)
                out["baseline_mb"] = reset_peak()
                t0 = time.perf_counter()
            clf = make_classifier(exp["classifier"], seed).set_params(**params).fit(Xe, y_in)
            out["n_rows_out"] = len(Xe)
            import pickle
            out["model_size_mb"] = round(len(pickle.dumps(clf)) / 2 ** 20, 1)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["peak_mb"] = status_mb("VmHWM")
    out["step_increase_mb"] = round(out["peak_mb"] - out["baseline_mb"], 0)
    print(json.dumps(out))


def status_mb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return round(int(line.split()[1]) / 1024, 0)
    return float("nan")


def reset_peak():
    """Reset VmHWM to the current RSS and return the current RSS in MB."""
    import gc
    gc.collect()
    Path("/proc/self/clear_refs").write_text("5")
    return status_mb("VmRSS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--experiment", default="adult_dbim_protected_kept")
    ap.add_argument("--step")
    args = ap.parse_args()
    if args.step:
        return child(args)

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    space = yaml.safe_load((REPO_ROOT / cfg["search_space"]).read_text())
    exp = next(e for e in cfg["tabular_experiments"] if e["name"] == args.experiment)
    from experiments.pipeline import candidate_resamplers
    from experiments.protocol import load_experiment_data
    n_outer_train = int(len(load_experiment_data(exp)[1]) * (1 - 1 / cfg["outer"]["n_splits"]))
    samplers = candidate_resamplers(space, n_outer_train)
    grid = space["classifiers"][exp["classifier"]]
    biggest = {k: (max(v, key=lambda x: float("inf") if x is None else x) if k != "min_samples_leaf" else min(v))
               for k, v in grid.items()}
    steps = ["impute_fit_transform", "impute_transform_validation"]
    steps += [f"resample:{s}" for s in samplers]
    steps += ["clf:" + json.dumps(biggest), "clf:" + json.dumps({**biggest, "sampler": "SMOTE"})]

    rows = []
    for step in steps:
        p = subprocess.run([sys.executable, __file__, "--config", args.config, "--experiment", args.experiment,
                            "--step", step], capture_output=True, text=True, timeout=3600)
        if p.returncode != 0:
            rows.append({"step": step, "error": (p.stderr.strip().splitlines() or ["?"])[-1][:200]})
        else:
            rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)

    import pandas as pd
    out_dir = REPO_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"memory_profile_{args.experiment}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
