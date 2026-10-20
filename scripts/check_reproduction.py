"""Phase 9.3 (author decision P2): rerun one slice of every experiment and compare it with the saved outputs.

    venv/bin/python scripts/check_reproduction.py --work /tmp/repro      # ~1 h on an i5-14500

Meant to be run in a clean clone with a fresh environment (README, "Checking the results"). Each rerun goes
to --work; the comparison reads the saved outputs in results/raw/ of the same checkout.

Slices:
  main, baselines, ablation, synthetic   outer fold repeat 0 / fold 0 of every experiment (--only-folds 0:0)
  timeseries                             fold 0, scenarios reference + proposed, seed 0 (6 grid points + 1 fit each)
  detector                               grouped CV repeat 0 fold 0, final model seed 0, checkpoint-830 predictions
  cost                                   Bank fold 0: DBIM fit (checks the stored accuracy) and DBIM model
                                         selection (checks the stored choice); times are not compared

Comparison: every CSV the rerun writes is matched with the saved file of the same name, restricted to the
rerun's folds/seeds, rows sorted, and compared value by value (exact equality; NaN equals NaN). Columns that
measure time or memory are not compared. Writes results/reproduction_check/{report.csv, report.md, run.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results" / "reproduction_check"
NOT_COMPARED = re.compile(r"(second|elapsed|wall|rss|peak|_gb$|^cpu$|duration|started|finished|timing|^n_jobs$)", re.I)
KEYS = ["repeat", "fold", "seed", "scenario", "split", "sub_seed"]


def git_commit():
    """HEAD of this checkout; inside the container (no .git) the commit passed in at build time."""
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or os.environ.get("DEBIAS_GIT_COMMIT", "unknown")


def sh(cmd, log):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    t = time.perf_counter()
    r = subprocess.run(list(map(str, cmd)), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"[exit {r.returncode}, {time.perf_counter() - t:.0f} s]\n")
    log.flush()
    return r.returncode


def shown(path):
    """Path relative to the repository when it is inside it (scratch files in --work are shown as they are)."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def compare(new_path, old_path, restrict=None):
    """One report row: is `new` equal to the matching rows of `old`?"""
    row = {"file": shown(old_path), "rows_rerun": None, "rows_compared": None,
           "identical": False, "columns_compared": 0, "columns_skipped": "", "max_abs_diff": None, "note": ""}
    if not old_path.exists():
        row["note"] = "no saved file of this name"
        return row
    new, old = read(new_path), read(old_path)
    row["rows_rerun"] = len(new)
    keys = [k for k in KEYS if k in new.columns and k in old.columns]
    for k in keys:                                   # restrict the saved rows to the rerun's slice
        old = old[old[k].isin(new[k].unique())]
    if restrict:
        old = restrict(old)
    cols = [c for c in new.columns if c in old.columns and not NOT_COMPARED.search(c)]
    row["columns_skipped"] = ";".join(c for c in new.columns if c not in cols)
    row["columns_compared"] = len(cols)
    row["rows_compared"] = len(old)
    if len(old) != len(new):
        row["note"] = f"row count differs: saved {len(old)} vs rerun {len(new)}"
        return row
    sort = lambda d: d[cols].astype(str).sort_values(cols, kind="mergesort").index   # noqa: E731
    a = new.loc[sort(new), cols].reset_index(drop=True)
    b = old.loc[sort(old), cols].reset_index(drop=True)
    diffs = []
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            same = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
            if not same.all():
                d = np.abs(xv - yv)[~same]
                diffs.append((c, float(np.nanmax(d)) if np.isfinite(d).any() else np.inf))
        elif not ((x.astype(str) == y.astype(str)).fillna(False) | (x.isna() & y.isna())).all():
            # missing equals missing (pandas 3 keeps missing values missing in astype(str))
            diffs.append((c, np.nan))
    row["identical"] = not diffs
    if diffs:
        row["max_abs_diff"] = max((d for _, d in diffs if not np.isnan(d)), default=np.nan)
        row["note"] = "differs in " + ", ".join(c for c, _ in diffs[:8])
    return row


def compare_dir(new_dir, old_dir, check, restrict=None):
    rows = []
    for p in sorted(Path(new_dir).iterdir()):
        if p.suffix == ".csv" or p.name.endswith(".csv.gz"):
            if p.name in ("timing.csv",):
                continue
            rows.append({"check": check, **compare(p, Path(old_dir) / p.name, restrict)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="scratch directory for the reruns")
    ap.add_argument("--skip", nargs="*", default=[], help="checks to skip (main baselines ablation synthetic "
                                                          "timeseries detector cost)")
    ap.add_argument("--reuse", action="store_true",
                    help="do not rerun a step whose output folder already exists in --work (e.g. after the "
                         "comparison itself failed); the report lists such steps as reused")
    args = ap.parse_args(argv)
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    rows, steps = [], []
    log = open(OUT / "run_log.txt", "a")
    log.write(f"\n===== {started} check_reproduction in {REPO_ROOT} =====\n")

    def step(name, cmd, out=None):
        if args.reuse and out is not None and Path(out).exists():
            log.write(f"\n[{name}: reusing the earlier rerun in {out}]\n")
            steps.append({"check": name, "exit_code": 0, "seconds": "reused"})
            return True
        t = time.perf_counter()
        code = sh(cmd, log)
        steps.append({"check": name, "exit_code": code, "seconds": round(time.perf_counter() - t, 1)})
        return code == 0

    for name in ("main", "baselines", "ablation", "synthetic"):
        if name in args.skip:
            continue
        cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text())
        if step(name, [py, "run_experiments.py", "--config", f"configs/{name}.yaml", "--out", work / name,
                       "--only-folds", "0:0"], out=work / name):
            exps = [e["name"] for k in ("tabular_experiments", "baseline_experiments", "synthetic_experiments")
                    for e in cfg.get(k, [])]
            for e in exps:
                rows += compare_dir(work / name / e, RAW / name / e, f"{name}/{e}")

    if "timeseries" not in args.skip:
        cfg = yaml.safe_load((REPO_ROOT / "configs" / "timeseries.yaml").read_text())
        exp = cfg["timeseries_experiments"][0]
        exp.update(folds=exp["folds"][:1], methods=["proposed"], seeds=[0])
        cfg["search_space"] = str(REPO_ROOT / cfg["search_space"])
        path = work / "timeseries_check.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        if step("timeseries", [py, "run_experiments.py", "--config", path, "--out", work / "timeseries"], out=work / "timeseries"):
            rows += compare_dir(work / "timeseries" / exp["name"], RAW / "timeseries" / exp["name"],
                                "timeseries", restrict=lambda d: d[d["scenario"].isin(["reference", "proposed"])]
                                if "scenario" in d.columns else d)

    if "detector" not in args.skip:
        cfg = yaml.safe_load((REPO_ROOT / "configs" / "detector.yaml").read_text())
        cfg["internal_cv"]["grouped"].update(repeats=[0], folds=[0])
        cfg["internal_cv"]["random"]["repeats"] = []
        cfg["final_models"]["seeds"] = [0]
        path = work / "detector_check.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        if step("detector", [py, "scripts/detector_validation.py", "--config", path, "--out", work / "detector"],
                out=work / "detector" / "raw" / "detector" / "external_predictions.csv"):
            new_dir, old_dir = work / "detector" / "raw" / "detector", RAW / "detector"
            for f in ("internal_cv_folds.csv", "internal_cv_predictions.csv", "internal_cv_epochs.csv",
                      "training_descriptions_groups.csv"):
                rows.append({"check": "detector", **compare(new_dir / f, old_dir / f,
                                                             lambda d: d[d["split"] == "grouped"]
                                                             if "split" in d.columns else d)})
            new, old = read(new_dir / "external_predictions.csv"), read(old_dir / "external_predictions.csv")
            cols = [c for c in ("dataset", "column", "keyword_on_name", "p_bert_seed0", "p_checkpoint830")
                    if c in new.columns and new[c].notna().any()]   # checkpoint-830 is absent from clean clones
            new[cols].to_csv(work / "detector_external_subset.csv", index=False)
            old[cols].to_csv(work / "detector_external_saved_subset.csv", index=False)
            rows.append({"check": "detector", **compare(work / "detector_external_subset.csv",
                                                        work / "detector_external_saved_subset.csv")})
            rows[-1]["file"] = "results/raw/detector/external_predictions.csv (seed 0, checkpoint-830, keyword)"

    if "cost" not in args.skip:
        # one output folder per task: each cost_run.py call rewrites its measurements.csv
        common = [py, "scripts/cost_run.py", "--config", "configs/cost.yaml", "--match",
                  "part=cost", "dataset=bank", "fold=0"]
        ok = (step("cost", common + ["task=dbim_fit", "--out", work / "cost_fit"])
              and step("cost", common + ["task=search", "method=dbim", "--out", work / "cost_search"]))
        if ok:
            m = pd.concat([read(work / d / "measurements.csv") for d in ("cost_fit", "cost_search")])
            for flag in ("reproduces_stored_accuracy", "reproduces_stored_selection"):
                vals = m[flag].dropna() if flag in m.columns else pd.Series([], dtype=object)
                rows.append({"check": "cost", "file": f"{flag} (Bank fold 0, DBIM)", "rows_rerun": len(vals),
                             "rows_compared": len(vals), "identical": bool(len(vals)) and bool(vals.astype(bool).all()),
                             "columns_compared": 1, "note": f"values: {sorted(set(vals.astype(str)))}"})

    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "report.csv", index=False)
    st = pd.DataFrame(steps)
    ok = bool(len(rep)) and bool(rep["identical"].all()) and bool((st["exit_code"] == 0).all())
    lines = [f"# Reproduction check ({started})", "",
             f"Checkout: `{git_commit()}`; "
             f"Python {platform.python_version()}; {round(time.perf_counter() - t0)} s.", "",
             f"**{'All compared outputs are identical.' if ok else 'Differences or failures found (see below).'}** "
             f"{int(rep['identical'].sum()) if len(rep) else 0} of {len(rep)} files identical; "
             f"{int((st['exit_code'] != 0).sum()) if len(st) else 0} failed steps.", "",
             "| check | file | rows compared | identical | note |", "|---|---|---|---|---|"]
    for r in rep.itertuples():
        lines.append(f"| {r.check} | `{r.file}` | {r.rows_compared} | {'yes' if r.identical else '**no**'} | "
                     f"{r.note or ''} |")
    lines += ["", "Steps:", "", "| check | exit code | seconds |", "|---|---|---|"]
    lines += [f"| {s.check} | {s.exit_code} | {s.seconds} |" for s in st.itertuples()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    (OUT / "run.json").write_text(json.dumps({
        "script": "scripts/check_reproduction.py", "started_utc": started,
        "wall_clock_seconds": round(time.perf_counter() - t0, 1), "work": str(work), "skip": args.skip,
        "reuse": args.reuse,
        "git_commit": git_commit(),
        "python": platform.python_version(), "steps": steps, "all_identical": ok}, indent=2) + "\n")
    print(f"{'OK' if ok else 'DIFFERENCES'}: {OUT / 'report.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
