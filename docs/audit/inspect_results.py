"""Read-only inspection of the stored experiment outputs behind the paper (Phase 0 audit).

Traces the paper's reported numbers back to the files in the research repo
(debias_ops): fairness reports, the cached detector outputs, the t-test script
and the time-series cleaning output. Nothing is retrained and no file is written
except this script's stdout.

Usage:
    python docs/audit/inspect_results.py <debias_ops_dir> > docs/audit/results_inspection.log
"""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def as_dict(v):
    """Some reports store dicts as strings (str() of a Python dict)."""
    if isinstance(v, str):
        import ast
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def fairness_reports(root):
    section("FAIRNESS REPORTS  debias_ops/fairness_evaluation/*.json")
    print("Each report = one train_test_split(test_size=0.2, random_state=42) of a "
          "RandomForest (Models/model_trainer.py).\n")
    rows = []
    for f in sorted((root / "fairness_evaluation").glob("*.json")):
        d = json.load(open(f))
        fair = d.get("fairness", {})
        eo = as_dict(fair.get("equalized_odds", {}))
        dp_groups = as_dict(fair.get("demographic_parity_by_group", {}))
        classes = [k for k in d if k not in ("accuracy", "macro avg", "weighted avg", "cross_validation_score",
                                             "roc_auc_score", "execution_time_seconds", "fairness")]
        print(f"--- {f.name}")
        print(f"  classes in report: {classes}")
        for c in classes:
            print(f"    class {c}: precision={d[c]['precision']:.4f} recall={d[c]['recall']:.4f} "
                  f"support={int(d[c]['support'])}")
        print(f"  test-set size (support): {int(d['weighted avg']['support'])}")
        print(f"  accuracy={d['accuracy']:.4f}  weighted precision={d['weighted avg']['precision']:.4f}  "
              f"macro F1={d['macro avg']['f1-score']:.4f}")
        print(f"  cross_validation_score (5-fold on all X)={d.get('cross_validation_score')}")
        print(f"  roc_auc_score (computed on ALL rows incl. training rows)={d.get('roc_auc_score')}")
        print(f"  execution_time_seconds={d.get('execution_time_seconds')}")
        print(f"  DP difference={fair.get('demographic_parity_difference')}")
        print(f"  DP by group (positive-prediction rate)={dp_groups}")
        if isinstance(eo, dict):
            print(f"  TPR diff={eo.get('tpr_difference')}  FPR diff={eo.get('fpr_difference')}")
            print(f"  TPR by group={eo.get('tpr_by_group')}")
            print(f"  FPR by group={eo.get('fpr_by_group')}")
        rows.append(f.name)
    print(f"\n{len(rows)} reports found.")


def cached_detector_outputs(root):
    section("CACHED DATASET INSPECTION (domain + sensitive attributes)  bias_identification/generated_data_info/")
    print("main.py loads these files instead of re-running the detectors when they exist (main.py:158-163).\n")
    for name in ["adult", "bank_marketing", "air_quality_modified_with_missing",
                 "individual_household_electric_power_consumption_with_missing"]:
        p = root / "bias_identification" / "generated_data_info" / f"{name}.json"
        if not p.exists():
            print(f"{name}: NOT FOUND")
            continue
        d = json.load(open(p))
        last = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %ad", "--date=short", "--", str(p)],
                              capture_output=True, text=True).stdout.strip()
        print(f"{name}: last committed {last}")
        print(f"  domain={d.get('domain')!r}  type={d.get('type_of_dataset')!r}")
        print(f"  protected_attributes={d.get('protected_attributes')}")
        for item in d.get("imbalanced_protected_attributes", []) + d.get("imbalanced_target_attributes", []):
            print(f"  imbalance: {item}")


def ttest(root):
    section("PAIRED T-TESTS  1st_Journal_2025/ttest_adult_bank.py (run unchanged)")
    script = root / "1st_Journal_2025" / "ttest_adult_bank.py"
    # The author's own script, run unchanged; pass this interpreter's package path so scipy is found.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run([sys.executable, script.name], cwd=script.parent, capture_output=True, text=True, env=env)
    print(out.stdout.strip())
    if out.returncode != 0:
        print("STDERR:", out.stderr.strip())
    print("\nEach input array has 4 elements = (mean, std, min, max) of a cross_validate run "
          "(see dbim_ttest.txt), so ttest_rel uses n=4 and df=3.")
    print("\n--- 1st_Journal_2025/dbim_ttest.txt (source of the arrays) ---")
    print((root / "1st_Journal_2025" / "dbim_ttest.txt").read_text().strip())


def timeseries(root):
    section("TIME-SERIES CLEANING OUTPUT  results/*_filled_missing_values_proposed_approach.csv")
    for name in ["air_quality_modified_with_missing", "individual_household_electric_power_consumption_with_missing"]:
        p = root / "results" / f"{name}_filled_missing_values_proposed_approach.csv"
        if not p.exists():
            print(f"{name}: NOT FOUND")
            continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number")
        print(f"{p.name}: shape={d.shape}")
        print(f"  NaN cells={int(d.isna().sum().sum())}  cells equal to -200 (UCI missing code)={int((num == -200).sum().sum())}")
        per_col = {c: int(v) for c, v in (num == -200).sum().items() if v}
        if per_col:
            print(f"  -200 per column: {per_col}")
    p = root / "results" / "cleaned_timeseries_experiments.txt"
    if p.exists():
        datasets = sorted({l.split(":", 1)[1].strip() for l in p.read_text().splitlines() if l.startswith("dataset:")})
        print(f"\n{p.name}: datasets mentioned = {datasets}")


def intermediate_files(root):
    section("INTERMEDIATE FILES REFERENCED BY THE EXPERIMENT SCRIPTS")
    refs = [
        "Datasets/air_quality_modified.csv",
        "Datasets/air_quality_modified_with_missing.csv",
        "Datasets/individual_household_electric_power_consumption_modified.csv",
        "Datasets/individual_household_electric_power_consumption_with_missing.csv",
        "Datasets/synthetic_individual_household_power_consumption.csv",
        "Datasets/adult_debiased.json",
        "Datasets/bank_marketing_debiased.json",
        "bert-sensitive-features-classifier",
        "results/fairlearn_sex_adult_comparison.csv",
        "results/fairlearn_race_adult_comparison.csv",
    ]
    for r in refs:
        print(f"  {'present' if (root / r).exists() else 'MISSING'}: {r}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    section("ENVIRONMENT")
    print(f"run at (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"python: {platform.python_version()}  pandas: {pd.__version__}")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    print(f"debias_ops: {root}  HEAD={head.stdout.strip()}  working tree clean={not dirty.stdout.strip()}")
    print("randomness: none (deterministic, read-only)")
    fairness_reports(root)
    cached_detector_outputs(root)
    ttest(root)
    timeseries(root)
    intermediate_files(root)


if __name__ == "__main__":
    main()
