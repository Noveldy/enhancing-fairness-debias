"""Phase 3.3-3.4: summaries and paired tests, DBIM vs every baseline (and the time-series fillers).

    venv/bin/python scripts/compare_baselines.py [--config configs/baselines.yaml]

Reads only stored per-fold outputs (results/raw/main, results/raw/baselines,
results/raw/timeseries) and writes:

  results/raw/comparison/<dbim experiment>_complete_rows_fold_metrics.csv
      DBIM metrics recomputed from its stored predictions on the complete rows
      of each test fold, i.e. the exact test rows the baselines were scored on.
  results/summary/baselines_summary.csv    mean, SD, 95% CI per method and metric
  results/summary/baselines_tests.csv      DBIM - baseline, paired by outer fold
  results/summary/timeseries_summary.csv   per cleaning scenario and horizon
  results/summary/timeseries_tests.csv     proposed - other filler, paired by fold and seed
  results/summary/literature_reported.csv  external numbers, not directly comparable (C5)
  results/summary/comparison_run.json      inputs (sha256), versions, time

Tests (experiments/stats.py): Nadeau-Bengio corrected resampled t-test (primary;
df = pairs - 1) and Wilcoxon signed-rank (secondary). Holm correction within each
family = one experiment x one metric, across the baselines compared with DBIM.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import stats  # noqa: E402
from experiments.metrics import fold_metrics  # noqa: E402
from experiments.protocol import load_experiment_data  # noqa: E402

UTILITY = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
FAIRNESS = ["dp_diff", "eo_diff"]
LOWER_IS_BETTER = {"dp_diff", "eo_diff", "tpr_diff", "fpr_diff", "brier", "test_mae_scaled", "test_smape",
                   "mae_scaled"}

# Paper numbers taken from other publications (AUDIT section 3; author decision C5).
LITERATURE = [
    {"dataset": "adult", "classifier": "random_forest", "metric": "accuracy", "value": 0.85216,
     "source": "paper section 3.2.3 / Table 7, reference [82]"},
    {"dataset": "adult", "classifier": "random_forest", "metric": "precision", "value": 0.80246,
     "source": "paper section 3.2.3 / Table 7, reference [82]"},
    {"dataset": "bank", "classifier": "decision_tree", "metric": "roc_auc", "value": 0.833,
     "source": "paper section 3.2.4 / Table 8, reference [83]"},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(method, mitigated):
    return f"{method}[{mitigated}]" if isinstance(mitigated, str) and mitigated else method


def dbim_complete_rows(exp, dbim_dir):
    """DBIM fold metrics on the complete test rows (the baselines' test rows)."""
    X, y, groups, strata = load_experiment_data(exp)
    complete = ~X.isna().any(axis=1).to_numpy()
    pred = pd.read_csv(dbim_dir / "predictions.csv.gz")
    pred = pred[complete[pred["row_id"].to_numpy()]]
    rows = []
    for (r, f), p in pred.groupby(["repeat", "fold"]):
        rows.append({"method": "dbim", "mitigated": "", "repeat": r, "fold": f,
                     **fold_metrics(p["y_true"].to_numpy(), p["y_pred"].to_numpy(), p["y_score"].to_numpy(),
                                    p[exp["protected"]].reset_index(drop=True))})
    return pd.DataFrame(rows), pred


def tabular(cfg, out_raw, out_sum, inputs):
    summaries, tests = [], []
    for exp in cfg["baseline_experiments"]:
        bdir = REPO_ROOT / "results" / "raw" / cfg["name"] / exp["name"]
        ddir = REPO_ROOT / "results" / "raw" / exp["compare_with"]
        if not (bdir / "fold_metrics.csv").exists():
            print(f"skip {exp['name']}: no baseline results yet")
            continue
        base = pd.read_csv(bdir / "fold_metrics.csv", keep_default_na=False, na_values=[""])
        base["mitigated"] = base["mitigated"].fillna("")
        dbim_cc, dbim_pred = dbim_complete_rows(exp, ddir)
        bpred = pd.read_csv(bdir / "predictions.csv.gz", usecols=["method", "repeat", "fold", "row_id"])
        for (r, f), p in bpred[bpred.method == "none"].groupby(["repeat", "fold"]):
            d = dbim_pred[(dbim_pred.repeat == r) & (dbim_pred.fold == f)]["row_id"]
            if set(d) != set(p["row_id"]):
                raise RuntimeError(f"{exp['name']} r{r} f{f}: DBIM and baseline test rows differ")
        dbim_cc.to_csv(out_raw / f"{Path(exp['compare_with']).name}_complete_rows_fold_metrics.csv", index=False)
        dbim_all = pd.read_csv(ddir / "fold_metrics.csv").assign(method="dbim (all test rows)", mitigated="")
        sel = pd.read_csv(bdir / "selection.csv")
        ratio = float((sel["n_test"] / sel["n_train"]).mean())
        inputs += [bdir / "fold_metrics.csv", bdir / "predictions.csv.gz", ddir / "predictions.csv.gz",
                   ddir / "fold_metrics.csv"]

        metrics = UTILITY + [f"{a}__{m}" for a in exp["protected"] for m in FAIRNESS]
        allm = pd.concat([dbim_cc.assign(method="dbim"), dbim_all, base])
        allm["label"] = [label(m, g) for m, g in zip(allm["method"], allm["mitigated"])]
        for lab, g in allm.groupby("label", sort=False):
            for met in metrics + ["precision", "recall", "brier"] + [f"{a}__{m}" for a in exp["protected"]
                                                                     for m in ["tpr_diff", "fpr_diff"]]:
                summaries.append({"experiment": exp["name"], "dbim_experiment": exp["compare_with"],
                                  "method": lab, "test_rows": "all" if "all test rows" in lab else "complete",
                                  "metric": met, **stats.mean_ci(g[met], ratio)})

        d_idx = dbim_cc.set_index(["repeat", "fold"])
        for met in metrics:
            family = []
            for lab, g in base.assign(label=[label(m, x) for m, x in zip(base["method"], base["mitigated"])]
                                      ).groupby("label", sort=False):
                b = g.set_index(["repeat", "fold"])[met]
                diff = (d_idx[met] - b).dropna()
                row = {"experiment": exp["name"], "metric": met,
                       "direction": "lower is better" if met.split("__")[-1] in LOWER_IS_BETTER
                       else "higher is better", "comparison": f"dbim - {lab}", "baseline": lab,
                       "dbim_mean": d_idx[met].mean(), "baseline_mean": b.mean(),
                       "test_train_ratio": ratio,
                       **stats.corrected_resampled_ttest(diff.to_numpy(), ratio),
                       **stats.wilcoxon_signed_rank(diff.to_numpy())}
                family.append(row)
            nb = stats.holm([r["nb_p"] for r in family])
            wx = stats.holm([r["wilcoxon_p"] for r in family])
            for r, a, b in zip(family, nb, wx):
                r["nb_p_holm"], r["wilcoxon_p_holm"], r["holm_family_size"] = a, b, len(family)
            tests += family
    pd.DataFrame(summaries).to_csv(out_sum / "baselines_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(out_sum / "baselines_tests.csv", index=False)
    return len(tests)


def timeseries(out_sum, inputs):
    d = REPO_ROOT / "results" / "raw" / "timeseries" / "air_quality_cleaning"
    t = pd.read_csv(d / "test_metrics.csv")
    rec = pd.read_csv(d / "reconstruction.csv")
    inputs += [d / "test_metrics.csv", d / "reconstruction.csv"]
    rows = []
    for (sc, h), g in t.groupby(["scenario", "horizon"]):
        for met in ["test_mae_scaled", "test_smape"]:
            v = g[met]
            fold_means = g.groupby("fold")[met].mean()
            rows.append({"scenario": sc, "horizon": h, "metric": met, "n": len(v), "mean": v.mean(),
                         "sd": v.std(ddof=1), "mean_of_fold_means": fold_means.mean(),
                         "sd_of_fold_means": fold_means.std(ddof=1)})
    for sc, g in rec.groupby("scenario"):
        per_fold = g.groupby("fold")["mae_scaled"].mean()
        rows.append({"scenario": sc, "horizon": "gap reconstruction", "metric": "mae_scaled", "n": len(per_fold),
                     "mean": per_fold.mean(), "sd": per_fold.std(ddof=1), "mean_of_fold_means": per_fold.mean(),
                     "sd_of_fold_means": per_fold.std(ddof=1)})
    pd.DataFrame(rows).to_csv(out_sum / "timeseries_summary.csv", index=False)

    tests = []
    piv = t.set_index(["fold", "seed", "horizon", "scenario"])
    for h in sorted(t["horizon"].unique()):
        for met in ["test_mae_scaled", "test_smape"]:
            family = []
            p = piv.xs(h, level="horizon")[met].unstack("scenario")
            for other in [c for c in p.columns if c != "proposed"]:
                diff = (p["proposed"] - p[other]).dropna().to_numpy()
                fold_diff = (p["proposed"] - p[other]).groupby(level="fold").mean()
                tt = stats.corrected_resampled_ttest(diff, 0.0)  # ratio 0: plain paired t, see note
                family.append({"horizon": h, "metric": met, "direction": "lower is better",
                               "comparison": f"proposed - {other}", "n_pairs": len(diff),
                               "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1),
                               "paired_t": tt["nb_t"], "df": tt["nb_df"], "t_p": tt["nb_p"],
                               "ci_low": tt["nb_ci_low"], "ci_high": tt["nb_ci_high"],
                               "cohens_dz": tt["cohens_dz"], **stats.wilcoxon_signed_rank(diff),
                               "fold_mean_diffs": json.dumps([round(x, 5) for x in fold_diff]),
                               "folds_proposed_better": int((fold_diff < 0).sum()),
                               "note": "pairs = 3 folds x 5 seeds; seeds within a fold share training data, "
                                       "so the 15 pairs are not independent and p-values are optimistic"})
            for r, a, b in zip(family, stats.holm([r["t_p"] for r in family]),
                               stats.holm([r["wilcoxon_p"] for r in family])):
                r["t_p_holm"], r["wilcoxon_p_holm"], r["holm_family_size"] = a, b, len(family)
            tests += family
    pd.DataFrame(tests).to_csv(out_sum / "timeseries_tests.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    out_raw = REPO_ROOT / "results" / "raw" / "comparison"
    out_sum = REPO_ROOT / "results" / "summary"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_sum.mkdir(parents=True, exist_ok=True)
    inputs = []
    n_tests = tabular(cfg, out_raw, out_sum, inputs)
    timeseries(out_sum, inputs)
    pd.DataFrame(LITERATURE).assign(
        comparable="no: literature-reported, not directly comparable (different splits, preprocessing, "
                   "and no access to per-fold results)").to_csv(out_sum / "literature_reported.csv", index=False)
    record = {"script": "scripts/compare_baselines.py", "config": args.config,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "n_tabular_tests": n_tests,
              "inputs": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in inputs},
              "packages": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]}}
    (out_sum / "comparison_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
