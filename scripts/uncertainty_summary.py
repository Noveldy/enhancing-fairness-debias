"""Phase 7.1: uncertainty of every reported metric, and of the choices model selection makes (R1.2).

    venv/bin/python scripts/uncertainty_summary.py

No new training: everything is computed from the per-fold files of Phases 2-6.

1. ``uncertainty_ci.csv``: mean, SD and 95% CI for the results that had no CI yet: synthetic-data
   fidelity and privacy, per-group selection rate / TPR / FPR, the detector's internal CV and the Air
   Quality forecasts. The other summaries (baselines_summary, fairness_utility_summary, ablation_summary,
   synthetic_utility) already give mean, SD and the corrected 95% CI; ``uncertainty_index.csv`` lists
   every summary table with its unit of replication, n and CI method.
   CI methods: repeated k-fold CV results use the Nadeau-Bengio corrected t interval
   (experiments/stats.mean_ci; training sets of different folds overlap). Air Quality: the 3
   expanding-window folds are the independent unit, so the primary CI is a t interval over the 3 fold
   means (df 2); a t interval over all 15 fold x seed runs is given too, marked as optimistic.
2. ``selection_frequencies.csv``: how often each resampler and each hyperparameter value was chosen
   across the 25 outer folds, per experiment and method; ``selection_stability.csv``: per experiment,
   the number of distinct configurations chosen, the share of the most frequent one, and how close the
   runners-up were in the inner CV (candidates within one standard error of the winner's inner score).
   Air Quality: the LSTM configuration chosen per fold and scenario.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from experiments import stats  # noqa: E402

RAW, SUM = REPO_ROOT / "results" / "raw", REPO_ROOT / "results" / "summary"
KFOLD_RATIO = 1 / 4        # n_test / n_train of 5-fold CV
INNER_FOLDS = 3


def _read(path, inputs):
    inputs.append(path)
    return pd.read_csv(path, float_precision="round_trip")


def t_ci(v, alpha=0.05):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    n, mean, sd = len(v), v.mean(), v.std(ddof=1)
    h = st.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - h, "ci_high": mean + h}


# ----------------------------------------------------------------------------- 1. confidence intervals
def ci_tables(inputs):
    rows = []
    nb = "Nadeau-Bengio corrected t, 25 outer folds"
    for exp in ["adult_synthetic", "bank_synthetic"]:
        fid = _read(RAW / "synthetic" / exp / "fidelity.csv", inputs)
        ratio = float((fid["n_real"] / fid["n_synth"]).mean())   # test rows / training rows
        for gen, g in fid.groupby("generator"):
            for met in ["mean_ks_numeric", "max_ks_numeric", "mean_tv", "max_tv", "pairwise_dependence_mae",
                        "pairwise_dependence_max", "spearman_diff_frobenius", "c2st_auc"]:
                rows.append({"source": "synthetic fidelity", "experiment": exp, "method": gen, "setting": "",
                             "metric": met, **stats.mean_ci(g[met], ratio), "ci_method": nb})
        priv = _read(RAW / "synthetic" / exp / "privacy.csv", inputs)
        for gen, g in priv.groupby("generator"):
            for met in ["dcr_train_median", "dcr_holdout_median", "exact_copies_train", "share_closer_to_train",
                        "mia_auc"]:
                rows.append({"source": "synthetic privacy", "experiment": exp, "method": gen, "setting": "",
                             "metric": met, **stats.mean_ci(g[met], ratio), "ci_method": nb})
    for path in sorted((RAW / "fairness_utility").glob("*_group_rates.csv")):
        exp = path.name.replace("_group_rates.csv", "")
        gr = _read(path, inputs)
        for (m, tr, a, grp), g in gr.groupby(["method", "test_rows", "attribute", "group"]):
            for met in ["selection_rate", "tpr", "fpr"]:
                if g[met].notna().sum() >= 2:
                    rows.append({"source": "group rates", "experiment": exp, "method": m,
                                 "setting": f"{tr} test rows; {a}={grp}", "metric": met,
                                 **stats.mean_ci(g[met], KFOLD_RATIO), "ci_method": nb})
    det = _read(RAW / "detector" / "internal_cv_folds.csv", inputs)
    for split, g in det.groupby("split"):
        ratio = float((g["n_test"] / (g["n_train"] + g["n_val"])).mean())
        k = g.groupby("repeat").size().iloc[0]
        for met in ["accuracy", "precision", "recall", "f1", "roc_auc", "keyword_on_text_accuracy",
                    "keyword_on_text_f1"]:
            rows.append({"source": "detector internal CV", "experiment": "detector", "method": "bert" if
                         not met.startswith("keyword") else "keyword rule on description", "setting": split,
                         "metric": met.replace("keyword_on_text_", ""), **stats.mean_ci(g[met], ratio),
                         "ci_method": f"Nadeau-Bengio corrected t, {len(g)} folds ({len(g) // k} x {k}-fold)"})
    ts = _read(RAW / "timeseries" / "air_quality_cleaning" / "test_metrics.csv", inputs)
    for (sc, h), g in ts.groupby(["scenario", "horizon"]):
        for met in ["test_mae_scaled", "test_smape"]:
            rows.append({"source": "air quality forecast", "experiment": "air_quality_cleaning", "method": sc,
                         "setting": f"horizon {h}", "metric": met, **t_ci(g.groupby("fold")[met].mean()),
                         "ci_method": "t over 3 fold means (df 2); folds are the independent unit"})
            rows.append({"source": "air quality forecast", "experiment": "air_quality_cleaning", "method": sc,
                         "setting": f"horizon {h}", "metric": met, **t_ci(g[met]),
                         "ci_method": "t over 15 fold x seed runs (optimistic: seeds share a fold's data)"})
    rec = _read(RAW / "timeseries" / "air_quality_cleaning" / "reconstruction.csv", inputs)
    for sc, g in rec.groupby("scenario"):
        rows.append({"source": "air quality gap reconstruction", "experiment": "air_quality_cleaning",
                     "method": sc, "setting": "artificial gap", "metric": "mae_scaled",
                     **t_ci(g.groupby("fold")["mae_scaled"].mean()),
                     "ci_method": "t over 3 fold means (df 2); folds are the independent unit"})
    return pd.DataFrame(rows)


def index_table():
    """Every summary table with its replication unit and CI method (for the manuscript's methods text)."""
    nb25 = ("25 outer folds (5 seeds x 5 folds)", "Nadeau-Bengio corrected t")
    return pd.DataFrame([
        ("baselines_summary.csv", "DBIM and 8 baselines, utility and fairness", *nb25),
        ("fairness_utility_summary.csv", "all methods incl. trivial predictors, utility, calibration, gaps", *nb25),
        ("ablation_summary.csv", "3-stage ablation", *nb25),
        ("synthetic_utility.csv", "TSTR / TRTR / TSTS", *nb25),
        ("uncertainty_ci.csv", "synthetic fidelity and privacy; per-group rates", *nb25),
        ("uncertainty_ci.csv", "detector internal CV", "15 grouped folds (3 x 5) / 5 random folds",
         "Nadeau-Bengio corrected t"),
        ("uncertainty_ci.csv", "Air Quality forecasts and gap reconstruction", "3 folds x 5 seeds",
         "t over 3 fold means (primary); t over 15 runs (optimistic)"),
        ("detector_external.csv", "detector on 52 external columns", "5 seeds (one column set)",
         "none: fixed column set; per-seed values shown"),
        ("*_tests.csv", "paired comparisons", "paired folds", "Nadeau-Bengio t + Wilcoxon, Holm-adjusted"),
    ], columns=["table", "content", "replication", "ci_method"])


# ----------------------------------------------------------------------------- 2. selection stability
def _params(s):
    d = json.loads(s) if isinstance(s, str) else s
    return {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in d.items()}


def _config(sampler, params):
    return f"{sampler} " + json.dumps(params, sort_keys=True)


def selection_tables(inputs):
    freq, stab = [], []
    runs = [("main", p) for p in sorted((RAW / "main").glob("*_dbim_*")) if p.is_dir()]
    runs += [("ablation", p) for p in sorted((RAW / "ablation").glob("*_impute_only_*")) if p.is_dir()]
    for source, d in runs:
        sel = _read(d / "selection.csv", inputs)
        inner = pd.read_csv(d / "inner_scores.csv.gz", float_precision="round_trip")
        inputs.append(d / "inner_scores.csv.gz")
        method = "dbim" if source == "main" else "cleaning only"
        params = sel["selected_clf_params"].map(_params)
        configs = [_config(s, p) for s, p in zip(sel["selected_sampler"], params)]
        _freq_rows(freq, d.name, method, sel["selected_sampler"], params)
        ties, gaps, tie_samplers = [], [], []
        for (r, f), g in inner.groupby(["repeat", "fold"]):
            g = g.dropna(subset=["mean_inner_score"]).sort_values("mean_inner_score", ascending=False)
            best = g.iloc[0]
            se = best["std_inner_score"] / np.sqrt(INNER_FOLDS)
            near = g[g["mean_inner_score"] >= best["mean_inner_score"] - se]
            ties.append(len(near) - 1)
            gaps.append(best["mean_inner_score"] - g.iloc[1]["mean_inner_score"] if len(g) > 1 else np.nan)
            tie_samplers.append(near["sampler"].nunique() > 1)
        c = Counter(configs)
        stab.append({"experiment": d.name, "method": method, "n_folds": len(sel), "distinct_configs": len(c),
                     "most_frequent_config": c.most_common(1)[0][0], "most_frequent_share": c.most_common(1)[0][1] /
                     len(sel), "most_frequent_sampler_share": sel["selected_sampler"].value_counts(normalize=True)
                     .iloc[0], "candidates_per_fold": int(inner.groupby(["repeat", "fold"]).size().iloc[0]),
                     "median_runner_up_gap": float(np.nanmedian(gaps)),
                     "median_candidates_within_1se": float(np.median(ties)),
                     "share_folds_with_tie_across_samplers": float(np.mean(tie_samplers)),
                     "selection_metric": "accuracy (inner 3-fold CV)"})
    for p in sorted((RAW / "baselines").glob("*_baselines_*")):
        if not p.is_dir():
            continue
        sel = _read(p / "selection.csv", inputs)
        sel = sel[sel["method"].isin(["none", "random_over", "random_under", "smote"])]
        for m, g in sel.groupby("method"):
            params = g["clf_params"].map(_params)
            _freq_rows(freq, p.name, m, None, params)
            c = Counter(_config("", q) for q in params)
            stab.append({"experiment": p.name, "method": m, "n_folds": len(g), "distinct_configs": len(c),
                         "most_frequent_config": c.most_common(1)[0][0].strip(),
                         "most_frequent_share": c.most_common(1)[0][1] / len(g),
                         "selection_metric": "accuracy (inner 3-fold CV); inner scores of runners-up not stored"})
    ls = _read(RAW / "timeseries" / "air_quality_cleaning" / "lstm_selection.csv", inputs)
    chosen = ls.loc[ls.groupby(["fold", "scenario"])["val_mae_scaled"].idxmin()]
    for (lr, icl), g in chosen.groupby(["lr", "input_chunk_length"]):
        freq.append({"experiment": "air_quality_cleaning", "method": "lstm (all scenarios)", "choice":
                     "lr, input_chunk_length", "value": f"{lr:g}, {icl}", "count": len(g), "share": len(g) /
                     len(chosen), "n": len(chosen)})
    return pd.DataFrame(freq), pd.DataFrame(stab)


def _freq_rows(freq, exp, method, samplers, params):
    n = len(params)
    if samplers is not None:
        for v, k in samplers.value_counts().items():
            freq.append({"experiment": exp, "method": method, "choice": "resampler", "value": v, "count": int(k),
                         "share": k / n, "n": n})
    for key in sorted({k for p in params for k in p}):
        for v, k in Counter(json.dumps(p.get(key)) for p in params).most_common():
            freq.append({"experiment": exp, "method": method, "choice": key, "value": v, "count": k,
                         "share": k / n, "n": n})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    inputs = []
    ci = ci_tables(inputs)
    ci.to_csv(SUM / "uncertainty_ci.csv", index=False)
    index_table().to_csv(SUM / "uncertainty_index.csv", index=False)
    freq, stab = selection_tables(inputs)
    freq.to_csv(SUM / "selection_frequencies.csv", index=False)
    stab.to_csv(SUM / "selection_stability.csv", index=False)
    from run_experiments import git
    record = {"script": "scripts/uncertainty_summary.py",
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git("rev-parse", "HEAD"),
              "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
              "inputs": {str(Path(p).relative_to(REPO_ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in sorted(set(map(str, inputs)))}}
    (SUM / "uncertainty_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(ci)} CI rows, {len(freq)} frequency rows, {len(stab)} stability rows")


if __name__ == "__main__":
    main()
