"""Nested, repeated, stratified cross-validation for the tabular experiments.

Outer loop: ``n_repeats`` x ``n_splits`` stratified folds (target x one protected
attribute), seed ``seeds[r]`` for repeat r. It gives the test estimate.

Inner loop: ``inner.n_splits`` stratified folds of the *outer training set only*.
It chooses the resampler and the classifier hyperparameters (GridSearchCV over
configs/search_space.yaml). The winner is refitted on the outer training set
and predicts the untouched outer test fold.

Everything data-dependent (imputation, encoding, resampling, the imbalance
check, the candidate list) is computed from the outer training set; the outer
test fold is only passed to ``predict``.
"""
import json
import logging
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import loaders
from experiments.memory_guard import GB, MemoryWatchdog, available_bytes, safe_n_jobs, tree_memory
from experiments.metrics import fold_metrics
from experiments.pipeline import candidate_resamplers, make_pipeline, param_grid

log = logging.getLogger(__name__)

LOADERS = {"adult": loaders.load_adult, "bank": loaders.load_bank}


def load_experiment_data(exp):
    """Returns X (model inputs), y, groups (protected attributes) and strata."""
    name = exp["dataset"]
    df = LOADERS[name](exp["variant"])
    if exp.get("subsample"):  # smoke tests only
        df = df.groupby(loaders.INFO[name]["target"], group_keys=False).sample(
            frac=exp["subsample"] / len(df), random_state=0).reset_index(drop=True)
    target = loaders.INFO[name]["target"]
    df = df.drop(columns=list(exp.get("drop_features", [])))
    y = df[target].to_numpy()
    groups = df[exp["protected"]].copy()
    X = df.drop(columns=[target])
    if not exp["protected_as_feature"]:
        X = X.drop(columns=exp["protected"])
    strata = pd.Series(y).astype(str).str.cat(df[exp["stratify_by"]].astype(str).to_numpy(), sep="|").to_numpy()
    return X, y, groups, strata


def outer_splits(strata, n_repeats, n_splits, seeds, only=None):
    """All outer folds; `only` = [[repeat, fold], ...] keeps just those (reproduction checks).
    The splits themselves do not change: a kept fold has exactly the rows it has in the full run."""
    keep = {tuple(x) for x in only} if only else None
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seeds[r])
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(strata)), strata)):
            if keep is None or (r, f) in keep:
                yield r, f, seeds[r], tr, te


def inner_splits(strata_train, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1000)
    return list(skf.split(np.zeros(len(strata_train)), strata_train))


def imbalance_check(y_train, groups_train, threshold_pct):
    """The original rule (datasets.py:identify_imbalance), on the training fold only."""
    def minority_pct(s):
        s = pd.Series(s)
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:  # numeric attributes were binned into 5
            s = pd.cut(s, bins=5)
        return 100 * s.value_counts(normalize=True).min()

    found = {"target": minority_pct(y_train)}
    for col in groups_train.columns:
        found[col] = minority_pct(groups_train[col])
    imbalanced = {k: round(v, 2) for k, v in found.items() if v < threshold_pct}
    return bool(imbalanced), imbalanced


def selection_scoring(space, domain):
    return space["selection_metric"].get(domain, space["selection_metric"]["default"])


def memory_settings(cfg):
    """The config's ``memory`` block with defaults (see experiments/memory_guard.py)."""
    m = {"per_worker_gb": 1.0, "main_gb": 2.0, "reserve_gb": 2.0, "budget_gb": None, **(cfg.get("memory") or {})}
    if m["budget_gb"] is None:  # everything free at start plus this run's own (reusable) memory, minus the reserve
        # Counting the run's own memory matters from the second experiment on: idle joblib workers of
        # the previous one still hold memory, which the watchdog counts as used. Without it the budget
        # was 3.8 GB too small and stopped a healthy run (2026-10-18 22:02, 5.7 GB still available).
        m["budget_gb"] = round((available_bytes() + tree_memory()[0]) / GB - m["reserve_gb"], 1)
    return m


def run_tabular_experiment(exp, cfg, space, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups, strata = load_experiment_data(exp)
    impute = exp["variant"] == "full" and bool(X.isna().any().any())
    scoring = selection_scoring(space, exp["domain"])
    log.info("%s: X %s, impute=%s, selection metric=%s", exp["name"], X.shape, impute, scoring)

    mem = memory_settings(cfg)
    log.info("%s: memory guard: budget %.1f GB, reserve %.1f GB, est. %.2f GB/worker",
             exp["name"], mem["budget_gb"], mem["reserve_gb"], mem["per_worker_gb"])

    predictions, selections, inner_rows, metric_rows = [], [], [], []
    start_all = time.perf_counter()
    for r, f, seed, tr, te in outer_splits(strata, cfg["outer"]["n_repeats"], cfg["outer"]["n_splits"], cfg["seeds"],
                                     cfg["outer"].get("only")):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
        resample, imbalanced = imbalance_check(y_tr, groups.iloc[tr], space["imbalance_threshold_pct"])
        # exp["resampling"] = false: the Phase 4 ablation's "cleaning only" stage (imputer + classifier).
        names = candidate_resamplers(space, len(tr)) if resample and exp.get("resampling", True) else []
        cache = tempfile.mkdtemp(prefix="pipe_cache_")
        context = f"{exp['name']} repeat {r} fold {f}"
        n_jobs = safe_n_jobs(cfg["n_jobs"], mem["per_worker_gb"], mem["main_gb"], mem["reserve_gb"], context)
        guard = MemoryWatchdog(mem["budget_gb"], mem["reserve_gb"], context)
        try:
            pipe = make_pipeline(X_tr, exp["classifier"], seed, impute)
            pipe.set_params(memory=cache)
            search = GridSearchCV(
                pipe, param_grid(space, exp["classifier"], names, seed), scoring=scoring,
                cv=inner_splits(strata[tr], cfg["inner"]["n_splits"], seed),
                refit=True, n_jobs=n_jobs, error_score=np.nan,
            )
            with guard:
                guard.stage = f"inner search + refit ({len(names) or 1} resamplers, n_jobs={n_jobs})"
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FitFailedWarning)
                    warnings.filterwarnings("ignore", message=".*test scores are non-finite.*")
                    search.fit(X_tr, y_tr)
                fit_seconds = time.perf_counter() - t0
                guard.stage = "predict outer test fold"
                t0 = time.perf_counter()
                score = search.predict_proba(X_te)[:, 1]
                pred = search.predict(X_te)
                predict_seconds = time.perf_counter() - t0
        finally:
            shutil.rmtree(cache, ignore_errors=True)

        res = pd.DataFrame(search.cv_results_)
        res["sampler_name"] = [type(s).__name__ if s != "passthrough" else "none" for s in res["param_sampler"]]
        failed = res[res["mean_test_score"].isna()]
        best = search.best_params_
        best_sampler = type(best["sampler"]).__name__ if best["sampler"] != "passthrough" else "none"
        clf_params = {k[5:]: v for k, v in best.items() if k.startswith("clf__")}
        selections.append({
            "repeat": r, "fold": f, "seed": seed, "n_train": len(tr), "n_test": len(te),
            "resampling_triggered": resample, "resampling_enabled": exp.get("resampling", True),
            "imbalanced": json.dumps(imbalanced),
            "candidates": len(res), "failed_candidates": len(failed),
            "failed_samplers": ";".join(sorted(set(failed["sampler_name"]))),
            "selected_sampler": best_sampler, "selected_clf_params": json.dumps(clf_params),
            "inner_best_score": search.best_score_,
            "fit_seconds": round(fit_seconds, 2), "predict_seconds": round(predict_seconds, 3),
            "n_jobs": n_jobs, "peak_memory_gb": round(guard.peak / GB, 2),
        })
        for _, row in res.iterrows():
            inner_rows.append({
                "repeat": r, "fold": f, "sampler": row["sampler_name"],
                "clf_params": json.dumps({k[11:]: row[k] for k in res.columns
                                          if k.startswith("param_clf__")}, default=str),
                "mean_inner_score": row["mean_test_score"], "std_inner_score": row["std_test_score"],
                "rank": row["rank_test_score"],
            })
        g_te = groups.iloc[te].reset_index(drop=True)
        fold_pred = pd.DataFrame({"repeat": r, "fold": f, "seed": seed, "row_id": te,
                                  "y_true": y_te, "y_pred": pred, "y_score": score})
        predictions.append(pd.concat([fold_pred, g_te], axis=1))
        metric_rows.append({"repeat": r, "fold": f, **fold_metrics(y_te, pred, score, g_te)})
        log.info("%s r%d f%d: %s %s inner=%.4f test acc=%.4f (fit %.0fs, n_jobs %d, peak %.1f GB)", exp["name"],
                 r, f, best_sampler, clf_params, search.best_score_, metric_rows[-1]["accuracy"], fit_seconds,
                 n_jobs, guard.peak / GB)

    pd.concat(predictions).to_csv(out_dir / "predictions.csv.gz", index=False)
    pd.DataFrame(selections).to_csv(out_dir / "selection.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(out_dir / "inner_scores.csv.gz", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    return {"experiment": exp["name"], "wall_clock_seconds": round(time.perf_counter() - start_all, 1),
            "n_rows": len(X), "n_features": X.shape[1], "imputation": impute, "selection_metric": scoring,
            "memory_guard": mem, "peak_memory_gb": max(s["peak_memory_gb"] for s in selections)}
