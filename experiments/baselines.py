"""Baselines under the same outer splits, seeds and preprocessing as the proposed pipeline (Phase 3).

For one dataset and one feature convention (protected attributes kept as model
inputs or dropped, author decision C6), every baseline is evaluated on the
outer folds of ``experiments.protocol.outer_splits``: the same seeds and the
same row indices as the DBIM run in results/raw/main/.

Data policy (author decision B2): baselines are trained on the complete rows
of each outer training fold (rows with a missing value are dropped, no
imputation). They are scored on the complete rows of the outer test fold, so
the statistical comparison with DBIM (scripts/compare_baselines.py) pairs
the same test rows in every fold.

Methods (configs/baselines.yaml):
  none            no mitigation; classifier hyperparameters chosen by inner CV
  random_over     RandomOverSampler + classifier, chosen jointly by inner CV
  random_under    RandomUnderSampler + classifier, chosen jointly by inner CV
  smote           SMOTE + classifier, chosen jointly by inner CV
  reweighing      AIF360 Reweighing weights (experiments/fair_baselines.py)
  dir             Disparate Impact Remover, repair level 1.0, fitted on the training fold
  exp_gradient    Fairlearn ExponentiatedGradient, demographic-parity constraint
  threshold_opt   Fairlearn ThresholdOptimizer, demographic-parity constraint
The four fairness methods mitigate one protected attribute at a time (as in
the original Table 25 scripts) and use the classifier hyperparameters that the
``none`` inner CV chose in the same outer fold. Resamplers balance the target,
as the DBIM resamplers do. Library defaults otherwise.

The inner CV uses ``inner_splits`` on the complete training rows, the same
grid as DBIM (configs/search_space.yaml) and the same selection metric.
"""
import logging
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

from experiments.fair_baselines import TrainFittedDIR, reweighing_weights
from experiments.memory_guard import GB, MemoryWatchdog, safe_n_jobs
from experiments.metrics import fold_metrics
from experiments.pipeline import make_classifier, make_encoder, make_pipeline, numeric_columns, param_grid
from experiments.protocol import (inner_splits, load_experiment_data, memory_settings, outer_splits,
                                  selection_scoring)

log = logging.getLogger(__name__)

RESAMPLING = {"none": None, "random_over": "RandomOverSampler", "random_under": "RandomUnderSampler",
              "smote": "SMOTE"}
FAIRNESS = ["reweighing", "dir", "exp_gradient", "threshold_opt"]


def _search(X, y, strata, classifier, sampler, space, scoring, seed, n_inner):
    pipe = make_pipeline(X, classifier, seed, impute=False)
    grid = param_grid(space, classifier, [sampler] if sampler else [], seed)
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner_splits(strata, n_inner, seed), refit=True, n_jobs=1)
    t0 = time.perf_counter()
    gs.fit(X, y)
    params = {k[5:]: v for k, v in gs.best_params_.items() if k.startswith("clf__")}
    return gs, params, time.perf_counter() - t0


def _clip_score(p, info):
    """Fairlearn's randomized predictors mix probabilities with weights that sum to 1 only up to
    floating-point rounding (seen: 1.0000000000000013), which sklearn's Brier score rejects.
    Clip to [0, 1] and record the largest excess."""
    info["score_clip_excess"] = float(max(p.max() - 1.0, -p.min(), 0.0))
    return np.clip(p, 0.0, 1.0)


def _fairness_fit_predict(method, Xe_tr, y_tr, a_tr, Xe_te, a_te, base, numeric_pos, seed, cfg):
    """Fit one fairness method on the encoded training fold; returns (pred, score, info)."""
    info = {}
    if method == "reweighing":
        w = reweighing_weights(y_tr, a_tr)
        base.fit(Xe_tr, y_tr, sample_weight=w)
        return base.predict(Xe_te), base.predict_proba(Xe_te)[:, 1], info
    if method == "dir":
        rep = TrainFittedDIR(numeric_pos, repair_level=cfg["dir_repair_level"]).fit(Xe_tr, a_tr)
        base.fit(rep.transform(Xe_tr, a_tr), y_tr)
        Xr = rep.transform(Xe_te, a_te)
        return base.predict(Xr), base.predict_proba(Xr)[:, 1], info
    if method == "threshold_opt":
        from fairlearn.postprocessing import ThresholdOptimizer
        to = ThresholdOptimizer(estimator=base, constraints=cfg["constraint"], predict_method="predict_proba",
                                prefit=False)
        to.fit(Xe_tr, y_tr, sensitive_features=a_tr)
        pred = to.predict(Xe_te, sensitive_features=a_te, random_state=seed)
        return pred, _clip_score(to._pmf_predict(Xe_te, sensitive_features=a_te)[:, 1], info), info
    if method == "exp_gradient":
        from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
        constraint = {"demographic_parity": DemographicParity, "equalized_odds": EqualizedOdds}[cfg["constraint"]]()
        eg = ExponentiatedGradient(base, constraint, eps=cfg["eg_eps"], max_iter=cfg["eg_max_iter"])
        eg.fit(Xe_tr, y_tr, sensitive_features=a_tr)
        info = {"eg_iterations": int(len(eg.predictors_)), "eg_nonzero_weights": int((eg.weights_ > 0).sum())}
        pred = eg.predict(Xe_te, random_state=seed)
        return pred, _clip_score(eg._pmf_predict(Xe_te)[:, 1], info), info
    raise ValueError(method)


def _fold_rows(method, attr, r, f, seed, te, y_te, pred, score, g_te):
    p = pd.DataFrame({"method": method, "mitigated": attr, "repeat": r, "fold": f, "seed": seed, "row_id": te,
                      "y_true": y_te, "y_pred": np.asarray(pred).astype(int), "y_score": score})
    p = pd.concat([p, g_te.reset_index(drop=True)], axis=1)
    m = {"method": method, "mitigated": attr, "repeat": r, "fold": f, **fold_metrics(y_te, pred, score, g_te)}
    return p, m


def run_fold(stage, exp, cfg, space, data, r, f, seed, tr, te):
    """One outer fold. stage 'main': inner searches + cheap fairness methods; 'eg': ExponentiatedGradient."""
    X, y, groups, strata = data
    bcfg = cfg["baselines"]
    scoring = selection_scoring(space, exp["domain"])
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
    g_te = groups.iloc[te]
    preds, metrics, sel = [], [], []
    none_params = None
    if stage == "main":
        for method in [m for m in exp["methods"] if m in RESAMPLING]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                gs, params, secs = _search(X_tr, y_tr, strata[tr], exp["classifier"], RESAMPLING[method], space,
                                           scoring, seed, cfg["inner"]["n_splits"])
            t0 = time.perf_counter()
            pred, score = gs.predict(X_te), gs.predict_proba(X_te)[:, 1]
            p, m = _fold_rows(method, "", r, f, seed, te, y_te, pred, score, g_te)
            preds.append(p), metrics.append(m)
            sel.append({"method": method, "mitigated": "", "repeat": r, "fold": f, "seed": seed,
                        "n_train": len(tr), "n_test": len(te), "clf_params": params,
                        "inner_best_score": gs.best_score_, "fit_seconds": round(secs, 2),
                        "predict_seconds": round(time.perf_counter() - t0, 3)})
            if method == "none":
                none_params = params
    fair = [m for m in exp["methods"] if m in FAIRNESS and ((m == "exp_gradient") == (stage == "eg"))]
    if fair:
        if none_params is None:   # eg stage: the params the 'none' search chose (passed in)
            none_params = exp["_none_params"][(r, f)]
        enc = make_encoder(X_tr).fit(X_tr)
        Xe_tr, Xe_te = enc.transform(X_tr), enc.transform(X_te)
        out_names = list(enc.get_feature_names_out())
        numeric_pos = [out_names.index(c) for c in numeric_columns(X_tr)]
        for attr in exp["mitigate"]:
            a_tr, a_te = groups[attr].to_numpy()[tr], groups[attr].to_numpy()[te]
            for method in fair:
                base = make_classifier(exp["classifier"], seed).set_params(**none_params)
                if stage == "eg" and "n_jobs" in base.get_params():
                    base.set_params(n_jobs=bcfg["eg_classifier_threads"])  # same trees for any thread count
                t0 = time.perf_counter()
                pred, score, info = _fairness_fit_predict(method, Xe_tr, y_tr, a_tr, Xe_te, a_te, base,
                                                          numeric_pos, seed, bcfg)
                secs = time.perf_counter() - t0
                p, m = _fold_rows(method, attr, r, f, seed, te, y_te, pred, score, g_te)
                preds.append(p), metrics.append(m)
                sel.append({"method": method, "mitigated": attr, "repeat": r, "fold": f, "seed": seed,
                            "n_train": len(tr), "n_test": len(te), "clf_params": none_params,
                            "fit_seconds": round(secs, 2), **info})
    return r, f, preds, metrics, sel


def _checkpointed(path, stage, exp, cfg, space, data, fold):
    """run_fold, but reuse the saved result of an earlier, interrupted run of the same fold."""
    import pickle
    if path.exists():
        return pickle.loads(path.read_bytes())
    res = run_fold(stage, exp, cfg, space, data, *fold)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(res))
    tmp.rename(path)
    return res


def run_baseline_experiment(exp, cfg, space, out_dir):
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups, strata = load_experiment_data(exp)
    if groups.isna().any().any():
        raise ValueError("protected attributes have missing values; the complete-case rule would drop groups")
    complete = ~X.isna().any(axis=1).to_numpy()
    log.info("%s: %d rows, %d complete (%d dropped for training and testing), X %s", exp["name"], len(X),
             complete.sum(), (~complete).sum(), X.shape)
    data = (X, y, groups, strata)
    folds = [(r, f, seed, tr[complete[tr]], te[complete[te]])
             for r, f, seed, tr, te in outer_splits(strata, cfg["outer"]["n_repeats"], cfg["outer"]["n_splits"],
                                                    cfg["seeds"], cfg["outer"].get("only"))]
    mem = memory_settings(cfg)
    # Each finished fold is saved, so a crashed run resumes where it stopped (same code and config:
    # the folder is keyed by the git commit; results are deterministic given the seeds).
    from run_experiments import git
    ckpt = out_dir / "partial" / git("rev-parse", "--short", "HEAD")
    ckpt.mkdir(parents=True, exist_ok=True)
    resumed = sorted(p.name for p in ckpt.glob("*.pkl"))
    if resumed:
        log.info("%s: resuming, %d fold results already saved in %s", exp["name"], len(resumed), ckpt)
    preds, metrics, sel = [], [], []
    t_all = time.perf_counter()

    # Stage 1: every method except ExponentiatedGradient, outer folds in parallel.
    context = f"{exp['name']} baselines (folds in parallel)"
    n_jobs = min(len(folds), safe_n_jobs(cfg["n_jobs"], mem["per_worker_gb"], mem["main_gb"], mem["reserve_gb"],
                                         context))
    with MemoryWatchdog(mem["budget_gb"], mem["reserve_gb"], context) as guard:
        guard.stage = f"inner searches + fairness methods, {len(folds)} folds, n_jobs={n_jobs}"
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_checkpointed)(ckpt / f"main_r{fold[0]}_f{fold[1]}.pkl", "main", exp, cfg, space, data, fold)
            for fold in folds)
        for r, f, p, m, s in results:
            preds += p
            metrics += m
            sel += s
            log.info("%s r%d f%d: %s", exp["name"], r, f, "  ".join(
                f"{x['method']}{'[' + x['mitigated'] + ']' if x['mitigated'] else ''} acc={x['accuracy']:.4f}"
                for x in m))
    peak = guard.peak

    # Stage 2: ExponentiatedGradient keeps every classifier it fits (up to eg_max_iter), so it
    # runs one fold at a time; the classifier itself uses eg_classifier_threads threads.
    if "exp_gradient" in exp["methods"]:
        exp = {**exp, "_none_params": {(s["repeat"], s["fold"]): s["clf_params"] for s in sel
                                       if s["method"] == "none"}}
        context = f"{exp['name']} ExponentiatedGradient (one fold at a time)"
        with MemoryWatchdog(mem["budget_gb"], mem["reserve_gb"], context) as guard:
            for fold in folds:
                guard.stage = f"ExponentiatedGradient r{fold[0]} f{fold[1]}"
                r, f, p, m, s = _checkpointed(ckpt / f"eg_r{fold[0]}_f{fold[1]}.pkl", "eg", exp, cfg, space, data,
                                              fold)
                preds += p
                metrics += m
                sel += s
                log.info("%s r%d f%d: %s", exp["name"], r, f, "  ".join(
                    f"exp_gradient[{x['mitigated']}] acc={x['accuracy']:.4f} ({y['eg_iterations']} iter, "
                    f"{y['fit_seconds']:.0f}s)" for x, y in zip(m, s)))
        peak = max(peak, guard.peak)

    keys = ["method", "mitigated", "repeat", "fold"]
    pd.concat(preds).sort_values(keys + ["row_id"]).to_csv(out_dir / "predictions.csv.gz", index=False)
    pd.DataFrame(metrics).sort_values(keys).to_csv(out_dir / "fold_metrics.csv", index=False)
    s = pd.DataFrame(sel)
    s["clf_params"] = s["clf_params"].map(lambda d: pd.Series(d).to_json())
    s.sort_values(keys).to_csv(out_dir / "selection.csv", index=False)
    import shutil
    shutil.rmtree(out_dir / "partial", ignore_errors=True)   # final outputs written; drop the fold cache
    return {"experiment": exp["name"], "wall_clock_seconds": round(time.perf_counter() - t_all, 1),
            "n_rows": int(len(X)), "n_complete_rows": int(complete.sum()), "n_features": X.shape[1],
            "methods": exp["methods"], "mitigate": exp["mitigate"], "n_jobs_stage1": n_jobs,
            "resumed_fold_results": resumed,
            "peak_memory_gb": round(peak / GB, 2)}
