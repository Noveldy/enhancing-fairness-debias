"""Phase 7.2-7.3: wall-clock time and peak memory of every pipeline stage, and how they grow with rows.

Measurement rules, so that numbers of different methods are comparable:

* Every task runs in a fresh process (multiprocessing 'spawn'), pinned to one performance core, with
  every thread pool (OpenMP, BLAS, torch) limited to one thread and scikit-learn n_jobs=1. All times
  are therefore single-core times. The experiment runs of Phases 2-5 used 5-18 parallel workers and
  their recorded times are not comparable between methods (DBIM's inner search ran 6 fits at once,
  each baseline fold ran one fit at a time next to other folds).
* Wall clock: ``time.perf_counter`` around the timed call only; data loading and untimed preparation
  are excluded. CPU time: ``time.process_time`` over the same call.
* Memory: resident set size (RSS) of the process, sampled every 5 ms by a background thread.
  Reported as the peak during the call and the increase over the RSS just before it.
* The driver (scripts/cost_run.py) kills a task that exceeds its timeout and records 'timeout'.

Tasks (``TASKS``): dbim_fit (final DBIM fit with the configuration the Phase 2 inner CV chose, stage
by stage), search (the whole inner-CV model selection, as in the experiments), resampler (one
resampler on the cleaned, encoded training rows), baseline_fit (Phase 3 baselines with the chosen
hyperparameters), generator (Phase 5 generators), detection (Phase 6 detector), ts_cleaning and
ts_lstm (Air Quality). Each returns a list of measurement rows.
"""
import os
import pickle
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = os.sysconf("SC_PAGE_SIZE")
GB = 1024 ** 3
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


# ----------------------------------------------------------------------------- measurement
def _rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE


class Meter:
    """Times a block and samples the process RSS every ``interval`` seconds while it runs."""

    def __init__(self, interval=0.005):
        self.interval = interval

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, _rss())

    def __enter__(self):
        self.rss0 = self.peak = _rss()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.c0, self.t0 = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.cpu_seconds = time.process_time() - self.c0
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss())
        return False

    def record(self):
        return {"seconds": self.seconds, "cpu_seconds": self.cpu_seconds, "peak_rss_gb": self.peak / GB,
                "rss_increase_gb": max(self.peak - self.rss0, 0) / GB}


def timed(rows, base, stage, fn, **extra):
    with Meter() as m:
        out = fn()
    rows.append({**base, "stage": stage, **m.record(), **extra})
    return out


def setup_process(cpu):
    """Child-process setup: pin to one CPU, one thread everywhere."""
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data
def _main_cfg():
    return yaml.safe_load((REPO_ROOT / "configs" / "main.yaml").read_text())


def experiment(dataset, convention="protected_kept"):
    cfg = _main_cfg()
    name = f"{dataset}_dbim_{convention}"
    return next(e for e in cfg["tabular_experiments"] if e["name"] == name), cfg


def fold_data(dataset, repeat, fold, frac=1.0, sub_seed=0):
    """Rows of one outer fold of configs/main.yaml; with frac < 1, a stratified subsample of its
    training rows (same strata as the outer split). The test rows are always the whole outer test fold."""
    from sklearn.model_selection import train_test_split
    from experiments.protocol import load_experiment_data, outer_splits
    exp, cfg = experiment(dataset)
    X, y, groups, strata = load_experiment_data(exp)
    tr, te = next((tr, te) for r, f, s, tr, te in
                  outer_splits(strata, cfg["outer"]["n_repeats"], cfg["outer"]["n_splits"], cfg["seeds"])
                  if r == repeat and f == fold)
    seed = cfg["seeds"][repeat]
    if frac < 1:
        tr, _ = train_test_split(tr, train_size=frac, stratify=strata[tr], random_state=sub_seed)
        tr = np.sort(tr)
    return {"exp": exp, "cfg": cfg, "X": X, "y": y, "groups": groups, "strata": strata, "tr": tr, "te": te,
            "seed": seed}


def dbim_selection(dataset, repeat, fold):
    s = pd.read_csv(REPO_ROOT / "results/raw/main" / f"{dataset}_dbim_protected_kept" / "selection.csv")
    row = s[(s["repeat"] == repeat) & (s["fold"] == fold)].iloc[0]
    params = {k: int(v) if isinstance(v, float) and v.is_integer() else v
              for k, v in __import__("json").loads(row["selected_clf_params"]).items()}
    return row["selected_sampler"], params, row


def _base(spec, d, **kw):
    return {"task_id": spec["task_id"], "part": spec["part"], "task": spec["task"], "dataset": spec.get("dataset"),
            "repeat": spec.get("repeat"), "fold": spec.get("fold"), "frac": spec.get("frac", 1.0),
            "sub_seed": spec.get("sub_seed", 0), "n_train_rows": len(d["tr"]) if d else None,
            "n_test_rows": len(d["te"]) if d else None, **kw}


def cache_path(cache_dir, spec):
    return Path(cache_dir) / f"{spec['dataset']}_r{spec['repeat']}_f{spec['fold']}_p{spec.get('frac', 1.0)}" \
                             f"_s{spec.get('sub_seed', 0)}.pkl"


# ----------------------------------------------------------------------------- tasks
def task_dbim_fit(spec, cache_dir):
    """The final DBIM fit, one stage at a time: what the experiments refit after model selection."""
    from imblearn.base import BaseSampler  # noqa: F401  (import cost outside the timed calls)
    from experiments.imputation import SimilarityImputer
    from experiments.metrics import extended_metrics, group_rate_rows
    from experiments.pipeline import make_classifier, make_encoder, make_resampler
    from experiments.protocol import imbalance_check
    d = fold_data(spec["dataset"], spec["repeat"], spec["fold"], spec.get("frac", 1.0), spec.get("sub_seed", 0))
    sel_r, sel_f = spec.get("config_from", (spec["repeat"], spec["fold"]))
    sampler_name, params, sel = dbim_selection(spec["dataset"], sel_r, sel_f)
    space = yaml.safe_load((REPO_ROOT / "configs" / "search_space.yaml").read_text())
    X_tr, X_te = d["X"].iloc[d["tr"]], d["X"].iloc[d["te"]]
    y_tr, y_te = d["y"][d["tr"]], d["y"][d["te"]]
    impute = d["exp"]["variant"] == "full" and bool(d["X"].isna().any().any())
    rows = []
    b = _base(spec, d, method="dbim", config=f"{sampler_name} {params}")
    timed(rows, b, "imbalance_check",
          lambda: imbalance_check(y_tr, d["groups"].iloc[d["tr"]], space["imbalance_threshold_pct"]))
    imp = SimilarityImputer() if impute else None
    Xc = timed(rows, b, "cleaning", lambda: imp.fit_transform(X_tr)) if impute else X_tr
    enc = make_encoder(X_tr)
    Xe = timed(rows, b, "encoding", lambda: enc.fit_transform(Xc))
    with open(cache_path(cache_dir, spec), "wb") as fh:   # input of the resampler tasks
        pickle.dump({"X": Xe, "y": y_tr}, fh)
    sampler = make_resampler(sampler_name, d["seed"]) if sampler_name != "none" else None
    Xr, yr = timed(rows, b, "resampling", lambda: sampler.fit_resample(Xe, y_tr),
                   resampler=sampler_name) if sampler else (Xe, y_tr)
    clf = make_classifier(d["exp"]["classifier"], d["seed"]).set_params(**params)
    timed(rows, b, "classifier_fit", lambda: clf.fit(Xr, yr), n_fit_rows=len(yr))

    def infer():
        Xt = enc.transform(imp.transform(X_te) if impute else X_te)
        return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]
    pred, score = timed(rows, b, "inference", infer)
    g_te = d["groups"].iloc[d["te"]].reset_index(drop=True)
    timed(rows, b, "fairness_evaluation",
          lambda: (extended_metrics(y_te, pred, score, g_te), group_rate_rows(y_te, pred, g_te)))
    # Check: with the stored configuration on the full fold, this fit reproduces the stored predictions.
    if spec.get("frac", 1.0) == 1.0 and (sel_r, sel_f) == (spec["repeat"], spec["fold"]):
        stored = pd.read_csv(REPO_ROOT / "results/raw/main" / f"{spec['dataset']}_dbim_protected_kept" /
                             "fold_metrics.csv")
        acc = stored[(stored["repeat"] == sel_r) & (stored["fold"] == sel_f)]["accuracy"].iloc[0]
        for r in rows:
            r["reproduces_stored_accuracy"] = bool(abs(float((pred == y_te).mean()) - acc) < 1e-12)
    return rows


def task_resampler(spec, cache_dir):
    from experiments.pipeline import make_resampler
    from experiments.protocol import load_experiment_data  # noqa: F401
    with open(cache_path(cache_dir, spec), "rb") as fh:
        c = pickle.load(fh)
    _, cfg = experiment(spec["dataset"])
    seed = cfg["seeds"][spec["repeat"]]
    rows = []
    b = {**_base(spec, None, method="dbim"), "n_train_rows": len(c["y"])}
    sampler = make_resampler(spec["resampler"], seed)
    try:
        Xr, yr = timed(rows, b, "resampling", lambda: sampler.fit_resample(c["X"], c["y"]),
                       resampler=spec["resampler"])
        rows[-1]["n_out_rows"] = len(yr)
    except Exception as e:   # e.g. KMeansSMOTE finds no cluster to oversample: record, do not hide
        rows.append({**b, "stage": "resampling", "resampler": spec["resampler"], "status": "error",
                     "error": f"{type(e).__name__}: {e}"[:300]})
    return rows


def task_search(spec, cache_dir):
    """The whole model selection of one outer fold, as run in the experiments but with n_jobs=1."""
    import shutil
    import tempfile
    from sklearn.exceptions import FitFailedWarning
    from sklearn.model_selection import GridSearchCV
    from experiments.pipeline import candidate_resamplers, make_pipeline, param_grid
    from experiments.protocol import imbalance_check, inner_splits, selection_scoring
    method = spec["method"]
    d = fold_data(spec["dataset"], spec["repeat"], spec["fold"], spec.get("frac", 1.0), spec.get("sub_seed", 0))
    space = yaml.safe_load((REPO_ROOT / "configs" / "search_space.yaml").read_text())
    exp, cfg = d["exp"], d["cfg"]
    tr = d["tr"]
    if method != "dbim":   # baselines: complete rows (B2)
        complete = ~d["X"].isna().any(axis=1).to_numpy()
        tr = tr[complete[tr]]
    X_tr, y_tr = d["X"].iloc[tr], d["y"][tr]
    rows = []
    b = _base(spec, d, method=method)
    b["n_train_rows"] = len(tr)
    if method == "dbim":
        resample, _ = imbalance_check(y_tr, d["groups"].iloc[tr], space["imbalance_threshold_pct"])
        names = candidate_resamplers(space, len(tr)) if resample else []
        impute = exp["variant"] == "full" and bool(d["X"].isna().any().any())
    else:
        from experiments.baselines import RESAMPLING
        names = [RESAMPLING[method]] if RESAMPLING[method] else []
        impute = False
    pipe = make_pipeline(X_tr, exp["classifier"], d["seed"], impute)
    cache = tempfile.mkdtemp(prefix="pipe_cache_")
    if method == "dbim":
        pipe.set_params(memory=cache)   # as in experiments/protocol.py (the imputer is cached across candidates)
    gs = GridSearchCV(pipe, param_grid(space, exp["classifier"], names, d["seed"]),
                      scoring=selection_scoring(space, exp["domain"]),
                      cv=inner_splits(d["strata"][tr], cfg["inner"]["n_splits"], d["seed"]), refit=True,
                      n_jobs=1, error_score=np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            timed(rows, b, "model_selection", lambda: gs.fit(X_tr, y_tr), n_candidates=len(names or [0]) * int(
                np.prod([len(v) for v in space["classifiers"][exp["classifier"]].values()])),
                  n_resamplers=len(names), resamplers=";".join(names))
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    best = gs.best_params_
    rows[-1]["selected_sampler"] = type(best["sampler"]).__name__ if best["sampler"] != "passthrough" else "none"
    rows[-1]["selected_clf_params"] = str({k[5:]: v for k, v in best.items() if k.startswith("clf__")})
    rows[-1]["fit_failures"] = int(pd.DataFrame(gs.cv_results_)["mean_test_score"].isna().sum())
    if spec.get("frac", 1.0) == 1.0 and method == "dbim":
        s, p, _ = dbim_selection(spec["dataset"], spec["repeat"], spec["fold"])
        rows[-1]["reproduces_stored_selection"] = (rows[-1]["selected_sampler"] == s and
                                                   {k[5:]: v for k, v in best.items() if k.startswith("clf__")} == p)
    return rows


def task_baseline_fit(spec, cache_dir):
    """A Phase 3 baseline fitted once with the hyperparameters its inner CV chose in this fold."""
    import json
    from experiments.baselines import FAIRNESS, RESAMPLING, _fairness_fit_predict
    from experiments.metrics import extended_metrics, group_rate_rows
    from experiments.pipeline import make_classifier, make_encoder, make_resampler, numeric_columns
    method = spec["method"]
    d = fold_data(spec["dataset"], spec["repeat"], spec["fold"])
    complete = ~d["X"].isna().any(axis=1).to_numpy()
    tr, te = d["tr"][complete[d["tr"]]], d["te"][complete[d["te"]]]
    d = {**d, "tr": tr, "te": te}
    bcfg = yaml.safe_load((REPO_ROOT / "configs" / "baselines.yaml").read_text())["baselines"]
    sel = pd.read_csv(REPO_ROOT / "results/raw/baselines" / f"{spec['dataset']}_baselines_protected_kept" /
                      "selection.csv")
    key = "none" if method in FAIRNESS else method
    row = sel[(sel["method"] == key) & (sel["repeat"] == spec["repeat"]) & (sel["fold"] == spec["fold"])].iloc[0]
    params = {k: int(v) if isinstance(v, float) and v.is_integer() else v
              for k, v in json.loads(row["clf_params"]).items()}
    X_tr, X_te, y_tr, y_te = d["X"].iloc[tr], d["X"].iloc[te], d["y"][tr], d["y"][te]
    g_te = d["groups"].iloc[te].reset_index(drop=True)
    rows = []
    b = _base(spec, d, method=method, mitigated=spec.get("mitigate", ""), config=str(params))
    enc = make_encoder(X_tr)
    Xe_tr = timed(rows, b, "encoding", lambda: enc.fit_transform(X_tr))
    clf = make_classifier(d["exp"]["classifier"], d["seed"]).set_params(**params)
    if method in RESAMPLING:
        Xr, yr = (timed(rows, b, "resampling", lambda: make_resampler(RESAMPLING[method], d["seed"])
                        .fit_resample(Xe_tr, y_tr), resampler=RESAMPLING[method])
                  if RESAMPLING[method] else (Xe_tr, y_tr))
        timed(rows, b, "classifier_fit", lambda: clf.fit(Xr, yr), n_fit_rows=len(yr))

        def infer():
            Xt = enc.transform(X_te)
            return clf.predict(Xt), clf.predict_proba(Xt)[:, 1]
        pred, score = timed(rows, b, "inference", infer)
    else:
        attr = spec["mitigate"]
        a_tr, a_te = d["groups"][attr].to_numpy()[tr], d["groups"][attr].to_numpy()[te]
        Xe_te = enc.transform(X_te)
        out_names = list(enc.get_feature_names_out())
        numeric_pos = [out_names.index(c) for c in numeric_columns(X_tr)]
        # fit and inference together: the fairness methods' fit and predict are one call in experiments/baselines.py
        pred, score, _ = timed(rows, b, "fit_and_inference", lambda: _fairness_fit_predict(
            method, Xe_tr, y_tr, a_tr, Xe_te, a_te, clf, numeric_pos, d["seed"], bcfg))
    timed(rows, b, "fairness_evaluation",
          lambda: (extended_metrics(y_te, pred, score, g_te), group_rate_rows(y_te, pred, g_te)))
    return rows


def task_generator(spec, cache_dir):
    from data import loaders
    from experiments.synthetic import make_generator
    d = fold_data(spec["dataset"], spec["repeat"], spec["fold"], spec.get("frac", 1.0), spec.get("sub_seed", 0))
    complete = ~d["X"].isna().any(axis=1).to_numpy()
    tr = d["tr"][complete[d["tr"]]]
    target = loaders.INFO[spec["dataset"]]["target"]
    train = d["X"].iloc[tr].assign(**{target: d["y"][tr]}).reset_index(drop=True)
    syn_cfg = yaml.safe_load((REPO_ROOT / "configs" / "synthetic.yaml").read_text())
    cond = next(e for e in syn_cfg["synthetic_experiments"] if e["dataset"] == spec["dataset"])["condition_on"]
    rows = []
    b = _base(spec, d, method=spec["generator"])
    b["n_train_rows"] = len(tr)
    gen = make_generator(spec["generator"], target, cond, fit_seed=1)
    timed(rows, b, "generation_fit", lambda: gen.fit(train))
    timed(rows, b, "generation_sample", lambda: gen.sample(len(train), 2), n_sampled=len(train))
    return rows


def task_detection(spec, cache_dir):
    import torch
    from data import loaders
    from experiments.detector import BertDetector, keyword_detect
    torch.set_num_threads(1)
    desc = {"adult": loaders.ADULT_DESCRIPTIONS, "bank": loaders.BANK_DESCRIPTIONS,
            "german": loaders.GERMAN_DESCRIPTIONS}[spec["dataset"]]
    rows = []
    b = {**_base(spec, None, method="bert"), "n_columns": len(desc)}
    det = BertDetector(seed=spec["head_seed"], threads=1)

    def load():
        model = det._load()
        model.load_state_dict(torch.load(REPO_ROOT / "results/raw/detector/heads" /
                                         f"bert_head_seed{spec['head_seed']}.pt"), strict=False)
        det.model_ = model.eval()
    timed(rows, b, "detection_model_load", load)
    timed(rows, b, "detection_bert", lambda: det.predict_proba(list(desc.values())))
    timed(rows, {**b, "method": "keyword"}, "detection_keyword", lambda: [keyword_detect(c) for c in desc])
    return rows


def _ts_setup(spec):
    from experiments.ts_cleaning import remove_period
    from experiments.ts_protocol import fold_periods, load_series, scenarios
    exp = yaml.safe_load((REPO_ROOT / "configs" / "timeseries.yaml").read_text())["timeseries_experiments"][0]
    df = load_series()
    train_raw, val_raw, test_raw = fold_periods(df, exp["folds"][spec["fold"]])
    scen = {s: (m, g) for s, m, g in scenarios(exp)}
    method, with_gap = scen[spec["scenario"]]
    gap = exp["artificial_gap"]
    train_in = remove_period(train_raw, gap["start"], gap["end"])[0] if with_gap else train_raw
    return exp, train_raw, val_raw, test_raw, train_in, method


def task_ts_cleaning(spec, cache_dir):
    from experiments.ts_cleaning import fill_training_period
    exp, train_raw, _, _, train_in, method = _ts_setup(spec)
    rows = []
    b = {**_base(spec, None, method=method, scenario=spec["scenario"]), "dataset": "air_quality",
         "n_train_rows": len(train_raw), "cells_filled": int(train_in.isna().sum().sum())}
    timed(rows, b, "cleaning",
          lambda: fill_training_period(train_in, method, exp.get("method_params", {}).get(method)))
    return rows


def task_ts_lstm(spec, cache_dir):
    """One LSTM fit (the configuration chosen in this fold and scenario, one seed) and its test forecasts."""
    import shutil
    import tempfile
    import torch
    from experiments.ts_cleaning import fill_training_period
    from experiments.ts_protocol import _darts, causal_fill, fit_lstm, forecast
    torch.set_num_threads(1)
    TimeSeries, Scaler, _, _ = _darts()
    exp, train_raw, val_raw, test_raw, train_in, method = _ts_setup(spec)
    space = yaml.safe_load((REPO_ROOT / "configs" / "search_space_timeseries.yaml").read_text())
    train_cfg = {**space["lstm"]["training"], "torch_threads": 1}
    sel = pd.read_csv(REPO_ROOT / "results/raw/timeseries/air_quality_cleaning/lstm_selection.csv")
    sel = sel[(sel["fold"] == spec["fold"]) & (sel["scenario"] == spec["scenario"])]
    best = sel.loc[sel["val_mae_scaled"].idxmin()]
    params = {"lr": float(best["lr"]), "input_chunk_length": int(best["input_chunk_length"])}
    filled, _, _ = fill_training_period(train_in, method, exp.get("method_params", {}).get(method))
    full = causal_fill(filled, val_raw, test_raw)
    cols = list(full.columns)
    series = TimeSeries.from_dataframe(full.reset_index(), time_col="datetime", value_cols=cols, freq="h")
    scaler = Scaler()
    train_s = scaler.fit_transform(series.slice(train_raw.index[0], train_raw.index[-1]))
    series_s = scaler.transform(series)
    val_s = series_s.slice(val_raw.index[0], val_raw.index[-1])
    rows = []
    b = {**_base(spec, None, method="lstm", scenario=spec["scenario"], config=str(params)),
         "dataset": "air_quality", "n_train_rows": len(train_raw)}
    work = Path(tempfile.mkdtemp(prefix="lstm_cost_"))
    try:
        model, info = timed(rows, b, "forecaster_fit", lambda: fit_lstm(train_s, val_s, params, train_cfg,
                                                                       spec["seed"], work))
        rows[-1]["epochs"] = info["epochs"]
        timed(rows, b, "inference", lambda: forecast(model, series_s, test_raw.index[0], 1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return rows


TASKS = {"dbim_fit": task_dbim_fit, "resampler": task_resampler, "search": task_search,
         "baseline_fit": task_baseline_fit, "generator": task_generator, "detection": task_detection,
         "ts_cleaning": task_ts_cleaning, "ts_lstm": task_ts_lstm}


def run_task(spec, cache_dir, cpu, conn):
    """Entry point of a child process; sends ('ok', rows) or ('error', message)."""
    try:
        setup_process(cpu)
        rows = TASKS[spec["task"]](spec, cache_dir)
        for r in rows:
            r.setdefault("status", "ok")
            r["cpu"] = cpu
        conn.send(("ok", rows))
    except Exception as e:  # reported by the driver; the run continues with the other tasks
        import traceback
        conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"))
    finally:
        conn.close()
