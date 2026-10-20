"""Leakage-free evaluation of the time-series cleaning methods (paper §3.2.5, Table 9).

What changed from the original experiment (docs/AUDIT.md §5.3, §6):

* -200 is NaN (loader), so the "no missing values" baseline no longer trains on
  14,290 cells of -200.
* Chronological folds (expanding window): train | validation | test. Each
  cleaning method is fitted on the training period only.
* The LSTM is early-stopped on the validation window, and its learning rate and
  input length are chosen there (revision Phase 2.3; R1.7). The test window is
  used once, for the reported scores. The original early-stopped and scored on
  the same split, and scored only 12 steps.
* Validation and test inputs that are missing are forward-filled from the past
  only, identically in every scenario, so scenarios differ only in how the
  *training* period was cleaned. Scores use only timestamps whose true value
  was observed.
* MAE is reported in original units and on a common [0, 1] scale (min-max of
  the observed training values, the same for every method), so methods are
  comparable. SMAPE is computed in original units. The paper scaled each
  scenario with its own scaler, so its numbers are not comparable across rows.
* New: reconstruction error of each method on the artificial gap, measured
  against the observed true values the gap removed.
* Several seeds per configuration (R1.2).

Scenarios per fold: ``reference`` (no artificial gap; natural gaps filled by
``reference_fill``), then for each method the artificial gap is removed and
all training gaps (natural + artificial) are filled by that method.
"""
import hashlib
import itertools
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import loaders
from experiments.memory_guard import GB, MemoryWatchdog
from experiments.protocol import memory_settings
from experiments.ts_cleaning import fill_training_period, remove_period

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- data and splits
def load_series():
    return loaders.load_air_quality().set_index("datetime").asfreq("h")


def fold_periods(df, fold):
    """Slices of the series for one fold (dates inclusive, as written in the config)."""
    t = lambda k: pd.Timestamp(fold[k])
    train = df.loc[:t("train_end")]
    val = df.loc[t("train_end") + pd.Timedelta("1h"):t("val_end")]
    test = df.loc[t("val_end") + pd.Timedelta("1h"):t("test_end")]
    return train, val, test


def causal_fill(filled_train, *later):
    """Forward-fill later periods from the past only (identical for every scenario)."""
    full = pd.concat([filled_train, *later])
    return full.ffill()


def common_scale(train_raw):
    """Min-max of the observed training values: one scale for every method and scenario."""
    lo, hi = train_raw.min(), train_raw.max()
    return lo, (hi - lo).replace(0, 1.0)


# ---------------------------------------------------------------------- metrics
def forecast_metrics(y_true, y_pred, lo, span):
    """Per-column and macro MAE (original units and common scale), RMSE and SMAPE on observed targets."""
    rows = {}
    for col in y_true.columns:
        t, p = y_true[col], y_pred[col]
        ok = t.notna() & p.notna()
        t, p = t[ok].to_numpy(), p[ok].to_numpy()
        err = np.abs(t - p)
        denom = np.abs(t) + np.abs(p)
        smape = 100 * np.mean(np.where(denom > 0, 2 * err / np.where(denom > 0, denom, 1), 0.0)) if len(t) else np.nan
        rows[col] = {"n": len(t), "mae": err.mean() if len(t) else np.nan,
                     "mae_scaled": (err / span[col]).mean() if len(t) else np.nan,
                     "rmse": np.sqrt((err ** 2).mean()) if len(t) else np.nan, "smape": smape}
    per = pd.DataFrame(rows).T
    macro = per[["mae_scaled", "smape"]].mean().to_dict()
    return per, macro


def reconstruction_metrics(filled, truth, gap_mask, lo, span):
    """Error of the filled values on artificial-gap cells whose true value was observed."""
    sel = pd.Series(gap_mask, index=truth.index)
    per, macro = forecast_metrics(truth[sel.to_numpy()], filled[sel.to_numpy()], lo, span)
    return per, macro


# ---------------------------------------------------------------------- LSTM (darts, as in the original)
def _darts():
    try:
        from darts import TimeSeries
        from darts.dataprocessing.transformers import Scaler
        from darts.models import RNNModel
        from pytorch_lightning.callbacks import EarlyStopping
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("The LSTM part needs darts, torch and pytorch-lightning "
                          "(requirements-timeseries.txt; not installed in venv/ yet).") from e
    return TimeSeries, Scaler, RNNModel, EarlyStopping


def fit_lstm(train_s, val_s, params, train_cfg, seed, work_dir):
    """Fit an LSTM with early stopping on the validation window; returns the best-epoch model."""
    _, _, RNNModel, EarlyStopping = _darts()
    import warnings
    import torch
    torch.set_num_threads(train_cfg["torch_threads"])
    warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but no accelerator.*")
    name = f"lstm_{seed}_" + hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    stopper = EarlyStopping(monitor="val_loss", patience=train_cfg["patience"],
                            min_delta=train_cfg["min_delta"], mode="min")
    model = RNNModel(
        model="LSTM", input_chunk_length=params["input_chunk_length"],
        training_length=max(train_cfg["training_length"], params["input_chunk_length"]),
        hidden_dim=train_cfg["hidden_dim"], n_epochs=train_cfg["max_epochs"], batch_size=train_cfg["batch_size"],
        optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": params["lr"]},
        random_state=seed, model_name=name, work_dir=str(work_dir), save_checkpoints=True,
        force_reset=True, log_tensorboard=False,
        pl_trainer_kwargs={"callbacks": [stopper], "accelerator": "cpu", "deterministic": True,
                           "enable_progress_bar": False, "logger": False},
    )
    t0 = time.perf_counter()
    model.fit(train_s, val_series=val_s, verbose=False)
    epochs = model.epochs_trained
    best = RNNModel.load_from_checkpoint(name, work_dir=str(work_dir), best=True)
    return best, {"fit_seconds": time.perf_counter() - t0, "epochs": epochs}


def forecast(model, series_s, start, horizon):
    """Forecasts over [start, end of series] at a given horizon, with the model never retrained."""
    fc = model.historical_forecasts(series_s, start=start, forecast_horizon=horizon, stride=horizon,
                                    retrain=False, last_points_only=False, verbose=False)
    return fc


def _to_frame(fcs, scaler, columns):
    parts = [scaler.inverse_transform(f).to_dataframe() if hasattr(f, "to_dataframe") else
             scaler.inverse_transform(f).pd_dataframe() for f in fcs]
    out = pd.concat(parts)
    out.columns = columns
    return out[~out.index.duplicated(keep="first")]


# ---------------------------------------------------------------------- main loop
def scenarios(exp):
    out = [("reference", exp["reference_fill"], False)]
    out += [(m, m, True) for m in exp["methods"]]
    return out


def run_timeseries_experiment(exp, cfg, space, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    TimeSeries, Scaler, _, _ = _darts()
    df = load_series()
    gap_start, gap_end = exp["artificial_gap"]["start"], exp["artificial_gap"]["end"]
    grid = [dict(zip(space["lstm"]["grid"], v)) for v in itertools.product(*space["lstm"]["grid"].values())]
    train_cfg = space["lstm"]["training"]
    horizons = exp["horizons"]
    mem = memory_settings(cfg)
    peak = 0
    cleaning_rows, recon_rows, select_rows, metric_rows, forecasts = [], [], [], [], []
    t_all = time.perf_counter()

    for fold_id, fold in enumerate(exp["folds"]):
        train_raw, val_raw, test_raw = fold_periods(df, fold)
        if not (pd.Timestamp(gap_end) <= train_raw.index[-1]):
            raise ValueError(f"fold {fold_id}: the artificial gap must lie inside the training period")
        lo, span = common_scale(train_raw)
        for scen, method, with_gap in scenarios(exp):
            train_in, gap_mask = (remove_period(train_raw, gap_start, gap_end) if with_gap
                                  else (train_raw, np.zeros(len(train_raw), bool)))
            filled, seconds, filler = fill_training_period(train_in, method, exp.get("method_params", {}).get(method))
            cleaning_rows.append({"fold": fold_id, "scenario": scen, "method": method, "seconds": round(seconds, 4),
                                  "cells_filled": int(train_in.isna().sum().sum())})
            if hasattr(filler, "donor_log_"):
                filler.donor_log_.assign(fold=fold_id, scenario=scen).to_csv(
                    out_dir / f"donor_log_fold{fold_id}_{scen}.csv", index=False)
            if with_gap:
                per, macro = reconstruction_metrics(filled, train_raw, gap_mask, lo, span)
                for col, r in per.iterrows():
                    recon_rows.append({"fold": fold_id, "scenario": scen, "column": col, **r.to_dict()})

            full = causal_fill(filled, val_raw, test_raw)
            cols = list(full.columns)
            series = TimeSeries.from_dataframe(full.reset_index(), time_col="datetime", value_cols=cols, freq="h")
            scaler = Scaler()  # fitted on this scenario's filled training period, as in the original
            train_s = scaler.fit_transform(series.slice(train_raw.index[0], train_raw.index[-1]))
            series_s = scaler.transform(series)
            val_s = series_s.slice(val_raw.index[0], val_raw.index[-1])          # early stopping only
            val_series_s = series_s.slice(train_raw.index[0], val_raw.index[-1])  # inputs for validation forecasts
            work = Path(tempfile.mkdtemp(prefix="lstm_"))
            context = f"{exp['name']} fold {fold_id} {scen}"
            try:
                with MemoryWatchdog(mem["budget_gb"], mem["reserve_gb"], context) as guard:
                    # 1. Choose lr and input length on the validation window (one seed).
                    best = None
                    for params in grid:
                        guard.stage = f"select {params}"
                        model, info = fit_lstm(train_s, val_s, params, train_cfg, space["lstm"]["selection_seed"], work)
                        fc = _to_frame(forecast(model, val_series_s, val_raw.index[0], 1), scaler, cols)
                        _, macro = forecast_metrics(val_raw.reindex(fc.index), fc, lo, span)
                        select_rows.append({"fold": fold_id, "scenario": scen, **params, **info,
                                            "val_mae_scaled": macro["mae_scaled"], "val_smape": macro["smape"]})
                        if best is None or macro["mae_scaled"] < best[1]:
                            best = (params, macro["mae_scaled"])
                    # 2. Refit the chosen configuration with every evaluation seed; score the test window once.
                    for seed in exp["seeds"]:
                        guard.stage = f"evaluate seed {seed}"
                        model, info = fit_lstm(train_s, val_s, best[0], train_cfg, seed, work)
                        for h in horizons:
                            fc = _to_frame(forecast(model, series_s, test_raw.index[0], h), scaler, cols)
                            fc = fc.loc[:test_raw.index[-1]]
                            per, macro = forecast_metrics(test_raw.reindex(fc.index), fc, lo, span)
                            metric_rows.append({"fold": fold_id, "scenario": scen, "seed": seed, "horizon": h,
                                                **best[0], **info, **{f"test_{k}": v for k, v in macro.items()}})
                            long = pd.concat([fc.stack().rename("y_pred"),
                                              test_raw.reindex(fc.index).stack().rename("y_true")], axis=1)
                            long.index.names = ["datetime", "column"]
                            forecasts.append(long.reset_index().assign(fold=fold_id, scenario=scen, seed=seed,
                                                                       horizon=h))
                        log.info("%s seed %d: %s test MAE(scaled) %.4f (h=%d)", context, seed, best[0],
                                 metric_rows[-len(horizons)]["test_mae_scaled"], horizons[0])
                peak = max(peak, guard.peak)
            finally:
                shutil.rmtree(work, ignore_errors=True)

    pd.DataFrame(cleaning_rows).to_csv(out_dir / "cleaning.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(out_dir / "reconstruction.csv", index=False)
    pd.DataFrame(select_rows).to_csv(out_dir / "lstm_selection.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "test_metrics.csv", index=False)
    pd.concat(forecasts).to_csv(out_dir / "forecasts.csv.gz", index=False)
    return {"experiment": exp["name"], "wall_clock_seconds": round(time.perf_counter() - t_all, 1),
            "n_rows": len(df), "columns": list(df.columns), "memory_guard": mem,
            "peak_memory_gb": round(peak / GB, 2)}
