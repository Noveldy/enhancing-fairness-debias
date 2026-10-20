"""Time-series cleaning (experiments/ts_cleaning.py) and the parts of the protocol that need no LSTM."""
import numpy as np
import pandas as pd
import pytest

from experiments.ts_cleaning import ContextGapFiller, InterpolationFiller, MeanFiller, find_gaps, remove_period
from experiments import ts_protocol as tp


def hourly(values, start="2004-01-01"):
    return pd.DataFrame(values, index=pd.date_range(start, periods=len(values), freq="h"), columns=["a", "b"])


def paper_example():
    """Paper §2.5.1: 20 records t0..t19, t5-t7 missing, t11 ~ t4 and t15 ~ t8 -> fill from t12-t14."""
    v = np.ones((20, 2))
    v[4] = [1.0, 0.0]
    v[11] = [1.0, 0.01]
    v[8] = [0.0, 1.0]
    v[15] = [0.01, 1.0]
    v[12:15] = [[12, 112], [13, 113], [14, 114]]
    v[5:8] = np.nan
    return hourly(v)


def test_find_gaps():
    assert find_gaps([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert find_gaps([0, 0]) == []


def test_paper_example_is_filled_from_t12_to_t14():
    df = paper_example()
    filler = ContextGapFiller().fit(df)
    out = filler.transform(df)
    np.testing.assert_array_equal(out.iloc[5:8].to_numpy(), [[12, 112], [13, 113], [14, 114]])
    log = filler.donor_log_
    assert set(log["donor_start"]) == {df.index[12]} and (log["route"] == "donor window").all()


def test_donors_are_original_training_observations_only():
    rng = np.random.default_rng(0)
    df = hourly(rng.random((300, 2)) + 1)
    df.iloc[50:60, 0] = np.nan
    df.iloc[120:125, :] = np.nan
    out = ContextGapFiller().fit(df).transform(df)
    observed = set(np.round(df["a"].dropna(), 12))
    assert set(np.round(out["a"].iloc[50:60], 12)) <= observed
    assert not out.isna().any().any()
    with pytest.raises(ValueError, match="period it was fitted on"):
        ContextGapFiller().fit(df).transform(df.iloc[:-1])


def test_long_gap_without_complete_donor_uses_passes_then_fallback():
    rng = np.random.default_rng(1)
    v = rng.random((200, 2)) + 1
    df = hourly(v)
    df.iloc[::7, 0] = np.nan            # no fully observed window of length 30 in column a
    df.iloc[100:130, :] = np.nan
    filler = ContextGapFiller(max_passes=3).fit(df)
    out = filler.transform(df)
    assert not out.isna().any().any()
    routes = set(filler.donor_log_.query("column == 'a'")["route"])
    assert "donor window, partly missing" in routes


def test_donor_window_does_not_touch_the_gap():
    df = paper_example()
    filler = ContextGapFiller().fit(df)
    filler.transform(df)
    for _, r in filler.donor_log_.iterrows():
        start = df.index.get_loc(r["donor_start"])
        assert start + r["gap_length"] < 5 or start - 1 > 7


def test_mean_and_interpolation_fillers():
    df = hourly(np.arange(20, dtype=float).repeat(2).reshape(20, 2))
    gappy = df.copy()
    gappy.iloc[3:6, 0] = np.nan
    assert MeanFiller().fit(gappy).transform(gappy).iloc[4, 0] == pytest.approx(gappy["a"].mean())
    lin = InterpolationFiller("linear").fit(gappy).transform(gappy)
    np.testing.assert_allclose(lin["a"], df["a"])


def test_remove_period_and_causal_fill():
    df = hourly(np.arange(40, dtype=float).repeat(2).reshape(40, 2))
    gapped, mask = remove_period(df, df.index[10], df.index[15])
    assert mask.sum() == 5 and gapped.iloc[10:15].isna().all().all()
    train, test = df.iloc[:20], df.iloc[20:].copy()
    test.iloc[3:6] = np.nan
    full = tp.causal_fill(train, test)
    assert (full.iloc[23:26]["a"] == df.iloc[22]["a"]).all()  # last observed value, never a later one


def test_forecast_metrics_use_observed_targets_only():
    idx = pd.date_range("2005-01-01", periods=4, freq="h")
    t = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]}, index=idx)
    p = pd.DataFrame({"a": [2.0, 100.0, 3.0, 4.0]}, index=idx)
    per, macro = tp.forecast_metrics(t, p, pd.Series({"a": 0.0}), pd.Series({"a": 10.0}))
    assert per.loc["a", "n"] == 3 and per.loc["a", "mae"] == pytest.approx(1 / 3)
    assert macro["mae_scaled"] == pytest.approx(1 / 30)
    assert macro["smape"] == pytest.approx(100 * (2 / 3) / 3)


def test_air_quality_paper_gap_is_filled_inside_training_period():
    """Real data, fold 0 of configs/timeseries.yaml: fit + fill only, no LSTM."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/timeseries.yaml").read_text())
    exp = cfg["timeseries_experiments"][0]
    df = tp.load_series()
    train, val, test = tp.fold_periods(df, exp["folds"][0])
    assert train.index[-1] < val.index[0] and val.index[-1] < test.index[0]
    gapped, mask = remove_period(train, exp["artificial_gap"]["start"], exp["artificial_gap"]["end"])
    assert mask.sum() == 1296
    filler = ContextGapFiller(**exp["method_params"]["proposed"]).fit(gapped)
    out = filler.transform(gapped)
    assert not out.isna().any().any() and out.index.equals(train.index)
    for method in ["mean", "linear", "spline"]:
        filled, _, _ = tp.fill_training_period(gapped, method)
        assert not filled.isna().any().any()
