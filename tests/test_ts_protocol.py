"""End-to-end check of the Air Quality LSTM protocol on a tiny configuration (needs darts/torch)."""
import json
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("darts")


def test_timeseries_smoke_run(tmp_path):
    p = subprocess.run([sys.executable, "run_experiments.py", "--config", "configs/timeseries_smoke.yaml",
                        "--out", str(tmp_path)], capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    out = tmp_path / "air_quality_smoke"
    m = pd.read_csv(out / "test_metrics.csv")
    assert set(m["scenario"]) == {"reference", "linear", "proposed"} and set(m["horizon"]) == {1, 12}
    assert m["test_mae_scaled"].notna().all()
    rec = pd.read_csv(out / "reconstruction.csv")
    assert set(rec["scenario"]) == {"linear", "proposed"} and (rec["n"] > 0).all()
    f = pd.read_csv(out / "forecasts.csv.gz")
    test_start = pd.Timestamp("2005-03-21 00:00")
    assert (pd.to_datetime(f["datetime"]) >= test_start).all()   # only test-window forecasts are scored
    run = json.loads((tmp_path / "run_air_quality_smoke.json").read_text())
    assert run["environment"]["packages"]["darts"] == "0.46.1"
