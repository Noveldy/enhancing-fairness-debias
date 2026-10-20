"""End-to-end run of the nested protocol on the smoke configuration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import run_experiments
from experiments.protocol import imbalance_check, inner_splits, load_experiment_data, outer_splits

REPO = Path(__file__).resolve().parent.parent
SMOKE = REPO / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    run_experiments.main(["--config", str(SMOKE), "--out", str(out)])
    return out


def test_outputs_written(smoke_run):
    for exp in ("adult_dbim_smoke", "bank_dbim_smoke"):
        d = smoke_run / exp
        for f in ("predictions.csv.gz", "selection.csv", "inner_scores.csv.gz", "fold_metrics.csv"):
            assert (d / f).exists(), f
        record = json.loads((smoke_run / f"run_{exp}.json").read_text())
        assert record["seeds"] == [0]
        assert record["environment"]["packages"]["scikit-learn"]
        assert len(record["data"][record["experiment"]["dataset"]]["sha256"]) == 64


def test_every_row_tested_exactly_once_per_repeat_and_never_resampled(smoke_run):
    for exp in ("adult_dbim_smoke", "bank_dbim_smoke"):
        pred = pd.read_csv(smoke_run / exp / "predictions.csv.gz")
        sel = pd.read_csv(smoke_run / exp / "selection.csv")
        for _, rep in pred.groupby("repeat"):
            assert rep["row_id"].is_unique
            assert set(rep["row_id"]) == set(range(2000))
        # Test folds keep their original size whatever resampler was selected.
        sizes = pred.groupby(["repeat", "fold"]).size().to_numpy()
        np.testing.assert_array_equal(sizes, sel["n_test"].to_numpy())


def test_bank_smoke_respects_feature_choices(smoke_run):
    cfg = yaml.safe_load(SMOKE.read_text())
    bank = next(e for e in cfg["tabular_experiments"] if e["dataset"] == "bank")
    X, _, groups, _ = load_experiment_data(bank)
    assert "duration" not in X.columns         # F1 (a)
    assert "marital" not in X.columns          # protected_as_feature: false
    assert list(groups.columns) == ["marital"]  # still available for fairness metrics


def test_inner_folds_stay_inside_outer_training_set():
    strata = np.array(["0|a", "1|a", "0|b", "1|b"] * 50)
    for _, _, seed, tr, te in outer_splits(strata, n_repeats=2, n_splits=5, seeds=[0, 1]):
        assert not set(tr) & set(te)
        for itr, iva in inner_splits(strata[tr], 3, seed):
            # inner indices address positions within the outer training set only
            assert max(itr.max(), iva.max()) < len(tr)


def test_imbalance_check_uses_original_thresholds():
    y = np.array([0] * 70 + [1] * 30)
    groups = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50})
    triggered, found = imbalance_check(y, groups, 40)
    assert triggered and found == {"target": 30.0}


def test_resampling_false_runs_imputer_without_resampler(tmp_path):
    """Phase 4.3 'cleaning only' stage: same pipeline, empty resampler slot."""
    cfg = yaml.safe_load(SMOKE.read_text())
    exp = dict(next(e for e in cfg["tabular_experiments"] if e["dataset"] == "adult"),
               name="adult_impute_only_smoke", method="impute_only", resampling=False)
    cfg["tabular_experiments"] = [exp]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_experiments.main(["--config", str(path), "--out", str(tmp_path / "out")])
    sel = pd.read_csv(tmp_path / "out" / exp["name"] / "selection.csv")
    assert sel["resampling_triggered"].all()        # the data are imbalanced ...
    assert not sel["resampling_enabled"].any()      # ... but resampling is switched off
    assert (sel["selected_sampler"] == "none").all()
    inner = pd.read_csv(tmp_path / "out" / exp["name"] / "inner_scores.csv.gz")
    assert set(inner["sampler"]) == {"none"}
