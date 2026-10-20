"""Phase 3 baselines: AIF360 equivalence, train-only fitting, identical splits."""
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from data import loaders
from experiments.fair_baselines import TrainFittedDIR, reweighing_weights
from experiments.protocol import load_experiment_data, outer_splits

logging.getLogger().setLevel(logging.ERROR)


def _aif_dataset(X, a, y):
    from aif360.datasets import BinaryLabelDataset
    cols = [f"f{j}" for j in range(X.shape[1])]
    return BinaryLabelDataset(df=pd.DataFrame(X, columns=cols).assign(p=a, y=y), label_names=["y"],
                              protected_attribute_names=["p"]), cols


def _toy(seed, n_groups, n=1200):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_groups, n)
    X = np.c_[np.round(rng.normal(0, 1, n) + 0.5 * a, 2), rng.integers(0, 40, n) * (1 + a), rng.exponential(1, n)]
    y = (X[:, 0] + rng.normal(0, 1, n) > 0.5).astype(int)
    return X, a, y


def test_reweighing_equals_aif360():
    from aif360.algorithms.preprocessing import Reweighing
    X, a, y = _toy(0, 2)
    d, _ = _aif_dataset(X, a, y)
    w = Reweighing([{"p": 0}], [{"p": 1}]).fit(d).transform(d).instance_weights
    np.testing.assert_allclose(reweighing_weights(y, a), w)


@pytest.mark.parametrize("n_groups,level", [(2, 1.0), (3, 1.0), (5, 0.5)])
def test_dir_equals_aif360_on_training_rows(n_groups, level):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    X, a, y = _toy(n_groups, n_groups)
    d, cols = _aif_dataset(X, a, y)
    r = DisparateImpactRemover(repair_level=level, sensitive_attribute="p").fit_transform(d)
    aif = r.features[:, [r.feature_names.index(c) for c in cols]]
    np.testing.assert_array_equal(TrainFittedDIR([0, 1, 2], level).fit(X, a).transform(X, a), aif)


def test_dir_repairs_test_rows_with_training_distribution_only():
    X, a, y = _toy(1, 2)
    rep = TrainFittedDIR([0, 1, 2]).fit(X[:800], a[:800])
    one = rep.transform(X[800:801], a[800:801])
    batch = rep.transform(X[800:], a[800:])       # other test rows must not change row 800's repair
    np.testing.assert_array_equal(one[0], batch[0])
    out = rep.transform(X[800:], a[800:])
    for j in range(3):  # repaired test values are training values of that column
        assert set(np.unique(out[:, j])) <= set(np.unique(X[:800, j]))


@pytest.mark.parametrize("name", ["adult", "bank"])
def test_complete_rows_equal_the_baseline_variant(name):
    full = getattr(loaders, f"load_{name}")("full")
    base = getattr(loaders, f"load_{name}")("baseline")
    assert (~full.isna().any(axis=1)).sum() == len(base)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    import run_experiments
    out = tmp_path_factory.mktemp("bsmoke")
    run_experiments.main(["--config", "configs/baselines_smoke.yaml", "--out", str(out)])
    return out, yaml.safe_load(open("configs/baselines_smoke.yaml"))


def test_smoke_uses_dbim_splits_and_complete_rows(smoke):
    out, cfg = smoke
    for exp in cfg["baseline_experiments"]:
        pred = pd.read_csv(out / exp["name"] / "predictions.csv.gz")
        X, y, groups, strata = load_experiment_data(exp)
        complete = ~X.isna().any(axis=1).to_numpy()
        expected_methods = set(exp["methods"])
        assert set(pred["method"]) == expected_methods
        for r, f, seed, tr, te in outer_splits(strata, cfg["outer"]["n_repeats"], cfg["outer"]["n_splits"],
                                               cfg["seeds"]):
            for method, p in pred[(pred.repeat == r) & (pred.fold == f)].groupby(["method", "mitigated"]):
                ids = p["row_id"].to_numpy()
                assert set(ids) == set(te[complete[te]]), method   # same test rows as DBIM, complete only
                assert not set(ids) & set(tr)
                np.testing.assert_array_equal(p.sort_values("row_id")["y_true"], y[np.sort(ids)])
        sel = pd.read_csv(out / exp["name"] / "selection.csv")
        eg = sel[sel.method == "exp_gradient"]
        none = sel[sel.method == "none"].set_index(["repeat", "fold"])["clf_params"]
        assert (eg.set_index(["repeat", "fold"])["clf_params"] == none.loc[eg.set_index(["repeat", "fold"]).index]).all()


def test_smoke_leaves_no_fold_cache(smoke):
    out, cfg = smoke
    for exp in cfg["baseline_experiments"]:
        assert not (out / exp["name"] / "partial").exists()


def test_saved_fold_is_reused(tmp_path, monkeypatch):
    from experiments import baselines
    path = tmp_path / "main_r0_f0.pkl"
    calls = []
    monkeypatch.setattr(baselines, "run_fold", lambda *a: calls.append(1) or ("result",))
    assert baselines._checkpointed(path, "main", None, None, None, None, (0, 0, 0, [], [])) == ("result",)
    assert baselines._checkpointed(path, "main", None, None, None, None, (0, 0, 0, [], [])) == ("result",)
    assert calls == [1]


def test_clip_score_records_rounding_excess():
    from experiments.baselines import _clip_score
    info = {}
    out = _clip_score(np.array([0.2, 1.0000000000000013]), info)
    assert out.max() == 1.0 and 0 < info["score_clip_excess"] < 1e-12
