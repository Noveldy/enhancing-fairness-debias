"""The train-fold SimilarityImputer must match the original algorithm and must not leak."""
import numpy as np
import pandas as pd
import pytest

from data import loaders
from experiments.imputation import SimilarityImputer
from experiments.legacy_protocol import legacy_similarity_imputation


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    # The original function appends to results/logs.txt in the working directory.
    (tmp_path / "results").mkdir()
    monkeypatch.chdir(tmp_path)


def toy_frame():
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0, 11.0, 12.0],
        "cat": ["a", np.nan, "b", "a", "c", "b", "a", "c", np.nan, "b", "a", "c"],
        "num2": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
    })


@pytest.mark.parametrize("detect_outliers", [False, True])
def test_fit_transform_matches_original_on_toy_data(in_tmp, detect_outliers):
    df = toy_frame()
    expected = legacy_similarity_imputation()(df.copy(), detect_outliers=detect_outliers, z_thresh=1.5)
    got = SimilarityImputer(detect_outliers=detect_outliers, z_thresh=1.5).fit_transform(df.copy())
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)


def test_fit_transform_matches_original_on_adult_sample(in_tmp):
    X = loaders.load_adult("full").drop(columns="income").sample(3000, random_state=0).reset_index(drop=True)
    expected = legacy_similarity_imputation()(X.copy())
    got = SimilarityImputer().fit_transform(X.copy())
    # The only allowed difference is the categorical fallback bug fix.
    diff = (got.astype(str) != expected.astype(str))
    assert not diff[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]].any().any()
    assert int(diff.to_numpy().sum()) == int(expected.isna().sum().sum())  # original fallback produced NaN
    assert not got.isna().any().any()


def test_test_rows_only_borrow_from_training_rows():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    test = pd.DataFrame({"a": [2.1, 2.2], "b": [np.nan, np.nan]})
    imp = SimilarityImputer(n_neighbors=3).fit(train)
    out = imp.transform(test)
    assert set(out["b"]) <= set(train["b"])
    # Transforming the test rows must not change the fitted state.
    before = imp.donors_.copy()
    imp.transform(test)
    np.testing.assert_array_equal(before, imp.donors_)


def test_target_not_used():
    X = toy_frame()
    y1 = pd.Series(np.zeros(len(X)))
    y2 = pd.Series(np.arange(len(X)))
    a = SimilarityImputer().fit_transform(X.copy(), y1)
    b = SimilarityImputer().fit_transform(X.copy(), y2)
    pd.testing.assert_frame_equal(a, b)


def test_unseen_category_is_kept_not_imputed():
    train = toy_frame().dropna()
    test = pd.DataFrame({"num": [np.nan], "cat": ["zzz"], "num2": [3.0]})
    out = SimilarityImputer(n_neighbors=3).fit(train).transform(test)
    assert out.loc[0, "cat"] == "zzz"
    assert not np.isnan(out.loc[0, "num"])


def test_working_memory_does_not_change_output():
    """The block size of the neighbour search is a memory setting only."""
    import numpy as np
    import pandas as pd
    from experiments.imputation import SimilarityImputer
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000),
                      "c": rng.choice(["x", "y", "z"], size=3000)})
    X = X.mask(rng.random(X.shape) < 0.1)
    outs = [SimilarityImputer(working_memory=wm).fit_transform(X) for wm in (1024, 8, 1)]
    assert outs[0].equals(outs[1]) and outs[0].equals(outs[2])
