"""Phase 5 generators, fidelity and privacy metrics on small controlled data."""
import random

import numpy as np
import pandas as pd
import pytest

from experiments.synthetic import (EmpiricalMarginalSampler, GaussianCopulaSampler, MarginalSampler,
                                   association_matrices, column_fidelity, fidelity_block, privacy_block)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    a = rng.choice(["F", "M"], n, p=[0.35, 0.65])
    z = rng.standard_normal(n)
    x1 = np.round(40 + 10 * z).astype(int)                       # strongly tied to x2 and y
    x2 = np.round(5 * z + rng.standard_normal(n) * 2).astype(int)
    cat = np.where(z + rng.standard_normal(n) * 0.5 > 0, "hi", "lo")
    y = ((z + 0.8 * (a == "M") + rng.standard_normal(n) * 0.7) > 0.8).astype(int)
    w = rng.integers(10_000, 1_000_000, n)                        # high-cardinality, like Adult's fnlwgt
    return pd.DataFrame({"x1": x1, "x2": x2, "w": w, "cat": cat, "sex": a, "y": y})


def spearman(df, a, b):
    return df[[a, b]].corr(method="spearman").iloc[0, 1]


def test_copula_keeps_dependence_and_cells_independent_samplers_lose_it(data):
    cop = GaussianCopulaSampler(["y", "sex"]).fit(data).sample(len(data), seed=1)
    ind = EmpiricalMarginalSampler().fit(data).sample(len(data), seed=1)
    real = spearman(data, "x1", "x2")
    assert abs(spearman(cop, "x1", "x2") - real) < 0.05
    assert abs(spearman(ind, "x1", "x2")) < 0.05
    # P(y, sex) is reproduced exactly (counts per cell, up to rounding)
    pr = data.groupby(["y", "sex"]).size() / len(data)
    ps = cop.groupby(["y", "sex"]).size() / len(cop)
    assert (pr - ps).abs().max() < 1e-3
    # only observed values and categories are produced; dtypes are kept
    assert set(cop["x1"]) <= set(data["x1"]) and set(cop["cat"]) <= set(data["cat"])
    assert (cop.dtypes == data.dtypes).all()


def test_generators_are_deterministic_and_marginal_sampler_restores_global_rng(data):
    for gen in (MarginalSampler("y"), EmpiricalMarginalSampler(), GaussianCopulaSampler(["y", "sex"])):
        gen.fit(data)
        pd.testing.assert_frame_equal(gen.sample(500, 7), gen.sample(500, 7))
    random.seed(123)
    np.random.seed(123)
    expect = (random.random(), np.random.random())
    random.seed(123)
    np.random.seed(123)
    MarginalSampler("y").fit(data).sample(100, 5)
    assert (random.random(), np.random.random()) == expect


def test_marginal_sampler_schema_follows_the_original_generator(data):
    gen = MarginalSampler("y").fit(data)
    s = gen.schema_
    assert s["x1"]["distribution"] == "truncnorm" and s["x1"]["type"] == "int"
    assert s["y"]["type"] == "category" and abs(sum(s["y"]["weights"]) - 1) < 1e-9
    out = gen.sample(2000, 3)
    assert out["x1"].between(data["x1"].min(), data["x1"].max()).all()
    assert abs(out["y"].mean() - data["y"].mean()) < 0.03


def test_fidelity_is_zero_for_identical_data_and_detects_independence(data):
    cf = column_fidelity(data, data)
    assert cf["tv"].max() == 0 and np.nanmax(cf["ks"]) == 0
    _, Ar, As = association_matrices(data, data)
    np.testing.assert_allclose(Ar, As)
    ind = EmpiricalMarginalSampler().fit(data).sample(len(data), seed=2)
    fb, _ = fidelity_block(data, ind, seed=0)
    assert fb["pairwise_dependence_mae"] > 0.1 and fb["c2st_auc"] > 0.8
    assert fb["mean_tv"] < 0.05


def test_privacy_flags_a_copying_generator(data):
    train, holdout = data.iloc[:2000], data.iloc[2000:]
    copy = privacy_block(train, holdout, train, seed=0, n_query=500)
    # the training reference is subsampled to the holdout size (1000 of 2000 rows), so a copy finds
    # its exact source only when that row (or a duplicate) is in the subsample
    assert copy["exact_copies_train"] >= 0.45 and copy["exact_copies_holdout"] < 0.05
    # expected share: 0.5 (source row in the reference: closer) + 0.5 * 0.5 (coin flip) = 0.75
    assert copy["share_closer_to_train"] > 0.65 and copy["mia_auc"] > 0.9
    cop = GaussianCopulaSampler(["y", "sex"]).fit(train).sample(2000, 1)
    fresh = privacy_block(train, holdout, cop, seed=0, n_query=500)
    assert 0.35 < fresh["share_closer_to_train"] < 0.65 and fresh["mia_auc"] < 0.6
