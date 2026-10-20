"""Deliberately inject the original leakage and check that results change (revision Phase 2.5).

If fitting the resampler on all data before splitting gave the same numbers
as the new pipeline, this check would prove nothing. It doesn't: the leaky
protocol scores higher on held-out data, and its test sets have a different
composition. Margins were measured on five synthetic datasets before being
fixed here (leaky minus proper accuracy: +0.023 to +0.055; REVISION_LOG Phase 2).
"""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from experiments.pipeline import make_pipeline


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=3000, n_features=10, n_informative=4, weights=[0.85],
                               flip_y=0.05, class_sep=0.8, random_state=0)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(10)]), y


def proper_protocol(X, y, sampler):
    """New protocol: the resampler is a pipeline step, fitted on training folds only."""
    accs, bals, n_neg = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        pipe = make_pipeline(X.iloc[tr], "decision_tree", seed=0, impute=False).set_params(sampler=sampler)
        pred = pipe.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        accs.append(accuracy_score(y[te], pred))
        bals.append(balanced_accuracy_score(y[te], pred))
        n_neg.append(int((y[te] == 0).sum()))
    return np.mean(accs), np.mean(bals), sum(n_neg)


def leaky_protocol(X, y, sampler):
    """Original order: resample everything, then split."""
    Xr, yr = sampler.fit_resample(X, y)
    accs, bals, n_neg = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xr, yr):
        pred = DecisionTreeClassifier(random_state=0).fit(Xr.iloc[tr], yr[tr]).predict(Xr.iloc[te])
        accs.append(accuracy_score(yr[te], pred))
        bals.append(balanced_accuracy_score(yr[te], pred))
        n_neg.append(int((yr[te] == 0).sum()))
    return np.mean(accs), np.mean(bals), sum(n_neg)


def test_undersampling_leak_inflates_scores_and_changes_test_set(data):
    X, y = data
    p_acc, p_bal, p_neg = proper_protocol(X, y, AllKNN())
    l_acc, l_bal, l_neg = leaky_protocol(X, y, AllKNN())
    assert l_acc - p_acc > 0.015
    assert l_bal - p_bal > 0.02
    assert p_neg == int((y == 0).sum())   # proper: every original row is tested once
    assert l_neg < p_neg                  # leaky: "hard" majority rows were removed from the test sets


def test_oversampling_leak_inflates_scores(data):
    X, y = data
    p_acc, p_bal, _ = proper_protocol(X, y, SMOTE(random_state=0))
    l_acc, l_bal, _ = leaky_protocol(X, y, SMOTE(random_state=0))
    assert l_acc - p_acc > 0.015
    assert l_bal - p_bal > 0.05           # synthetic minority rows land in the test sets
