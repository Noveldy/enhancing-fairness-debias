"""Build the evaluated pipelines.

Every data-dependent step is a pipeline step, so ``fit`` on a training fold is
the only place anything is learned and ``predict`` on a test fold only applies it:

    SimilarityImputer -> ordinal encoding -> resampler -> classifier

The resampler is an imblearn step: it runs inside ``fit`` only, so test folds
are never resampled.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import (
    AllKNN,
    CondensedNearestNeighbour,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from experiments.imputation import SimilarityImputer


def make_resampler(name, seed):
    """Resamplers with library defaults, as in dataset_resampling.py:30-35."""
    if name == "SMOTE":
        return SMOTE(random_state=seed)
    if name == "ADASYN":
        return ADASYN(random_state=seed)
    if name == "BorderlineSMOTE":
        return BorderlineSMOTE(random_state=seed)
    if name == "KMeansSMOTE":
        return KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=seed)
    if name == "SVMSMOTE":
        return SVMSMOTE(random_state=seed)
    if name == "CondensedNearestNeighbour":
        return CondensedNearestNeighbour(random_state=seed)
    if name == "AllKNN":
        return AllKNN()
    if name == "NeighbourhoodCleaningRule":
        return NeighbourhoodCleaningRule()
    if name == "OneSidedSelection":
        return OneSidedSelection(random_state=seed)
    if name == "TomekLinks":
        return TomekLinks()
    # Plain resampling baselines (Phase 3), library defaults.
    if name == "RandomOverSampler":
        return RandomOverSampler(random_state=seed)
    if name == "RandomUnderSampler":
        return RandomUnderSampler(random_state=seed)
    raise ValueError(f"unknown resampler {name!r}")


def candidate_resamplers(space, n_train_rows):
    """Names of the resamplers the proposed method may choose, for this training set size."""
    rs = space["resamplers"]
    names = list(rs["oversampling"]) + list(rs["undersampling"])
    if n_train_rows >= rs["large_training_rows"]:
        names = [n for n in names if n not in rs["excluded_when_large"]]
    return names


def make_classifier(name, seed):
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    raise ValueError(f"unknown classifier {name!r}")


def make_encoder(X):
    """Ordinal codes for categoricals (categories learned on the training fold), numerics passed through.

    Output columns: categoricals first, then numerics (``numeric_columns(X)``).
    """
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        [
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
            ("numeric", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )


def numeric_columns(X):
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def make_pipeline(X, classifier, seed, impute):
    """Imputer (optional) -> ordinal encoder for categoricals -> resampler slot -> classifier."""
    encoder = make_encoder(X)
    steps = []
    if impute:
        steps.append(("impute", SimilarityImputer()))
    steps += [("encode", encoder), ("sampler", "passthrough"), ("clf", make_classifier(classifier, seed))]
    return Pipeline(steps)


def param_grid(space, classifier, resampler_names, seed):
    grid = {f"clf__{k}": list(v) for k, v in space["classifiers"][classifier].items()}
    grid["sampler"] = [make_resampler(n, seed) for n in resampler_names] if resampler_names else ["passthrough"]
    return grid
