"""The ORIGINAL (leaky) evaluation order, kept only to show that the new protocol differs.

Original order (legacy/original/main.py:29-71, docs/AUDIT.md §2):
    1. similarity imputation on all rows, target included
    2. resampling of all rows
    3. train/test split of the resampled data, then train and evaluate

Never used for reported results; only by tests/test_leakage.py and
scripts/leakage_sanity_check.py.
"""
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors

LEGACY_DATASETS_PY = Path(__file__).resolve().parent.parent / "legacy/original/bias_identification/datasets.py"


def legacy_similarity_imputation():
    """Compile the original ``similarity_based_imputation`` from legacy/.

    Its module loads a transformer model at import time, so only this function's
    source is compiled. It appends to ``results/logs.txt`` in the working directory.
    """
    tree = ast.parse(LEGACY_DATASETS_PY.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "similarity_based_imputation")
    namespace = {"np": np, "pd": pd, "Path": Path, "logging": logging, "zscore": zscore,
                 "norm": norm, "NearestNeighbors": NearestNeighbors}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(LEGACY_DATASETS_PY), "exec"), namespace)
    return namespace["similarity_based_imputation"]


def label_encode_all(df):
    """Original model_trainer.py:88-98: label codes by order of appearance, fitted on all rows."""
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            codes = {v: i for i, v in enumerate(pd.unique(df[col])) if not pd.isna(v)}
            df[col] = df[col].map(codes).astype(float)
    return df
