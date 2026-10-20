"""The proposed similarity-based imputation (paper §2.5.1, Fig. 3) as a train-fold transformer.

Port of ``legacy/original/bias_identification/datasets.py:similarity_based_imputation``
with the fit/transform split the original lacked (docs/AUDIT.md §2, item 1):

* ``fit`` sees only the training rows. Label maps, column means, the
  nearest-neighbour index and the donor values all come from them.
* ``transform`` fills a missing cell with the value of the most similar
  *training* row that has it, so test rows never donate to each other.
* The target is never passed in (the pipeline only gives it X), so it can no
  longer drive the neighbour search.

Unchanged from the original: label encoding by order of appearance, unscaled
features, temporary mean-filling for the neighbour search, cosine distance
with brute-force search, 10 neighbours with the row itself skipped, and the
optional z-score outlier masking.

One bug fix: when no neighbour has a value, the original fell back to the
column mean even for categorical columns. The mean of label codes is not a
valid code, so the reverse mapping produced NaN. Categorical columns now fall
back to the most frequent training category.

Memory: the neighbour search computes distances in blocks of at most
``working_memory`` MB (scikit-learn's default is 1024 MB). The first full run
was killed by the OOM killer because 18 parallel workers each held a ~1 GB
block at the same time. A smaller block gives identical neighbours (each row's
distances are computed on their own); checked on full Adult outer folds and in
tests/test_imputation.py.
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


class SimilarityImputer(TransformerMixin, BaseEstimator):
    def __init__(self, n_neighbors=10, detect_outliers=False, z_thresh=3.0, working_memory=128):
        self.n_neighbors = n_neighbors
        self.detect_outliers = detect_outliers
        self.z_thresh = z_thresh
        self.working_memory = working_memory

    # ------------------------------------------------------------------ helpers
    def _encode(self, X):
        """Label-encode categoricals with the training maps (unseen -> NaN)."""
        enc = pd.DataFrame(index=X.index)
        for col in self.columns_:
            if col in self.label_maps_:
                enc[col] = X[col].map(self.label_maps_[col]).astype(float)
            else:
                enc[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        return enc

    def _mask_outliers(self, X, enc):
        """Mark numeric values with |z| > z_thresh (training mean/std) as missing."""
        if not self.detect_outliers:
            return X
        X = X.copy()
        for col in self.numeric_cols_:
            std = self.num_std_[col]
            if std > 0:
                z = (enc[col] - self.num_mean_[col]) / std
                X.loc[z.abs() > self.z_thresh, col] = np.nan
        return X

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _fit(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SimilarityImputer expects a pandas DataFrame")
        self.columns_ = list(X.columns)
        self.numeric_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]
        # Label maps in order of appearance, exactly as in the original: it enumerated
        # unique() *including* NaN, so a NaN that appears early takes a code and shifts
        # the codes of later categories. The codes feed the cosine distances, so keep that.
        self.label_maps_ = {
            col: {v: i for i, v in enumerate(pd.unique(X[col])) if not pd.isna(v)}
            for col in self.categorical_cols_
        }
        self.inverse_maps_ = {col: {i: v for v, i in m.items()} for col, m in self.label_maps_.items()}
        enc = self._encode(X)
        # Outlier statistics (scipy.stats.zscore uses ddof=0).
        self.num_mean_ = enc[self.numeric_cols_].mean()
        self.num_std_ = enc[self.numeric_cols_].std(ddof=0)
        X = self._mask_outliers(X, enc)
        enc = self._encode(X)
        self.col_means_ = enc.mean()
        self.donors_ = enc.to_numpy()
        filled = enc.fillna(self.col_means_).to_numpy()
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine", algorithm="brute").fit(filled)
        self.fallback_ = {}
        for col in self.columns_:
            if col in self.categorical_cols_:
                mode = X[col].mode(dropna=True)
                self.fallback_[col] = mode.iloc[0] if len(mode) else np.nan
            else:
                self.fallback_[col] = self.col_means_[col]
        self.n_features_in_ = len(self.columns_)
        self.feature_names_in_ = np.array(self.columns_, dtype=object)
        return X, enc

    # ------------------------------------------------------------------ transform
    def fit_transform(self, X, y=None):
        X_masked, enc = self._fit(X)
        return self._impute(X_masked, enc, skip_self=True)

    def transform(self, X):
        X = X[self.columns_]
        enc = self._encode(X)
        X = self._mask_outliers(X, enc)
        enc = self._encode(X)
        return self._impute(X, enc, skip_self=False)

    def _impute(self, X, enc, skip_self):
        out = X.copy()
        missing = X.isna().to_numpy()
        rows = np.flatnonzero(missing.any(axis=1))
        if len(rows) == 0:
            return out
        query = enc.fillna(self.col_means_).to_numpy()[rows]
        # The original queried 10 neighbours and skipped the row itself, so at most
        # 9 other rows donate. New rows are not in the index, so ask for 9.
        k = self.n_neighbors if skip_self else self.n_neighbors - 1
        with sklearn.config_context(working_memory=self.working_memory):
            _, neighbors = self.nn_.kneighbors(query, n_neighbors=k)
        col_pos = {c: j for j, c in enumerate(self.columns_)}
        for r, row in enumerate(rows):
            for col in X.columns[missing[row]]:
                j = col_pos[col]
                value = None
                for donor in neighbors[r]:
                    if skip_self and donor == row:
                        continue
                    v = self.donors_[donor, j]
                    if not np.isnan(v):
                        value = self.inverse_maps_[col][int(v)] if col in self.inverse_maps_ else v
                        break
                if value is None:
                    value = self.fallback_[col]
                out.iat[row, j] = value
        return out
