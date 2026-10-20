"""Synthetic tabular data (revision Phase 5): generators, fidelity and privacy metrics.

Generators (all fitted on the complete rows of one outer training fold; every
column of the experiment's data, target included, is generated):

* ``MarginalSampler`` ("marginal_sampler"): the paper's FSDG code path,
  ``bias_mitigation.synthetic_data.AdvancedSyntheticDataGenerator``, unchanged.
  It samples every column independently from a schema. The paper's Adult/Bank
  schemas were not stored, so the schema is derived from the training fold
  (author decision K1): categoricals and the target -> "category" with the
  observed frequencies; numerics -> "int"/"truncnorm" with the training mean,
  SD, min and max. The original code draws from the global ``random`` and
  ``np.random`` states; they are seeded per call and restored afterwards.
* ``EmpiricalMarginalSampler`` ("independent_marginals"): every column drawn
  independently from its exact training distribution. It isolates what is lost
  by sampling columns independently, without the truncnorm approximation.
* ``GaussianCopulaSampler`` ("gaussian_copula"): own lightweight implementation
  (author decision D3, scipy/numpy only). One copula per cell of the
  conditioning columns (target x one protected attribute), so P(target,
  attribute) is kept exactly and P(X | target, attribute) is modelled per cell.
  Inside a cell: each column is mapped to normal scores (numeric: ranks with
  random tie-breaking; categorical: the category's interval of the CDF, ordered
  by frequency, with a uniform draw inside it); the normal scores' correlation
  matrix is the copula; samples are mapped back through the exact empirical
  inverse CDF (``inverted_cdf``, only observed values) or the category intervals.

Metrics compare a synthetic sample with real held-out rows (the complete rows of
the outer test fold), so a generator that copies its training rows gains nothing.
"""
import random
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors

from bias_mitigation.synthetic_data.synthetic_data_generator import AdvancedSyntheticDataGenerator


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _cast_like(synth, real):
    """Give the synthetic frame the real frame's column order and dtypes."""
    out = synth[list(real.columns)].copy()
    for c in real.columns:
        if _is_numeric(real[c]):
            out[c] = pd.to_numeric(out[c]).astype(real[c].dtype)
        else:
            out[c] = out[c].astype(real[c].dtype)
    return out


# ---------------------------------------------------------------- generators

class MarginalSampler:
    name = "marginal_sampler"

    def __init__(self, target):
        self.target = target

    def fit(self, df):
        schema = {}
        for c in df.columns:
            s = df[c]
            if _is_numeric(s) and c != self.target:
                kind = "int" if (s % 1 == 0).all() else "float"
                if s.std() > 0:
                    schema[c] = {"type": kind, "distribution": "truncnorm", "mean": float(s.mean()),
                                 "std": float(s.std()), "min": float(s.min()), "max": float(s.max()), "clip": True}
                else:  # constant column: uniform on [v, v]
                    schema[c] = {"type": kind, "min": float(s.min()), "max": float(s.max())}
            else:
                freq = s.value_counts(normalize=True)
                schema[c] = {"type": "category", "categories": [v.item() if hasattr(v, "item") else v
                                                                for v in freq.index],
                             "weights": [float(w) for w in freq.to_numpy()]}
        self.schema_ = schema
        self.template_ = df.iloc[:0]
        return self

    def sample(self, n, seed):
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed % 2 ** 32)
            out = AdvancedSyntheticDataGenerator(self.schema_).generate(n_rows=n)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        return _cast_like(out, self.template_)


class EmpiricalMarginalSampler:
    name = "independent_marginals"

    def fit(self, df):
        self.df_ = df.reset_index(drop=True)
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({c: self.df_[c].to_numpy()[rng.integers(0, len(self.df_), n)]
                             for c in self.df_.columns}).pipe(_cast_like, self.df_)


class _CellCopula:
    """Gaussian copula with empirical margins for one conditioning cell."""

    def fit(self, df, rng):
        n = len(df)
        self.columns_ = list(df.columns)
        self.margins_ = {}
        Z = np.empty((n, len(self.columns_)))
        for j, c in enumerate(self.columns_):
            x = df[c].to_numpy()
            if _is_numeric(df[c]):
                order = np.lexsort((rng.random(n), x))       # ties broken at random
                ranks = np.empty(n)
                ranks[order] = np.arange(n)
                u = (ranks + 0.5) / n
                self.margins_[c] = ("num", np.sort(x))
            else:
                freq = df[c].value_counts()                   # most frequent first
                cats = freq.index.to_numpy()
                upper = np.cumsum(freq.to_numpy()) / n
                lower = np.concatenate([[0.0], upper[:-1]])
                pos = pd.Index(cats).get_indexer(x)
                u = lower[pos] + rng.random(n) * (upper[pos] - lower[pos])
                self.margins_[c] = ("cat", cats, upper)
            Z[:, j] = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
        R = np.corrcoef(Z, rowvar=False) if len(self.columns_) > 1 else np.ones((1, 1))
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh(R)                              # nearest positive-definite correlation
        R = (V * np.maximum(w, 1e-6)) @ V.T
        d = np.sqrt(np.diag(R))
        self.corr_ = R / np.outer(d, d)
        self.chol_ = np.linalg.cholesky(self.corr_)
        return self

    def sample(self, n, rng):
        U = stats.norm.cdf(rng.standard_normal((n, len(self.columns_))) @ self.chol_.T)
        out = {}
        for j, c in enumerate(self.columns_):
            m = self.margins_[c]
            if m[0] == "num":
                out[c] = np.quantile(m[1], U[:, j], method="inverted_cdf")
            else:
                cats, upper = m[1], m[2]
                out[c] = cats[np.minimum(np.searchsorted(upper, U[:, j], side="right"), len(cats) - 1)]
        return pd.DataFrame(out)


def _allocate(n, shares):
    """Integer counts summing to n, proportional to shares (largest remainder)."""
    raw = np.asarray(shares, float) * n
    k = np.floor(raw).astype(int)
    k[np.argsort(-(raw - k))[: n - k.sum()]] += 1
    return k


class GaussianCopulaSampler:
    name = "gaussian_copula"

    def __init__(self, condition_on, min_cell_rows=30, fit_seed=0):
        self.condition_on = list(condition_on)
        self.min_cell_rows = min_cell_rows
        self.fit_seed = fit_seed

    def fit(self, df):
        rng = np.random.default_rng(self.fit_seed)
        self.template_ = df.iloc[:0]
        self.cells_ = []
        others = [c for c in df.columns if c not in self.condition_on]
        for key, cell in df.groupby(self.condition_on, sort=True, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            if len(cell) < self.min_cell_rows:
                raise ValueError(f"cell {dict(zip(self.condition_on, key))} has only {len(cell)} training rows")
            self.cells_.append((key, len(cell) / len(df), _CellCopula().fit(cell[others], rng)))
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        parts = []
        for (key, _, cop), k in zip(self.cells_, _allocate(n, [s for _, s, _ in self.cells_])):
            if k == 0:
                continue
            part = cop.sample(k, rng)
            for c, v in zip(self.condition_on, key):
                part[c] = v
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)
        return _cast_like(out.iloc[rng.permutation(len(out))].reset_index(drop=True), self.template_)


def make_generator(name, target, condition_on, fit_seed):
    if name == "marginal_sampler":
        return MarginalSampler(target)
    if name == "independent_marginals":
        return EmpiricalMarginalSampler()
    if name == "gaussian_copula":
        return GaussianCopulaSampler([target, condition_on], fit_seed=fit_seed)
    raise ValueError(f"unknown generator {name!r}")


# ---------------------------------------------------------------- fidelity

def column_fidelity(real, synth, n_bins=20):
    """Per column: KS statistic (numeric), total-variation distance (categorical; numeric on
    real-quantile bins), and summary statistics."""
    rows = []
    for c in real.columns:
        r, s = real[c], synth[c]
        if _is_numeric(r):
            edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            hr = np.histogram(r, edges)[0] / len(r)
            hs = np.histogram(s, edges)[0] / len(s)
            rows.append({"column": c, "type": "numeric", "ks": stats.ks_2samp(r, s).statistic,
                         "tv": 0.5 * np.abs(hr - hs).sum(), "real_mean": r.mean(), "synth_mean": s.mean(),
                         "real_sd": r.std(), "synth_sd": s.std()})
        else:
            pr, ps = r.value_counts(normalize=True), s.value_counts(normalize=True)
            idx = pr.index.union(ps.index)
            rows.append({"column": c, "type": "categorical", "ks": np.nan,
                         "tv": 0.5 * (pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).abs().sum(),
                         "real_n_categories": int(r.nunique()), "synth_n_categories": int(s.nunique())})
    return pd.DataFrame(rows)


def _codes(real, synth, c):
    cats = pd.Index(pd.concat([real[c], synth[c]]).unique())
    return cats.get_indexer(real[c]), cats.get_indexer(synth[c]), len(cats)


def _cramers_v(a, b, ka, kb):
    t = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    t = t[t.sum(1) > 0][:, t.sum(0) > 0]
    if min(t.shape) < 2:
        return 0.0
    e = np.outer(t.sum(1), t.sum(0)) / t.sum()
    chi2 = ((t - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def _corr_ratio(codes, x, k):
    if np.var(x) == 0:
        return 0.0
    n = np.bincount(codes, minlength=k)
    means = np.bincount(codes, weights=x, minlength=k) / np.maximum(n, 1)
    return float(np.sqrt((n * (means - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum()))


def association_matrices(real, synth):
    """Mixed association matrices in [0, 1]: |Spearman| (numeric-numeric), Cramer's V
    (categorical-categorical), correlation ratio (numeric-categorical)."""
    cols = list(real.columns)
    num = {c: _is_numeric(real[c]) for c in cols}
    codes = {c: _codes(real, synth, c) for c in cols if not num[c]}
    ranks_r = {c: stats.rankdata(real[c]) for c in cols if num[c]}
    ranks_s = {c: stats.rankdata(synth[c]) for c in cols if num[c]}
    k = len(cols)
    Ar, As = np.eye(k), np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cols[i], cols[j]
            if num[a] and num[b]:
                vr = abs(np.corrcoef(ranks_r[a], ranks_r[b])[0, 1])
                vs = abs(np.corrcoef(ranks_s[a], ranks_s[b])[0, 1])
            elif not num[a] and not num[b]:
                ra, sa, ka = codes[a]
                rb, sb, kb = codes[b]
                vr, vs = _cramers_v(ra, rb, ka, kb), _cramers_v(sa, sb, ka, kb)
            else:
                cat, x = (a, b) if not num[a] else (b, a)
                rc, sc, kc = codes[cat]
                vr = _corr_ratio(rc, real[x].to_numpy(float), kc)
                vs = _corr_ratio(sc, synth[x].to_numpy(float), kc)
            Ar[i, j] = Ar[j, i] = np.nan_to_num(vr)
            As[i, j] = As[j, i] = np.nan_to_num(vs)
    return cols, Ar, As


def spearman_numeric(df):
    num = [c for c in df.columns if _is_numeric(df[c])]
    return df[num].corr(method="spearman").fillna(0.0).to_numpy()


def c2st_auc(real, synth, seed, n_max=5000, n_splits=5):
    """Classifier two-sample test: cross-validated ROC AUC of a gradient-boosting classifier that
    tells real from synthetic rows (equal-size samples). 0.5 = indistinguishable, 1 = fully separable."""
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth), n_max)
    r = real.iloc[rng.choice(len(real), n, replace=False)]
    s = synth.iloc[rng.choice(len(synth), n, replace=False)]
    both = pd.concat([r, s], ignore_index=True)
    is_cat = []
    X = np.empty((len(both), both.shape[1]))
    for j, c in enumerate(both.columns):
        if _is_numeric(both[c]):
            X[:, j] = both[c].to_numpy(float)
            is_cat.append(False)
        else:
            X[:, j] = pd.factorize(both[c])[0]
            is_cat.append(True)
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(categorical_features=np.array(is_cat), random_state=seed, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        auc = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                              scoring="roc_auc")
    return float(auc.mean())


def fidelity_block(real, synth, seed, c2st=True):
    """Joint and marginal fidelity of one synthetic sample against real rows."""
    cf = column_fidelity(real, synth)
    _, Ar, As = association_matrices(real, synth)
    iu = np.triu_indices(len(Ar), 1)
    d = np.abs(Ar - As)[iu]
    Sr, Ss = spearman_numeric(real), spearman_numeric(synth)
    out = {"n_real": len(real), "n_synth": len(synth),
           "mean_ks_numeric": cf["ks"].mean(), "max_ks_numeric": cf["ks"].max(),
           "mean_tv": cf["tv"].mean(), "max_tv": cf["tv"].max(),
           "pairwise_dependence_mae": float(d.mean()) if len(d) else np.nan,
           "pairwise_dependence_max": float(d.max()) if len(d) else np.nan,
           "spearman_diff_frobenius": float(np.linalg.norm(Sr - Ss)),
           "spearman_diff_mean_abs": float(np.abs(Sr - Ss)[np.triu_indices(len(Sr), 1)].mean())
           if len(Sr) > 1 else np.nan}
    if c2st:
        out["c2st_auc"] = c2st_auc(real, synth, seed)
    return out, cf


# ---------------------------------------------------------------- privacy

class GowerEncoder:
    """Rows -> vectors whose L1 distance / n_columns is the Gower distance: numeric columns scaled
    by their training range, categorical columns one-hot with weight 1/2 (a mismatch costs 1)."""

    def fit(self, df):
        self.columns_ = list(df.columns)
        self.num_ = {c: (float(df[c].min()), float(df[c].max() - df[c].min()) or 1.0)
                     for c in df.columns if _is_numeric(df[c])}
        self.cats_ = {c: pd.Index(df[c].unique()) for c in df.columns if c not in self.num_}
        return self

    def transform(self, df):
        parts = []
        for c in self.columns_:
            if c in self.num_:
                lo, rng = self.num_[c]
                parts.append(((df[c].to_numpy(float) - lo) / rng)[:, None])
            else:
                idx = self.cats_[c].get_indexer(df[c])
                oh = np.zeros((len(df), len(self.cats_[c]) + 1))   # last column: category unseen in training
                oh[np.arange(len(df)), np.where(idx < 0, len(self.cats_[c]), idx)] = 0.5
                parts.append(oh)
        return np.hstack(parts) / len(self.columns_)


def nearest_distance(query, reference):
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="brute").fit(reference)
    return nn.kneighbors(query, return_distance=True)[0][:, 0]


def privacy_block(train, holdout, synth, seed, n_query=2000):
    """Distance to closest record (DCR) and a distance-based membership-inference attack.

    DCR: for n_query synthetic rows, the Gower distance to the closest training row and to the
    closest holdout row, with equally large reference sets (training rows subsampled to the
    holdout size). A generator that copies training rows is closer to the training set; its
    exact-copy share is about (holdout size / training size) plus duplicates, because only the
    subsampled training rows can be found.
    MIA: members = n_query training rows, non-members = n_query holdout rows; the attacker scores a
    real row by minus its distance to the closest synthetic row; AUC 0.5 = no membership signal.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    enc = GowerEncoder().fit(train)
    m = min(len(train), len(holdout))
    tr_ref = enc.transform(train.iloc[rng.choice(len(train), m, replace=False)])
    ho_ref = enc.transform(holdout.iloc[rng.choice(len(holdout), m, replace=False)])
    q = enc.transform(synth.iloc[rng.choice(len(synth), min(n_query, len(synth)), replace=False)])
    d_tr, d_ho = nearest_distance(q, tr_ref), nearest_distance(q, ho_ref)
    k = min(n_query, len(train), len(holdout))
    members = enc.transform(train.iloc[rng.choice(len(train), k, replace=False)])
    non = enc.transform(holdout.iloc[rng.choice(len(holdout), k, replace=False)])
    S = enc.transform(synth)
    score = -np.r_[nearest_distance(members, S), nearest_distance(non, S)]
    return {"dcr_train_median": float(np.median(d_tr)), "dcr_holdout_median": float(np.median(d_ho)),
            "dcr_train_p05": float(np.quantile(d_tr, 0.05)), "dcr_holdout_p05": float(np.quantile(d_ho, 0.05)),
            "exact_copies_train": float((d_tr == 0).mean()), "exact_copies_holdout": float((d_ho == 0).mean()),
            "share_closer_to_train": float((d_tr < d_ho).mean() + 0.5 * (d_tr == d_ho).mean()),
            "mia_auc": float(roc_auc_score(np.r_[np.ones(k), np.zeros(k)], score)),
            "n_query": int(len(q)), "n_reference": int(m), "n_mia_each": int(k)}
