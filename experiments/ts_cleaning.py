"""Time-series cleaning: the proposed context-based gap filler and the paper's comparison methods.

All fillers follow the same rule as the tabular imputer: ``fit`` sees only the
training period, and nothing after it is used (revision Phase 2.1).

Proposed algorithm (paper §2.5.1, Fig. 4; author decision G1: "implement it as
the paper describes, per gap and per column, donors from the training period
only"). For every column and every contiguous gap of length L in it:

1. Anchors: the record (all columns at that timestamp) just before the gap and
   the record just after it. A gap at the start or end of the period has one
   anchor.
2. Every candidate donor window of length L elsewhere in the training period
   is scored by cosine similarity: sim(record before gap, record before window)
   + sim(record after gap, record after window). This is the paper's example:
   t4 ~ t11 and t8 ~ t15, so t5..t7 are filled from t12..t14.
3. The best-scoring window donates its values for this column.

The paper's text leaves three details open. The choices below are decisions
G2-G4 (REVISION_LOG.md, design decisions):

* G2, donor windows with missing values. In Air Quality no column has a fully
  observed window as long as the paper's 1,296-hour artificial gap (longest
  run: 1,100 h; CO(GT): 95 h). Rule: if fully observed windows exist, take the
  most similar one. Otherwise take the window with the fewest missing donor
  cells (ties: most similar), copy its observed values, and treat the cells
  still missing as new, shorter gaps (up to ``max_passes`` passes; donors are
  always original observations, never filled values). Cells left after the
  last pass are filled by linear interpolation and counted in the donor log.
* G3, similarity space. The record vectors are used unscaled, as in the
  tabular imputer (a faithful port of the original). Missing entries in a
  record are replaced by the training mean for the similarity only.
  ``standardize=True`` z-scores them with training statistics instead.
* G4, donor windows may not overlap the gap or its anchors.

Every filled gap is recorded in ``donor_log_`` (the paper's donor log L).
"""
import time

import numpy as np
import pandas as pd


def find_gaps(mask):
    """Contiguous runs of True in a boolean array, as (start, end) inclusive positions."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def remove_period(df, start, end):
    """Set every value in [start, end) to NaN (the paper's artificial gap). Returns (df_with_gap, gap_mask)."""
    gap = (df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))
    out = df.copy()
    out.loc[gap, :] = np.nan
    return out, gap


# ---------------------------------------------------------------------- comparison methods
class MeanFiller:
    """Column mean of the training period (paper: "mean interpolation")."""
    name = "mean"

    def fit(self, train):
        self.means_ = train.mean()
        return self

    def transform(self, train):
        return train.fillna(self.means_)


class InterpolationFiller:
    """pandas interpolation inside the training period only (paper: linear, spline order 3).

    Same calls as legacy/original/bias_mitigation/data_cleaning.py. Gaps at the very
    start or end of the period, which interpolation cannot reach, take the nearest
    observed value (logged as ``edge_cells_``).
    """

    def __init__(self, method="linear", order=None):
        self.method = method
        self.order = order
        self.name = method if order is None else f"{method}{order}"

    def fit(self, train):
        return self

    def transform(self, train):
        kw = {"order": self.order} if self.order is not None else {}
        out = train.copy()
        for col in out.columns:
            out[col] = out[col].interpolate(method=self.method, **kw)
        self.edge_cells_ = int(out.isna().sum().sum())
        return out.ffill().bfill()


# ---------------------------------------------------------------------- proposed method
class ContextGapFiller:
    """The proposed context-based time-series cleaning (see module docstring)."""
    name = "proposed"

    def __init__(self, max_passes=3, standardize=False):
        self.max_passes = max_passes
        self.standardize = standardize

    def fit(self, train):
        self.columns_ = list(train.columns)
        self.index_ = train.index
        self.values_ = train.to_numpy(dtype=float)          # donors: original observations only
        self.means_ = np.nanmean(self.values_, axis=0)
        records = np.where(np.isnan(self.values_), self.means_, self.values_)
        if self.standardize:
            std = np.nanstd(self.values_, axis=0)
            records = (records - self.means_) / np.where(std > 0, std, 1.0)
        norms = np.linalg.norm(records, axis=1)
        self.records_ = records / np.where(norms > 0, norms, 1.0)[:, None]  # unit rows: dot = cosine
        return self

    def transform(self, train):
        if not train.index.equals(self.index_):
            raise ValueError("ContextGapFiller fills the period it was fitted on (the training period)")
        t0 = time.perf_counter()
        values = self.values_.copy()
        log = []
        for j, col in enumerate(self.columns_):
            observed = ~np.isnan(self.values_[:, j])
            todo = ~observed
            for p in range(1, self.max_passes + 1):
                gaps = find_gaps(todo)
                if not gaps:
                    break
                for s, e in gaps:
                    entry = self._fill_gap(values, observed, j, s, e)
                    entry.update(column=col, passes=p)
                    log.append(entry)
                todo = np.isnan(values[:, j])
            leftover = np.isnan(values[:, j])
            if leftover.any():
                filled = pd.Series(values[:, j], index=self.index_).interpolate("linear").ffill().bfill()
                for s, e in find_gaps(leftover):
                    log.append({"column": col, "gap_start": self.index_[s], "gap_length": e - s + 1,
                                "donor_start": pd.NaT, "score": np.nan, "donor_missing": np.nan,
                                "passes": self.max_passes + 1, "route": "linear fallback"})
                values[:, j] = filled.to_numpy()
        self.donor_log_ = pd.DataFrame(log)
        self.seconds_ = time.perf_counter() - t0
        return pd.DataFrame(values, index=self.index_, columns=self.columns_)

    def _fill_gap(self, values, observed, j, s, e):
        n, L = len(values), e - s + 1
        before = s - 1 if s > 0 else None
        after = e + 1 if e + 1 < n else None
        # Candidate window [i+1, i+L] with anchors i and i+L+1, kept inside the period.
        lo = 0 if before is not None else -1
        hi = n - L - 2 if after is not None else n - L - 1
        cand = np.arange(lo, hi + 1)
        # G4: the window and its anchors may not touch the gap.
        cand = cand[(cand + L + 1 < s) | (cand > e)]
        if before is None and after is None or len(cand) == 0:
            return {"gap_start": self.index_[s], "gap_length": L, "donor_start": pd.NaT, "score": np.nan,
                    "donor_missing": np.nan, "route": "no candidate window"}
        score = np.zeros(len(cand))
        if before is not None:
            score += self.records_[cand] @ self.records_[before]
        if after is not None:
            score += self.records_[cand + L + 1] @ self.records_[after]
        # Missing donor cells per window, from the original observations of this column.
        miss = np.concatenate([[0], np.cumsum(~observed)])
        donor_missing = miss[cand + L + 1] - miss[cand + 1]
        best_missing = donor_missing.min()
        ok = donor_missing == best_missing
        k = np.flatnonzero(ok)[np.argmax(score[ok])]
        i = cand[k]
        donor = self.values_[i + 1:i + L + 1, j]
        target = values[s:e + 1, j]
        take = np.isnan(target) & ~np.isnan(donor)
        target[take] = donor[take]
        return {"gap_start": self.index_[s], "gap_length": L, "donor_start": self.index_[i + 1],
                "score": float(score[k]), "donor_missing": int(best_missing),
                "route": "donor window" if best_missing == 0 else "donor window, partly missing"}


METHODS = {
    "mean": lambda cfg: MeanFiller(),
    "linear": lambda cfg: InterpolationFiller("linear"),
    "spline": lambda cfg: InterpolationFiller("spline", order=3),
    "proposed": lambda cfg: ContextGapFiller(**(cfg or {})),
}


def fill_training_period(train, method, method_cfg=None):
    """Fit a filler on the training period and fill it. Returns (filled, seconds, filler)."""
    filler = METHODS[method](method_cfg)
    t0 = time.perf_counter()
    filled = filler.fit(train).transform(train)
    seconds = time.perf_counter() - t0
    if filled.isna().any().any():
        raise RuntimeError(f"{method}: training period still has missing values after filling")
    return filled, seconds, filler
