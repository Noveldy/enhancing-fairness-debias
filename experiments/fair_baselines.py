"""Fairness-mitigation baselines that must be fitted on the training fold only (Phase 3).

* ``reweighing_weights``: AIF360's Reweighing (Kamiran & Calders 2012),
  w(a, y) = P(A=a) P(Y=y) / P(A=a, Y=y), estimated on the training fold. AIF360
  only takes one privileged/unprivileged split; this form also covers attributes
  with more than two groups (race, marital). For a binary attribute it equals
  AIF360's weights (tests/test_baselines.py).
* ``TrainFittedDIR``: the Disparate Impact Remover (Feldman et al. 2015, AIF360's
  ``DisparateImpactRemover``) with separate fit and transform. AIF360 only has
  ``fit_transform``, which repairs a dataset with its own per-group distributions;
  the original scripts therefore repaired the test set with test-set quantiles
  (AUDIT section 2, item 7). Here the per-group distributions are learned on the
  training fold and test rows are mapped through them, with AIF360's own repair
  rule (see the class). Categorical codes are not repaired (an ordinal code has no meaningful
  quantile); the original repaired one-hot columns, which full repair maps to 0/1
  again.
"""
import numpy as np
import pandas as pd


def reweighing_weights(y, a):
    """Instance weights making the label independent of the group in the weighted training set."""
    df = pd.DataFrame({"y": np.asarray(y), "a": np.asarray(a)})
    n = len(df)
    p_a = df["a"].value_counts() / n
    p_y = df["y"].value_counts() / n
    p_ay = df.groupby(["a", "y"]).size() / n
    w = p_a.loc[df["a"]].to_numpy() * p_y.loc[df["y"]].to_numpy() / p_ay.loc[list(zip(df["a"], df["y"]))].to_numpy()
    return w


def _lower_median(values):
    """BlackBoxAuditing's median: the left-of-centre value for even lengths, so always a data value."""
    v = sorted(values)
    return v[len(v) // 2 - 1] if len(v) % 2 == 0 else v[len(v) // 2]


class TrainFittedDIR:
    """Disparate Impact Remover fitted on training rows; ``columns`` are the positions to repair.

    Same rule as AIF360 0.6.1 / BlackBoxAuditing 0.1.54 (``CategoricRepairer.repair``,
    numeric columns, kdd=False): each group's sorted distinct values are cut into
    K = min over groups of the number of distinct values quantile buckets; bucket k's
    target is the lower median, over groups, of each group's lower-median value in
    bucket k; a value moves ``repair_level`` of the way to its bucket's target in the
    index space of the column's sorted distinct training values. Fitted on training
    rows, this reproduces AIF360's output on those rows exactly (tests/test_baselines.py).
    A test value is placed in the bucket of the nearest distinct training value of
    its group.
    """

    def __init__(self, columns, repair_level=1.0):
        self.columns = list(columns)
        self.repair_level = repair_level

    def fit(self, X, a):
        X = np.asarray(X, dtype=float)
        a = np.asarray(a)
        self.groups_ = list(np.unique(a))
        self.col_values_, self.group_values_, self.group_targets_ = {}, {}, {}
        for j in self.columns:
            self.col_values_[j] = np.unique(X[:, j])
            uniq = {g: np.unique(X[a == g, j]) for g in self.groups_}
            n_q = min(len(u) for u in uniq.values())
            unit = 1.0 / n_q
            offsets = {g: 0.0 for g in self.groups_}
            target_of = {g: np.empty(len(uniq[g])) for g in self.groups_}
            for _ in range(n_q):
                medians, spans = [], {}
                for g in self.groups_:
                    n_vals = len(uniq[g])
                    off = int(round(offsets[g] * n_vals))
                    cnt = int(round((offsets[g] + unit) * n_vals) - off)
                    offsets[g] += unit
                    if cnt > 0:
                        spans[g] = (off, off + cnt)
                        medians.append(_lower_median(uniq[g][off:off + cnt].tolist()))
                target = _lower_median(medians)
                for g, (lo, hi) in spans.items():
                    target_of[g][lo:hi] = target
            for g in self.groups_:
                self.group_values_[(j, g)] = uniq[g]
                self.group_targets_[(j, g)] = target_of[g]
        return self

    @staticmethod
    def _nearest(sorted_vals, x):
        i = np.clip(np.searchsorted(sorted_vals, x), 1, len(sorted_vals) - 1) if len(sorted_vals) > 1 else 0
        if len(sorted_vals) == 1:
            return np.zeros(len(x), dtype=int)
        left = np.abs(x - sorted_vals[i - 1]) <= np.abs(sorted_vals[i] - x)
        return np.where(left, i - 1, i)

    def transform(self, X, a):
        X = np.array(X, dtype=float, copy=True)
        a = np.asarray(a)
        for j in self.columns:
            col = self.col_values_[j]
            for g in np.unique(a):
                if g not in self.groups_:      # group never seen in training: leave unrepaired
                    continue
                m = a == g
                x = X[m, j]
                target = self.group_targets_[(j, g)][self._nearest(self.group_values_[(j, g)], x)]
                pos = self._nearest(col, x)
                tpos = self._nearest(col, target)
                X[m, j] = col[pos + np.round((tpos - pos) * self.repair_level).astype(int)]
        return X
