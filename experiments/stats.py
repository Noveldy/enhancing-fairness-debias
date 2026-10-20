"""Paired tests for repeated cross-validation (Phase 3.3).

Replaces the original paired t-test, which paired the (mean, std, min, max)
summaries of two runs (AUDIT section 4). Here each pair is one outer fold
(repeat r, fold f) evaluated by both methods on the same test rows.

* ``corrected_resampled_ttest``: Nadeau & Bengio (2003) variance correction for
  repeated k-fold CV, as in Bouckaert & Frank (2004): var = (1/n + n_test/n_train) * s^2,
  df = n - 1, where n = r * k fold differences. The training sets of different
  folds overlap, so the plain paired t-test underestimates the variance. This is
  the primary test.
* ``wilcoxon_signed_rank``: scipy's two-sided test on the same differences, with
  the matched-pairs rank-biserial correlation as effect size. It ignores the
  overlap between training sets, so its p-values are optimistic; reported as a
  secondary, distribution-free check.
* ``holm``: Holm-Bonferroni adjusted p-values within one family of comparisons.
"""
import numpy as np
from scipy import stats


def corrected_resampled_ttest(diff, test_train_ratio, alpha=0.05):
    """diff: per-fold differences (method A - method B). Returns a dict of statistics."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = np.sqrt((1.0 / n + test_train_ratio) * sd ** 2)
    df = n - 1
    if se == 0:
        t, p = (0.0, 1.0) if mean == 0 else (np.sign(mean) * np.inf, 0.0)
    else:
        t = mean / se
        p = 2 * stats.t.sf(abs(t), df)
    h = stats.t.ppf(1 - alpha / 2, df) * se
    return {"n_pairs": n, "mean_diff": mean, "sd_diff": sd, "nb_t": t, "nb_df": df, "nb_p": p,
            "nb_ci_low": mean - h, "nb_ci_high": mean + h,
            "cohens_dz": mean / sd if sd > 0 else np.nan}


def wilcoxon_signed_rank(diff):
    d = np.asarray(diff, dtype=float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"wilcoxon_W": np.nan, "wilcoxon_p": 1.0, "rank_biserial": 0.0, "n_nonzero": 0}
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nz))
    r_plus, r_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {"wilcoxon_W": float(res.statistic), "wilcoxon_p": float(res.pvalue),
            "rank_biserial": (r_plus - r_minus) / (r_plus + r_minus), "n_nonzero": int(len(nz))}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order (NaN stays NaN)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    order = np.argsort(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def mean_ci(values, test_train_ratio, alpha=0.05):
    """Mean, SD and a 95% CI of one method's per-fold metric, with the same variance correction."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    mean, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * np.sqrt((1.0 / n + test_train_ratio) * sd ** 2)
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - h, "ci_high": mean + h}
