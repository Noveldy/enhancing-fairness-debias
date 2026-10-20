"""Paired tests for repeated CV (experiments/stats.py)."""
import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from experiments import stats


def test_corrected_ttest_by_hand():
    d = np.array([0.01, 0.02, 0.00, 0.03, 0.01, 0.02, 0.015, 0.005, 0.02, 0.01])
    r = stats.corrected_resampled_ttest(d, test_train_ratio=0.25)
    se = np.sqrt((1 / 10 + 0.25) * d.var(ddof=1))
    assert r["nb_t"] == pytest.approx(d.mean() / se)
    assert r["nb_df"] == 9
    assert r["nb_p"] == pytest.approx(2 * sps.t.sf(d.mean() / se, 9))
    assert r["nb_ci_low"] == pytest.approx(d.mean() - sps.t.ppf(0.975, 9) * se)


def test_correction_is_more_conservative_than_plain_paired_t():
    rng = np.random.default_rng(0)
    d = rng.normal(0.005, 0.01, 25)
    plain = sps.ttest_1samp(d, 0).pvalue
    assert stats.corrected_resampled_ttest(d, 0.25)["nb_p"] > plain


def test_holm_matches_statsmodels():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.001])
    np.testing.assert_allclose(stats.holm(p), multipletests(p, method="holm")[1])
    out = stats.holm([0.01, np.nan, 0.02])
    assert np.isnan(out[1]) and out[0] == pytest.approx(0.02)


def test_wilcoxon_effect_size_sign():
    assert stats.wilcoxon_signed_rank(np.arange(1, 11) / 100)["rank_biserial"] == 1.0
    r = stats.wilcoxon_signed_rank(-np.arange(1, 11) / 100)
    assert r["rank_biserial"] == -1.0 and r["wilcoxon_p"] < 0.01
    assert stats.wilcoxon_signed_rank(np.zeros(5))["wilcoxon_p"] == 1.0
