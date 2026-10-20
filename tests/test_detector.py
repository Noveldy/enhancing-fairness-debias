"""Phase 6: keyword rule, training-data handling, paraphrase grouping, proxy association."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.loaders import GERMAN_COLUMNS, load_german
from experiments.detector import keyword_detect, load_training_data, paraphrase_groups

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from detector_validation import association  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_keyword_rule_matches_the_original_including_substring_errors():
    assert keyword_detect("sex") and keyword_detect("Race") and keyword_detect("age_in_years")
    assert keyword_detect("percentage")            # 'age' inside 'percentage': the documented false positive
    assert not keyword_detect("marital-status")    # 'marital status' (with a space) is the listed word
    assert not keyword_detect("income")


def test_training_data_duplicates_removed_without_label_conflicts():
    d, n_dup = load_training_data(REPO / "bias_identification" / "sensitive_features_dataset.csv")
    assert n_dup == 9 and len(d) == 816
    assert not d["text"].str.lower().str.strip().str.rstrip(".").duplicated().any()
    assert set(d["y"]) == {0, 1}


def test_paraphrase_groups_keep_templates_together_and_split_unrelated_texts():
    texts = ["Percentage of Black residents", "Percentage of Black inhabitants", "Proportion of Black residents",
             "Annual income of the household", "Number of rooms per dwelling", "Credit amount requested"]
    g = paraphrase_groups(texts)
    assert g[0] == g[1] == g[2]
    assert len({g[3], g[4], g[5], g[0]}) == 4


def test_association_measures_and_german_loader():
    a = pd.Series(["x", "y"] * 50)
    assert abs(association(a, a) - 1.0) < 1e-12
    assert abs(association(pd.Series(np.arange(100)), pd.Series(np.arange(100) * 2)) - 1.0) < 1e-12
    df = load_german()
    assert list(df.columns) == GERMAN_COLUMNS and df.shape == (1000, 21)
    assert set(df["personal_status_and_sex"]) <= {"A91", "A92", "A93", "A94", "A95"}
