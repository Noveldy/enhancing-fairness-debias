"""Shape, schema and missingness checks for the revised dataset loaders (revision Phase 1).

Expected numbers were derived independently from the raw files
(docs/audit/dataset_inspection.log and a separate pandas check) before the
loaders were written; see REVISION_LOG.md, Phase 1.
"""
import shutil

import pandas as pd
import pytest

from data import loaders
from data.summary import build_summary


# --------------------------------------------------------------------------- raw files

@pytest.mark.parametrize("name", sorted(loaders.RAW_FILES))
def test_raw_checksum_matches(name):
    assert loaders.sha256(loaders.raw_path(name)) == loaders.RAW_FILES[name]["sha256"]


def test_modified_raw_file_is_rejected(tmp_path):
    for spec in loaders.RAW_FILES.values():
        (tmp_path / spec["file"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(loaders.RAW_DIR / spec["file"], tmp_path / spec["file"])
    with open(tmp_path / "adult.csv", "a") as f:
        f.write("\n")
    with pytest.raises(loaders.ChecksumError):
        loaders.load_adult(raw_dir=tmp_path)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        loaders.load_bank("raw")


# --------------------------------------------------------------------------- Adult

def test_adult_shape_and_schema():
    full = loaders.load_adult("full")
    base = loaders.load_adult("baseline")
    assert full.shape == (48842, 15)
    assert base.shape == (45222, 15)
    assert list(full.columns) == loaders.ADULT_COLUMNS
    assert set(loaders.ADULT_DESCRIPTIONS) == set(loaders.ADULT_COLUMNS)


def test_adult_missingness():
    full = loaders.load_adult("full")
    missing = full.isna().sum()
    assert missing[missing > 0].to_dict() == {"workclass": 2799, "occupation": 2809, "native-country": 857}
    assert int(full.isna().any(axis=1).sum()) == 3620
    assert not (full.astype("string") == "?").any().any()
    assert not loaders.load_adult("baseline").isna().any().any()


def test_adult_target_is_binary():
    full = loaders.load_adult("full")
    assert full["income"].dtype == "int64"
    assert set(full["income"].unique()) == {0, 1}
    assert int(full["income"].sum()) == 11687  # '>50K' (7,841) + '>50K.' (3,846)


def test_adult_protected_attributes():
    full = loaders.load_adult("full")
    assert full["sex"].value_counts().to_dict() == {"Male": 32650, "Female": 16192}
    assert set(full["race"].unique()) == {"White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"}


# --------------------------------------------------------------------------- Bank Marketing

def test_bank_shape_and_schema():
    full = loaders.load_bank("full")
    base = loaders.load_bank("baseline")
    assert full.shape == (45211, 17)
    assert base.shape == (43193, 17)
    assert list(full.columns) == loaders.BANK_COLUMNS
    assert set(loaders.BANK_DESCRIPTIONS) == set(loaders.BANK_COLUMNS)


def test_bank_is_single_variant():
    cols = set(loaders.load_bank().columns)
    assert {"balance", "day"} <= cols  # bank-full columns
    assert not cols & {"day_of_week", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"}
    full = loaders.load_bank()
    assert full["day"].between(1, 31).all()
    assert set(full["education"].dropna()) == {"primary", "secondary", "tertiary"}


def test_bank_missingness_and_unknown_categories():
    full = loaders.load_bank("full")
    missing = full.isna().sum()
    assert missing[missing > 0].to_dict() == {"job": 288, "education": 1857}
    assert int(full.isna().any(axis=1).sum()) == 2018
    assert set(full["poutcome"]) == {"failure", "other", "success", "unknown"}
    assert set(full["contact"]) == {"cellular", "telephone", "unknown"}
    assert int((full["poutcome"] == "unknown").sum()) == 36959
    assert int((full["contact"] == "unknown").sum()) == 13020
    assert not loaders.load_bank("baseline").isna().any().any()


def test_bank_numeric_columns_untouched():
    raw = loaders.read_raw("bank")
    full = loaders.load_bank()
    assert int((full["balance"] == -200).sum()) == 3  # real balances, not a missing code
    pd.testing.assert_series_equal(full["balance"], raw["balance"])


def test_bank_target():
    full = loaders.load_bank()
    assert set(full["y"].unique()) == {0, 1}
    assert int(full["y"].sum()) == 5289


# --------------------------------------------------------------------------- Air Quality

AQ_MISSING = {
    "CO(GT)": 1683, "PT08.S1(CO)": 366, "C6H6(GT)": 366, "PT08.S2(NMHC)": 366, "NOx(GT)": 1639,
    "PT08.S3(NOx)": 366, "NO2(GT)": 1642, "PT08.S4(NO2)": 366, "PT08.S5(O3)": 366,
    "T": 366, "RH": 366, "AH": 366,
}


def test_air_quality_shape_and_schema():
    aq = loaders.load_air_quality()
    assert aq.shape == (9357, 13)
    assert list(aq.columns) == loaders.AIR_QUALITY_COLUMNS
    assert "NMHC(GT)" not in aq.columns
    assert set(loaders.AIR_QUALITY_DESCRIPTIONS) == set(loaders.AIR_QUALITY_COLUMNS)


def test_air_quality_missing_code_converted():
    aq = loaders.load_air_quality()
    sensors = aq.drop(columns="datetime")
    assert not (sensors == -200).any().any()
    missing = sensors.isna().sum()
    assert missing[missing > 0].to_dict() == AQ_MISSING
    assert int(sensors.isna().sum().sum()) == 8258
    assert int(sensors.isna().any(axis=1).sum()) == 2416


def test_air_quality_hourly_index():
    aq = loaders.load_air_quality()
    assert aq["datetime"].iloc[0] == pd.Timestamp("2004-03-10 18:00:00")
    assert aq["datetime"].iloc[-1] == pd.Timestamp("2005-04-04 14:00:00")
    assert (aq["datetime"].diff().dropna() == pd.Timedelta("1h")).all()


# --------------------------------------------------------------------------- summary table

def test_data_summary_numbers():
    summary, missingness = build_summary()
    s = summary.set_index("dataset")
    assert s.loc["adult", "rows_full"] == 48842
    assert s.loc["adult", "rows_baseline"] == 45222
    assert s.loc["bank", "rows_full"] == 45211
    assert s.loc["bank", "raw_missing_marker_cells"] == 52124  # the paper's '52,124 missing values'
    assert s.loc["bank", "unknown_kept_as_category_cells"] == 49979
    assert s.loc["bank", "missing_cells_full"] == 2145
    assert s.loc["air_quality", "raw_missing_marker_cells"] == 16701
    assert s.loc["air_quality", "missing_cells_full"] == 8258
    aq = missingness[missingness["dataset"] == "air_quality"].set_index("column")
    assert aq.loc["NMHC(GT)", "status"] == "dropped"
    assert aq.loc["NMHC(GT)", "raw_missing_markers"] == 8443
