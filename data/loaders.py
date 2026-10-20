"""Dataset loaders for the revised experiments.

Each loader reads a raw UCI file from ``Datasets/``, checks its SHA-256 and
applies a fixed schema and missing-value policy (documented in data/README.md).

The loaders never learn anything from the data: no imputation, scaling or
encoding happens here. Those steps are fitted on training folds only, inside
the evaluation pipeline (revision Phase 2).

Two variants per tabular dataset:
    "full"      true missing values are left as NaN, for the proposed cleaning
                algorithm (fitted later on training folds).
    "baseline"  rows with true missing values are dropped (complete cases).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "Datasets"
VARIANTS = ("full", "baseline")

# Raw files exactly as fetched with ucimlrepo (debias_ops/1st_Journal_2025/fetch_datasets.py).
RAW_FILES = {
    "adult": {
        "file": "adult.csv",
        "sha256": "7bf09a7484d84f92c7e69582078bb01b2f7c4862c8fb27990c0dd848a8a029f4",
    },
    "bank": {
        "file": "bank_marketing.csv",
        "sha256": "592cda3d5f6b704f0ce23477e8375acf2301460efe76cd192e2a789325ccdc13",
    },
    "air_quality": {
        "file": "air_quality.csv",
        "sha256": "d8196847f231fc141ba16c1376aeb90d747a95353203e20ca3b4ae135438157d",
    },
    # Phase 6 external validation set (author decision D2); downloaded 2026-10-20 from
    # https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
    "german": {
        "file": "german_credit/german.data",
        "sha256": "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
        "doc": "german_credit/german.doc",
        "doc_sha256": "69b92cb551111bc37098698288849c96f0fd87ce81a2e54451ff39aa1e53fe49",
    },
}

# Facts used by the summary script, the tests and data/README.md.
INFO = {
    "adult": {
        "source": "UCI Adult (id 2), adult.data + adult.test merged, via ucimlrepo",
        "url": "https://archive.ics.uci.edu/dataset/2/adult",
        "target": "income",
        "positive_label": ">50K",
        "protected": ["sex", "race"],
        "missing_policy": (
            "Both empty cells (2,203) and literal '?' (4,262) in workclass/occupation/native-country "
            "are true missing values (NaN): dropped in 'baseline', left for the cleaning algorithm in 'full'. "
            "Income labels from adult.test carry a trailing '.' and are merged with the adult.data labels."
        ),
    },
    "bank": {
        "source": "UCI Bank Marketing (id 222), variant bank-full, via ucimlrepo",
        "url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "target": "y",
        "positive_label": "yes",
        "protected": ["marital", "age"],
        "missing_policy": (
            "ucimlrepo stores UCI 'unknown' as empty cells. job/education 'unknown' = true missing "
            "(NaN; dropped in 'baseline', left for the cleaning algorithm in 'full'). "
            "contact/poutcome 'unknown' = explicit category 'unknown' in both variants "
            "(poutcome 'unknown' is 'not previously contacted': 36,954 of its 36,959 rows have pdays = -1)."
        ),
    },
    "air_quality": {
        "source": "UCI Air Quality (id 360), via ucimlrepo",
        "url": "https://archive.ics.uci.edu/dataset/360/air+quality",
        "target": None,
        "positive_label": None,
        "protected": [],
        "missing_policy": (
            "The documented missing-value code -200 is converted to NaN in every sensor column before any "
            "other step. NMHC(GT) is dropped (90.2% missing; author decision B5). Remaining gaps are filled "
            "later by the compared methods (mean/linear/spline interpolation, proposed algorithm), fitted on "
            "training data only."
        ),
    },
}


INFO["german"] = {
    "source": "UCI Statlog (German Credit Data) (id 144), german.data (categorical codes) + german.doc",
    "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    "target": "credit_risk",
    "positive_label": "good (1)",
    "protected": ["personal_status_and_sex", "age_in_years", "foreign_worker"],
    "missing_policy": "No missing values. Used only for the sensitive-attribute detector (Phase 6).",
}


class ChecksumError(RuntimeError):
    """Raised when a raw file does not match its recorded SHA-256."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(name, raw_dir=None):
    return Path(raw_dir or RAW_DIR) / RAW_FILES[name]["file"]


def read_raw(name, raw_dir=None):
    """Read a raw file after verifying its checksum."""
    path = raw_path(name, raw_dir)
    digest = sha256(path)
    if digest != RAW_FILES[name]["sha256"]:
        raise ChecksumError(f"{path}: sha256 {digest} != expected {RAW_FILES[name]['sha256']}")
    return pd.read_csv(path)


def _check_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


# --------------------------------------------------------------------------- Adult

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
ADULT_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country",
]
ADULT_DESCRIPTIONS = {
    "age": "Age of the individual (years)",
    "workclass": "Employment type (e.g. Private, Self-emp-inc, Federal-gov)",
    "fnlwgt": "Census sampling weight (final weight)",
    "education": "Highest education level attained",
    "education-num": "Highest education level as an ordinal number",
    "marital-status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Relationship to the household head (e.g. Husband, Wife, Own-child)",
    "race": "Race of the individual",
    "sex": "Sex of the individual (Female, Male)",
    "capital-gain": "Capital gains (USD)",
    "capital-loss": "Capital losses (USD)",
    "hours-per-week": "Hours worked per week",
    "native-country": "Country of origin",
    "income": "Whether annual income exceeds 50K USD",
}


def load_adult(variant="full", raw_dir=None):
    """UCI Adult (48,842 rows; 45,222 complete rows).

    Schema fixes:
    - '?' is a missing-value marker, converted to NaN like the empty cells.
    - target ``income``: '>50K' / '>50K.' -> 1, '<=50K' / '<=50K.' -> 0
      (the raw file has 4 labels; see docs/AUDIT.md section 5.1).
    """
    _check_variant(variant)
    df = read_raw("adult", raw_dir)
    for col in ADULT_CATEGORICAL:
        df[col] = df[col].str.strip().replace("?", np.nan)
    labels = df["income"].str.strip().str.rstrip(".")
    df["income"] = labels.map({"<=50K": 0, ">50K": 1})
    if df["income"].isna().any():
        raise ValueError(f"unexpected labels in Adult target: {sorted(labels[df['income'].isna()].unique())}")
    df["income"] = df["income"].astype(int)
    df = df[ADULT_COLUMNS]
    if variant == "baseline":
        df = df.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- Bank Marketing

BANK_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y",
]
BANK_TRUE_MISSING = ["job", "education"]
# One description per feature, from the UCI documentation. Replaces the list in
# legacy/debias_ops/main.py:73-74, which was shifted by one from 'housing' onward.
BANK_DESCRIPTIONS = {
    "age": "Age of the client (years)",
    "job": "Type of job (e.g. admin., blue-collar, management, student)",
    "marital": "Marital status (married, single, divorced; divorced includes widowed)",
    "education": "Education level (primary, secondary, tertiary)",
    "default": "Whether the client has credit in default",
    "balance": "Average yearly balance (euros)",
    "housing": "Whether the client has a housing loan",
    "loan": "Whether the client has a personal loan",
    "contact": "Contact communication type (cellular, telephone, unknown)",
    "day": "Last contact day of the month",
    "month": "Last contact month of the year",
    "duration": "Last contact duration in seconds (known only after the call; see data/README.md)",
    "campaign": "Number of contacts performed during this campaign for this client",
    "pdays": "Days since the client was last contacted in a previous campaign (-1 = not previously contacted)",
    "previous": "Number of contacts performed before this campaign for this client",
    "poutcome": "Outcome of the previous marketing campaign (failure, other, success, unknown)",
    "y": "Whether the client subscribed to a term deposit",
}
BANK_UNKNOWN_AS_CATEGORY = ["contact", "poutcome"]


def load_bank(variant="full", raw_dir=None):
    """UCI Bank Marketing, bank-full (45,211 rows).

    Schema fixes:
    - ucimlrepo names the day-of-month column ``day_of_week``; it holds 1-31, so it
      is renamed back to the UCI name ``day``.
    - target ``y``: 'yes' -> 1, 'no' -> 0.
    """
    _check_variant(variant)
    df = read_raw("bank", raw_dir).rename(columns={"day_of_week": "day"})
    for col in BANK_UNKNOWN_AS_CATEGORY:
        df[col] = df[col].fillna("unknown")
    df["y"] = df["y"].map({"no": 0, "yes": 1})
    if df["y"].isna().any():
        raise ValueError("unexpected labels in Bank target 'y'")
    df["y"] = df["y"].astype(int)
    df = df[BANK_COLUMNS]
    if variant == "baseline":
        df = df.dropna(subset=BANK_TRUE_MISSING).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- Air Quality

AIR_QUALITY_MISSING_CODE = -200
AIR_QUALITY_DROPPED = ["NMHC(GT)"]
AIR_QUALITY_COLUMNS = [
    "datetime", "CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)",
    "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
AIR_QUALITY_DESCRIPTIONS = {
    "datetime": "Date and hour of the measurement (hourly averages)",
    "CO(GT)": "True hourly averaged CO concentration (mg/m^3), reference analyzer",
    "PT08.S1(CO)": "Tin oxide sensor response, nominally CO targeted",
    "C6H6(GT)": "True hourly averaged benzene concentration (microg/m^3), reference analyzer",
    "PT08.S2(NMHC)": "Titania sensor response, nominally NMHC targeted",
    "NOx(GT)": "True hourly averaged NOx concentration (ppb), reference analyzer",
    "PT08.S3(NOx)": "Tungsten oxide sensor response, nominally NOx targeted",
    "NO2(GT)": "True hourly averaged NO2 concentration (microg/m^3), reference analyzer",
    "PT08.S4(NO2)": "Tungsten oxide sensor response, nominally NO2 targeted",
    "PT08.S5(O3)": "Indium oxide sensor response, nominally O3 targeted",
    "T": "Temperature (degrees C)",
    "RH": "Relative humidity (%)",
    "AH": "Absolute humidity",
}


def load_air_quality(raw_dir=None):
    """UCI Air Quality (9,357 hourly rows, 2004-03-10 18:00 to 2005-04-04 14:00).

    Schema fixes:
    - ``Date`` (m/d/Y, as written by ucimlrepo) and ``Time`` are combined into ``datetime``.
    - -200 (the documented missing-value code) -> NaN in all sensor columns.
    - NMHC(GT) is dropped (author decision B5).
    The series is checked to be a regular hourly index with no duplicates.
    """
    df = read_raw("air_quality", raw_dir)
    df.insert(0, "datetime", pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M:%S"))
    df = df.drop(columns=["Date", "Time"] + AIR_QUALITY_DROPPED)
    sensors = [c for c in df.columns if c != "datetime"]
    df[sensors] = df[sensors].astype(float).mask(df[sensors] == AIR_QUALITY_MISSING_CODE)
    steps = df["datetime"].diff().dropna()
    if not df["datetime"].is_unique or not (steps == pd.Timedelta("1h")).all():
        raise ValueError("Air Quality datetime index is not a regular hourly series")
    return df[AIR_QUALITY_COLUMNS]


# --------------------------------------------------------------------------- German Credit (Phase 6 only)

# Column names = the attribute titles of german.doc in snake_case; descriptions = the titles verbatim
# (german.doc section 7). Neither is hand-picked, because the detector reads both.
GERMAN_DESCRIPTIONS = {
    "status_of_existing_checking_account": "Status of existing checking account",
    "duration_in_month": "Duration in month",
    "credit_history": "Credit history",
    "purpose": "Purpose",
    "credit_amount": "Credit amount",
    "savings_account_bonds": "Savings account/bonds",
    "present_employment_since": "Present employment since",
    "installment_rate_in_percentage_of_disposable_income": "Installment rate in percentage of disposable income",
    "personal_status_and_sex": "Personal status and sex",
    "other_debtors_guarantors": "Other debtors / guarantors",
    "present_residence_since": "Present residence since",
    "property": "Property",
    "age_in_years": "Age in years",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "number_of_existing_credits_at_this_bank": "Number of existing credits at this bank",
    "job": "Job",
    "number_of_people_being_liable_to_provide_maintenance_for":
        "Number of people being liable to provide maintenance for",
    "telephone": "Telephone",
    "foreign_worker": "foreign worker",
    # The target has no attribute title in german.doc; this description paraphrases its section 8.
    "credit_risk": "Credit risk (1 = good, 2 = bad)",
}
GERMAN_COLUMNS = list(GERMAN_DESCRIPTIONS)


def load_german(raw_dir=None):
    """UCI Statlog German Credit, 1,000 rows, 20 attributes + target (1 = good, 2 = bad).

    Categorical attributes keep their UCI codes (e.g. A93 = male : single); numeric ones are ints.
    """
    path = Path(raw_dir or RAW_DIR) / RAW_FILES["german"]["file"]
    digest = sha256(path)
    if digest != RAW_FILES["german"]["sha256"]:
        raise ChecksumError(f"{path}: sha256 {digest} != expected {RAW_FILES['german']['sha256']}")
    df = pd.read_csv(path, sep=" ", header=None, names=GERMAN_COLUMNS)
    if df.shape != (1000, 21) or not set(df["credit_risk"]) <= {1, 2}:
        raise ValueError(f"unexpected German Credit file: shape {df.shape}")
    return df
