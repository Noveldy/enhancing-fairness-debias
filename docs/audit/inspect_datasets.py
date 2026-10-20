"""Read-only dataset inspection for the Phase 0 audit (docs/AUDIT.md).

Reads the CSV/JSON files the original experiments used and prints facts about
them: checksum, shape, columns, missing values, placeholder codes ("?",
"unknown", -200, empty strings) and the class balance of targets and
protected attributes. It never writes to the data folder.

Usage:
    python docs/audit/inspect_datasets.py <datasets_dir> [<sensitive_features_csv>] > docs/audit/dataset_inspection.log
"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Placeholder values that may stand for "missing" in the UCI files.
STRING_CODES = ["?", "unknown", ""]
NUMERIC_CODES = [-200]

# What each file is expected to be, so the log shows target / protected columns.
DATASETS = {
    "adult.csv": {"target": "income", "protected": ["sex", "race", "age", "marital-status", "relationship"]},
    "bank_marketing.csv": {"target": "y", "protected": ["marital", "age"]},
    "air_quality.csv": {"target": None, "protected": []},
    "individual_household_electric_power_consumption.csv": {"target": None, "protected": []},
}
# Pipeline output (DBIM "debiased" Adult) that the paper's numbers were derived from.
OUTPUTS = {"adult_debiased.json": {"target": "income", "protected": ["sex", "race"]}}

# Columns that identify the Bank Marketing variant.
BANK_FULL_ONLY = {"balance", "day"}
BANK_ADDITIONAL_ONLY = {"day_of_week", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def describe(df, name, meta):
    print(f"shape (rows, cols): {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("\ndtypes:")
    for c, t in df.dtypes.items():
        print(f"  {c}: {t}")

    print("\nper-column counts  [NaN | '?' | 'unknown' | empty-string | -200]:")
    totals = {"nan": 0, "?": 0, "unknown": 0, "": 0, -200: 0}
    for c in df.columns:
        s = df[c]
        n_nan = int(s.isna().sum())
        as_str = s.dropna().astype(str).str.strip()
        n_q = int((as_str == "?").sum())
        n_unk = int((as_str.str.lower() == "unknown").sum())
        n_empty = int((as_str == "").sum())
        num = pd.to_numeric(s, errors="coerce")
        n_200 = int((num == -200).sum())
        totals["nan"] += n_nan
        totals["?"] += n_q
        totals["unknown"] += n_unk
        totals[""] += n_empty
        totals[-200] += n_200
        if n_nan or n_q or n_unk or n_empty or n_200:
            print(f"  {c}: {n_nan} | {n_q} | {n_unk} | {n_empty} | {n_200}")
    print(f"  TOTAL: NaN={totals['nan']}  '?'={totals['?']}  'unknown'={totals['unknown']}  "
          f"empty={totals['']}  -200={totals[-200]}")
    print(f"  rows with >=1 NaN: {int(df.isna().any(axis=1).sum())}")
    print(f"  duplicate rows: {int(df.duplicated().sum())}")

    target = meta.get("target")
    if target and target in df.columns:
        print(f"\ntarget '{target}' value counts (raw, including NaN):")
        vc = df[target].value_counts(dropna=False)
        for k, v in vc.items():
            print(f"  {k!r}: {v} ({100 * v / len(df):.2f}%)")
    for p in meta.get("protected", []):
        if p not in df.columns:
            print(f"\nprotected '{p}': NOT PRESENT")
            continue
        s = df[p]
        if p == "age" and pd.api.types.is_numeric_dtype(s):
            print(f"\nprotected 'age': min={s.min()} max={s.max()} mean={s.mean():.2f}")
            # Same 5-bin equal-width scheme as bias_identification/datasets.py (BIN_SIZE = 5)
            binned = pd.cut(s, 5).value_counts(sort=False)
            for k, v in binned.items():
                print(f"  bin {k}: {v} ({100 * v / len(df):.2f}%)")
            continue
        print(f"\nprotected '{p}' value counts (raw, including NaN):")
        for k, v in s.value_counts(dropna=False).items():
            print(f"  {k!r}: {v} ({100 * v / len(df):.2f}%)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    data_dir = Path(sys.argv[1])
    sf_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    section("ENVIRONMENT")
    print(f"run at (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"python: {platform.python_version()}  pandas: {pd.__version__}  numpy: {np.__version__}")
    print(f"platform: {platform.platform()}")
    print(f"data dir: {data_dir.resolve()}")
    print("randomness: none (deterministic, read-only)")

    for fname, meta in {**DATASETS, **OUTPUTS}.items():
        path = data_dir / fname
        section(fname)
        if not path.exists():
            print("FILE NOT FOUND")
            continue
        print(f"sha256: {sha256(path)}")
        print(f"size (bytes): {path.stat().st_size}")
        if fname.endswith(".json"):
            df = pd.read_json(path, orient="records")
        else:
            # keep_default_na=True mirrors how the pipeline loads data (pd.read_csv defaults)
            df = pd.read_csv(path, low_memory=False)
        describe(df, fname, meta)
        if fname == "bank_marketing.csv":
            cols = set(df.columns)
            print("\nBank variant check:")
            print(f"  columns only in bank-full:              {sorted(cols & BANK_FULL_ONLY)}")
            print(f"  columns only in bank-additional(-full): {sorted(cols & BANK_ADDITIONAL_ONLY)}")
            if "day_of_week" in df.columns:
                print(f"  'day_of_week' unique values: {sorted(df['day_of_week'].dropna().unique().tolist())}")
            for c in ["education", "poutcome", "contact"]:
                if c in df.columns:
                    print(f"  '{c}' categories: {sorted(df[c].dropna().astype(str).unique().tolist())}")

    if sf_csv is not None:
        section(f"sensitive-features training data: {sf_csv.name}")
        print(f"sha256: {sha256(sf_csv)}")
        df = pd.read_csv(sf_csv)
        print(f"shape: {df.shape}  columns: {list(df.columns)}")
        for c in df.columns:
            if df[c].nunique() <= 10:
                print(f"'{c}' value counts: {json.dumps({str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()})}")
        print(f"duplicate rows: {int(df.duplicated().sum())}")
        first = df.columns[0]
        print(f"duplicate values in '{first}': {int(df[first].duplicated().sum())}")


if __name__ == "__main__":
    main()
