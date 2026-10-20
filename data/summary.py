"""Corrected dataset facts for the paper's dataset tables (revision Phase 1).

Everything is computed from the raw files and the loaders, except the
``PAPER_REPORTED`` values, which are copied from the submitted manuscript
(paper_v1.6) only so the corrected numbers can be shown next to them.
"""
import numpy as np
import pandas as pd

from data import loaders

# Values stated in the submitted manuscript, for comparison only.
PAPER_REPORTED = {
    "adult": {"rows": "48842 (§3.1, Table 7)", "missing": "2203 (§3.2.3, Table 7)"},
    "bank": {"rows": "45211 (§3.1); 48842 (Table 8)", "missing": "52124 (§3.2.4, Table 8)"},
    "air_quality": {"rows": "9358 (§3.1)", "missing": "0, 'did not contain any missing values' (§3.2.5)"},
}


def _raw_markers(name, raw):
    """Cells in the raw file that encode 'missing/unknown', per column."""
    if name == "adult":
        as_str = raw.astype("string").apply(lambda s: s.str.strip())
        return raw.isna() | (as_str == "?")
    if name == "bank":
        return raw.isna()  # ucimlrepo stores UCI 'unknown' as empty cells
    if name == "air_quality":
        num = raw.drop(columns=["Date", "Time"])
        return num == loaders.AIR_QUALITY_MISSING_CODE
    raise KeyError(name)


def _load(name, variant):
    if name == "adult":
        return loaders.load_adult(variant)
    if name == "bank":
        return loaders.load_bank(variant)
    if name == "air_quality":
        return loaders.load_air_quality() if variant == "full" else None
    raise KeyError(name)


def build_summary():
    """Return (summary, missingness) DataFrames."""
    summary_rows, miss_rows = [], []
    for name in ("adult", "bank", "air_quality"):
        info = loaders.INFO[name]
        raw = loaders.read_raw(name)
        markers = _raw_markers(name, raw)
        full = _load(name, "full")
        base = _load(name, "baseline")
        target = info["target"]
        non_feature = {target, "datetime"}
        dropped = loaders.AIR_QUALITY_DROPPED if name == "air_quality" else []
        unknown_as_category = (
            int(markers[loaders.BANK_UNKNOWN_AS_CATEGORY].sum().sum()) if name == "bank" else 0
        )
        summary_rows.append({
            "dataset": name,
            "source": info["source"],
            "url": info["url"],
            "raw_file": loaders.RAW_FILES[name]["file"],
            "sha256": loaders.RAW_FILES[name]["sha256"],
            "raw_rows": len(raw),
            "raw_columns": raw.shape[1],
            "rows_full": len(full),
            "rows_baseline": len(base) if base is not None else np.nan,
            "n_features": len([c for c in full.columns if c not in non_feature]),
            "features": ";".join(c for c in full.columns if c not in non_feature),
            "target": target or "",
            "positive_label": info["positive_label"] or "",
            "positive_rate_full": round(full[target].mean(), 4) if target else np.nan,
            "positive_rate_baseline": round(base[target].mean(), 4) if base is not None else np.nan,
            "protected_attributes": ";".join(info["protected"]),
            "raw_missing_marker_cells": int(markers.sum().sum()),
            "unknown_kept_as_category_cells": unknown_as_category,
            "dropped_columns": ";".join(dropped),
            "missing_cells_full": int(full.isna().sum().sum()),
            "rows_with_missing_full": int(full.isna().any(axis=1).sum()),
            "paper_reported_rows": PAPER_REPORTED[name]["rows"],
            "paper_reported_missing": PAPER_REPORTED[name]["missing"],
            "missing_policy": info["missing_policy"],
        })
        for col in markers.columns:
            loaded_col = "day" if (name == "bank" and col == "day_of_week") else col
            in_full = loaded_col in full.columns
            n_missing = int(full[loaded_col].isna().sum()) if in_full else np.nan
            miss_rows.append({
                "dataset": name,
                "column": loaded_col,
                "raw_missing_markers": int(markers[col].sum()),
                "missing_in_full_variant": n_missing,
                "pct_missing_in_full_variant": round(100 * n_missing / len(full), 2) if in_full else np.nan,
                "status": "dropped" if col in dropped else (
                    "unknown kept as category" if name == "bank" and col in loaders.BANK_UNKNOWN_AS_CATEGORY
                    else "kept"),
            })
    summary = pd.DataFrame(summary_rows)
    missingness = pd.DataFrame(miss_rows)
    # Counts stay integers even where a value is not applicable (NaN).
    summary["rows_baseline"] = summary["rows_baseline"].astype("Int64")
    missingness["missing_in_full_variant"] = missingness["missing_in_full_variant"].astype("Int64")
    return summary, missingness
