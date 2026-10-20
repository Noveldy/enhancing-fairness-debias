"""Write the corrected dataset tables (revision Phase 1 acceptance).

Outputs, in results/summary/:
    data_summary.csv       one row per dataset (shape, target, missingness, checksum, paper values)
    data_missingness.csv   one row per dataset column
    data_summary_run.json  run record: time, git commit, versions, checksums, wall-clock

Usage (from the repository root):
    venv/bin/python scripts/make_data_summary.py
"""
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import loaders  # noqa: E402
from data.summary import build_summary  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "summary"


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def main():
    start = time.perf_counter()
    summary, missingness = build_summary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "data_summary.csv", index=False)
    missingness.to_csv(OUT_DIR / "data_missingness.csv", index=False)
    record = {
        "script": "scripts/make_data_summary.py",
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),  # tracked changes only
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "seeds": "none (deterministic, no randomness)",
        "raw_files": {name: spec for name, spec in loaders.RAW_FILES.items()},
        "wall_clock_seconds": round(time.perf_counter() - start, 3),
        "outputs": ["results/summary/data_summary.csv", "results/summary/data_missingness.csv"],
    }
    (OUT_DIR / "data_summary_run.json").write_text(json.dumps(record, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.drop(columns=["source", "url", "sha256", "features", "missing_policy"]).to_string(index=False))
    print(f"\nWrote {OUT_DIR.relative_to(REPO_ROOT)}/ in {record['wall_clock_seconds']} s")


if __name__ == "__main__":
    main()
