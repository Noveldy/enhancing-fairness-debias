"""Phase 9.4 (author decision P6): package the toolkit's detector checkpoint, checkpoint-830, for Zenodo.

    venv/bin/python scripts/package_checkpoint.py          # writes dist/ (git-ignored); ~1 min

The checkpoint is not in git (README, "Detector checkpoint"). This writes:
  dist/checkpoint-830.tar.gz          weights (safetensors), config, tokenizer files and trainer_state.json,
                                      under bert-sensitive-features-classifier/checkpoint-830/, so that
                                      `tar -xzf checkpoint-830.tar.gz` in the repository root puts it where
                                      configs/detector.yaml expects it. Deterministic: sorted entries, fixed
                                      times and owners, so rebuilding gives the same bytes.
  dist/checkpoint-830.SHA256SUMS      sha256 of every packaged file and of the archive
  dist/checkpoint-830.README.md       what it is, what was left out, how to use it
Left out: optimizer.pt, scheduler.pt, rng_state.pth (only needed to resume training) and training_args.bin
(a pickle, which can run code when loaded; the training settings are in trainer_state.json).

Then it checks the archive: unpacks it into a temporary folder, checks every sha256, loads the model from the
unpacked copy and from the original folder with experiments.detector.LegacyCheckpoint (as
scripts/detector_validation.py does) and requires identical probabilities on the external columns. It also reports
the difference from the saved p_checkpoint830 values in results/raw/detector/external_predictions.csv, which
come from another process (the Phase 6 run) and may differ by float32 rounding. Writes results/reproduction_check/checkpoint830_package.json.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import platform
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DIST = REPO_ROOT / "dist"
NAME = "checkpoint-830"
PREFIX = "bert-sensitive-features-classifier/checkpoint-830"
KEEP = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "trainer_state.json", "vocab.txt"]
LEFT_OUT = {"optimizer.pt": "optimizer state, only for resuming training",
            "scheduler.pt": "learning-rate scheduler state, only for resuming training",
            "rng_state.pth": "random-number state, only for resuming training",
            "training_args.bin": "pickled training arguments (a pickle can run code when loaded); "
                                 "the settings are in trainer_state.json"}


def sha256(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, bytes):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def build(src):
    missing = [f for f in KEEP if not (src / f).exists()]
    if missing:
        sys.exit(f"missing in {src}: {missing}")
    DIST.mkdir(exist_ok=True)
    tar_path = DIST / f"{NAME}.tar.gz"
    with open(tar_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in sorted(KEEP):
            info = tar.gettarinfo(str(src / f), arcname=f"{PREFIX}/{f}")
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
            with open(src / f, "rb") as fh:
                tar.addfile(info, fh)
    sums = {f"{PREFIX}/{f}": sha256(src / f) for f in sorted(KEEP)}
    sums[tar_path.name] = sha256(tar_path)
    (DIST / f"{NAME}.SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p, h in sums.items()))
    return tar_path, sums


def readme(sums, size):
    left = "\n".join(f"- `{f}`: {why}" for f, why in LEFT_OUT.items())
    return f"""# checkpoint-830: the DEBIAS toolkit's detector checkpoint

Fine-tuned `bert-base-uncased` sequence classifier (sensitive / not sensitive column descriptions), the
checkpoint loaded by the DEBIAS toolkit as submitted with "Enhancing Fairness: A Cross-Platform Application for
Mitigating Bias in Data Ecosystems" (Array, ARRAY-D-26-05264). The revised evaluation retrains the detector
(REVISION_LOG.md, Phase 6) and reports this checkpoint as a reference.

It is not the checkpoint behind the submitted paper's detector results: the research code loads checkpoint-610,
which is not available (docs/AUDIT.md, section 8).

## Use

In the root of the repository (branch `revision`):

    tar -xzf checkpoint-830.tar.gz
    sha256sum -c checkpoint-830.SHA256SUMS --ignore-missing
    venv/bin/python scripts/detector_validation.py --config configs/detector.yaml

The model files load with `transformers` (`AutoModelForSequenceClassification.from_pretrained`); the weights are
in safetensors format. Archive size: {size / 1e6:.0f} MB.

## Contents

{chr(10).join(f"- `{p}`" for p in sums if p.startswith(PREFIX))}

Left out of the original checkpoint folder:

{left}

## Training data and licence

Trained on `bias_identification/sensitive_features_dataset.csv`: column descriptions collected from Kaggle
dataset pages (data/README.md). The individual source datasets are not listed, so the licence of the training
data, and therefore of these weights, is not established; the repository's MIT licence covers the code only.
"""


def verify(tar_path, sums):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "detector.yaml").read_text())
    saved = pd.read_csv(REPO_ROOT / "results" / "raw" / "detector" / "external_predictions.csv",
                        float_precision="round_trip")
    from experiments.detector import LegacyCheckpoint
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp, filter="data")
        unpacked = Path(tmp) / PREFIX
        names = sorted(p.name for p in unpacked.iterdir())
        sums_ok = all(sha256(unpacked / f) == sums[f"{PREFIX}/{f}"] for f in KEEP)
        p = LegacyCheckpoint(unpacked, cfg["bert"]["threads"]).predict_proba(saved["description"])
    q = LegacyCheckpoint(REPO_ROOT / PREFIX, cfg["bert"]["threads"]).predict_proba(saved["description"])
    old = saved["p_checkpoint830"].to_numpy(float)
    t = cfg["threshold"]
    return {"files_unpacked": names, "sha256_match": sums_ok, "external_columns": len(saved),
            # the packaging test: the unpacked archive against the original folder, same process
            "archive_vs_original_folder_identical": bool(np.array_equal(p, q)),
            # against the Phase 6 run (another process, after training five models): float noise only
            "archive_vs_saved_identical": bool(np.array_equal(p, old)),
            "archive_vs_saved_max_abs_diff": float(np.abs(p - old).max()),
            "archive_vs_saved_same_decision": bool(np.array_equal(p >= t, old >= t)), "threshold": t}


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    src = REPO_ROOT / PREFIX
    if not src.exists():
        sys.exit(f"{src} not found (the checkpoint is not in git)")
    tar_path, sums = build(src)
    size = tar_path.stat().st_size
    (DIST / f"{NAME}.README.md").write_text(readme(sums, size))
    check = verify(tar_path, sums)
    import torch
    import transformers
    rec = {"script": "scripts/package_checkpoint.py", "started_utc": started, "archive": f"dist/{tar_path.name}",
           "archive_bytes": size, "sha256": sums, "left_out": list(LEFT_OUT), "check": check,
           "python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__}
    out = REPO_ROOT / "results" / "reproduction_check" / "checkpoint830_package.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(check, indent=2))
    ok = check["sha256_match"] and check["archive_vs_original_folder_identical"]
    print(f"{'OK' if ok else 'DIFFERENCES'}: {tar_path} ({size / 1e6:.0f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
