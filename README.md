# enhancing-fairness

Back end of the DEBIAS toolkit and the complete, reproducible evaluation behind the revised manuscript
**"Enhancing Fairness: A Cross-Platform Application for Mitigating Bias in Data Ecosystems"**
(Array, ARRAY-D-26-05264). The Flutter front end lives in a separate repository.

This branch (`main`) holds the revised code, every raw and summary output, and a log that maps each change
to a reviewer comment. The code and results behind the **submitted** manuscript are kept unchanged under
`legacy/` for comparison.

## Quick start

Requirements: Linux, Python 3.12 (tested with 3.12.3), about 3 GB of disk space for the environment.

```bash
git clone https://github.com/Noveldy/enhancing-fairness-debias.git && cd enhancing-fairness && git checkout release
make env       # venv/ from requirements-lock.txt (CPU-only torch)
make test      # test suite, about 1 minute
make verify    # rebuild every table, figure and LaTeX table from the saved raw outputs and
               # compare checksums with the committed files, about 1 minute
```

If `python3.12` on your PATH is not a working Python 3.12 (for example, a stale virtual environment), name it
explicitly: `make env PYTHON=/usr/bin/python3.12`. The same steps also run in Docker (see below).

## Documents

| File | What it is |
|---|---|
| `REVISION_LOG.md` | Every change, why it was made, and the reviewer comment it answers, phase by phase; the design decisions (B1, D1, …) cited in code and configs |
| `docs/AUDIT.md` | Audit of the submitted code and results (leakage, datasets, tuning, security, paper vs. outputs); `docs/audit/` holds its inspection scripts and logs |
| `data/README.md` | Datasets: source, checksum, licence, missing-value policy, detector training data |

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Dataset loaders (checksum check, schema, missing values; no imputation or encoding) |
| `experiments/` | Evaluation code: nested-CV protocol, pipeline, imputer, baselines, metrics, statistics, time series, synthetic data, detector, timing harness, memory guard |
| `run_experiments.py`, `scripts/` | Experiment drivers and the scripts that turn raw outputs into summaries, figures and tables |
| `configs/` | Every configuration: folds, seeds, search ranges (`search_space*.yaml`), methods |
| `results/raw/` | Raw outputs: per-fold predictions, inner-CV scores, selections, run records (seeds, versions, data checksums, hardware, git commit, wall clock) |
| `results/summary/` | Summary tables (CSV), LaTeX tables (`*.tex`, `latex/*.tex`), run records |
| `results/figures/` | Figures (PNG and PDF) |
| `results/hardware.json` | The machine used for the timing results |
| `Datasets/` | Raw UCI files (CC BY 4.0; see `data/README.md`) |
| `bias_identification/`, `bias_mitigation/`, `Models/`, `main.py`, `apis.py`, `service/` | The toolkit itself (pipeline and REST API) |
| `tests/` | Tests: datasets, leakage, imputer, protocol, baselines, statistics, synthetic data, detector, timing, security, storage |
| `legacy/` | Frozen copies of the original code and research outputs (`legacy/README.md`) |

## Reproducing the results

Every reported number comes from a script run that is logged (`run_*.json`, `run_log.txt`). Seeds are fixed in
the configs. `make experiments` reruns everything, then `make summaries` rebuilds the tables and figures. Times below are from the original runs on an Intel i5-14500 (14 cores, 15 GB RAM). The memory
guard (`experiments/memory_guard.py`) lowers parallelism, or stops a run, before the machine runs out of memory.

| Target | Command | Raw outputs | Time | Answers |
|---|---|---|---|---|
| `make data` | `scripts/make_data_summary.py` | `results/summary/data_*.csv` | seconds | R2 data inconsistencies |
| `make main` | `run_experiments.py --config configs/main.yaml` | `results/raw/main/` | 3.8 h | R2 leakage, R1.7 tuning ranges |
| `make timeseries` | `--config configs/timeseries.yaml` | `results/raw/timeseries/` | 3.6 h | Air Quality (Table 9) |
| `make baselines` | `--config configs/baselines.yaml` | `results/raw/baselines/` | 2.1 h | R2 baselines |
| `make ablation` | `--config configs/ablation.yaml` | `results/raw/ablation/` | 0.5 h | R2 ablation |
| `make synthetic` | `--config configs/synthetic.yaml` | `results/raw/synthetic/` | 0.3 h | R2 synthetic data |
| `make detector` | `scripts/detector_validation.py` | `results/raw/detector/` | 1.0 h | R2 detector validation |
| `make cost` | `scripts/cost_run.py --config configs/cost.yaml` | `results/raw/cost/`, `results/hardware.json` | 1.8 h | R1.2, R1.8 time and complexity |

`make summaries` runs, from the raw outputs only:

| Script | Writes |
|---|---|
| `scripts/compare_baselines.py` | `baselines_summary.csv`, `baselines_tests.csv`, `timeseries_summary.csv`, `timeseries_tests.csv` |
| `scripts/fairness_utility.py` | `fairness_utility_*.csv/.tex`, `ablation_*.csv/.tex`, `group_rates_summary.csv`, figures `tradeoff_*`, `ablation_*` |
| `scripts/synthetic_summary.py` | `synthetic_*.csv/.tex`, figures `synthetic_*` |
| `scripts/uncertainty_summary.py` | `uncertainty_ci.csv` (95% CI of every metric), `selection_frequencies.csv`, `selection_stability.csv` |
| `scripts/cost_summary.py` | `cost_*.csv`, figures `cost_methods_*`, `cost_scaling_*` |
| `scripts/make_latex_tables.py` | `results/summary/latex/*.tex` |

### Determinism

The tabular, time-series, synthetic and detector results are deterministic for a given environment and a
given number of parallel workers: seeds are fixed, and the same run repeats exactly. The memory guard chooses
the number of workers from the free memory, and that number sets how many BLAS threads each worker uses. In
one experiment this changes results. In the sensitivity run that keeps Bank's `duration` feature
(`bank_dbim_with_duration`), the inner-CV scores of some candidates move by up to 0.006 (`results/reproduction_check/thread_sensitivity.json`), because the
nearest-neighbour searches of SMOTE, ADASYN, BorderlineSMOTE and NCR break distance ties differently. In
`make check` the selected model and the predictions were unchanged. All other experiments gave identical
results with different numbers of workers (Phase 9 in `REVISION_LOG.md`). Timings (`make cost`) depend on the machine and are not expected to repeat exactly; the timed
runs instead check that they reproduce the stored model choices and accuracies. Figures are written with a
fixed `SOURCE_DATE_EPOCH`, so rebuilt PDFs are byte-identical.

## Checking the results

- `make verify` rebuilds every summary, figure and LaTeX table from `results/raw/`. It then compares checksums with the files present before the rebuild. It rewrites the `*_run.json` records, which hold timestamps.
- `make check` (about 1 h) reruns one slice of every experiment into `/tmp/debias-repro-check` and compares it value by value with the saved outputs. The slices are: outer fold 0 of repeat 0 of every tabular, baseline, ablation and synthetic experiment; one Air Quality fold with the proposed cleaning and one seed; one grouped detector fold and one final model; and the Bank DBIM fit and model selection of the timing run. The report goes to `results/reproduction_check/report.md`. The result of the check run from a clean clone is recorded there and in `REVISION_LOG.md` (Phase 9).

## Docker

```bash
make docker-build                         # image from the committed files only (git archive HEAD)
docker run --rm debias-revision make test
docker run --rm debias-revision make verify
docker run --rm debias-revision make check
```

Give the container at least 8 GB of memory (Docker Desktop: Settings, Resources). With less, the memory
guard stops the experiment smoke tests in `make test`; `make verify` needs about 1 GB. The image pins Python 3.12.3, installs `requirements-lock.txt`, and pre-fetches the detector's base model
(`google-bert/bert-base-uncased`, pinned revision), so it runs offline.

## Detector checkpoint (not in git)

The original detector checkpoints (`bert-sensitive-features-classifier/`, 4.2 GB) are git-ignored. The revised
detector is retrained from `bias_identification/sensitive_features_dataset.csv` by `make detector`. Its five
classifier heads are saved in `results/raw/detector/heads/`. Load them with `torch.load(..., weights_only=True)`.
`checkpoint-830` is the detector the submitted toolkit loads; the checkpoint behind the submitted paper's detector
results (`checkpoint-610`) is not available (`docs/AUDIT.md` §8). Without the folder, `scripts/detector_validation.py`
skips only the "checkpoint-830 (reference)" row. `scripts/package_checkpoint.py` packages it (weights, config,
tokenizer, about 0.4 GB, without the training-resume files) into `dist/checkpoint-830.tar.gz` for a separate
download; unpacking that archive in the repository root restores the folder.

## Running the toolkit's API

```bash
venv/bin/python apis.py        # http://127.0.0.1:5000, debugger off
```

The bias-check route (`/data-bias-api`) also needs the original application's dependencies (`requirements.txt`),
which are not part of the pinned evaluation environment.

### API security

The first version of the API executed user-submitted Python code (value functions), evaluated form fields with
`eval`, served any file by path and ran the Werkzeug debugger on all interfaces (`docs/AUDIT.md` §9, S1–S9).
The revised API:

- **runs no user code.** Value functions are declarative expressions over the index `i` and named numeric
  parameters (arithmetic, comparisons, conditionals and an allow-list of math and random functions). They are
  parsed, checked and evaluated by `service/expressions.py`, with limits on size, depth and output.
- **stores value functions as data.** They are JSON records with a SHA-256 hash, plus an HMAC when
  `DEBIAS_STORE_HMAC_KEY` is set (`service/value_store.py`), and are re-checked on every load.
- **uses storage keys, not file paths.** Keys are validated, and storage is local or MinIO (`service/storage.py`,
  `scripts/minio_connector.py`).
- **has safe network defaults.** It binds to `127.0.0.1`, the debugger is off, the API key is optional, CORS is
  limited to localhost, sizes are limited, and HTML reports are sandboxed.

`tests/test_security.py` and `tests/test_storage.py` (84 tests) send each attack, among them system calls, file and
network access, sandbox escapes, infinite loops, memory bombs, regex bombs, path traversal and tampered stored
functions, and check that it is rejected without side effects.

| Variable | Default | Meaning |
|---|---|---|
| `DEBIAS_HOST`, `DEBIAS_PORT` | `127.0.0.1`, `5000` | bind address |
| `DEBIAS_DEBUG` | `0` | Werkzeug debugger; never enable on a reachable host |
| `DEBIAS_API_KEY` | unset | if set, every request needs `X-API-Key` |
| `DEBIAS_CORS_ORIGINS` | localhost origins | comma-separated origins (regexes allowed) |
| `DEBIAS_STORE_HMAC_KEY` | unset | signs stored value functions; required on load when set |
| `DEBIAS_ENABLE_LEGACY_CODE_EXEC` | `0` | **restores arbitrary code execution** for the old code path; for local use only |
| `DEBIAS_MAX_ROWS`, `DEBIAS_MAX_UPLOAD_MB` | `1000000`, `200` | size limits |
| `DEBIAS_STORAGE_DIR` | `<repo>/storage` | local storage root (git-ignored) |
| `DEBIAS_MINIO_ENDPOINT`, `_ACCESS_KEY`, `_SECRET_KEY`, `_BUCKET`, `_PREFIX`, `_SECURE` | unset, –, –, `debias`, –, `1` | MinIO, used when the endpoint is set |

Not solved here, so deploy accordingly:
- there are no user accounts, only an optional shared key;
- `/data-bias-api` has no per-request time limit;
- without the HMAC key, the hash does not stop an attacker who can write to the store;
- there is no retention policy for uploaded data.

For use by more than one person, run the API behind TLS and a production WSGI server.

## Archiving

`CITATION.cff` and `.zenodo.json` hold the metadata for a Zenodo release; check them before publishing. A GitHub release of this branch, with the Zenodo integration
enabled, archives the repository and assigns a DOI. Before archiving:
- the detector training data come from Kaggle dataset pages whose licences are not listed (`data/README.md`);
- uploading `checkpoint-830` (optional): `venv/bin/python scripts/package_checkpoint.py` writes `dist/checkpoint-830.tar.gz`, its checksums and a README to upload as a separate file.

## License

Code: MIT (`LICENSE`). Datasets in `Datasets/`: CC BY 4.0, by their UCI authors (`data/README.md`).
