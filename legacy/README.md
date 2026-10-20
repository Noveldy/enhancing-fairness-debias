# legacy/

Frozen copies of the code and results behind the **submitted** manuscript, kept for side-by-side comparison with the revised implementation. Nothing here is maintained, imported or run by the revised code. Do not edit these files.

`MANIFEST.sha256` lists a checksum for every file. Verify with:

```bash
cd legacy && sha256sum -c MANIFEST.sha256
```

## `original/`

The in-scope pipeline code of **this repository at commit `9469d82`** (the last commit before the revision branch), extracted with `git archive`:

- `main.py`, `apis.py`, `requirements.txt`
- `bias_identification/` (data inspection, imbalance check, similarity-based imputation, sensitive-features training data)
- `Models/` (classifier/LSTM training, fairness metrics, BERT detector)
- `bias_mitigation/data_cleaning.py`, `bias_mitigation/dataset_resampling.py`, `bias_mitigation/synthetic_data/`
- `aif360_fairlearn/` (AIF360 / Fairlearn baselines, Table 25)
- `Resampling Algorithms Benchmarking/` (scripts and result CSVs, Tables 5–6)

Not included: the LLM, metric/optimizer recommenders and Scopus analysis, which are out of scope for this paper (decision A3). They are not part of this repository.

## `debias_ops/`

Copied (not moved) from the research repository `../debias_ops` at commit **`cebad9e`**, with a clean working tree, on 2026-10-18. That repository produced the paper's results (docs/AUDIT.md §1.1) and was left untouched. Contents:

| Path | What it is | Paper |
|---|---|---|
| `main.py` | Experiment driver; datasets/scenarios were selected by uncommenting blocks | §3.2 |
| `1st_Journal_2025/` | Dataset fetch, time-series preprocessing, synthetic-data evaluation, descriptive analysis, paired t-tests and their input log | §3.1–3.3 |
| `fairness_evaluation/*.json` | Classifier/fairness reports per dataset, attribute and scenario | Tables 10–12, 22–25 |
| `bias_identification/generated_data_info/*.json` | Cached domain / sensitive-attribute / imbalance outputs used by the experiments | Tables 7–8 |
| `results/` | Pipeline log, BERT training log, time-series results, filled time series | §2.5.1, §3.2.5 |
| `Datasets/adult_debiased.json` | DBIM output for Adult present in the research repo (22,713 rows, 4 income classes; does not match the reports, see AUDIT §5.1) | n/a |

The household-power files are kept here only as a record; the dataset was removed from the revision (decision B4).
