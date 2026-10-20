# Datasets

The revised experiments use three UCI datasets. The raw files live in `Datasets/` exactly as fetched with `ucimlrepo` (`legacy/debias_ops/1st_Journal_2025/fetch_datasets.py`). They are never edited; `data/loaders.py` checks each file's SHA-256 before use and applies the schema and missing-value policy below.

The loaders do **not** impute, scale or encode. Those steps are fitted on training folds only, inside the evaluation pipeline (Phase 2).

- **`full` variant:** true missing values stay as `NaN`, to be filled by the cleaning algorithm (or another imputer) fitted on training data.
- **`baseline` variant:** rows with true missing values are dropped (author decision B2).

Regenerate the summary tables and run the checks with (environment: `requirements-dev.txt`, Python 3.12):

```bash
venv/bin/python scripts/make_data_summary.py   # -> results/summary/data_summary.csv, data_missingness.csv, data_summary_run.json
venv/bin/python -m pytest tests/test_datasets.py
```

All numbers below come from `results/summary/data_summary.csv` and `data_missingness.csv`.

| | Adult | Bank Marketing | Air Quality |
|---|---|---|---|
| Source | [UCI id 2](https://archive.ics.uci.edu/dataset/2/adult) | [UCI id 222](https://archive.ics.uci.edu/dataset/222/bank+marketing) | [UCI id 360](https://archive.ics.uci.edu/dataset/360/air+quality) |
| Variant | `adult.data` + `adult.test` merged | **bank-full** | single file |
| Raw file | `Datasets/adult.csv` | `Datasets/bank_marketing.csv` | `Datasets/air_quality.csv` |
| SHA-256 | `7bf09a7484d84f92c7e69582078bb01b2f7c4862c8fb27990c0dd848a8a029f4` | `592cda3d5f6b704f0ce23477e8375acf2301460efe76cd192e2a789325ccdc13` | `d8196847f231fc141ba16c1376aeb90d747a95353203e20ca3b4ae135438157d` |
| Rows (full) | 48,842 | 45,211 | 9,357 hourly rows, 2004-03-10 18:00 → 2005-04-04 14:00 |
| Rows (baseline) | 45,222 | 43,193 | n/a (time series) |
| Features | 14 | 16 | 12 sensor series + `datetime` |
| Target | `income` (1 = `>50K`, 23.93%) | `y` (1 = `yes`, 11.70%) | none (forecasting) |
| Protected attributes in fairness evaluation | `sex`, `race` | `marital` (`age` is also flagged as protected) | none |

## Adult

- **Columns:** `age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income`
- **Missing values:**
  - The raw file marks missing values two ways: empty cells (2,203) and the literal string `"?"` (4,262). Both are treated as missing.
  - That gives **6,465 missing cells in 3,620 rows**: `workclass` 2,799, `occupation` 2,809, `native-country` 857.
  - The original pipeline imputed only the 2,203 empty cells and kept `"?"` as a category.
- **Target:** the raw file has four labels. `<=50K` and `>50K` come from `adult.data`; the same two with a trailing `.` come from `adult.test`. They are merged into 0/1.
- **Duplicates:** 29 duplicate rows are kept, as in the UCI release.

## Bank Marketing (bank-full)

- **Columns:** `age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, y`
- **Single variant.** This is `bank-full`; the `bank-additional` columns (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) do not exist here.
  - `ucimlrepo` renames UCI's `day` to `day_of_week`, but the values are **day of month (1–31)**. The loader renames it back to `day`.
  - `education` takes the bank-full values `primary / secondary / tertiary`.
  - `poutcome` takes `failure / other / success / unknown`.
- **Missing values:** `ucimlrepo` writes UCI's `"unknown"` as empty cells, 52,124 in total. These are the "52,124 missing values" in the submitted paper. Most of them are not missing data:
  - `poutcome` "unknown" (36,959) means *not previously contacted*: 36,954 of these rows have `pdays = -1`. Kept as the category `"unknown"`.
  - `contact` "unknown" (13,020) is kept as the category `"unknown"`.
  - `job` (288) and `education` (1,857) "unknown" are **true missing values**: 2,145 cells in 2,018 rows.
- **`balance`:** contains 3 legitimate values of -200. This dataset has no missing-value code, so they are kept unchanged.
- **`duration`:** UCI notes that call duration is only known after the call, so it strongly predicts the outcome and should be excluded from realistic predictive models. It is excluded from the main experiments; one run with it is kept for comparison (decision F1).

## Air Quality

- **Columns:** `datetime, CO(GT), PT08.S1(CO), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH`
- **Datetime:** `Date` (written as m/d/Y by `ucimlrepo`) and `Time` are combined into `datetime`. The index is checked to be a regular hourly series with no duplicates.
- **Missing values:**
  - The documented missing-value code **-200** is converted to `NaN` before any other step. The raw file has 16,701 such cells and no `NaN`.
  - `NMHC(GT)` is dropped: 8,443 of 9,357 values (90.2%) are missing (author decision B5).
  - The remaining **8,258 missing cells in 2,416 rows** are: `CO(GT)` 1,683, `NOx(GT)` 1,639, `NO2(GT)` 1,642, and 366 in each other column.
  - They are filled later by the compared methods (mean, linear and spline interpolation, and the proposed algorithm), fitted on training data only.

## Removed: individual household electric power consumption

Removed from the revision (author decision B4). Its outputs from the original experiments are kept only as a record in `legacy/debias_ops/`.

## Licensing and citations

Checked on the UCI dataset pages on 2026-10-20: all four datasets are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. They may be redistributed in this repository and its archive, provided they are cited:

| Dataset | Citation (as recommended by UCI) |
|---|---|
| Adult | Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 |
| Bank Marketing | Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 |
| Air Quality | Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F |
| Statlog (German Credit) | Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 |

The repository's MIT license (`LICENSE`) covers the code, not these files.

## Sensitive-attribute detector training data

- **File:** `bias_identification/sensitive_features_dataset.csv` (unchanged since the original repository). sha256 `2248d5cf…b247` (`docs/AUDIT.md` §5.5). 825 short column descriptions, labelled `Sensitive` (425) or `Not Sensitive` (400). 9 exact duplicates are removed before training (816 used).
- **Provenance:** column descriptions collected from the documentation of public datasets on Kaggle (confirmed by the author, 2026-10-20). The individual Kaggle datasets are not listed. Each Kaggle dataset has its own licence, so this file is not covered by the repository's MIT licence or by CC BY 4.0.
- **Use:** training and internal validation of the detector (Phase 6); `results/raw/detector/training_descriptions_groups.csv` lists each description with its paraphrase group.

## German Credit (Phase 6 only)

- **Source:** UCI Statlog (German Credit Data), id 144, `german.data` + `german.doc`, downloaded 2026-10-20 from https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/ into `Datasets/german_credit/`.
- **sha256:** `german.data` b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871; `german.doc` 69b92cb551111bc37098698288849c96f0fd87ce81a2e54451ff39aa1e53fe49.
- **Shape:** 1,000 rows, 20 attributes (7 numeric, 13 categorical with UCI codes such as A93) + target `credit_risk` (1 = good, 2 = bad). No missing values.
- **Column names / descriptions:** the attribute titles of `german.doc` (snake_case / verbatim), `data/loaders.py:GERMAN_DESCRIPTIONS`.
- **Protected (Phase 6 ground truth):** `personal_status_and_sex`, `age_in_years`, `foreign_worker`.
- **Use:** only as the external validation set of the sensitive-attribute detector (author decision D2); not used to develop it.
