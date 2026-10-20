# Phase 0 Audit

**Status:** discovery only. No pipeline code was changed and no experiment was rerun.
**Date:** 2026-10-18 · **Branch:** `revision` (from `main` @ 9469d82)
**Inputs read:** this repo (`enhancing-fairness`), the research repo `../debias_ops` @ cebad9e (clean tree, read-only), the Flutter UI `../debias` (security items only), the manuscript `paper_v1.6.docx` and the reviews `ARRAY-D-26-05264-reviews.pdf`.

## How to read this document

- Code references use `repo/path:line`. `EF/` means this repo and `OPS/` means `../debias_ops`. Except for `main.py`, the shared modules are functionally identical in both repos (see §11).
- Verdicts: **Confirmed**, **Refuted**, **Partly**, or **Cannot determine**.
- Every number below comes from one of two read-only scripts committed with this audit:
  - `docs/audit/inspect_datasets.py` → `docs/audit/dataset_inspection.log`
  - `docs/audit/inspect_results.py` → `docs/audit/results_inspection.log`

  Both are deterministic. Re-run them with the commands in §13.
- "Paper §x / Table y" means `paper_v1.6.docx`.

## Summary

| # | Area | Reviewer comment | Verdict | One line |
|---|------|------------------|---------|----------|
| 1 | Leakage | R2 (evaluation protocol) | **Confirmed**, and broader than the reviewer stated | Imputation, resampler selection *and* resampling all run on the full dataset before the train/test split. The resampler changes the test set (Bank: 1,660 test negatives removed). Imputation uses the target column. AUC is computed on training rows. |
| 2 | Baselines | R2 | **Confirmed** | The headline baselines (85.216% acc, 83.3% AUC) are copied from two papers. The AIF360/Fairlearn runs use a different data source, missing-value policy, split (70/30 vs 80/20) and feature set. |
| 3 | Statistics | R2 | **Confirmed**, and worse than stated | The "paired t-test" pairs the 4-tuple *(mean, std, min, max)*, giving n = 4 and df = 3. It compares complete pipeline vs no-cleaning, not vs the baseline. The script's p-values are 0.20–0.67; the paper reports p = 0.02. |
| 4 | Datasets | R2 (data inconsistencies) | **Confirmed** on all four points | Bank = `bank-full` read through `ucimlrepo`. The 52,124 "missing" values are exactly the UCI "unknown" entries. Table 8's row count is Adult's. Air Quality has 16,701 cells of -200. Household-power appears only in the Discussion. |
| 5 | Tuning | R1.7 | **Partly** (the quoted phrase is not in v1.6) | No hyperparameter search exists anywhere. The "best" resampler is chosen by max CV accuracy on the full data. The BERT checkpoint is chosen on its own test split. The LSTM is early-stopped and scored on the same validation split. |
| 6 | Security | R2 | **Confirmed**, and worse than stated | `exec()` of user code with full builtins, persisted into an imported `.py` file. Also: `eval()` on a form field, arbitrary file read via two GET endpoints, path injection via schema names, `debug=True` on `0.0.0.0`, no authentication. |
| 7 | Synthetic data | R2 | **Confirmed** | Every column is sampled independently (marginals only). Models trained on the synthetic data are at chance or near-constant. |
| 8 | Sensitive-attribute detector | R2 | **Partly** | The reported 95.2% accuracy is measured on the split used to select the checkpoint. The cached detector outputs used in the experiments date from July 2025 and contradict the paper. |
| 9 | Reproducibility | R2 | **Confirmed** | Experiments run by hand-editing `main.py`. Several intermediate files and the trained classifier are missing. Requirements are unpinned. The paper's Table 7/8 and time-series numbers can't be traced to any stored output. |

---

## 1. Repository map

### 1.1 The three code bases

| Repo | Role | Notes |
|------|------|-------|
| `../debias_ops` | Code of record for the paper's results. | Has `Datasets/` (not tracked by git), `fairness_evaluation/*.json` (Tables 10–12, 22–25), `1st_Journal_2025/` (t-tests, dataset fetch, time-series prep, synthetic eval), `results/`, and cached detector outputs. |
| `enhancing-fairness` (this repo) | Public copy for the paper. | No data, no t-test, no result JSONs. Adds the `aif360_fairlearn/` baselines (Table 25), which are **not** in `OPS`. |
| `../debias` | Flutter UI (desktop/web/Android). | Calls the Flask API. Relevant to security only. |

### 1.2 Entry points

| Entry point | What it does |
|-------------|--------------|
| `OPS/main.py:287` | Calls `check_dataset_bias(...)` at import time. Each dataset/scenario is selected by **uncommenting** a block (`OPS/main.py:60-123`). At HEAD every block is commented out, so the file raises `NameError: DATA` as-is. The "no pipeline" scenario is another commented block (`OPS/main.py:128-148`). The "no cleaning" scenario has no code at all; it implies manually commenting out the imputation call (`OPS/main.py:154`) and the output filename (`OPS/main.py:208-209`). |
| `EF/main.py` | Same function without the dataset blocks. `EF/main.py:117-118` no longer calls the time-series imputer: the timer starts and stops with nothing in between. |
| `EF/apis.py` | Flask app with the routes `/data-bias-api`, `/get-value-functions-api`, `/submit-value-function-api`, `/submit-data-schema-api`, `/get-data-schemas-api`, `/generate-synthetic-data-api`, `/get-data-api`, `/get-plot-api`, `/recommend-optimizer-api` and `/recommend-evaluation-metrics-api`. Run with `app.run(host="0.0.0.0", port=5000, debug=True)` (last line). |
| `EF/aif360_fairlearn/*.py` | Standalone baseline scripts for Table 25. |
| `EF/Resampling Algorithms Benchmarking/*.py`, `OPS/benchmarking/*.py` | Resampler timing on `make_classification` data (Tables 5–6, Figs 7–14). |
| `OPS/1st_Journal_2025/*.py` | `fetch_datasets.py`, `timeseries_preprocessing.py`, `synthetic_data_eval.py`, `descriptive analysis.py`, `ttest_adult_bank.py`. |
| `OPS/Models/sensitive_features_text_classifier.py:9` | Trains the BERT detector. Run by hand. |

> **Working-tree note:** on the `revision` branch, the author's 38 uncommitted file deletions (plus a README edit) from `main` are still in the working tree (left untouched, as agreed). With them applied, `EF/apis.py:6-7` imports two deleted modules (`evaluation_recommender`, `optimizer_recommendation`), so **the API cannot start**.

### 1.3 DBIM pipeline stages, in execution order (`EF/main.py:check_dataset_bias`)

| # | Stage | Code | Fitted on |
|---|-------|------|-----------|
| 1 | Similarity-based imputation | `EF/main.py:29` → `EF/bias_identification/datasets.py:176` | **All rows, target column included** |
| 2 | Copy | `EF/main.py:31` | n/a |
| 3 | Type, domain and sensitive-attribute detection (or load from cache) | `EF/main.py:33-43`; `datasets.py:43`, `:55`, `:114` | Column names and descriptions only |
| 4 | Imbalance check | `EF/main.py:48` → `datasets.py:132` | All rows |
| 5 | Classifier on the un-resampled data (overwritten, but **label-encodes `data_copy` in place**) | `EF/main.py:54` → `EF/Models/model_trainer.py:88-98` | All rows |
| 6 | Resampler selection + resampling | `EF/main.py:65` → `EF/bias_mitigation/dataset_resampling.py:28` | **All rows** |
| 7 | Train/test split + RandomForest + fairness metrics | `EF/main.py:71` → `model_trainer.py:108` | Split happens **here, after stages 1–6** |
| 8 | Save "debiased" dataset | `EF/main.py:73` | n/a |

Time-series branch: reindex to a regular frequency → `fill_timeseries_missing_values` (`EF/bias_mitigation/data_cleaning.py:8`, called only in `OPS/main.py:249`) → `train_dummy_forecaster` (`model_trainer.py:164`).

Other components:
- **FSDG (synthetic data):** `EF/bias_mitigation/synthetic_data/synthetic_data_generator.py`, with value functions in `.../value_functions.py`.
- **Fairness metrics:** `EF/Models/model_trainer.py:31-79`.
- **Dataset loading:** `EF/bias_identification/datasets.py:36` (`Datasets/<name>.csv`), API uploads at `EF/apis.py:54-65`, `OPS/1st_Journal_2025/fetch_datasets.py` (UCI via `ucimlrepo`), and `fetch_openml` / UCI zip downloads in `EF/aif360_fairlearn/*`.

---

## 2. Leakage audit (R2: "cleaning, resampling and selection … before cross-validation")

**Verdict: Confirmed.** The leakage is broader than the reviewer described.

1. **Imputation runs on the full dataset, including the target.**
   - `EF/main.py:29` runs `similarity_based_imputation` on all rows before any split.
   - Inside it, every non-numeric column (the target included) is label-encoded (`datasets.py:213`) and mean-filled (`datasets.py:231`).
   - A cosine k-NN is fitted on every column, **target included** (`datasets.py:234-238`), and missing cells are copied from neighbours.
   - So a test row's imputed features depend on its own label and on other rows' labels.
   - The features are also unscaled, so `fnlwgt` (~10⁵) dominates the cosine similarity.
2. **The resampler is selected on all data.**
   - `dataset_resampling.py:47` mean-fills on all rows.
   - `:79` and `:112` call `cross_validate(make_pipeline(sampler, DecisionTree), X, y)` on the full data. Resampling inside each fold is correct, but the folds are the same data later used for evaluation.
   - `cv` defaults to unshuffled 5-fold. Bank is ordered by date, so its folds vary widely (accuracy min 0.21, max 0.85; `results_inspection.log`, dbim_ttest section).
   - Selection: `:162-192` (max accuracy, or F1 if `domain == 'healthcare'`).
3. **The selected resampler is applied to all rows before the split.**
   - `dataset_resampling.py:200` / `:204` call `fit_resample(X, y)`, and `EF/main.py:71` → `model_trainer.py:108` then calls `train_test_split` on the resampled data.
   - Oversampling therefore puts synthetic rows into the test set. Undersampling removes "hard" majority rows from the test set.
   - Concrete evidence (`results_inspection.log`):

     | Report | Test negatives | Test positives |
     |---|---|---|
     | Bank, no pipeline | 7,985 | 1,058 |
     | Bank, complete pipeline (AllKNN) | 6,325 | 1,058 |
     | Adult, no pipeline | 7,431 | 2,338 |
     | Adult, complete pipeline (TomekLinks) | 6,758 | 2,338 |

   - Bank's minority-class precision rises from 0.47 to 0.70 largely because the test set changed.
4. **Other in-sample estimates.**
   - `model_trainer.py:128,160` compute `roc_auc_score(y, model.predict_proba(X))` on **all rows, training rows included**.
   - This is why every report shows AUC ≈ 0.94–0.99, including a model at 0.50 test accuracy (`synthetic_adult_balanced_sex`: accuracy 0.5002, "AUC" 0.98).
   - `model_trainer.py:123,155` run `cross_val_score` on the already resampled X, y.
5. **Encoding** is fitted on all rows, by order of appearance (`model_trainer.py:88-98`). This leaks little information, but it is not train-only. The function also mutates its caller's DataFrame, which is why `data_resampling` receives encoded data at all (`EF/main.py:54` before `:65`).
6. **Sensitive-attribute and domain detection** use only column names and descriptions, so no row-level leakage. The imbalance check (`datasets.py:132`) reads all rows; it only gates whether resampling runs.
7. **Baseline scripts** (`EF/aif360_fairlearn/`) split first, then fit preprocessing on train (`fairlearn_adult.py:114`). One exception: `aif360_adult_DIR.py:405` and `aif360_bank.py:717` call `DisparateImpactRemover.fit_transform` **on the test set**. DIR is label-free, but it then repairs test rows using test-set quantiles.
8. **Time series:** `OPS/1st_Journal_2025/timeseries_preprocessing.py:22` mean-imputes the household data using the full series before subsetting.

---

## 3. Baseline audit (R2: "baselines taken from external sources")

**Verdict: Confirmed.**

| Paper claim | Source | Split / preprocessing | Comparable to DBIM run? |
|---|---|---|---|
| Adult RF acc 85.216%, precision 80.246% (Paper §3.2.3, Table 7) | Literature, ref [82] | Unknown (not in code) | No |
| Bank DT AUC 83.3% (Paper §3.2.4, Table 8) | Literature, ref [83] | Unknown | No |
| AIF360 DIR, Adult (Table 25) | Own run: `EF/aif360_fairlearn/aif360_adult_DIR.py` | OpenML `adult` v2 (`:44`), `?` → NaN then **dropna** (`:71,76`), 70/30 split, seed 42 (`:20,217`), one-hot fitted on train, RF; DIR fitted on test (`:405`) | No (see below) |
| Fairlearn ThresholdOptimizer, Adult (Table 25) | Own run: `EF/aif360_fairlearn/fairlearn_adult.py` | OpenML v2, dropna (`:39`), labels normalised (`:48-55`), **sex and race removed from features** (`:88`), 70/30, seed 42, DP constraint, fitted on train (`:227-238`) | No |
| AIF360 / Fairlearn, Bank (Table 25) | Own runs: `aif360_bank.py`, `fairlearn_bank.py` | UCI `bank.zip` → `bank-full.csv` (`fairlearn_bank.py:35`), "unknown" **kept** as a category, `marital` removed from features (`:78`), 70/30, DT | No |
| DBIM (Tables 7, 8, 10–12, 25) | Own run, `OPS/main.py` | `ucimlrepo` CSVs (`OPS/1st_Journal_2025/fetch_datasets.py`), similarity imputation, resampling **before** an 80/20 split, label encoding, protected attribute **kept** as a feature, RF only (`model_trainer.py:113`; the DT line is commented out) | n/a |

Differences between DBIM and the baselines:
- Data source: `ucimlrepo` vs OpenML/UCI zip.
- Missing-value policy: impute vs dropna vs keep "unknown".
- Test size: 20% vs 30%.
- Feature set: protected attribute kept vs dropped.
- Classifier: the paper says Bank used DT, but the code at HEAD trains RF for every report. Which classifier produced the Bank reports **cannot be determined**. The revision uses DT for Bank, as the paper states (decision C2).

The baseline scripts write CSVs to `results/fairlearn_*.csv` etc., and none of them exist in either repo. Table 25 values therefore can't be traced to a stored file. The DBIM rows of Table 25 do match the stored reports: Adult/race DP 0.158, TPR 0.177, FPR 0.084; Bank TPR 0.017, FPR 0.019.

---

## 4. Statistics audit (R2: "paired t-tests reported without paired observations")

**Verdict: Confirmed, and worse than stated.**

- **Code:** `OPS/1st_Journal_2025/ttest_adult_bank.py:1-74`.
- **What is paired:** each array is the 4-tuple *(mean, std, min, max)* of one `cross_validate` run, copied by hand from the log `OPS/1st_Journal_2025/dbim_ttest.txt`. Example: `ttest_adult_bank.py:4-5` are the accuracy tuples of "no cleaning" and "complete pipeline". `ttest_rel` therefore pairs a mean with a mean, a std with a std, and so on, giving n = 4 and df = 3. This is not a valid test of anything.
- **What is compared:** "complete pipeline" vs "no cleaning", i.e. both arms use resampling. The paper (§3.2.3, §3.2.4) says the test compares "the proposed approach and the baseline one" across "10 independent runs".
- **Number of runs:** there are no 10 runs. Every seed is fixed (`random_state=42`, `model_trainer.py:108,113,121`) and there is no seed loop anywhere.
- **Reported vs actual p-values:** the paper reports p = 0.02 for both datasets. Running the script unchanged gives (`results_inspection.log`):
  - Adult accuracy: t = 1.30, p = 0.285
  - Adult precision: t = 1.65, p = 0.197
  - Bank accuracy: t = 0.65, p = 0.564
  - Bank precision: t = 0.54, p = 0.627
  - Every metric: p between 0.197 and 0.666

  **No stored script or output produces p = 0.02.**
- The source metrics are the resampler-selection CV scores (decision tree, selection stage), not the RF evaluation reported in Tables 7–8.

---

## 5. Dataset audit (R2: data inconsistencies)

All counts are from `docs/audit/dataset_inspection.log`. Files are in `OPS/Datasets/`, which is **not under version control**, so the SHA-256 values below are the only record of which bytes were used.

### 5.1 Adult

- **Source:** UCI id 2 via `ucimlrepo` (`fetch_datasets.py:5`). Train and test parts are merged.
- **sha256:** `7bf09a74…29f4`
- **Shape:** 48,842 × 15, which matches the paper.
- **Missing values:** 2,203 NaN (workclass 963, occupation 966, native-country 274). This **matches** the paper's "2203". There are also **4,262 literal `"?"` strings** (workclass 1,836, occupation 1,843, native-country 583). `pd.read_csv` does not treat these as missing, so the pipeline kept `"?"` as a real category.
- **Target `income` has four labels:** `<=50K` 24,720; `<=50K.` 12,435; `>50K` 7,841; `>50K.` 3,846. The trailing dot comes from the UCI test file.
  - The stored fairness reports are binary (classes 0.0/1.0; 2,338 positive test rows = 20% of 7,841 + 3,846). So the file that produced the reports **was not this file**, or the labels were merged somewhere not found in either repo.
  - `adult_debiased.json` (the pipeline output in `Datasets/`) has 22,713 rows and **4 income classes**, which matches neither the paper nor the reports.
  - **Cannot determine** how the binary labels were produced; the old Adult numbers are treated as not reproducible (decision B3).
- **Protected attributes:**
  - sex: Female 33.15%.
  - race: "Other" 0.83%.
  - age: 5 equal-width bins, top bin (75.4, 90] = 0.76%.

  These match Paper Table 7.
- **Duplicates:** 29 rows.

### 5.2 Bank Marketing

- **Source:** UCI id 222 via `ucimlrepo` (`fetch_datasets.py:19`). This is **`bank-full`** (45,211 × 17, including `balance`).
- **sha256:** `592cda3d…dc13`
- **Column renaming:** `ucimlrepo` renames `day` (day of month, values 1–31) to **`day_of_week`**.
- **Schema mixing in Paper Table 3 (confirmed):**
  - `balance` and the 45,211 row count come from `bank-full`.
  - Table 3 also lists `day_of_week` as "last contact day of the week". The data holds days 1–31, and that meaning belongs to `bank-additional`.
  - Education values `basic.4y … university degree` are from `bank-additional`. This file has `primary / secondary / tertiary`.
  - `poutcome = nonexistent` is from `bank-additional`. This file has `failure / other / success`.
  - The table has a typo: "compaign".
- **"52,124 missing values" (Paper §3.2.4, Table 8):** this is exactly the NaN count after `ucimlrepo` converts UCI's `"unknown"` to NaN:

  | Column | NaN |
  |---|---|
  | poutcome | 36,959 |
  | contact | 13,020 |
  | education | 1,857 |
  | job | 288 |
  | **Total** | **52,124** |

  These are not measurement gaps. For example, `poutcome` is "unknown" for clients who were never contacted before (`pdays = -1`). The pipeline imputed them with neighbours' values.
- **Paper Table 8, "Number of Rows" = 48,842:** that is Adult's count. **Confirmed error;** the correct value is 45,211.
- **Feature descriptions are misaligned.** The descriptions used for detection (`OPS/main.py:73-74`) contain a `balance` description at index 5, but `balance` is missing from `FEATURES`. Every later feature gets the previous feature's description, e.g. `housing` gets "Average yearly balance".
- **Marital:** divorced 11.52%. **y:** yes 11.70%. Both match Table 8.

### 5.3 Air Quality

- **Source:** UCI id 360 via `ucimlrepo` (`fetch_datasets.py:33`).
- **sha256:** `d8196847…157d`
- **Shape:** 9,357 × 15. The paper says 9,358.
- **Missing-value code:** **16,701 cells equal -200**. Per column: NMHC(GT) 8,443 (90%), CO(GT) 1,683, NO2(GT) 1,642, NOx(GT) 1,639, and 366 in each other sensor column. There are 0 NaN.
- **The paper's "did not contain any missing values" (§3.2.5) is refuted.**
- No code anywhere converts -200 to NaN. The "cleaned" series the LSTM was trained on still contains **14,290 cells of -200** (`results_inspection.log`, time-series section).
- The paper's artificial gap (08/11/2004–31/12/2004) is created by the commented lines `OPS/1st_Journal_2025/timeseries_preprocessing.py:35-39`. The intermediate files `air_quality_modified*.csv` are **missing**.

### 5.4 Individual household electric power consumption

- **Source:** UCI id 235 via `ucimlrepo` (`fetch_datasets.py:43`).
- **sha256:** `d3a1571b…a7f9a7`
- **Shape:** 2,075,259 × 9.
- **Missing values:** `"?"` in 25,979 rows (every measurement column).
- **Processing** (`timeseries_preprocessing.py:6-40`):
  1. `"?"` → NaN.
  2. **Mean-impute over the full series** (`:22`).
  3. Keep 2009-06-15 07:35 onwards, first 43,800 rows (`:26-28`).
  4. Delete 2009-06-18…25 to create a gap (`:37`).
- **Code and outputs that exist:**
  - Cached detector output `OPS/bias_identification/generated_data_info/individual_household_electric_power_consumption_with_missing.json` (domain "energy").
  - Filled output in `OPS/results/`.
  - Value functions `Global_active_power` / `Global_reactive_power` in `EF/bias_mitigation/synthetic_data/value_functions.py:174-190`.
  - A synthetic-vs-real evaluation, `OPS/1st_Journal_2025/synthetic_data_eval.py` (its last active block).
- **Paper:** mentioned **only in the Discussion** (MAE 0.02, SMAPE 81.86). Absent from Methods and Results. **Confirmed.**
- Neither number appears in any stored output. `OPS/results/cleaned_timeseries_experiments.txt` contains only the Delhi climate dataset.
- The intermediate files are missing.

### 5.5 Sensitive-features training data

`OPS/bias_identification/sensitive_features_dataset.csv`:
- sha256 `2248d5cf…b247`; 825 rows; 425 Sensitive / 400 Not Sensitive. This matches the paper.
- **6 duplicated texts.** These can fall on both sides of the 80/20 split.

---

## 6. Tuning audit (R1.7)

**Verdict: Partly.**

R1.7 quotes "until the model obtains the maximum accuracy". **That phrase does not appear in `paper_v1.6.docx`.** The closest text is Paper §2.5.1: "the dataset that enabled the algorithm to achieve the best performance possible". It may have been in the submitted version. What the code actually does:

| Model | What is tuned | Selection data | Search range |
|---|---|---|---|
| Resampler choice (DBIM) | 1 of 8–10 samplers | Max mean CV **accuracy** (F1 if domain = healthcare) over 5 unshuffled folds of the **full** dataset (`dataset_resampling.py:79,112,162-192`). The winner is applied to all data and then evaluated on a split of that data. | Fixed list (`:30-35`). SVMSMOTE and CNN are dropped if rows ≥ 20,000 (`:37`). The paper says 10,000 for CNN (§3.2.2). |
| RandomForest / DecisionTree (DBIM, baselines) | Nothing (sklearn defaults) | n/a | n/a |
| LSTM (time series) | Nothing: a single lr 1e-6, Adam, 100 epochs (`model_trainer.py:166-198`) | EarlyStopping on `val_loss` (`:170-175`) uses the **same** last-20% split (`:186`) that MAE/SMAPE are computed on (`:221-225`). The forecast is `predict(n=12)` (`:215`), so the scores cover 12 time steps. | None |
| BERT sensitive-feature detector | Checkpoint epoch (1–10) | `load_best_model_at_end=True` with `eval_dataset = test` (`OPS/Models/sensitive_features_text_classifier.py:65-85`). The best epoch is chosen on the test split, then the paper's 95.2% / 97.9% is measured on that same split (`:90-98`; `OPS/results/sensitive_features_classifier_results.txt`). Fixed lr 2e-4, batch 8, encoder frozen except the pooler (`:18-25`). | None |
| Domain classifier | None (zero-shot DeBERTa) | n/a | 22 labels in code (`EF/bias_identification/datasets.py:33`); the paper says 23, adding "physics" |

---

## 7. Synthetic data (R2)

**Verdict: Confirmed.**

- `AdvancedSyntheticDataGenerator.generate` (`synthetic_data_generator.py:210-212`) loops over the schema and calls `generate_column` (`:97`) **once per column, independently**.
  - There is no joint, target-conditional or protected-group-conditional sampling.
  - "Balanced" mode (`:31-36`, `:53-70`) replaces each column's weights with uniform ones, again per column.
  - The RNGs are unseeded globals (`random`, `np.random`).
- **Outcomes** (`results_inspection.log`):
  - `synthetic_adult_sex`: the model predicts the positive class for 0.1% of rows (minority recall 0.002). DP difference 0.00004 because it almost never predicts positive.
  - `synthetic_adult_balanced_sex`: accuracy **0.500**, macro F1 0.50, so chance level.
  - `synthetic_bank_balanced_marital`: TPR ≈ FPR in every group (0.178 vs 0.170; 0.167 vs 0.169; 0.161 vs 0.159), so the model **cannot separate the classes**. The paper (§3.3.5) says it "maintained meaningful predictive behavior", which this contradicts. Minority precision is 0.14.
  - `synthetic_adult_sex` and `synthetic_adult_race` are the same model on the same data, with identical metrics.
- **Fidelity code** (`OPS/1st_Journal_2025/synthetic_data_eval.py`): per-column KS and Wasserstein, plus the sum of absolute frequency differences. No joint, conditional, subgroup, utility (TSTR) or privacy metric exists. The synthetic files the paper compared (Adult, Bank) are not stored.

---

## 8. Sensitive-attribute detection (R2)

**Verdict: Partly.**

- **Methods** (`EF/bias_identification/datasets.py:43-70`):
  - (a) If descriptions are given, a BERT classifier labels each description (`:55` → `Models/sensitive_features_text_classifier.py:102`).
  - (b) Otherwise, substring matching against 16 legal protected-class words (`:29-31`). This matches e.g. "age" inside "percentage" or "language".
- **Trained model:** `bert-sensitive-features-classifier/` is gitignored. `OPS` loads `checkpoint-610`, `EF` loads `checkpoint-830`.
  - *Correction (Phase 1, 2026-10-18):* the Phase 0 text said this folder was missing. It is missing in `OPS`, but `EF` has a local copy (4.2 GB, not tracked) with checkpoints 83–830 from a training run dated 2026-08-03, i.e. one checkpoint per epoch for 10 epochs. `checkpoint-610`, which the research code loads, is not among them, so the checkpoint behind the paper's results is still unavailable. The decision to retrain (D1) is unaffected.
- **Results actually used in the experiments are cached** (`OPS/bias_identification/generated_data_info/*.json`, last committed July 2025). `main.py` loads them instead of re-running detection (`EF/main.py:33-38`). The cache disagrees with the paper:

  | Dataset | Field | Cached value | Paper |
  |---|---|---|---|
  | Adult | domain | **"education"** | "finance" (§3.2.3, Table 7) |
  | Adult | protected attributes | age, **workclass**, marital-status, **occupation**, relationship, race, sex | 5 attributes |
  | Air Quality | domain | **"transportation"** | "chemistry" (§3.2.5) |

  For Adult the domain error does not change the selection metric, since only "healthcare" switches to F1.
- **Evaluation:** a single 80/20 split, seed 0, with checkpoint selection on that split (§6). There is no external validation and no per-attribute or proxy evaluation.

---

## 9. Security audit (R2: user-defined functions)

**Verdict: Confirmed, and worse than stated.** Every place where user-supplied code or paths are stored, loaded or executed:

| # | Location | Issue |
|---|----------|-------|
| S1 | `EF/apis.py:133-201` (`/submit-value-function-api`) | Runs `exec(code, context)` (`:159`) on the JSON body. `context` has no `__builtins__` key, so Python injects the full builtins: `__import__('os').system(...)` works. The code runs **before** any validation (the `_factory`, docstring and `_params` checks come after), and `factory({})(5)` is then called (`:186-187`). |
| S2 | `EF/apis.py:193-194` | After validation the code is **appended to `bias_mitigation/synthetic_data/value_functions.py`**, which `EF/apis.py:4` and `synthetic_data_generator.py:14-16` import on every start. The code therefore persists and **runs on every server start** for all users. No hash, signature or review step exists. The paper (§2.5.2) says these files are stored in MinIO; there is **no MinIO code** in any repo. |
| S3 | `EF/apis.py:68` | `eval(request.form['detect_outliers'])`: code execution through a form field. |
| S4 | `EF/apis.py:389-410` (`/get-data-api`), `:412-425` (`/get-plot-api`) | `path` query parameter is used unchecked: **arbitrary file read** (`send_file` of any path; JSON read of any path). The Flutter UI calls both (`debias/lib/utilities/api_service.dart:58`, `debias/lib/views/html_viewer_screen.dart:30`). |
| S5 | `EF/apis.py:205-272`, `:357-368` | The schema `name` is any string and is interpolated into `Datasets/{timestamp}_{schema_name}_...json` and `plots/...`: **path traversal on write**. |
| S6 | `EF/apis.py` (last line) | `app.run(host="0.0.0.0", debug=True)`: the Werkzeug debugger and reloader are reachable from the network. |
| S7 | All routes | No authentication. CORS is open (`@cross_origin()` with defaults). |
| S8 | `debias/lib/utilities/api_service.dart:102-105` | The UI builds Python source by string interpolation of user fields (`name`, `doc`, function body) and posts it to S1. |
| S9 | `EF/apis.py:49-50` | Uploads are saved with `secure_filename`, so this one is OK. Filenames can still collide across users (no per-user storage), which contradicts the paper's "only available to the user that requested them" (§2.5.2). |

---

## 10. Paper numbers vs stored outputs

From `results_inspection.log`:

| Paper item | Stored output | Match? |
|---|---|---|
| Table 7 (Adult): acc 86%, precision **95%**, AUC **91%**, recall 86%, F1 86% | `adult_sex_complete_pipeline`: acc 0.857, weighted precision 0.853, in-sample AUC 0.990, F1 0.854 | Accuracy yes. **Precision and AUC: no stored output gives 95% or 91%.** |
| Table 8 (Bank): AUC 84%, acc 83%, precision 87%, recall 83%, F1 85% | `bank_marketing_marital_complete_pipeline`: acc 0.916, precision 0.916, AUC 0.966 | **No.** None of the Table 8 numbers matches. They may come from an earlier run (cannot determine). |
| Tables 10–11 (Adult DP/TPR/FPR by group) | `adult_{sex,race}_*` | Mostly yes, within rounding. Exceptions: Table 10 scenario i FPR group 1 = 0.01 vs 0.022; execution times 60/91 s vs 70/112 s. |
| Table 12 (Bank) | `bank_marketing_marital_*` | Rates match, except scenario i FPR group 0 = **0.59** vs 0.0599 (typo). Times 10/61/92 s vs 1.4/71/123 s. |
| Table 25 DBIM rows | complete-pipeline reports | Yes |
| Table 25 FSDG rows | `synthetic_*` reports | Partly. Adult/race (0.02, 0.02, 0.04) does not match `synthetic_adult_race` (0.012, 0.05, 0.0015); Bank matches only up to ordering. Needs author check. |
| Table 9 (Air Quality MAE/SMAPE) and Discussion household numbers | none | **No stored output.** |
| p = 0.02 (§3.2.3, §3.2.4) | `ttest_adult_bank.py` → p = 0.20–0.67 | **No.** |

---

## 11. Public repo vs research repo

`diff` of the shared modules shows comment-only differences in `dataset_resampling.py`, `data_cleaning.py`, `datasets.py` and `synthetic_data_generator.py`. Real differences:

- **`main.py`:** `EF` drops the dataset blocks and the time-series imputer call.
- **`sensitive_features_text_classifier.py:104`:** `checkpoint-830` vs `checkpoint-610`.
- **`apis.py`:** `EF` drops `check_algorithm_bias`.
- **`aif360_fairlearn/`:** exists only in `EF`.
- **Only in `OPS`:** t-tests, results, cached detector outputs, journal scripts.

---

## 12. Environment and reproducibility facts

- `requirements.txt` (both repos) is **unpinned** except plotly, transformers, onnxmltools and fairlearn. The library versions behind the published numbers are unknown.
- The project `.venv` is **broken**: `.venv/bin/python3 → /bin/python3`, which is now Python 3.14, while the packages are installed for 3.12. The audit scripts were run with `/usr/bin/python3.12` and `PYTHONPATH=.venv/lib/python3.12/site-packages` (pandas 3.0.5, numpy 2.4.6, scipy 1.18.0, scikit-learn 1.9.0).
- `OPS/Datasets/` is gitignored, so only the checksums above identify the data.
- **Missing artifacts:**
  - `air_quality_modified*.csv`, `individual_household_electric_power_consumption_{modified,with_missing}.csv`
  - `synthetic_*` CSVs
  - `bank_marketing_debiased.json`
  - the trained BERT checkpoint used by the research code (`checkpoint-610`); see the §8 correction
  - all baseline result CSVs

---

## 13. Re-running this audit

```bash
PY="env PYTHONPATH=.venv/lib/python3.12/site-packages /usr/bin/python3.12"
$PY docs/audit/inspect_datasets.py ../debias_ops/Datasets ../debias_ops/bias_identification/sensitive_features_dataset.csv > docs/audit/dataset_inspection.log
$PY docs/audit/inspect_results.py  ../debias_ops > docs/audit/results_inspection.log
```

Apart from the timestamp line, the output is deterministic. `inspect_results.py` runs the author's `ttest_adult_bank.py` unchanged.
