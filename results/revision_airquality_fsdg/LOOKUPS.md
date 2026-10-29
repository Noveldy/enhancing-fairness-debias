# Part B: read-only lookups

Nothing was rerun or edited. Values that are not stored as such were computed in seconds from stored outputs by
`lookups_compute.py` (locked environment, Docker image `debias-revision`). Its output is in
`lookups_computed.md` and `lookups_computed.json` (with the sha256 of every input), and
`lookups_dbim_vs_none_tests.csv`. "Computed" marks those values; all others are copied from the file named.
All paths are relative to the repository root; `S/` = `results/summary/`.

## 1. Bank ECE: no mitigation (stage 1) vs DBIM (stage 3)

Source: `S/ablation_tests.csv`, rows `bank_protected_kept` / `bank_protected_dropped`, `test_rows = complete`,
`metric = ece`, comparison `3 cleaning + resampling (DBIM) - 1 original data (complete rows)`. Corrected resampled
t (Nadeau–Bengio), 25 outer folds, df 24. The Holm family is the 3 stage comparisons of the ablation.

| Bank | ECE no mitigation | ECE DBIM | Difference [95% CI] | t (df 24) | p | Holm p |
|---|---|---|---|---|---|---|
| protected kept | 0.00866 | 0.00953 | +0.00088 [−0.00384, 0.00559] | 0.384 | 0.704 | 0.921 |
| protected dropped | 0.00830 | 0.00936 | +0.00106 [−0.00387, 0.00599] | 0.444 | 0.661 | 0.961 |

No significant difference. (For Adult, DBIM's ECE is higher: +0.0073, Holm p = 0.003 kept; +0.0092, Holm p = 0.0003 dropped.)

## 2. t statistics (df 24), DBIM vs no mitigation (Tables 7 and 9)

Sources: ROC AUC, accuracy, balanced accuracy and F1 from `S/baselines_tests.csv` (comparison `dbim - none`);
ECE from `S/ablation_tests.csv` (stage 3 − stage 1, complete test rows), because `baselines_tests.csv` has no ECE.
Both files compare the same predictions on the same complete test rows: the differences are identical. The t
values differ only around the 6th decimal (computed: largest |Δt| = 8.6e-6 for these metrics), because the two
files use a slightly different test/train ratio in the correction (0.2500006 or 0.2500005 vs 0.2500000).
Sign: DBIM − no mitigation.
Full table with CI, p and Holm p: `lookups_dbim_vs_none_tests.csv`.

| Metric | Adult, kept | Adult, dropped | Bank, kept | Bank, dropped |
|---|---|---|---|---|
| ROC AUC | 0.105 | −0.933 | −0.706 | −0.478 |
| Accuracy | −1.012 | −0.990 | 0.284 | −0.005 |
| Balanced accuracy | 7.688 | 2.094 | 1.218 | 1.573 |
| F1 | 5.353 | 1.679 | 1.286 | 1.634 |
| ECE | 3.578 | 4.685 | 0.384 | 0.444 |

"kept"/"dropped" = protected attributes used as features or not. `docs/COMPARISON_OLD_VS_NEW.md` uses "kept".
The Holm families differ: 12 comparisons (Adult) or 8 (Bank) per metric in `baselines_tests.csv` (DBIM vs every
baseline), but 3 for ECE in `ablation_tests.csv`. A table that mixes these rows should say so.

## 3. Do the Wilcoxon tests agree with the corrected t-tests on every significant comparison?

**Yes.** Every comparison that is significant by the primary test (Holm p < 0.05) is also significant by the
Wilcoxon signed-rank test (Holm p < 0.05), in all four test files (computed):

| File | Primary test | Comparisons | Significant (primary) | Also significant (Wilcoxon) | Significant by Wilcoxon only |
|---|---|---|---|---|---|
| `S/baselines_tests.csv` | corrected t | 288 | 136 | 136 | 65 |
| `S/ablation_tests.csv` | corrected t | 208 | 55 | 55 | 51 |
| `S/synthetic_utility_tests.csv` | corrected t | 156 | 119 | 119 | 20 |
| `S/timeseries_tests.csv` | paired t over 15 runs (not corrected; the file notes the p-values are optimistic) | 16 | 9 | 9 | 0 |

The reverse does not hold. The Wilcoxon test treats the 25 outer folds as independent and has no correction for
their overlapping training sets, so it flags 136 further comparisons. For DBIM vs no mitigation, 7 rows are
significant by Wilcoxon only: Adult (dropped) accuracy, balanced accuracy and F1; Bank (kept and dropped) balanced
accuracy and F1. Example: Bank balanced accuracy (kept), corrected-t Holm p = 1.000, Wilcoxon Holm p = 0.015.
The text should therefore report significance from the corrected t-test and the Wilcoxon only as a secondary check.

## 4. Synthetic data: pairwise dependence and the real-vs-real KS reference

Pairwise-dependence error (mean |Δ association| over column pairs, mean ± SD over 25 folds;
`S/synthetic_fidelity.csv`, `pairwise_dependence_mae_mean/_sd`):

| | Real train vs real test (floor) | FSDG (marginal sampler) | Gaussian copula | Independent marginals |
|---|---|---|---|---|
| Adult | 0.010 ± 0.001 | 0.153 ± 0.002 | 0.059 ± 0.001 | 0.153 ± 0.002 |
| Bank | 0.011 ± 0.001 | 0.116 ± 0.002 | 0.069 ± 0.002 | 0.116 ± 0.002 |

Real-vs-real KS reference: **0.006 or 0.007 depending on which columns are averaged.** Both values are correct;
the copula's value must be quoted on the same basis.

| Mean KS over | Real vs real | Copula | FSDG | Source |
|---|---|---|---|---|
| numeric columns **including the 0/1 target** | Adult 0.0062, Bank 0.0063 → **0.006** | 0.0067, 0.0068 → 0.007 | 0.348, 0.334 | `S/synthetic_fidelity.csv` (`mean_ks_numeric_mean`) |
| numeric **features only** (as the paper's FSDG means 0.405 / 0.390) | Adult 0.0071, Bank 0.0072 → **0.007** | 0.0078, 0.0079 → 0.008 | 0.405, 0.390 | real and FSDG: `S/synthetic_paper_means.csv`; copula: computed from `results/raw/synthetic/*/fidelity_columns.csv` |

`REVISION_LOG.md` Phase 5 ("mean KS 0.007 vs 0.006") uses the first basis. If the paper quotes FSDG at
0.405/0.390, the matching statement is copula 0.008 vs reference 0.007. The computed features-only values for the
real reference reproduce `S/synthetic_paper_means.csv` exactly.

## 5. Table 29: are the "race" rows of DIR and ThresholdOptimizer from runs that target race?

**Yes**, if Table 29 is built like the Table 25 section of `docs/COMPARISON_OLD_VS_NEW.md`. That section is
generated by `scripts/make_comparison.py:97`, which reads method `dir[race]` / `threshold_opt[race]` from
`S/fairness_utility_summary.csv`. These are separate runs with race as the sensitive feature passed to the method:
- `configs/baselines.yaml:43`: `mitigate: [sex, race]`, one attribute per run;
- `experiments/baselines.py:143-144`: `a_tr = groups[attr]`, passed to `TrainFittedDIR` and to
  `ThresholdOptimizer.fit(..., sensitive_features=a_tr)`.

The `[sex]` runs are separate rows.

Two differences from the original Table 25 matter:
1. Race keeps its 5 recorded categories, and each gap is max − min over the groups (`experiments/metrics.py:26-28`).
   The original scripts used White vs non-White (`legacy/original/aif360_fairlearn/aif360_adult_DIR.py:26-28`).
2. DIR repairs numeric columns only.

## 6. Is "no resampling" a candidate in the DBIM search?

**No.** When the imbalance check triggers, the search grid contains only resamplers
(`experiments/pipeline.py:109-111`, `candidate_resamplers` lines 62-68). The "passthrough" (no resampling) slot is
used only when the check does not trigger, or in the ablation's cleaning-only stage. The check triggered in all
25 outer folds of all 5 DBIM experiments, and each fold had 96 candidates: 8 resamplers × 12 classifier settings.
None of them was "none" (computed from `results/raw/main/*/selection.csv` and `inner_scores.csv.gz`).

The 8 resamplers are ADASYN, AllKNN, BorderlineSMOTE, KMeansSMOTE, NeighbourhoodCleaningRule, OneSidedSelection,
SMOTE and TomekLinks. SVMSMOTE and CNN are dropped at ≥ 20,000 training rows (`configs/search_space.yaml`).
KMeansSMOTE fails in every fold (REVISION_LOG Phase 2), so 7 resamplers are effectively compared.

## 7. Air Quality LSTM

| Question | Answer | Source |
|---|---|---|
| Target variable | All 12 columns: a multivariate LSTM forecasts every sensor column. Metrics are computed per column and macro-averaged over the 12. There is no single target. | `experiments/ts_protocol.py:198` (`value_cols=cols`), `:90` |
| Forecast horizons | 1 h and 12 h. For h = 12, `historical_forecasts(forecast_horizon=12, stride=12, last_points_only=False, retrain=False)` makes non-overlapping 12-step forecasts over the test window, and all 12 steps are scored. The h = 12 MAE is therefore the mean over steps 1–12, not the 12th step alone. | `configs/timeseries.yaml:24`; `experiments/ts_protocol.py:141-145, 223-226` |
| MAE scaling | `mae_scaled` = absolute error ÷ (max − min of that column's **observed training values**), per fold. The same scale is used for every method and scenario, then averaged per column and over the 12 columns. MAE in original units is also stored. Separately, the LSTM's inputs are min-max scaled by a darts `Scaler` fitted on each scenario's filled training period. That affects only the model, not the metric. | `experiments/ts_protocol.py:69-72, 76-91, 199-201` |
| Missing values outside the artificial gap | Training period: the natural gaps are filled by the method being compared (each scenario fills natural + artificial gaps with its own method); the reference scenario has no artificial gap and fills natural gaps by linear interpolation. Validation/test inputs: forward-filled from the past only, the same in every scenario. Scoring: only timestamps whose true value was observed. Reconstruction error: only artificial-gap cells with an observed true value. | `configs/timeseries.yaml:20`; `experiments/ts_protocol.py:63-66, 81, 94-98, 183-196` |

## 8. ECE bins; outer-fold stratification

- **ECE:** 10 equal-width bins on [0, 1]; a score of exactly 1.0 goes into the last bin
  (`experiments/metrics.py:70-79`, called with the default at `:86`).
- **Outer folds:** stratified by target × one protected attribute:
  - Adult: `income` × `sex`;
  - Bank: `y` × `marital` (`configs/main.yaml:36, 59`; the same in `baselines.yaml`, `ablation.yaml`,
    `synthetic.yaml`; strata built at `experiments/protocol.py:51`).

  The inner 3-fold CV uses the same strata (`experiments/protocol.py:67-69`). The Air Quality folds are
  chronological (expanding window), not stratified.

## 9. Detector external validation

- **The 52 columns:** every column of the three datasets, target included (computed from
  `S/detector_external_columns.csv`):
  - Adult: 14 of its 15 columns. `relationship` is the proxy: it is reported separately and not scored, so there
    are 53 columns in all, which is why the Phase 9 checkpoint check counts 53.
  - Bank: 17 (16 features including `duration`, plus `y`).
  - German Credit: 21 (20 attributes plus `credit_risk`).

  10 columns are sensitive:
  - Adult: age, marital-status, race, sex, native-country;
  - Bank: age, marital;
  - German: personal_status_and_sex, age_in_years, foreign_worker.

  The other 42 are not sensitive. Ground truth: `configs/detector.yaml` (decision L1).
- **"relationship associated with sex (0.65)":** Cramér's V (both columns categorical; chi-square without continuity
  correction), computed on the 45,222 complete Adult records. Value 0.6487. Proxy threshold 0.5.
  Sources: `scripts/detector_validation.py:113-128, 131-147`, `configs/detector.yaml:31`,
  `S/detector_external_columns.csv`.
- **Were the Adult/Bank descriptions in the detector's training data?** Not as written for the validation (they
  are the revised UCI descriptions in `data/loaders.py`, decision L4). Some are very close to training texts
  (computed; similarity = the stored TF-IDF cosine to the nearest training text):
  - Exact matches (case-insensitive): Adult "Marital status" (training "marital status", labelled Sensitive) and
    "Occupation" (training "Occupation", labelled Not Sensitive).
  - Near matches:
    - Adult age 1.00 ("The age of the individual in years");
    - Adult sex 0.82;
    - Adult race 0.78 ("race");
    - Adult fnlwgt 0.77 ("final weight", the UCI Adult name of this column);
    - Bank marital 0.77;
    - Bank age 0.74 ("Age (years)").

  The other Adult/Bank descriptions have similarity ≤ 0.57. The training file's sources (Kaggle dataset pages) are
  not listed (`data/README.md`, OPEN_QUESTIONS P4), so it cannot be established whether Adult or Bank
  documentation was itself a source. "final weight" suggests that Adult column names are among the training
  texts. German Credit has no such overlap (REVISION_LOG Phase 6).
- **Deployed toolkit:** one model. `bias_identification/datasets.py:55` calls
  `Models/sensitive_features_text_classifier.py:104`, which loads `bert-sensitive-features-classifier/checkpoint-830`.
  The 5 retrained models (`results/raw/detector/heads/`) and their majority vote exist only in the validation
  (`scripts/detector_validation.py`).

## 10. Similarity cleaning

- **Time-series method, scaling before the cosine similarity:** **no**. `standardize: false`
  (`configs/timeseries.yaml:23`, decision G3).
  - The record vectors (all columns at a timestamp) are used in original units, with missing entries replaced by
    the training mean for the similarity only. They are then normalised to unit length, which is the cosine step
    (`experiments/ts_cleaning.py:111-122`).
  - So the large-valued columns dominate the similarity: the PT08 sensors (~1000) and NOx (~250).
  - `standardize=True` (z-scores) exists but was not used.
- **What replaces outliers:** only the tabular imputer has an outlier step, and it is optional.
  - Numeric values with |z| > `z_thresh` (default 3; training mean and SD) are set to missing. They are then filled
    like any missing value: the value of the most similar (cosine) training row that has one, or the fallback
    below if none of the neighbours has one.
  - So an outlier is replaced by a donor value, not clipped and not replaced by the mean
    (`experiments/imputation.py:55-64`; toolkit `bias_identification/datasets.py:176-191`).
  - The step is **off** in every reported experiment (`SimilarityImputer()` default `detect_outliers=False`,
    `experiments/pipeline.py:104`, `experiments/imputation.py:38`) and off by default in the toolkit
    (`main.py:20`; the API takes it as a form field).
  - The time-series method has no outlier step.
- **Fallback for categorical columns:**
  - In the evaluation's `SimilarityImputer`, the **mode** (most frequent training category)
    (`experiments/imputation.py:96-102`).
  - The toolkit's own `bias_identification/datasets.py:259` still uses the **mean** for every column, including
    label-coded categoricals. That gives a non-existent code, which becomes NaN after decoding: the bug fixed only
    in the evaluation code (REVISION_LOG Phase 2.1).
  - Numeric fallback: the training mean in both.

## 11. "No mitigation" Adult training: 74 s vs 2.2 s

The two numbers measure different things; both are single-core.
- **74.3 s** (`S/cost_methods.csv`, `adult,none`: `total_training_s` = `model_selection_s` = 74.33 s, mean of folds
  0–2) is the whole training procedure: the inner-CV grid search (12 random-forest settings × 3 inner folds = 36
  fits) plus the refit on the outer training fold.
- **2.2 s** (`S/synthetic_fit_time.csv`, `adult_synthetic`, `real_train`; computed mean of 1.37, 1.35 and 3.96 s =
  2.23 s) is **one** fit of the random forest, with the hyperparameters the inner CV had already chosen and no
  search. Fold 2 chose 300 trees, hence 3.96 s.

The comparable number in the cost run is the final fit alone: 2.36 s (`final_fit_s`; classifier fit 2.33 s). The
FSDG timing (Tables 22–24) compares one fit on real data (2.2 s) with one fit on FSDG data (4.6 s).

## 12. Threat model: residual risks

Source: `docs/THREAT_MODEL.md` §6 (also summarised in REVISION_LOG Phase 8).
1. No user accounts. The paper's statement that stored files are "only available to the user that requested
   them" (§2.5.2) is still not true: there is one optional shared API key and no per-user storage.
2. The current Flutter UI does not send `X-API-Key`, so API-key mode breaks the UI until it is updated.
3. CPU cost of `/data-bias-api`: there is no per-request time limit and no process isolation. A shared deployment
   needs a WSGI server with worker timeouts, rate limits and TLS; the Flask development server is for local use.
4. Synthetic generation is bounded but not instant (1 million rows with a 300-node expression take seconds to
   minutes).
5. Without `DEBIAS_STORE_HMAC_KEY`, the SHA-256 detects accidental or naive changes, but not a deliberate attacker
   with write access to the store.
6. The legacy flag (`DEBIAS_ENABLE_LEGACY_CODE_EXEC=1`) re-enables remote code execution. Integrity checks prove
   only who wrote the stored code, not that it is safe.
7. No retention policy: uploads containing personal data are kept until deleted by hand.
8. Model files are loaded from the local repository (safetensors, no pickle); replacing them needs write access to
   the server.
9. The UI still builds Python source by string interpolation (S8). The server no longer runs it, but the UI/API
   contract should move to the JSON form.

Also stated there:
- the supply chain (A5) is out of scope, beyond pinned versions;
- the test against a real MinIO server is still open (§8; it ran against an S3-compatible mock).

## 13. Repository

| Item | Value | Source |
|---|---|---|
| Remote of this working copy | `https://github.com/Noveldy/enhancing-fairness-working.git` | `git remote -v` |
| Its default branch | `main` (9469d82 "updated README", the pre-revision code). The revision is **not** on it: `revision` (8b0079f, identical to this working copy) is pushed as its own branch and not merged; there is also `release` (b209f03, one squashed commit). | `git ls-remote --symref origin HEAD`, `git ls-remote --heads origin` (2026-10-27) |
| URL cited in the package | `https://github.com/Noveldy/enhancing-fairness` | `CITATION.cff:8`, `README.md:16`, REVISION_LOG Phase 9 (P5) |
| That repository | Has only `main` (default; 373564c, 2026-10-23, "Fix JSON formatting in .zenodo.json"). Its tree equals the `release` snapshot except for `README.md`, `.zenodo.json` and the two `docs/audit/*.log` files. | `git ls-remote` and a shallow, blob-less clone into a scratch folder (no change to any local repository) |
| What that repository lacks compared with `revision` | 20 files, including `docs/THREAT_MODEL.md`, `docs/COMPARISON_OLD_VS_NEW.md`, `docs/NOTES_FOR_MANUSCRIPT.md`, `docs/OPEN_QUESTIONS.md`, `REVISION_PLAN.md`, `scripts/synthetic_paper_metrics.py`, `scripts/make_comparison.py` and all Phase 10 outputs (`S/synthetic_paper_*.csv`, `synthetic_data_description.csv`, `synthetic_fit_time.csv`, `paper_metrics_fold_columns.csv`). If the paper cites the threat model or the Phase 10 FSDG tables through the public repository, they are not there. | same |
| Its README | Says `git clone … && git checkout release`, but that repository has no `release` branch, so the quick start fails at that step. | `README.md:16` of the public `main` |
| Zenodo DOI | **None.** `CITATION.cff`, `.zenodo.json` (here and in the public repository) and the README contain no DOI; the only DOIs in the repository are those of the UCI datasets (`data/README.md:75-78`). Nothing has been published to Zenodo from this package (REVISION_LOG Phase 9). | grep over the repository |
