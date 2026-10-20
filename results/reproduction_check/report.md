# Reproduction check (2026-10-20T14:02:35+00:00)

Checkout: `8744cd827c00841df231305a58fe2892a9a5eb0f`; Python 3.12.3; 446 s.

**Differences or failures found (see below).** 76 of 77 files identical; 0 failed steps.

| check | file | rows compared | identical | note |
|---|---|---|---|---|
| main/adult_dbim_protected_kept | `results/raw/main/adult_dbim_protected_kept/fold_metrics.csv` | 1 | yes |  |
| main/adult_dbim_protected_kept | `results/raw/main/adult_dbim_protected_kept/inner_scores.csv.gz` | 96 | yes |  |
| main/adult_dbim_protected_kept | `results/raw/main/adult_dbim_protected_kept/predictions.csv.gz` | 9769 | yes |  |
| main/adult_dbim_protected_kept | `results/raw/main/adult_dbim_protected_kept/selection.csv` | 1 | yes |  |
| main/adult_dbim_protected_dropped | `results/raw/main/adult_dbim_protected_dropped/fold_metrics.csv` | 1 | yes |  |
| main/adult_dbim_protected_dropped | `results/raw/main/adult_dbim_protected_dropped/inner_scores.csv.gz` | 96 | yes |  |
| main/adult_dbim_protected_dropped | `results/raw/main/adult_dbim_protected_dropped/predictions.csv.gz` | 9769 | yes |  |
| main/adult_dbim_protected_dropped | `results/raw/main/adult_dbim_protected_dropped/selection.csv` | 1 | yes |  |
| main/bank_dbim_protected_kept | `results/raw/main/bank_dbim_protected_kept/fold_metrics.csv` | 1 | yes |  |
| main/bank_dbim_protected_kept | `results/raw/main/bank_dbim_protected_kept/inner_scores.csv.gz` | 96 | yes |  |
| main/bank_dbim_protected_kept | `results/raw/main/bank_dbim_protected_kept/predictions.csv.gz` | 9043 | yes |  |
| main/bank_dbim_protected_kept | `results/raw/main/bank_dbim_protected_kept/selection.csv` | 1 | yes |  |
| main/bank_dbim_protected_dropped | `results/raw/main/bank_dbim_protected_dropped/fold_metrics.csv` | 1 | yes |  |
| main/bank_dbim_protected_dropped | `results/raw/main/bank_dbim_protected_dropped/inner_scores.csv.gz` | 96 | yes |  |
| main/bank_dbim_protected_dropped | `results/raw/main/bank_dbim_protected_dropped/predictions.csv.gz` | 9043 | yes |  |
| main/bank_dbim_protected_dropped | `results/raw/main/bank_dbim_protected_dropped/selection.csv` | 1 | yes |  |
| main/bank_dbim_with_duration | `results/raw/main/bank_dbim_with_duration/fold_metrics.csv` | 1 | yes |  |
| main/bank_dbim_with_duration | `results/raw/main/bank_dbim_with_duration/inner_scores.csv.gz` | 96 | **no** | differs in mean_inner_score, std_inner_score, rank |
| main/bank_dbim_with_duration | `results/raw/main/bank_dbim_with_duration/predictions.csv.gz` | 9043 | yes |  |
| main/bank_dbim_with_duration | `results/raw/main/bank_dbim_with_duration/selection.csv` | 1 | yes |  |
| baselines/adult_baselines_protected_kept | `results/raw/baselines/adult_baselines_protected_kept/fold_metrics.csv` | 12 | yes |  |
| baselines/adult_baselines_protected_kept | `results/raw/baselines/adult_baselines_protected_kept/predictions.csv.gz` | 108408 | yes |  |
| baselines/adult_baselines_protected_kept | `results/raw/baselines/adult_baselines_protected_kept/selection.csv` | 12 | yes |  |
| baselines/adult_baselines_protected_dropped | `results/raw/baselines/adult_baselines_protected_dropped/fold_metrics.csv` | 12 | yes |  |
| baselines/adult_baselines_protected_dropped | `results/raw/baselines/adult_baselines_protected_dropped/predictions.csv.gz` | 108408 | yes |  |
| baselines/adult_baselines_protected_dropped | `results/raw/baselines/adult_baselines_protected_dropped/selection.csv` | 12 | yes |  |
| baselines/bank_baselines_protected_kept | `results/raw/baselines/bank_baselines_protected_kept/fold_metrics.csv` | 8 | yes |  |
| baselines/bank_baselines_protected_kept | `results/raw/baselines/bank_baselines_protected_kept/predictions.csv.gz` | 69224 | yes |  |
| baselines/bank_baselines_protected_kept | `results/raw/baselines/bank_baselines_protected_kept/selection.csv` | 8 | yes |  |
| baselines/bank_baselines_protected_dropped | `results/raw/baselines/bank_baselines_protected_dropped/fold_metrics.csv` | 8 | yes |  |
| baselines/bank_baselines_protected_dropped | `results/raw/baselines/bank_baselines_protected_dropped/predictions.csv.gz` | 69224 | yes |  |
| baselines/bank_baselines_protected_dropped | `results/raw/baselines/bank_baselines_protected_dropped/selection.csv` | 8 | yes |  |
| ablation/adult_impute_only_protected_kept | `results/raw/ablation/adult_impute_only_protected_kept/fold_metrics.csv` | 1 | yes |  |
| ablation/adult_impute_only_protected_kept | `results/raw/ablation/adult_impute_only_protected_kept/inner_scores.csv.gz` | 12 | yes |  |
| ablation/adult_impute_only_protected_kept | `results/raw/ablation/adult_impute_only_protected_kept/predictions.csv.gz` | 9769 | yes |  |
| ablation/adult_impute_only_protected_kept | `results/raw/ablation/adult_impute_only_protected_kept/selection.csv` | 1 | yes |  |
| ablation/adult_impute_only_protected_dropped | `results/raw/ablation/adult_impute_only_protected_dropped/fold_metrics.csv` | 1 | yes |  |
| ablation/adult_impute_only_protected_dropped | `results/raw/ablation/adult_impute_only_protected_dropped/inner_scores.csv.gz` | 12 | yes |  |
| ablation/adult_impute_only_protected_dropped | `results/raw/ablation/adult_impute_only_protected_dropped/predictions.csv.gz` | 9769 | yes |  |
| ablation/adult_impute_only_protected_dropped | `results/raw/ablation/adult_impute_only_protected_dropped/selection.csv` | 1 | yes |  |
| ablation/bank_impute_only_protected_kept | `results/raw/ablation/bank_impute_only_protected_kept/fold_metrics.csv` | 1 | yes |  |
| ablation/bank_impute_only_protected_kept | `results/raw/ablation/bank_impute_only_protected_kept/inner_scores.csv.gz` | 12 | yes |  |
| ablation/bank_impute_only_protected_kept | `results/raw/ablation/bank_impute_only_protected_kept/predictions.csv.gz` | 9043 | yes |  |
| ablation/bank_impute_only_protected_kept | `results/raw/ablation/bank_impute_only_protected_kept/selection.csv` | 1 | yes |  |
| ablation/bank_impute_only_protected_dropped | `results/raw/ablation/bank_impute_only_protected_dropped/fold_metrics.csv` | 1 | yes |  |
| ablation/bank_impute_only_protected_dropped | `results/raw/ablation/bank_impute_only_protected_dropped/inner_scores.csv.gz` | 12 | yes |  |
| ablation/bank_impute_only_protected_dropped | `results/raw/ablation/bank_impute_only_protected_dropped/predictions.csv.gz` | 9043 | yes |  |
| ablation/bank_impute_only_protected_dropped | `results/raw/ablation/bank_impute_only_protected_dropped/selection.csv` | 1 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/fidelity.csv` | 4 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/fidelity_columns.csv` | 60 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/fidelity_subgroups.csv` | 36 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/privacy.csv` | 4 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/tstr_fold_metrics.csv` | 8 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/tstr_group_rates.csv` | 56 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/tstr_predictions.csv.gz` | 72272 | yes |  |
| synthetic/adult_synthetic | `results/raw/synthetic/adult_synthetic/tsts_fold_metrics.csv` | 6 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/fidelity.csv` | 4 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/fidelity_columns.csv` | 64 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/fidelity_subgroups.csv` | 20 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/privacy.csv` | 4 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/tstr_fold_metrics.csv` | 8 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/tstr_group_rates.csv` | 24 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/tstr_predictions.csv.gz` | 69224 | yes |  |
| synthetic/bank_synthetic | `results/raw/synthetic/bank_synthetic/tsts_fold_metrics.csv` | 6 | yes |  |
| timeseries | `results/raw/timeseries/air_quality_cleaning/cleaning.csv` | 2 | yes |  |
| timeseries | `results/raw/timeseries/air_quality_cleaning/donor_log_fold0_proposed.csv` | 928 | yes |  |
| timeseries | `results/raw/timeseries/air_quality_cleaning/forecasts.csv.gz` | 32256 | yes |  |
| timeseries | `results/raw/timeseries/air_quality_cleaning/lstm_selection.csv` | 12 | yes |  |
| timeseries | `results/raw/timeseries/air_quality_cleaning/reconstruction.csv` | 12 | yes |  |
| timeseries | `results/raw/timeseries/air_quality_cleaning/test_metrics.csv` | 4 | yes |  |
| detector | `results/raw/detector/internal_cv_folds.csv` | 1 | yes |  |
| detector | `results/raw/detector/internal_cv_predictions.csv` | 164 | yes |  |
| detector | `results/raw/detector/internal_cv_epochs.csv` | 10 | yes |  |
| detector | `results/raw/detector/training_descriptions_groups.csv` | 816 | yes |  |
| detector | `results/raw/detector/external_predictions.csv (seed 0, checkpoint-830, keyword)` | 53 | yes |  |
| cost | `reproduces_stored_accuracy (Bank fold 0, DBIM)` | 7 | yes | values: ['True'] |
| cost | `reproduces_stored_selection (Bank fold 0, DBIM)` | 1 | yes | values: ['True'] |

Steps:

| check | exit code | seconds |
|---|---|---|
| main | 0 | reused |
| baselines | 0 | reused |
| ablation | 0 | reused |
| synthetic | 0 | reused |
| timeseries | 0 | reused |
| detector | 0 | reused |
| cost | 0 | 3.4 |
| cost | 0 | 430.2 |
