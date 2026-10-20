# synthetic_data_eval.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np
from pathlib import Path

def compare_numeric_column(real_df, synth_df, column):
    print(f"\n--- {column} ---")
    print("Real Summary:")
    print(real_df[column].describe())
    print("\nSynthetic Summary:")
    print(synth_df[column].describe())

    sns.kdeplot(real_df[column], label='Real', fill=True)
    sns.kdeplot(synth_df[column], label='Synthetic', fill=True)
    plt.title(f'Distribution Comparison: {column}')
    plt.legend()
    # store the plot
    # plt.savefig(Path(f'bias_mitigation/synthetic_data/adult_evaluation/{column}_distribution_comparison.png'))
    # plt.show()

def compare_categorical_column(real_df, synth_df, column):
    real_counts = real_df[column].value_counts(normalize=True)
    synth_counts = synth_df[column].value_counts(normalize=True)

    comparison = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
    comparison.columns = ['Real', 'Synthetic']
    comparison.plot(kind='bar', title=f"Category Distribution: {column}")
    plt.ylabel("Proportion")
    # store the plot
    # plt.savefig(Path(f'bias_mitigation/synthetic_data/adult_evaluation/{column}_category_distribution_comparison.png'))
    # plt.show()

def compare_statistical_distance(real_df, synth_df, column):
    ks_stat, ks_p = ks_2samp(real_df[column], synth_df[column])
    w_dist = wasserstein_distance(real_df[column], synth_df[column])

    print(f"{column}:")
    print(f"  KS-statistic = {ks_stat:.4f} (p = {ks_p:.4f})")
    print(f"  Wasserstein distance = {w_dist:.4f}")

# def compare_correlation_structure(real_df, synth_df):
#     real_corr = real_df.corr(numeric_only=True)
#     synth_corr = synth_df.corr(numeric_only=True)
#     diff_corr = real_corr - synth_corr

#     sns.heatmap(diff_corr, cmap='coolwarm', center=0)
#     plt.title("Correlation Difference (Real - Synthetic)")
#     plt.show()

def evaluate_dataset(real_df, synth_df, numeric_cols, categorical_cols):
    for col in numeric_cols:
        compare_numeric_column(real_df, synth_df, col)
        compare_statistical_distance(real_df, synth_df, col)

    for col in categorical_cols:
        compare_categorical_column(real_df, synth_df, col)

    # calculate avereage ks-statistic and wasserstein distance for numeric columns
    ks_stats = []
    wasserstein_dists = []
    for col in numeric_cols:
        ks_stat, _ = ks_2samp(real_df[col], synth_df[col])
        w_dist = wasserstein_distance(real_df[col], synth_df[col])
        ks_stats.append(ks_stat)
        wasserstein_dists.append(w_dist)
    avg_ks_stat = np.mean(ks_stats)
    avg_wasserstein_dist = np.mean(wasserstein_dists)
    print(f"\nAverage KS-statistic for numeric columns: {avg_ks_stat:.4f}"
          f"\nAverage Wasserstein distance for numeric columns: {avg_wasserstein_dist:.4f}")
    
    # calculate avearege category distribution difference for categorical columns
    category_diffs = []
    for col in categorical_cols:
        real_counts = real_df[col].value_counts(normalize=True)
        synth_counts = synth_df[col].value_counts(normalize=True)
        diff = np.abs(real_counts - synth_counts).sum()
        category_diffs.append(diff)
        # print category difference for each column
        print(f"{col}: Category distribution difference = {diff:.4f}")
    avg_category_diff = np.mean(category_diffs)
    print(f"\nAverage category distribution difference for categorical columns: {avg_category_diff:.4f}")

    # compare_correlation_structure(real_df, synth_df)


# ----------------------------------------------------------------- ADULT DATASET EVALUATION -----------------------------------------------------------------------------
# # Load your real and synthetic data
# real_df = pd.read_json(Path('Datasets/adult_debiased.json'))
# synth_df = pd.read_csv(Path('Datasets/synthetic_adult.csv'))
# # convert age,workclass,education,marital-status,occupation,relationship,race,sex,native-country,income to categorical
# categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
# real_df[categorical_columns] = real_df[categorical_columns].astype('category')
# synth_df[categorical_columns] = synth_df[categorical_columns].astype('category')

# # Define columns
# numeric_cols = real_df.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = real_df.select_dtypes(include=['category']).columns

# # Run evaluation
# evaluate_dataset(real_df, synth_df, numeric_cols, categorical_cols)

# ----------------------------------------------------------------- BANK MARKETING DATASET EVALUATION -----------------------------------------------------------------------------
# # Load your real and synthetic data
# real_df = pd.read_json(Path('Datasets/bank_marketing_debiased.json'))
# synth_df = pd.read_csv(Path('Datasets/synthetic_bank.csv'))
# # convert age, job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome to categorical
# categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'poutcome', 'y']
# real_df[categorical_columns] = real_df[categorical_columns].astype('category')
# synth_df[categorical_columns] = synth_df[categorical_columns].astype('category')
# # Define columns
# numeric_cols = real_df.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = real_df.select_dtypes(include=['category']).columns
# # Run evaluation
# evaluate_dataset(real_df, synth_df, numeric_cols, categorical_cols)

# ----------------------------------------------------------------- AIR QUALITY DATASET EVALUATION -----------------------------------------------------------------------------
# # Load your real and synthetic data
# real_df = pd.read_csv(Path('Datasets/air_quality_modified.csv'))
# synth_df = pd.read_csv(Path('Datasets/synthetic_air_quality.csv'))
# # convert timestamp to datetime
# real_df['datetime'] = pd.to_datetime(real_df['datetime'])
# synth_df['timestamp'] = pd.to_datetime(synth_df['timestamp'])
# # set timestamp as index
# real_df.set_index('datetime', inplace=True)
# synth_df.set_index('timestamp', inplace=True)
# # Define columns
# numeric_cols = real_df.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = real_df.select_dtypes(include=['category']).columns

# # Run evaluation
# # evaluate_dataset(real_df, synth_df, numeric_cols, categorical_cols)
# # plot synthetic data and real data in the same plot to compare them
# print("Real Data Info:")
# print(real_df.info())
# print("Synthetic Data Info:")
# print(synth_df.info())
# real_df.plot(subplots=True, figsize=(10, 15), title='Air Quality Dataset Ground Truth')
# synth_df.plot(subplots=True, figsize=(10, 15), title='Synthetic Air Quality Dataset')
# plt.show()


# ----------------------------------------------------------------- INDIVIDUAL HOUSEHOLD POWER CONSUMPTION DATASET EVALUATION -----------------------------------------------------------------------------

# Load your real and synthetic data
real_df = pd.read_csv(Path('Datasets/individual_household_electric_power_consumption_modified.csv'))
synth_df = pd.read_csv(Path('Datasets/synthetic_individual_household_power_consumption.csv'))
# convert timestamp to datetime
real_df['datetime'] = pd.to_datetime(real_df['datetime'])
synth_df['timestamp'] = pd.to_datetime(synth_df['timestamp'])
# set timestamp as index
real_df.set_index('datetime', inplace=True)
synth_df.set_index('timestamp', inplace=True)
# Define columns
numeric_cols = real_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = real_df.select_dtypes(include=['category']).columns

# Run evaluation
# evaluate_dataset(real_df, synth_df, numeric_cols, categorical_cols)
# plot synthetic data and real data in the same plot to compare them
print("Real Data Info:")
print(real_df.info())
print("Synthetic Data Info:")
print(synth_df.info())
real_df.plot(subplots=True, figsize=(10, 15), title='Individual Household Power Consumption Dataset Ground Truth')
synth_df.plot(subplots=True, figsize=(10, 15), title='Synthetic Individual Household Power Consumption Dataset')
plt.show()