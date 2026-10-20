import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import (
    CondensedNearestNeighbour,
    AllKNN,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    TomekLinks
)
from sklearn.datasets import make_classification
from collections import defaultdict

# ---------------------------
# 1. Synthetic Data Generator
# ---------------------------

def generate_synthetic_data(n_samples, imbalance_ratio=0.1):
    """Generate imbalanced synthetic binary classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        random_state=42,
    )
    return X, y

# ---------------------------
# 2. Benchmark Function
# ---------------------------

def benchmark_samplers(sizes, ratios):
    samplers = {
        'CondensedNN': CondensedNearestNeighbour(),
        'AllKNN': AllKNN(),
        'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule(),
        'OneSidedSelection': OneSidedSelection(),
        'TomekLinks': TomekLinks(),
    }

    results = []

    for ratio in ratios:
        print(f"\n=== Imbalance Ratio: {ratio} ===")
        for size in sizes:
            print(f"  Generating dataset with {size} samples...")
            X, y = generate_synthetic_data(n_samples=size, imbalance_ratio=ratio)

            for name, sampler in samplers.items():
                print(f"    Running {name}...")
                start = time.time()
                try:
                    X_res, y_res = sampler.fit_resample(X, y)
                    elapsed = round(time.time() - start, 2)
                    removed = round((1 - len(y_res) / len(y)) * 100, 2)
                    results.append({
                        'Sampler': name,
                        'Dataset Size': size,
                        'Imbalance Ratio': ratio,
                        'Execution Time (s)': elapsed,
                        'Majority Removed (%)': removed
                    })
                except Exception as e:
                    print(f"      {name} failed: {e}")
                    results.append({
                        'Sampler': name,
                        'Dataset Size': size,
                        'Imbalance Ratio': ratio,
                        'Execution Time (s)': None,
                        'Majority Removed (%)': None
                    })

    return pd.DataFrame(results)

# ---------------------------
# 3. Visualization Function
# ---------------------------

def plot_results_by_ratio(df):
    ratios = sorted(df['Imbalance Ratio'].unique())
    for ratio in ratios:
        subset = df[df['Imbalance Ratio'] == ratio]
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Imbalance Ratio = {ratio}', fontsize=16)

        for name, group in subset.groupby('Sampler'):
            ax[0].plot(group['Dataset Size'], group['Execution Time (s)'], label=name, marker='o')
            ax[1].plot(group['Dataset Size'], group['Majority Removed (%)'], label=name, marker='x')

        ax[0].set_title('Execution Time vs Dataset Size')
        ax[0].set_xlabel('Dataset Size')
        ax[0].set_ylabel('Time (s)')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].set_title('Majority Removed (%) vs Dataset Size')
        ax[1].set_xlabel('Dataset Size')
        ax[1].set_ylabel('Majority Removed (%)')
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # store the plot
        plt.savefig(f'under_sampling_results_ratio_{ratio}.png')
        # plt.show()

# ---------------------------
# 4. Main Execution
# ---------------------------

if __name__ == "__main__":
    dataset_sizes = [1000, 5000, 10000, 20000]
    imbalance_ratios = [0.01, 0.05, 0.1, 0.2]

    print("Benchmarking under-sampling algorithms on synthetic datasets...")

    benchmark_df = benchmark_samplers(dataset_sizes, imbalance_ratios)
    print("\nUnder-samling Benchmark Results:\n", benchmark_df)
    # store results to CSV
    benchmark_df.to_csv('under_sampling_benchmark_results.csv', index=False)

    plot_results_by_ratio(benchmark_df)
