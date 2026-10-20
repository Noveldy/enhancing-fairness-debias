import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
import os
from pathlib import Path
from scipy import stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# # load the adult.csv dataset
# data = pd.read_csv(Path('Datasets/adult.csv'))
# # get unique values of income column
# unique_values = data['income'].unique()
# print(f"Unique values in 'income' column: {unique_values}")

# # filter data to create a dataframe with 1000 rows and only keep the columns: age, race, sex, occupation and income
# demo_data = data[['age', 'race', 'sex', 'occupation', 'income']].sample(n=1000, random_state=42)
# # save the demo_data dataframe to a csv file
# demo_data.to_csv(Path('Datasets/demo_dataset.csv'), index=False)

# for testing
# age, race, sex, occupation
# the age of the individual, the race of the individual, the sex of the individual, the occupation of the individual
# income
# the income of the individual

def extract_category_weights(df, column):
    """
    Returns a tuple: (categories, weights)
    where weights are normalized frequencies.
    """
    value_counts = df[column].value_counts(normalize=True).sort_index()
    categories = value_counts.index.tolist()
    weights = value_counts.values.tolist()
    # keep two decimal places for weights
    weights = [round(weight, 5) for weight in weights]
    return categories, weights

def compare_distributions(df, candidate_dists=None, top_n=3, sample_size=5000):
    if candidate_dists is None:
        candidate_dists = [
            'norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min', 
            'pareto', 'triang', 'chi2'
        ]

    results = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        data = df[col].dropna().values
        if len(data) > sample_size:
            data = np.random.choice(data, size=sample_size, replace=False)
        
        dist_scores = []

        for dist_name in candidate_dists:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                D, p = stats.kstest(data, dist_name, args=params)

                # Get parameter names
                shape_names = []
                if dist.shapes:
                    shape_names = [s.strip() for s in dist.shapes.split(',')]
                param_names = shape_names + ['loc', 'scale']
                param_dict = dict(zip(param_names, params))

                dist_scores.append((dist_name, D, p, param_dict))
            except Exception:
                dist_scores.append((dist_name, np.inf, 0.0, None))  # failed fit

        dist_scores.sort(key=lambda x: x[1])  # sort by D-statistic (lower is better)
        results[col] = dist_scores[:top_n]    # keep top N matches

    return results


def plot_fit(data, dist_fits, col_name):
    plt.figure(figsize=(10, 6))

    # Plot histogram of the actual data
    count, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', edgecolor='black', label='Data Histogram')
    x = np.linspace(min(data), max(data), 1000)

    for dist_name, D, p, params in dist_fits:
        if params is None:
            continue

        try:
            dist = getattr(stats, dist_name)

            # Split parameters into shape, loc, scale
            if len(params) >= 2:
                *shape_params, loc, scale = params
            else:
                # Not enough parameters to define loc and scale
                continue

            # Ensure scale is valid
            if not np.isfinite(scale) or scale <= 0:
                continue

            # Compute PDF
            if shape_params:
                pdf_vals = dist.pdf(x, *shape_params, loc=loc, scale=scale)
            else:
                pdf_vals = dist.pdf(x, loc=loc, scale=scale)

            # Validate PDF
            if np.any(np.isnan(pdf_vals)) or np.all(pdf_vals == 0):
                continue

            # Plot the fitted distribution
            plt.plot(x, pdf_vals, lw=2, label=f"{dist_name} (KS={D:.3f}, p={p:.3f})")

        except Exception as e:
            print(f"Skipping {dist_name} for {col_name} due to error: {e}")
            continue

    plt.title(f"Distribution Fit for: {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_symbolic_regression(series: pd.Series):
    """
    Fit symbolic regression model to a single time series column.
    """
    y = series.values
    i = np.arange(len(series)).reshape(-1, 1)  # time index

    model = SymbolicRegressor(
        population_size=1000,
        generations=20,
        stopping_criteria=0.01,
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'),
        parsimony_coefficient=0.001,
        verbose=1,
        random_state=42
    )

    model.fit(i, y)
    return model._program, model

def register_symbolic_value_function(expression_str: str, file_path: str, func_name: str = None):
    """
    Appends a symbolic regression expression as a registered value function.
    
    Args:
        expression_str (str): The symbolic expression (Python-compatible string).
        file_path (str): Path to the value_functions.py file.
        func_name (str): Optional name for the value function. Auto-generated if None.
    """
    # Sanitize and default func name
    if not func_name:
        base_name = "auto_expr"
        idx = 1
        while True:
            candidate = f"{base_name}_{idx}"
            if candidate not in open(file_path).read():
                func_name = candidate
                break
            idx += 1
    
    func_code = f"""
@register_value_func("{func_name}")
def {func_name}_factory(params=None):
    \"\"\"
    Symbolic regression-generated value function.
    Expression: {expression_str}
    \"\"\"
    return lambda i: {expression_str}

{func_name}_factory._params = []
"""
    # Append to file
    with open(file_path, "a") as f:
        f.write(func_code.strip() + "\n\n")

    print(f"✅ Value function '{func_name}' registered in {file_path}.")

def symbolic_summary(df: pd.DataFrame, register: bool = False, vf_file_path: str = ""):
    results = {}
    for col in df.columns:
        print(f"\nFitting symbolic regression for column: {col}")
        symbol, model = run_symbolic_regression(df[col])
        i = np.arange(len(df)).reshape(-1, 1)
        pred = model._program.execute(i)
        mse = np.mean((pred - df[col].values) ** 2)

        expression_str = str(symbol)

        if register and vf_file_path:
            register_symbolic_value_function(
                expression_str=expression_str,
                file_path=vf_file_path,
                func_name=f"auto_{col}"
            )

        results[col] = {
            "expression": expression_str,
            "lambda": model._program.execute,
            "mse": mse
        }
    return results


import numpy as np

# Helper function to wrap a Fourier lambda with noise
def build_noisy_fourier_lambda(expr_func, params):
    noise_std = params.get("noise_std", 0.0)
    seed = params.get("seed", None)
    
    if seed is not None:
        np.random.seed(seed)

    def wrapped(i):
        base = expr_func(i)
        noise = np.random.normal(0, noise_std) if noise_std > 0 else 0.0
        return base + noise

    return wrapped

# ----------------------------------------
# Individual factory functions start here
# ----------------------------------------

def generate_fourier_lambdas_with_noise(
    df,
    datetime_col="datetime",
    top_n=10,
    timestep_hours=1,
    max_columns_to_plot=13,
    random_seed=42
):
    """
    Generate clean and noisy Fourier-based lambda approximations for numeric time series in a DataFrame.

    Returns:
        clean_lambdas: dict of column -> { "func": callable, "expr": string }
        noisy_lambdas: dict of column -> { "func": callable, "expr": string }
    """
    np.random.seed(random_seed)
    df = df.sort_values(datetime_col)

    data_columns = df.select_dtypes(include=[np.number]).columns
    clean_lambdas = {}
    noisy_lambdas = {}

    for i, col in enumerate(data_columns):
        signal = df[col].values
        n = len(signal)

        yf = fft(signal)
        xf = fftfreq(n, d=timestep_hours)

        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = yf[pos_mask]

        magnitudes = np.abs(yf_pos)
        if len(magnitudes) < top_n:
            continue

        top_indices = np.argsort(magnitudes)[-top_n:][::-1]
        amps = 2.0 / n * np.abs(yf_pos[top_indices])
        freqs = xf_pos[top_indices]
        phases = np.angle(yf_pos[top_indices])
        dc_offset = np.mean(signal)

        # Build the expression string for the lambda
        terms = [
            f"{amp:.6f} * np.sin(2 * np.pi * {freq:.6f} * t + {phase:.6f})"
            for amp, freq, phase in zip(amps, freqs, phases)
        ]
        expr = f"{dc_offset:.6f} + " + " + ".join(terms)
        lambda_code = f"lambda t: {expr}"
        base_lambda = eval(lambda_code, {"np": np, "pi": np.pi})

        # Estimate noise
        residuals = signal - np.array([base_lambda(t) for t in range(n)])
        noise_std = np.std(residuals)
        # print noise standard deviation
        print(f"Column '{col}': Noise Std Dev = {noise_std:.4f}")

        # Create noisy wrapper
        def make_noisy_func(base_func, noise_std):
            return lambda t: base_func(t) + np.random.normal(0, noise_std)

        noisy_lambda = make_noisy_func(base_lambda, noise_std)

        # Store both clean and noisy
        clean_lambdas[col] = {"func": base_lambda, "expr": lambda_code}
        noisy_lambdas[col] = {"func": noisy_lambda, "expr": lambda_code}

        # Optional plotting
        if i < max_columns_to_plot:
            t_vals = np.arange(n)
            clean_vals = [base_lambda(t) for t in t_vals]
            noisy_vals = [noisy_lambda(t) for t in t_vals]

            plt.figure(figsize=(10, 4))
            plt.plot(t_vals, signal, label="Original", alpha=0.7)
            plt.plot(t_vals, clean_vals, label="Fourier Approx (clean)", linestyle='--')
            plt.plot(t_vals, noisy_vals, label="Fourier Approx (noisy)", linestyle=':', alpha=0.8)
            plt.title(f"{col} - Original vs Approximations")
            plt.xlabel("Time Index (t)")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # plt.show()

    return clean_lambdas, noisy_lambdas


#---------------------------------------------------------------------- ADULT DATASET ----------------------------------------------------------------------
# load the adult_debiased.json as a dataframe
# data = pd.read_json(Path('Datasets/adult_debiased.json'))
# print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
# print(data.info())
# # convert age,workclass,education,marital-status,occupation,relationship,race,sex,native-country,income to categorical
# categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
# data[categorical_columns] = data[categorical_columns].astype('category')
# # print range of values per column
# for column in data.columns:
#     if pd.api.types.is_numeric_dtype(data[column]):
#         print(f"{column}: min={data[column].min()}, max={data[column].max()}")
#     else:
#         print(f"{column}: unique values={data[column].nunique()}")  # For categorical columns, show unique counts

# # print distribution of each column
# for column in data.columns:
#     if pd.api.types.is_numeric_dtype(data[column]):
#         print(f"{column} distribution:\n{data[column].describe()}\n")
#     else:
#         categories, weights = extract_category_weights(data, column)
#         print(f"{column} categories and weights:")
#         print("Categories:", categories)
#         print("Weights:", weights)
#         # print(f"{column} value counts:\n{data[column].value_counts()}\n")

# # Run the distribution comparison
# dist_results = compare_distributions(data)

# print("Best-Fitting Distributions per Column:\n")
# for col_name, fits in dist_results.items():
#     if not fits:
#         print(f"Column '{col_name}' has no fitting results.")
#         continue

#     best_fit = min(fits, key=lambda x: x[1])  # x[1] = D-statistic
#     dist_name, D_stat, p_val, params = best_fit

#     print(f"Column: {col_name}")
#     print(f"   Best Distribution: {dist_name}")
#     print(f"   KS D-statistic   : {D_stat:.6f}")
#     print(f"   p-value          : {p_val:.6e}")
#     print(f"   Parameters       : {params}\n")
#---------------------------------------------------------------------- BANK MARKETING DATASET ----------------------------------------------------------------------
# load the bank_marketing_debiased.json as a dataframe
# data = pd.read_json(Path('Datasets/bank_marketing_debiased.json'))
# print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
# print(data.info())
# # convert age,workclass,education,marital-status,occupation,relationship,race,sex,native-country,income to categorical
# categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'poutcome', 'y']
# data[categorical_columns] = data[categorical_columns].astype('category')
# # print range of values per column
# for column in data.columns:
#     if pd.api.types.is_numeric_dtype(data[column]):
#         print(f"{column}: min={data[column].min()}, max={data[column].max()}")
#     else:
#         print(f"{column}: unique values={data[column].nunique()}")  # For categorical columns, show unique counts

# # print distribution of each column
# for column in data.columns:
#     if pd.api.types.is_numeric_dtype(data[column]):
#         print(f"{column} distribution:\n{data[column].describe()}\n")
#     else:
#         categories, weights = extract_category_weights(data, column)
#         print(f"{column} categories and weights:")
#         print("Categories:", categories)
#         print("Weights:", weights)
#         # print(f"{column} value counts:\n{data[column].value_counts()}\n")

# # Run the distribution comparison
# dist_results = compare_distributions(data)

# print("Best-Fitting Distributions per Column:\n")
# for col_name, fits in dist_results.items():
#     if not fits:
#         print(f"Column '{col_name}' has no fitting results.")
#         continue

#     best_fit = min(fits, key=lambda x: x[1])  # x[1] = D-statistic
#     dist_name, D_stat, p_val, params = best_fit

#     print(f"Column: {col_name}")
#     print(f"   Best Distribution: {dist_name}")
#     print(f"   KS D-statistic   : {D_stat:.6f}")
#     print(f"   p-value          : {p_val:.6e}")
#     print(f"   Parameters       : {params}\n")

#---------------------------------------------------------------------- AIR QUALITY DATASET ----------------------------------------------------------------------
# # load the air_quality_modified.csv as a dataframe
# data = pd.read_csv(Path('Datasets/air_quality_modified.csv'))
# clean_lambdas, noisy_lambdas = generate_fourier_lambdas_with_noise(data, top_n=5)
# print("Generated Fourier lambdas:")
# # iterate over lambdas and print them
# for col, lambda_code in clean_lambdas.items():
#     print(f"{col}: {lambda_code['expr']}")
# for col, lambda_code in noisy_lambdas.items():
#     print(f"{col} (noisy): {lambda_code['expr']}")

#---------------------------------------------------------------------- INDIVIDUAL HOUSEHOLD POWER CONSUMPTION DATASET ----------------------------------------------------------------------
data = pd.read_csv(Path('Datasets/individual_household_electric_power_consumption_modified.csv'))
clean_lambdas, noisy_lambdas = generate_fourier_lambdas_with_noise(data, top_n=5)
print("Generated Fourier lambdas:")
# iterate over lambdas and print them
for col, lambda_code in clean_lambdas.items():
    print(f"{col}: {lambda_code['expr']}")
for col, lambda_code in noisy_lambdas.items():
    print(f"{col} (noisy): {lambda_code['expr']}")