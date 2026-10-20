import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import rstr
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import truncnorm, beta, gamma, weibull_min, triang, pareto, cauchy, chi2
try:
    from value_functions import VALUE_FUNC_FACTORY_REGISTRY
except:
    from bias_mitigation.synthetic_data.value_functions import VALUE_FUNC_FACTORY_REGISTRY

# ============================
# Helper Functions
# ============================

def generate_random_string(length):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=length))

def generate_pattern_string(pattern, n):
    return [rstr.xeger(pattern) for _ in range(n)]

def inject_nulls(values, null_prob):
    return [val if random.random() > null_prob else None for val in values]

def generate_skewed_category(categories, weights, n, balanced=False):
    # If I want to generate a balanced dataset (fairness across categories), I will use equal weights
    # for all categories
    if balanced:
        weights = [1.0 / len(categories)] * len(categories)
    return random.choices(categories, weights=weights, k=n)

# generate balanced boolean values
def balanced_boolean(n):
    values = [True] * (n // 2) + [False] * (n - n // 2)
    random.shuffle(values)
    return values

def balanced_ref_source(source_list, n):
    ref_pool = source_list.copy()
    random.shuffle(ref_pool)
    return [ref_pool[i % len(ref_pool)] for i in range(n)]


def generate_ids(n, prefix='ID'):
    return [f"{prefix}{i:06d}" for i in range(n)]

def generate_uniform_category(categories, n, balanced=False):
    """
    Generate n samples from given categories with uniform probability.
    
    If balanced=True, tries to distribute samples equally across all categories.
    """
    categories = list(categories)
    num_categories = len(categories)

    if balanced:
        # Repeat categories equally
        reps = n // num_categories
        remainder = n % num_categories
        values = categories * reps + list(np.random.choice(categories, remainder, replace=True))
        np.random.shuffle(values)
    else:
        # Uniform random choice
        values = np.random.choice(categories, n, replace=True)

    return values


# ============================
# Value Function Registry
# ============================

def sin_wave_factory():
    return lambda i: 10 + 5 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 1)

def trend_noise_factory(slope=1.0, intercept=0):
    return lambda i: intercept + slope * i + np.random.normal(0, 1)

def flat_random_factory(min_val=0, max_val=10):
    return lambda i: np.random.uniform(min_val, max_val)

# ============================
# Base Generator
# ============================

class AdvancedSyntheticDataGenerator:
    def __init__(self, schema, constraints=None):
        self.schema = schema
        self.constraints = constraints or []

    def generate_column(self, col_def, n):
        col_type = col_def['type']

        if col_type == 'string':
            values = [generate_random_string(col_def.get('length', 10)) for _ in range(n)]
        elif col_type == 'pattern':
            values = generate_pattern_string(col_def['pattern'], n)
        elif col_type == 'int':
            dist = col_def.get('distribution', 'uniform')
            if dist == 'normal':
                values = np.random.normal(loc=col_def['mean'], scale=col_def['std'], size=n)
            elif dist == 'truncnorm':
                a = (col_def['min'] - col_def['mean']) / col_def['std']
                b = (col_def['max'] - col_def['mean']) / col_def['std']
                values = truncnorm.rvs(a, b, loc=col_def['mean'], scale=col_def['std'], size=n)
            elif dist == 'gamma':
                values = gamma.rvs(a=col_def['a'], scale=col_def['scale'], size=n)
            elif dist == 'weibull_min':
                values = weibull_min.rvs(c=col_def['c'], scale=col_def['scale'], size=n)
            elif dist == 'beta':
                raw = beta.rvs(a=col_def['a'], b=col_def['b'], size=n)
                values = col_def['min'] + raw * (col_def['max'] - col_def['min'])
            elif dist == 'triangular':
                c = (col_def['mode'] - col_def['min']) / (col_def['max'] - col_def['min'])
                values = triang.rvs(c, loc=col_def['min'], scale=col_def['max'] - col_def['min'], size=n)
            elif dist == 'pareto':
                values = pareto.rvs(b=col_def['shape'], size=n) + col_def['min']
            elif dist == 'cauchy':
                values = cauchy.rvs(loc=col_def['loc'], scale=col_def['scale'], size=n)
            elif dist == 'chi2':
                values = chi2.rvs(df=col_def['df'], size=n)
            else:
                values = np.random.randint(col_def['min'], col_def['max'] + 1, size=n)

            values = np.round(values).astype(int).tolist()
            if col_def.get('clip'):
                values = np.clip(values, col_def['min'], col_def['max'])

        elif col_type == 'float':
            dist = col_def.get('distribution', 'uniform')
            if dist == 'normal':
                values = np.random.normal(loc=col_def['mean'], scale=col_def['std'], size=n)
            elif dist == 'lognormal':
                values = np.random.lognormal(mean=col_def['mean'], sigma=col_def['std'], size=n)
            elif dist == 'truncnorm':
                a = (col_def['min'] - col_def['mean']) / col_def['std']
                b = (col_def['max'] - col_def['mean']) / col_def['std']
                values = truncnorm.rvs(a, b, loc=col_def['mean'], scale=col_def['std'], size=n)
            elif dist == 'trunclognorm':
                raw = np.random.lognormal(mean=col_def['mean'], sigma=col_def['std'], size=n)
                values = np.clip(raw, col_def['min'], col_def['max'])
            elif dist == 'exponential':
                lam = 1 / col_def['mean']
                raw = np.random.exponential(scale=1/lam, size=n)
                values = np.clip(raw, col_def['min'], col_def['max'])
            elif dist == 'gamma':
                values = gamma.rvs(a=col_def['shape'], scale=col_def['scale'], size=n)
            elif dist == 'weibull':
                values = weibull_min.rvs(c=col_def['shape'], scale=col_def['scale'], size=n)
            elif dist == 'beta':
                raw = beta.rvs(a=col_def['a'], b=col_def['b'], size=n)
                values = col_def['min'] + raw * (col_def['max'] - col_def['min'])
            elif dist == 'triangular':
                c = (col_def['mode'] - col_def['min']) / (col_def['max'] - col_def['min'])
                values = triang.rvs(c, loc=col_def['min'], scale=col_def['max'] - col_def['min'], size=n)
            elif dist == 'pareto':
                values = pareto.rvs(b=col_def['shape'], size=n) + col_def['min']
            elif dist == 'cauchy':
                values = cauchy.rvs(loc=col_def['mean'], scale=col_def['std'], size=n)
            elif dist == 'chi2':
                values = chi2.rvs(df=col_def['df'], size=n)
            else:
                values = np.random.uniform(col_def['min'], col_def['max'], size=n)

            values = values.tolist()
            if col_def.get('clip'):
                values = np.clip(values, col_def['min'], col_def['max'])

        elif col_type == 'bool':
            values = balanced_boolean(n) if col_def.get('balanced') else np.random.choice([True, False], size=n).tolist()
        elif col_type == 'date':
            start = datetime.strptime(col_def['start'], '%Y-%m-%d')
            end = datetime.strptime(col_def['end'], '%Y-%m-%d')
            values = [(start + timedelta(days=random.randint(0, (end - start).days))).strftime('%Y-%m-%d') for _ in range(n)]
        elif col_type == 'category':
            # values = generate_skewed_category(col_def['categories'], col_def['weights'], n, balanced=col_def.get('balanced', False))
            categories = col_def['categories']
            weights = col_def.get('weights')  # Safe access
            balanced = col_def.get('balanced', False)
            if weights:
                values = generate_skewed_category(categories, weights, n, balanced=balanced)
            else:
                values = generate_uniform_category(categories, n, balanced=balanced)
        elif col_type == 'id':
            values = generate_ids(n, prefix=col_def.get('prefix', 'ID'))
        elif col_type == 'ref':
            parent_col = col_def['source']
            source_vals = self.generated_data.get(parent_col)
            if not source_vals:
                raise ValueError(f"Reference source column '{parent_col}' must be defined before '{col_def}'")
            use_balanced = col_def.get('balanced', False)
            if use_balanced:
                values = balanced_ref_source(source_vals, n)
            else:
                values = random.choices(source_vals, k=n)
        else:
            raise ValueError(f"Unsupported column type: {col_type}")

        if 'null_prob' in col_def:
            values = inject_nulls(values, col_def['null_prob'])

        return values

    def generate(self, n_rows=10):
        self.generated_data = {}
        for col_name, col_def in self.schema.items():
            if col_def['type'] != 'timeseries':  # defer timeseries to subclass
                self.generated_data[col_name] = self.generate_column(col_def, n_rows)
        df = pd.DataFrame(self.generated_data)

        for func in self.constraints:
            df = func(df)

        return df

# ============================
# Time Series Generator
# ============================

class TimeSeriesSyntheticGenerator(AdvancedSyntheticDataGenerator):
    def __init__(self, schema, constraints=None):
        super().__init__(schema, constraints)

    def generate(self, n_rows=1):
        base_df = super().generate(n_rows=n_rows)
        ts_defs = {col: defn for col, defn in self.schema.items() if defn['type'] == 'timeseries'}

        if not ts_defs:
            return base_df  # no timeseries required

        full_data = []
        for _, row in base_df.iterrows():
            base_row = row.to_dict()
            for ts_col, ts_def in ts_defs.items():
                entity_id = base_row[ts_def['entity_col']]
                start = pd.to_datetime(ts_def['start'])
                periods = ts_def['periods']
                freq = ts_def['freq']
                value_func_def = ts_def.get('value_func')

                if isinstance(value_func_def, str):
                    factory = VALUE_FUNC_FACTORY_REGISTRY.get(value_func_def)
                    if not factory:
                        raise ValueError(f"Unknown value function: {value_func_def}")
                    value_func = factory({})

                elif isinstance(value_func_def, dict):
                    name = value_func_def.get("name")
                    params = value_func_def.get("params", {})
                    factory = VALUE_FUNC_FACTORY_REGISTRY.get(name)
                    if not factory:
                        raise ValueError(f"Unknown value function: {name}")
                    value_func = factory(params)

                else:
                    raise ValueError(f"Invalid value_func definition: {value_func_def}")


                timestamps = pd.date_range(start=start, periods=periods, freq=freq)
                values = [value_func(i) for i in range(periods)]
                for t, v in zip(timestamps, values):
                    new_row = base_row.copy()
                    new_row['timestamp'] = t
                    new_row[ts_col] = v
                    full_data.append(new_row)
        # Group by date and take the max (or first non-null) of each group
        timeseries_data = pd.DataFrame(full_data).groupby('timestamp', as_index=False).agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)
        # check if the n_rows is equal to 1
        if n_rows == 1:
            # get the column that contains only one value and is not numeric
            entity_col = [
                col for col in timeseries_data.columns
                if timeseries_data[col].nunique(dropna=True) == 1 and not pd.api.types.is_numeric_dtype(timeseries_data[col])
            ]
            # drop the entity column
            if entity_col:
                timeseries_data = timeseries_data.drop(columns=entity_col)
        # drop row with all null values
        timeseries_data = timeseries_data.dropna(how='any')

        return timeseries_data

# ============================
# Save / Load Support
# ============================

class SyntheticIO:
    @staticmethod
    def save_schema(schema, filepath):
        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2, default=str)

    @staticmethod
    def load_schema(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_data(df, filepath, format='csv'):
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', date_format='iso', indent=2)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'.")

    @staticmethod  
    def save_plotly_as_html(fig, file):
        """Append Plotly figure HTML to a file."""
        fig.write_html(file, full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def audit_bias(df, categorical_threshold=10, visualize=False, save_dir="plots", html_output="data_generator_plots.html"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create or clear the output HTML file
        html_path = os.path.join(save_dir, html_output)
        with open(html_path, 'w') as f:
            f.write("<html><head><title>Bias Audit Plots</title></head><body>")  # Start the HTML file with <body>

        report = {}

        for col in df.columns:
            col_data = df[col]

            null_ratio = col_data.isnull().mean()
            non_null = col_data.dropna()

            if non_null.empty:
                report[col] = {
                    "note": "All values are null",
                    "null_ratio": 1.0
                }
                continue

            n_unique = non_null.nunique()

            try:
                if pd.api.types.is_bool_dtype(non_null) or n_unique <= categorical_threshold:
                    counts = non_null.value_counts(normalize=True)
                    entropy = -sum(p * np.log2(p) for p in counts if p > 0)
                    max_class_ratio = counts.iloc[0]

                    report[col] = {
                        "type": "categorical",
                        "unique_values": int(n_unique),
                        "most_common_ratio": round(max_class_ratio, 3),
                        "entropy": round(entropy, 3),
                        "is_skewed": bool(max_class_ratio > 0.7 or entropy < 1.0),
                        "null_ratio": round(null_ratio, 3)
                    }

                    if visualize:
                        # Create Plotly bar chart for categorical feature
                        fig = px.bar(counts, x=counts.index, y=counts.values, labels={'x': col, 'y': 'Proportion'})
                        fig.update_layout(title=f'Categorical Distribution of {col}')
                        # add x label
                        fig.update_xaxes(title_text=col)
                        # Append the HTML for this plot to the file
                        with open(html_path, 'a') as f:  # Open in append mode
                            SyntheticIO.save_plotly_as_html(fig, f)

                elif pd.api.types.is_numeric_dtype(non_null):
                    stats = {
                        "mean": round(non_null.mean(), 2),
                        "std": round(non_null.std(), 2),
                        "skew": round(non_null.skew(), 2),
                        "kurtosis": round(non_null.kurtosis(), 2)
                    }
                    report[col] = {
                        "type": "numeric",
                        **stats,
                        "is_skewed": abs(stats["skew"]) > 1.0,
                        "null_ratio": round(null_ratio, 3)
                    }

                    if visualize:
                        # Create Plotly histogram for numeric feature
                        fig = px.histogram(non_null, x=non_null, nbins=20, labels={col: 'Frequency'})
                        fig.update_layout(title=f'Numeric Distribution of {col}')
                        # add x label
                        fig.update_xaxes(title_text=col)
                        # Append the HTML for this plot to the file
                        with open(html_path, 'a') as f:  # Open in append mode
                            SyntheticIO.save_plotly_as_html(fig, f)

                else:
                    report[col] = {
                        "type": "unsupported",
                        "note": f"Type {col_data.dtype} not audited",
                        "null_ratio": round(null_ratio, 3)
                    }

            except Exception as e:
                report[col] = {
                    "type": "error",
                    "error": str(e),
                    "null_ratio": round(null_ratio, 3)
                }

        return report