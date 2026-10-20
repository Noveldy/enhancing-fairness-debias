import io
import zipfile
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from fairlearn.metrics import MetricFrame
from fairlearn.postprocessing import ThresholdOptimizer


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30


# ============================================================
# 1. LOAD BANK MARKETING DATASET
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

response = requests.get(url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open("bank-full.csv") as f:
        df = pd.read_csv(f, sep=";")


print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. CLEAN DATA
# ============================================================

df = df.replace("?", np.nan).dropna().reset_index(drop=True)

print("Shape after removing missing values:", df.shape)


# ============================================================
# 3. TARGET
# ============================================================

df["target"] = df["y"].map({"yes": 1, "no": 0}).astype(int)
df = df.drop(columns=["y"])


# ============================================================
# 4. THREE MARITAL GROUPS
# ============================================================

marital_mapping = {"divorced": 0, "married": 1, "single": 2}

df["marital_group"] = df["marital"].map(marital_mapping).astype(int)

GROUP_NAMES = {0: "Divorced", 1: "Married", 2: "Single"}

print("\nMarital groups:")
print(df["marital"].value_counts())


# ============================================================
# 5. FEATURES / TARGET / PROTECTED ATTRIBUTE
# ============================================================

X = df.drop(columns=["target", "marital", "marital_group"])
y = df["target"]
protected = df["marital_group"]


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train_raw, X_test_raw, y_train, y_test, protected_train, protected_test = train_test_split(
    X, y, protected, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)


# ============================================================
# 7. ENCODE CATEGORICAL VARIABLES
# ============================================================

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ("numerical", "passthrough", numerical_columns)
])

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

X_train = np.asarray(X_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)


# ============================================================
# 8. METRIC FUNCTIONS
# ============================================================

def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan


def fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan


def tpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (fn + tp) if (fn + tp) > 0 else np.nan


metrics = {
    "FPR": fpr,
    "FNR": fnr,
    "TPR": tpr,
    "DP": lambda y_true, y_pred: np.mean(y_pred == 1)
}


# ============================================================
# 9. FUNCTION TO EVALUATE MODEL
# ============================================================

def evaluate_model(name, y_true, y_pred, protected_attribute):

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=protected_attribute
    )

    by_group = metric_frame.by_group.copy()
    by_group.index = by_group.index.map(GROUP_NAMES)

    dp = metric_frame.by_group["DP"]
    fpr_values = metric_frame.by_group["FPR"]
    tpr_values = metric_frame.by_group["TPR"]

    dp_gap = dp.max() - dp.min()
    dp_ratio = dp.min() / dp.max() if dp.max() > 0 else np.nan
    fpr_gap = fpr_values.max() - fpr_values.min()
    fnr_gap = metric_frame.by_group["FNR"].max() - metric_frame.by_group["FNR"].min()
    tpr_gap = tpr_values.max() - tpr_values.min()
    eo_gap = max(fpr_gap, tpr_gap)

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)
    print("\nMetrics per marital group:")
    print(by_group.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nOverall metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"DP Gap: {dp_gap:.4f}")
    print(f"DP Ratio: {dp_ratio:.4f}")
    print(f"FPR Gap: {fpr_gap:.4f}")
    print(f"FNR Gap: {fnr_gap:.4f}")
    print(f"TPR Gap: {tpr_gap:.4f}")
    print(f"EO Gap: {eo_gap:.4f}")

    return by_group, {
        "Accuracy": accuracy,
        "DP Gap": dp_gap,
        "DP Ratio": dp_ratio,
        "FPR Gap": fpr_gap,
        "FNR Gap": fnr_gap,
        "TPR Gap": tpr_gap,
        "EO Gap": eo_gap
    }


# ============================================================
# 10. BASELINE DECISION TREE
# ============================================================

dt = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    max_depth=5,
    min_samples_leaf=20
)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

baseline_group, baseline_overall = evaluate_model(
    "BASELINE DECISION TREE",
    y_test,
    y_pred,
    protected_test
)


# ============================================================
# 11. FAIRLEARN DEMOGRAPHIC PARITY MITIGATION
# ============================================================

threshold_optimizer = ThresholdOptimizer(
    estimator=dt,
    constraints="demographic_parity",
    predict_method="predict_proba",
    prefit=True
)

threshold_optimizer.fit(
    X_train,
    y_train,
    sensitive_features=protected_train
)


# ============================================================
# 12. MITIGATED PREDICTIONS
# ============================================================

y_pred_fair = threshold_optimizer.predict(
    X_test,
    sensitive_features=protected_test
)


fair_group, fair_overall = evaluate_model(
    "FAIRLEARN DEMOGRAPHIC PARITY + DECISION TREE",
    y_test,
    y_pred_fair,
    protected_test
)


# ============================================================
# 13. BASELINE VS FAIRLEARN
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "DP Gap",
        "DP Ratio",
        "FPR Gap",
        "FNR Gap",
        "TPR Gap",
        "EO Gap"
    ],
    "Baseline DT": [
        baseline_overall["Accuracy"],
        baseline_overall["DP Gap"],
        baseline_overall["DP Ratio"],
        baseline_overall["FPR Gap"],
        baseline_overall["FNR Gap"],
        baseline_overall["TPR Gap"],
        baseline_overall["EO Gap"]
    ],
    "Fairlearn DT": [
        fair_overall["Accuracy"],
        fair_overall["DP Gap"],
        fair_overall["DP Ratio"],
        fair_overall["FPR Gap"],
        fair_overall["FNR Gap"],
        fair_overall["TPR Gap"],
        fair_overall["EO Gap"]
    ]
})


print("\n" + "=" * 65)
print("BASELINE VS FAIRLEARN")
print("=" * 65)
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 14. PER-GROUP COMPARISON
# ============================================================

baseline_group = baseline_group.reset_index().rename(columns={"index": "Group"})
fair_group = fair_group.reset_index().rename(columns={"index": "Group"})

baseline_group.insert(0, "Model", "Baseline DT")
fair_group.insert(0, "Model", "Fairlearn DT")

per_group = pd.concat(
    [baseline_group, fair_group],
    ignore_index=True
)

print("\n" + "=" * 65)
print("PER-GROUP COMPARISON")
print("=" * 65)
print(per_group.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 15. SAVE RESULTS
# ============================================================

comparison.to_csv("results/fairlearn_bank_dt_comparison.csv", index=False)
per_group.to_csv("results/fairlearn_bank_dt_per_group.csv", index=False)

print("\nResults saved.")