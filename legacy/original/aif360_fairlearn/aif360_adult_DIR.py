import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Number of trees
N_ESTIMATORS = 200

# Protected attribute:
# "sex"  -> Female vs Male
# "race" -> Non-White vs White
PROTECTED_ATTRIBUTE = "race"

# Disparate Impact Remover repair level:
# 0.0 = no repair
# 1.0 = full repair
REPAIR_LEVEL = 1.0


# ============================================================
# 1. LOAD ADULT DATASET
# ============================================================

print("=" * 70)
print("1. Loading Adult dataset")
print("=" * 70)

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy()

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. CLEAN MISSING VALUES
# ============================================================

df = df.replace("?", np.nan)

print("\nMissing values:")
print(df.isna().sum())

df = df.dropna().reset_index(drop=True)

print("\nShape after removing missing values:", df.shape)


# ============================================================
# 4. IDENTIFY TARGET
# ============================================================

print("\nTarget column:")
print(adult.target_names)

print("\nTarget values:")
print(df["class"].value_counts())

df["income"] = (
    df["class"]
    .astype(str)
    .str.strip()
    .str.startswith(">50K")
    .astype(int)
)

df = df.drop(columns=["class"])


# ============================================================
# 5. PROTECTED ATTRIBUTE
# ============================================================

if PROTECTED_ATTRIBUTE == "sex":

    # Female = 0
    # Male   = 1

    df["protected"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .map({
            "Female": 0,
            "Male": 1
        })
    )

    group_names = {
        0: "Female",
        1: "Male"
    }

elif PROTECTED_ATTRIBUTE == "race":

    # Amer-Indian-Eskimo = 0
    # Asian-Pac-Islander = 1
    # Black = 2
    # Other = 3
    # White = 4

    df["protected"] = (
        df["race"]
        .astype(str)
        .str.strip()
        .map({
            "Amer-Indian-Eskimo": 0,
            "Asian-Pac-Islander": 1,
            "Black": 2,
            "Other": 3,
            "White": 4
        })
    )

    group_names = {
        0: "Amer-Indian-Eskimo",
        1: "Asian-Pac-Islander",
        2: "Black",
        3: "Other",
        4: "White"
    }

else:
    raise ValueError(
        "PROTECTED_ATTRIBUTE must be 'sex' or 'race'"
    )


if df["protected"].isna().any():
    raise ValueError(
        f"Unexpected values found in {PROTECTED_ATTRIBUTE}"
    )


print("\nProtected attribute:", PROTECTED_ATTRIBUTE)

print(
    df["protected"]
    .map(group_names)
    .value_counts()
)


# ============================================================
# 6. CREATE X AND y
# ============================================================

columns_to_remove = [
    "income",
    "protected",
    PROTECTED_ATTRIBUTE
]

X = df.drop(columns=columns_to_remove)

y = df["income"].astype(int)

protected = df["protected"].astype(int)


# ============================================================
# 7. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train_raw, X_test_raw, y_train, y_test, protected_train, protected_test = (
    train_test_split(
        X,
        y,
        protected,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print("\nTraining samples:", len(X_train_raw))
print("Testing samples:", len(X_test_raw))


# ============================================================
# 9. ONE-HOT ENCODE FEATURES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

print("\nEncoded training shape:", X_train.shape)
print("Encoded testing shape:", X_test.shape)

X_train = np.asarray(X_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)

y_train = np.asarray(y_train, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.float64)

protected_train = np.asarray(
    protected_train,
    dtype=np.float64
)

protected_test = np.asarray(
    protected_test,
    dtype=np.float64
)


# ============================================================
# 10. BASELINE RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("2. Training baseline Random Forest")
print("=" * 70)

rf_baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None
)

rf_baseline.fit(
    X_train,
    y_train
)

y_pred_baseline = rf_baseline.predict(
    X_test
)

baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy:.4f}"
)


# ============================================================
# 11. CREATE AIF360 TRAINING DATASET
# ============================================================

print("\n" + "=" * 70)
print("3. Creating AIF360 training dataset")
print("=" * 70)

feature_names = [
    f"feature_{i}"
    for i in range(X_train.shape[1])
]

aif_train_df = pd.DataFrame(
    X_train,
    columns=feature_names
)

aif_train_df["protected"] = protected_train
aif_train_df["income"] = y_train

aif_train_df = aif_train_df.astype(
    np.float64
)

aif_train = BinaryLabelDataset(
    favorable_label=1.0,
    unfavorable_label=0.0,
    df=aif_train_df,
    label_names=["income"],
    protected_attribute_names=["protected"]
)

print(
    "AIF360 dataset created successfully."
)

print(
    "Features:",
    aif_train.features.shape
)


# ============================================================
# 12. APPLY AIF360 DISPARATE IMPACT REMOVER
# ============================================================

print("\n" + "=" * 70)
print("4. Applying AIF360 Disparate Impact Remover")
print("=" * 70)

print(
    "Repair level:",
    REPAIR_LEVEL
)

dir_remover = DisparateImpactRemover(
    repair_level=REPAIR_LEVEL,
    sensitive_attribute="protected"
)

aif_train_repaired = dir_remover.fit_transform(
    aif_train
)

print(
    "Training features repaired successfully."
)


# ============================================================
# 13. CREATE AIF360 TEST DATASET
# ============================================================

aif_test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)

aif_test_df["protected"] = protected_test
aif_test_df["income"] = y_test

aif_test_df = aif_test_df.astype(
    np.float64
)

aif_test = BinaryLabelDataset(
    favorable_label=1.0,
    unfavorable_label=0.0,
    df=aif_test_df,
    label_names=["income"],
    protected_attribute_names=["protected"]
)

aif_test_repaired = dir_remover.fit_transform(
    aif_test
)

print(
    "Testing features repaired successfully."
)


# ============================================================
# 14. EXTRACT REPAIRED FEATURES
# ============================================================

protected_index = (
    aif_train_repaired
    .feature_names
    .index("protected")
)

X_train_repaired = np.delete(
    aif_train_repaired.features,
    protected_index,
    axis=1
)

X_test_repaired = np.delete(
    aif_test_repaired.features,
    protected_index,
    axis=1
)

X_train_repaired = np.asarray(
    X_train_repaired,
    dtype=np.float64
)

X_test_repaired = np.asarray(
    X_test_repaired,
    dtype=np.float64
)

print(
    "\nRepaired training shape:",
    X_train_repaired.shape
)

print(
    "Repaired testing shape:",
    X_test_repaired.shape
)


# ============================================================
# 15. TRAIN DIR RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("5. Training DIR Random Forest")
print("=" * 70)

rf_dir = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None
)

rf_dir.fit(
    X_train_repaired,
    y_train
)

y_pred_dir = rf_dir.predict(
    X_test_repaired
)

dir_accuracy = accuracy_score(
    y_test,
    y_pred_dir
)

print(
    f"DIR accuracy: "
    f"{dir_accuracy:.4f}"
)


# ============================================================
# 16. FUNCTION: GROUP METRICS
# ============================================================

def calculate_group_metrics(
    y_true,
    y_pred,
    protected_attribute
):

    results = []

    groups = np.unique(
        protected_attribute
    )

    for group in groups:

        mask = (
            protected_attribute == group
        )

        yt = np.asarray(
            y_true
        )[mask].astype(int)

        yp = np.asarray(
            y_pred
        )[mask].astype(int)

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
            labels=[0, 1]
        ).ravel()

        fpr_denominator = fp + tn

        if fpr_denominator > 0:
            fpr = fp / fpr_denominator
        else:
            fpr = np.nan

        fnr_denominator = fn + tp

        if fnr_denominator > 0:
            fnr = fn / fnr_denominator
        else:
            fnr = np.nan

        if fnr_denominator > 0:
            tpr = tp / fnr_denominator
        else:
            tpr = np.nan

        selection_rate = np.mean(
            yp == 1
        )

        results.append({
            "group_value": group,
            "group": group_names[group],
            "N": len(yt),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "FPR": fpr,
            "FNR": fnr,
            "TPR": tpr,
            "Selection_Rate": selection_rate
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# 17. FUNCTION: FAIRNESS METRICS
# ============================================================

def calculate_fairness_metrics(
    group_metrics
):

    privileged_value = 1
    unprivileged_value = 0

    privileged = group_metrics[
        group_metrics["group_value"] == privileged_value
    ].iloc[0]

    unprivileged = group_metrics[
        group_metrics["group_value"] == unprivileged_value
    ].iloc[0]

    dp_difference = (
        unprivileged["Selection_Rate"]
        -
        privileged["Selection_Rate"]
    )

    fpr_difference = (
        unprivileged["FPR"]
        -
        privileged["FPR"]
    )

    fnr_difference = (
        unprivileged["FNR"]
        -
        privileged["FNR"]
    )

    tpr_difference = (
        unprivileged["TPR"]
        -
        privileged["TPR"]
    )

    average_odds_difference = (
        fpr_difference
        +
        tpr_difference
    ) / 2

    equalized_odds_gap = max(
        abs(fpr_difference),
        abs(tpr_difference)
    )

    return {
        "DP_difference": dp_difference,
        "FPR_difference": fpr_difference,
        "FNR_difference": fnr_difference,
        "TPR_difference": tpr_difference,
        "Average_Odds_Difference":
            average_odds_difference,
        "Equalized_Odds_Gap":
            equalized_odds_gap
    }


# ============================================================
# 18. EVALUATE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("6. BASELINE FAIRNESS RESULTS")
print("=" * 70)

baseline_group_metrics = calculate_group_metrics(
    y_test,
    y_pred_baseline,
    protected_test
)

print("\nPer-group metrics:")

print(
    baseline_group_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

baseline_fairness = calculate_fairness_metrics(
    baseline_group_metrics
)

print("\nBaseline fairness:")

for metric, value in baseline_fairness.items():
    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 19. EVALUATE DIR MODEL
# ============================================================

print("\n" + "=" * 70)
print("7. DIR FAIRNESS RESULTS")
print("=" * 70)

dir_group_metrics = calculate_group_metrics(
    y_test,
    y_pred_dir,
    protected_test
)

print("\nPer-group metrics:")

print(
    dir_group_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

dir_fairness = calculate_fairness_metrics(
    dir_group_metrics
)

print("\nDIR fairness:")

for metric, value in dir_fairness.items():
    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 20. FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("8. BASELINE vs DIR")
print("=" * 70)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "DP Difference",
        "FPR Difference",
        "FNR Difference",
        "TPR Difference",
        "Average Odds Difference",
        "Equalized Odds Gap"
    ],

    "Baseline RF": [
        baseline_accuracy,
        baseline_fairness["DP_difference"],
        baseline_fairness["FPR_difference"],
        baseline_fairness["FNR_difference"],
        baseline_fairness["TPR_difference"],
        baseline_fairness["Average_Odds_Difference"],
        baseline_fairness["Equalized_Odds_Gap"]
    ],

    "DIR RF": [
        dir_accuracy,
        dir_fairness["DP_difference"],
        dir_fairness["FPR_difference"],
        dir_fairness["FNR_difference"],
        dir_fairness["TPR_difference"],
        dir_fairness["Average_Odds_Difference"],
        dir_fairness["Equalized_Odds_Gap"]
    ]
})

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 21. PER-GROUP COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("9. PER-GROUP COMPARISON")
print("=" * 70)

baseline_tmp = baseline_group_metrics[
    [
        "group",
        "N",
        "FPR",
        "FNR",
        "TPR",
        "Selection_Rate"
    ]
].copy()

baseline_tmp["Model"] = "Baseline RF"

dir_tmp = dir_group_metrics[
    [
        "group",
        "N",
        "FPR",
        "FNR",
        "TPR",
        "Selection_Rate"
    ]
].copy()

dir_tmp["Model"] = "DIR RF"

per_group_comparison = pd.concat(
    [
        baseline_tmp,
        dir_tmp
    ],
    ignore_index=True
)

per_group_comparison = per_group_comparison[
    [
        "Model",
        "group",
        "N",
        "FPR",
        "FNR",
        "TPR",
        "Selection_Rate"
    ]
]

print(
    per_group_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 22. SAVE RESULTS
# ============================================================

comparison.to_csv(
    f"results/aif360_{PROTECTED_ATTRIBUTE}_adult_rf_DIR_fairness_comparison.csv",
    index=False
)

per_group_comparison.to_csv(
    f"results/aif360_{PROTECTED_ATTRIBUTE}_adult_rf_DIR_per_group_metrics.csv",
    index=False
)

print("\nResults saved:")

print(
    f"  results/aif360_{PROTECTED_ATTRIBUTE}_adult_rf_DIR_fairness_comparison.csv"
)

print(
    f"  results/aif360_{PROTECTED_ATTRIBUTE}_adult_rf_DIR_per_group_metrics.csv"
)
