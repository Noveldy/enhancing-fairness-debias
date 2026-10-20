import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30

N_ESTIMATORS = 200

# Try 0.0, 0.5, 0.8, 1.0
#
# 0.0 = no repair
# 1.0 = full repair
#
REPAIR_LEVEL = 1.0


# ============================================================
# MARITAL GROUPS
# ============================================================

MARITAL_MAPPING = {
    "divorced": 0,
    "married": 1,
    "single": 2
}

GROUP_NAMES = {
    0: "Divorced",
    1: "Married",
    2: "Single"
}


# ============================================================
# 1. LOAD BANK MARKETING DATASET
# ============================================================

import pandas as pd

url = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/00222/"
    "bank.zip"
)

import zipfile
import io
import requests

response = requests.get(url)

with zipfile.ZipFile(
    io.BytesIO(response.content)
) as z:

    with z.open("bank-full.csv") as f:

        df = pd.read_csv(
            f,
            sep=";"
        )

print(df.shape)

print(df.columns.tolist())

print(df["y"].value_counts())

print(df["marital"].value_counts())


# ============================================================
# 2. CLEAN DATA
# ============================================================

# OpenML versions can contain ? as missing values

df = df.replace(
    "?",
    np.nan
)


# Remove missing values

df = (
    df
    .dropna()
    .reset_index(drop=True)
)


print(
    "\nShape after removing missing values:",
    df.shape
)


# ============================================================
# 3. TARGET VARIABLE
# ============================================================

print(
    "\nOriginal target distribution:"
)

print(
    df["y"].value_counts()
)


# Convert:
#
# yes -> 1
# no  -> 0

df["target"] = (
    df["y"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)


if df["target"].isna().any():

    raise ValueError(
        "Unexpected values found in target column."
    )


df["target"] = (
    df["target"]
    .astype(int)
)


# Remove original target

df = df.drop(
    columns=["y"]
)


# ============================================================
# 4. CREATE THREE MARITAL GROUPS
# ============================================================

print(
    "\nOriginal marital distribution:"
)

print(
    df["marital"].value_counts()
)


# Keep ALL THREE groups:
#
# divorced = 0
# married  = 1
# single   = 2

df["marital_group"] = (
    df["marital"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(MARITAL_MAPPING)
)


if df["marital_group"].isna().any():

    unexpected = (
        df.loc[
            df["marital_group"].isna(),
            "marital"
        ]
        .unique()
    )

    raise ValueError(
        f"Unexpected marital categories: {unexpected}"
    )


df["marital_group"] = (
    df["marital_group"]
    .astype(int)
)


print(
    "\nThree protected groups:"
)

print(
    df["marital_group"]
    .map(GROUP_NAMES)
    .value_counts()
)


# ============================================================
# 5. CREATE FEATURES AND TARGET
# ============================================================

# IMPORTANT:
#
# marital is removed from the model.
#
# marital_group is also removed from the model.
#
# The Decision Tree therefore does not directly receive
# marital status.

columns_to_remove = [
    "target",
    "marital",
    "marital_group"
]


X = df.drop(
    columns=columns_to_remove
)


y = df["target"]


protected = df["marital_group"]


# ============================================================
# 6. IDENTIFY FEATURE TYPES
# ============================================================

categorical_columns = (
    X
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)


numerical_columns = (
    X
    .select_dtypes(
        exclude=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)


print(
    "\nCategorical features:"
)

print(
    categorical_columns
)


print(
    "\nNumerical features:"
)

print(
    numerical_columns
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

(
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    protected_train,
    protected_test
) = train_test_split(

    X,
    y,
    protected,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train_raw)
)

print(
    "Testing samples:",
    len(X_test_raw)
)


# ============================================================
# 8. ONE-HOT ENCODE FEATURES
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


X_train = (
    preprocessor
    .fit_transform(X_train_raw)
)


X_test = (
    preprocessor
    .transform(X_test_raw)
)


# Explicit numerical conversion

X_train = np.asarray(
    X_train,
    dtype=np.float64
)


X_test = np.asarray(
    X_test,
    dtype=np.float64
)


y_train = np.asarray(
    y_train,
    dtype=np.float64
)


y_test = np.asarray(
    y_test,
    dtype=np.float64
)


protected_train = np.asarray(
    protected_train,
    dtype=np.float64
)


protected_test = np.asarray(
    protected_test,
    dtype=np.float64
)


print(
    "\nEncoded feature matrix:"
)

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)


# ============================================================
# 9. BASELINE DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("2. BASELINE DECISION TREE")
print("=" * 70)


dt_baseline = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)


dt_baseline.fit(
    X_train,
    y_train
)


y_pred_baseline = (
    dt_baseline
    .predict(X_test)
)


baseline_accuracy = (
    accuracy_score(
        y_test,
        y_pred_baseline
    )
)


print(
    f"Baseline accuracy: "
    f"{baseline_accuracy:.4f}"
)


# ============================================================
# 10. CREATE AIF360 TRAINING DATASET
# ============================================================

print("\n" + "=" * 70)
print("3. CREATING AIF360 DATASET")
print("=" * 70)


# Create column names

feature_names = [
    f"feature_{i}"
    for i in range(
        X_train.shape[1]
    )
]


aif_train_df = pd.DataFrame(
    X_train,
    columns=feature_names
)


# Add protected attribute

aif_train_df[
    "marital_group"
] = protected_train


# Add target

aif_train_df[
    "target"
] = y_train


# Make sure everything is numeric

aif_train_df = (
    aif_train_df
    .astype(np.float64)
)


# Create AIF360 dataset

aif_train = BinaryLabelDataset(

    favorable_label=1.0,

    unfavorable_label=0.0,

    df=aif_train_df,

    label_names=[
        "target"
    ],

    protected_attribute_names=[
        "marital_group"
    ]
)


print(
    "AIF360 dataset created."
)


print(
    "\nAIF360 protected attribute:"
)

print(
    aif_train.protected_attribute_names
)


# ============================================================
# 11. DISPARATE IMPACT REMOVER
# ============================================================

print("\n" + "=" * 70)
print("4. DISPARATE IMPACT REMOVER")
print("=" * 70)


print(
    f"Repair level: {REPAIR_LEVEL}"
)


print(
    "Protected groups:"
)

for value, name in GROUP_NAMES.items():

    print(
        f"  {value} = {name}"
    )


# Create DIR

dir_remover = (
    DisparateImpactRemover(

        repair_level=REPAIR_LEVEL,

        sensitive_attribute="marital_group"
    )
)


# Apply repair to training data

aif_train_repaired = (
    dir_remover
    .fit_transform(aif_train)
)


print(
    "\nDIR transformation completed."
)


# ============================================================
# 12. EXTRACT REPAIRED FEATURES
# ============================================================

# AIF360 keeps the protected attribute in the dataset.
#
# We remove it before training the Decision Tree.

protected_index = (
    aif_train_repaired
    .feature_names
    .index(
        "marital_group"
    )
)


X_train_repaired = (
    np.delete(
        aif_train_repaired.features,
        protected_index,
        axis=1
    )
)


y_train_repaired = (
    aif_train_repaired
    .labels
    .ravel()
)


X_train_repaired = np.asarray(
    X_train_repaired,
    dtype=np.float64
)


y_train_repaired = np.asarray(
    y_train_repaired,
    dtype=np.float64
)


print(
    "\nRepaired training shape:",
    X_train_repaired.shape
)


# ============================================================
# 13. REPAIR TEST DATA
# ============================================================

# Important:
#
# The test set must also be transformed before being given
# to the reweighted/repaired Decision Tree.
#
# AIF360's DIR documentation notes that test data should be
# transformed consistently with the training data. The
# distributions conditioned on the protected attribute need
# to be compatible for this transformation. 
#
# We therefore create an AIF360 test dataset with the exact
# same feature representation.

aif_test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)


aif_test_df[
    "marital_group"
] = protected_test


aif_test_df[
    "target"
] = y_test


aif_test_df = (
    aif_test_df
    .astype(np.float64)
)


aif_test = BinaryLabelDataset(

    favorable_label=1.0,

    unfavorable_label=0.0,

    df=aif_test_df,

    label_names=[
        "target"
    ],

    protected_attribute_names=[
        "marital_group"
    ]
)


# Apply the same DIR object

aif_test_repaired = (
    dir_remover
    .fit_transform(
        aif_test
    )
)


# Extract repaired test features

protected_index_test = (
    aif_test_repaired
    .feature_names
    .index(
        "marital_group"
    )
)


X_test_repaired = (
    np.delete(
        aif_test_repaired.features,
        protected_index_test,
        axis=1
    )
)


X_test_repaired = np.asarray(
    X_test_repaired,
    dtype=np.float64
)


print(
    "Repaired test shape:",
    X_test_repaired.shape
)


# ============================================================
# 14. REPAIRED DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("5. DECISION TREE AFTER DIR")
print("=" * 70)


dt_repaired = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)


dt_repaired.fit(

    X_train_repaired,

    y_train_repaired
)


y_pred_repaired = (
    dt_repaired
    .predict(
        X_test_repaired
    )
)


repaired_accuracy = (
    accuracy_score(
        y_test,
        y_pred_repaired
    )
)


print(
    f"DIR + DT accuracy: "
    f"{repaired_accuracy:.4f}"
)


# ============================================================
# 15. PER-GROUP METRICS
# ============================================================

def calculate_group_metrics(
    y_true,
    y_pred,
    protected_attribute
):

    results = []


    groups = sorted(
        np.unique(
            protected_attribute
        )
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


        tn, fp, fn, tp = (
            confusion_matrix(
                yt,
                yp,
                labels=[0, 1]
            )
            .ravel()
        )


        # ----------------------------------------------------
        # FPR
        # ----------------------------------------------------

        if (fp + tn) > 0:

            fpr = (
                fp /
                (fp + tn)
            )

        else:

            fpr = np.nan


        # ----------------------------------------------------
        # FNR
        # ----------------------------------------------------

        if (fn + tp) > 0:

            fnr = (
                fn /
                (fn + tp)
            )

        else:

            fnr = np.nan


        # ----------------------------------------------------
        # TPR
        # ----------------------------------------------------

        if (fn + tp) > 0:

            tpr = (
                tp /
                (fn + tp)
            )

        else:

            tpr = np.nan


        # ----------------------------------------------------
        # Selection rate
        # ----------------------------------------------------

        selection_rate = (
            np.mean(
                yp == 1
            )
        )


        results.append({

            "group_value":
                int(group),

            "group":
                GROUP_NAMES[int(group)],

            "N":
                len(yt),

            "TN":
                tn,

            "FP":
                fp,

            "FN":
                fn,

            "TP":
                tp,

            "FPR":
                fpr,

            "FNR":
                fnr,

            "TPR":
                tpr,

            "Selection_Rate":
                selection_rate
        })


    return pd.DataFrame(
        results
    )


# ============================================================
# 16. THREE-GROUP FAIRNESS METRICS
# ============================================================

def calculate_three_group_fairness(
    group_metrics
):

    selection_rates = (
        group_metrics[
            "Selection_Rate"
        ]
    )


    fprs = (
        group_metrics[
            "FPR"
        ]
    )


    fnrs = (
        group_metrics[
            "FNR"
        ]
    )


    tprs = (
        group_metrics[
            "TPR"
        ]
    )


    # ========================================================
    # DEMOGRAPHIC PARITY GAP
    # ========================================================
    #
    # Maximum difference in positive prediction rate
    # between any two marital groups.
    #
    # 0 = perfect DP

    dp_gap = (
        selection_rates.max()
        -
        selection_rates.min()
    )


    # ========================================================
    # DEMOGRAPHIC PARITY RATIO
    # ========================================================
    #
    # Minimum selection rate / maximum selection rate
    #
    # 1 = perfect DP

    if selection_rates.max() > 0:

        dp_ratio = (
            selection_rates.min()
            /
            selection_rates.max()
        )

    else:

        dp_ratio = np.nan


    # ========================================================
    # FPR GAP
    # ========================================================

    fpr_gap = (
        fprs.max()
        -
        fprs.min()
    )


    # ========================================================
    # FNR GAP
    # ========================================================

    fnr_gap = (
        fnrs.max()
        -
        fnrs.min()
    )


    # ========================================================
    # TPR GAP
    # ========================================================

    tpr_gap = (
        tprs.max()
        -
        tprs.min()
    )


    # ========================================================
    # EQUALIZED ODDS GAP
    # ========================================================
    #
    # EO requires equal TPR and equal FPR.
    #
    # Therefore use the larger of the two disparities.

    equalized_odds_gap = max(
        fpr_gap,
        tpr_gap
    )


    # ========================================================
    # AVERAGE ODDS GAP
    # ========================================================

    average_odds_gap = (
        (
            fpr_gap
            +
            tpr_gap
        )
        /
        2
    )


    return {

        "DP_Gap":
            dp_gap,

        "DP_Ratio":
            dp_ratio,

        "FPR_Gap":
            fpr_gap,

        "FNR_Gap":
            fnr_gap,

        "TPR_Gap":
            tpr_gap,

        "Equalized_Odds_Gap":
            equalized_odds_gap,

        "Average_Odds_Gap":
            average_odds_gap
    }


# ============================================================
# 17. BASELINE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("6. BASELINE RESULTS")
print("=" * 70)


baseline_group_metrics = (
    calculate_group_metrics(

        y_test,

        y_pred_baseline,

        protected_test
    )
)


print(
    "\nPer-group classification metrics:"
)


print(
    baseline_group_metrics.to_string(

        index=False,

        float_format=lambda x:
        f"{x:.4f}"
    )
)


baseline_fairness = (
    calculate_three_group_fairness(
        baseline_group_metrics
    )
)


print(
    "\nBaseline fairness metrics:"
)


for metric, value in (
    baseline_fairness.items()
):

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 18. DIR RESULTS
# ============================================================

print("\n" + "=" * 70)
print("7. DIR + DECISION TREE RESULTS")
print("=" * 70)


repaired_group_metrics = (
    calculate_group_metrics(

        y_test,

        y_pred_repaired,

        protected_test
    )
)


print(
    "\nPer-group classification metrics:"
)


print(
    repaired_group_metrics.to_string(

        index=False,

        float_format=lambda x:
        f"{x:.4f}"
    )
)


repaired_fairness = (
    calculate_three_group_fairness(
        repaired_group_metrics
    )
)


print(
    "\nDIR + DT fairness metrics:"
)


for metric, value in (
    repaired_fairness.items()
):

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 19. BASELINE VS DIR
# ============================================================

print("\n" + "=" * 70)
print("8. BASELINE vs DIR")
print("=" * 70)


comparison = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "DP Gap",

        "DP Ratio",

        "FPR Gap",

        "FNR Gap",

        "TPR Gap",

        "Equalized Odds Gap",

        "Average Odds Gap"
    ],


    "Baseline DT": [

        baseline_accuracy,

        baseline_fairness[
            "DP_Gap"
        ],

        baseline_fairness[
            "DP_Ratio"
        ],

        baseline_fairness[
            "FPR_Gap"
        ],

        baseline_fairness[
            "FNR_Gap"
        ],

        baseline_fairness[
            "TPR_Gap"
        ],

        baseline_fairness[
            "Equalized_Odds_Gap"
        ],

        baseline_fairness[
            "Average_Odds_Gap"
        ]
    ],


    "DIR + DT": [

        repaired_accuracy,

        repaired_fairness[
            "DP_Gap"
        ],

        repaired_fairness[
            "DP_Ratio"
        ],

        repaired_fairness[
            "FPR_Gap"
        ],

        repaired_fairness[
            "FNR_Gap"
        ],

        repaired_fairness[
            "TPR_Gap"
        ],

        repaired_fairness[
            "Equalized_Odds_Gap"
        ],

        repaired_fairness[
            "Average_Odds_Gap"
        ]
    ]
})


print(
    comparison.to_string(

        index=False,

        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 20. PER-GROUP COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("9. PER-GROUP COMPARISON")
print("=" * 70)


baseline_tmp = (
    baseline_group_metrics[
        [
            "group",
            "N",
            "FPR",
            "FNR",
            "TPR",
            "Selection_Rate"
        ]
    ]
    .copy()
)


baseline_tmp["Model"] = (
    "Baseline DT"
)


repaired_tmp = (
    repaired_group_metrics[
        [
            "group",
            "N",
            "FPR",
            "FNR",
            "TPR",
            "Selection_Rate"
        ]
    ]
    .copy()
)


repaired_tmp["Model"] = (
    "DIR + DT"
)


per_group_comparison = pd.concat(

    [
        baseline_tmp,
        repaired_tmp
    ],

    ignore_index=True
)


per_group_comparison = (
    per_group_comparison[
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
)


print(
    per_group_comparison.to_string(

        index=False,

        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

comparison.to_csv(

    "results/aif360_bank_marketing_DIR_fairness_comparison.csv",

    index=False
)


per_group_comparison.to_csv(

    "results/aif360_bank_marketing_DIR_per_group_metrics.csv",

    index=False
)


print("\n" + "=" * 70)

print("10. RESULTS SAVED")

print("=" * 70)


print(
    "results/aif360_bank_marketing_DIR_fairness_comparison.csv"
)


print(
    "results/aif360_bank_marketing_DIR_per_group_metrics.csv"
)


print("\nFinished.")