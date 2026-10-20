# load all the datasets located in the 'Time-Series' folder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import (
    RNNModel,
)
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
import logging
import time

# modules used for classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# modules for cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

def demographic_parity(y_pred, sensitive):
    """
    Demographic Parity Difference
    Difference between max and min positive prediction rates across groups.
    """
    groups = np.unique(sensitive)
    rates = {}

    for g in groups:
        mask = sensitive == g
        rates[g] = np.mean(y_pred[mask])

    dp_diff = max(rates.values()) - min(rates.values())
    return dp_diff, rates


def equalized_odds(y_true, y_pred, sensitive):
    """
    Equalized Odds Difference
    Measures disparity in TPR and FPR across groups.
    """
    groups = np.unique(sensitive)

    tpr = {}
    fpr = {}

    for g in groups:
        mask = sensitive == g

        yt = y_true[mask]
        yp = y_pred[mask]

        TP = np.sum((yp == 1) & (yt == 1))
        FP = np.sum((yp == 1) & (yt == 0))
        TN = np.sum((yp == 0) & (yt == 0))
        FN = np.sum((yp == 0) & (yt == 1))

        tpr[g] = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr[g] = FP / (FP + TN) if (FP + TN) > 0 else 0

    tpr_diff = max(tpr.values()) - min(tpr.values())
    fpr_diff = max(fpr.values()) - min(fpr.values())

    return {
        "tpr_difference": tpr_diff,
        "fpr_difference": fpr_diff,
        "tpr_by_group": tpr,
        "fpr_by_group": fpr
    }



# function to train a dummy classifier from scikit-learn
def train_dummy_classifier(data, target_variable, sensitive_attribute):
    # # if there are missing data, drop the rows with missing data
    # data = data.dropna()
    # get non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    one_hot_encoded_cols = {}
    for col in non_numeric_cols:
        print(f'Encoding column: {col}')
        # get unique values in the column
        unique_values = data[col].unique()
        # map unique values to integers
        value_map = {val: i for i, val in enumerate(unique_values)}
        # the mapping of nan should be nan
        value_map[np.nan] = np.nan  # ensure NaN is preserved
        data[col] = data[col].map(value_map)
        # keep track of the mapping to revert it later if needed
        one_hot_encoded_cols[col] = value_map
    # split the data into features and target variable
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    if sensitive_attribute is not None:
        sensitive = X[sensitive_attribute]
        print(f'The sensitive attribute is: {sensitive_attribute}')
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sensitive, test_size=0.2, random_state=42, stratify=y)
        # define the classifier
        # Decision Tree Classifier
        # model = DecisionTreeClassifier(random_state=42)
        # random forest classifier
        model = RandomForestClassifier(random_state=42)
        # train the classifier
        model.fit(X_train, y_train)
        # make predictions
        y_pred = model.predict(X_test)
        # calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Example: Using StratifiedKFold cross-validation to preserve the class distribution during splitting
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Evaluate your model (e.g., RandomForestClassifier) using cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        training_report = classification_report(y_test, y_pred, output_dict=True)
        # append scores.mean() to the training_report
        training_report['cross_validation_score'] = scores.mean()
        # calculate AUC-ROC
        training_report['roc_auc_score'] = roc_auc_score(y, model.predict_proba(X)[:, 1])
        # -------- FAIRNESS METRICS -------- #
        dp_diff, dp_rates = demographic_parity(y_pred, s_test.values)
        eq_odds = equalized_odds(y_test.values, y_pred, s_test.values)
        training_report["fairness"] = {
            "demographic_parity_difference": str(dp_diff),
            "demographic_parity_by_group": str(dp_rates),
            "equalized_odds": str(eq_odds)
        }
    else:
        print('No sensitive attribute provided. Skipping fairness metrics.')
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # define the classifier
        # Decision Tree Classifier
        # model = DecisionTreeClassifier(random_state=42)
        # random forest classifier
        model = RandomForestClassifier(random_state=42)
        # train the classifier
        model.fit(X_train, y_train)
        # make predictions
        y_pred = model.predict(X_test)
        # calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Example: Using StratifiedKFold cross-validation to preserve the class distribution during splitting
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Evaluate your model (e.g., RandomForestClassifier) using cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        training_report = classification_report(y_test, y_pred, output_dict=True)
        # append scores.mean() to the training_report
        training_report['cross_validation_score'] = scores.mean()
        # calculate AUC-ROC
        training_report['roc_auc_score'] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return training_report

# function to train a dummy LSTM model
def train_dummy_forecaster(series):
    # define learning rates
    learning_rates = [1e-6]
    # define optimizers
    optimizers = ['Adam']
    # define early stopping callback
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=3,
        min_delta=0.05,
        mode='min',
    )
    dataset_name_index = 0
    
    # find length of dataset
    dataset_length = len(series) - 1
    dataset_frequency = series.freq
    number_of_columns = len(series.columns)
    print(f'Number of rows: {dataset_length}')
    print(f'Frequency: {dataset_frequency}')
    print(f'Number of columns: {number_of_columns}')
    # split the dataset into training and validation sets
    train, val = series.split_after(0.8)
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    series_transformed = transformer.transform(series)
    # define an LSTM model
    model = RNNModel(
        model='LSTM',
        input_chunk_length=7,
        output_chunk_length=1,
        n_epochs=100,
        random_state=0,
        model_name='LSTM',
        log_tensorboard=False,
        pl_trainer_kwargs={"callbacks": [early_stopper]}
    )
    # iterate over the learning rates
    for learning_rate in learning_rates:
        # iterate over the optimizers
        for optimizer in optimizers:
            # define the optimizer
            model.optimizer_kwargs = {'lr': learning_rate}
            model.optimizer = optimizer
            # start the timer
            start = time.time()
            # train the model
            model.fit(train_transformed, val_series=val_transformed, verbose=True)
            # make predictions
            pred_series = model.predict(n=12)
            # end the timer
            end = time.time()
            training_time = end - start
            # calculate the error
            try:
                mape_error = mape(val_transformed, pred_series)
            except:
                mape_error = np.nan
            try:
                smape_error = smape(val_transformed, pred_series)
            except:
                smape_error = np.nan
            try:
                mae_error = mae(val_transformed, pred_series)
            except:
                mae_error = np.nan
            print(f'MAPE: {mape_error}')
            print(f'SMAPE: {smape_error}')
            print(f'MAE: {mae_error}')
            print(f'Learning rate: {learning_rate}')
            print(f'Optimizer: {optimizer}')
            print('\n')
