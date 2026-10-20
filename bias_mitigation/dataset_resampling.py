import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from transformers import pipeline

from Models.model_trainer import train_dummy_classifier
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
# modules for over-sampling
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
# modules for under-sampling
from imblearn.under_sampling import CondensedNearestNeighbour, AllKNN, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, balanced_accuracy_score
)
from collections import defaultdict

# function to perform over-sampling of the data (see https://imbalanced-learn.org/stable/references/over_sampling.html)
def data_resampling(data, target_variable, data_info, fileName=None):
    # define a list of over-sampling algorithms to use
    over_sampling_algorithms = [SMOTE(random_state=42), 
                                ADASYN(random_state=42), 
                                BorderlineSMOTE(random_state=42), KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42), 
                                SVMSMOTE(random_state=42)]
    # define a list of under-sampling algorithms to use
    under_sampling_algorithms = [CondensedNearestNeighbour(), AllKNN(), NeighbourhoodCleaningRule(), OneSidedSelection(), TomekLinks()]
    # check if the dataset has 20000 rows or more
    if len(data) >= 20000:
        logging.info(f'[DATA BIAS MITIGATION] The dataset has {len(data)} rows, SVMSMOTE and CondensedNearestNeighbour will not be used.')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'\n[DATA BIAS MITIGATION] The dataset has {len(data)} rows, SVMSMOTE and CondensedNearestNeighbour will not be used.')
        # remove SVMSMOTE from the list of over-sampling algorithms and CondensedNearestNeighbour from the list of under-sampling algorithms
        over_sampling_algorithms.pop()
        under_sampling_algorithms.pop(0)
    # split the data into features and target variable
    X = data.drop(columns=[target_variable])
    # fill missing values with the mean of the column
    X = X.fillna(X.mean())
    y = data[target_variable]

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True),
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
    }

    over_sampling_results = []
    under_sampling_results = []
    logging.info('Original dataset shape %s' % Counter(y))
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'Original dataset shape %s' % Counter(y))
    logging.info('Over-sampling the dataset')
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'\nOver-sampling the dataset')
    # oversample the dataset using each algorithm
    for algorithm in over_sampling_algorithms:
        logging.info(f'Using the {algorithm} algorithm')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'\nUsing the {algorithm} algorithm')
        try:      
            # Build the pipeline
            model = make_pipeline(
                algorithm,
                DecisionTreeClassifier(random_state=42)
            )
            # Cross-validation with multiple metrics
            cv_results = cross_validate(
                model, X, y,
                scoring=scoring,
                return_train_score=True,
                return_estimator=True,
                n_jobs=-1
            )
            # Print results
            for metric in scoring:
                print(f"{metric} mean: {cv_results[f'test_{metric}'].mean():.3f}, std:{cv_results[f'test_{metric}'].std():.3f}, min: {cv_results[f'test_{metric}'].min():.3f}, max: {cv_results[f'test_{metric}'].max():.3f}")
                with open(Path(f'results/logs.txt'), mode="a") as log_file:
                    log_file.write(f"\n{metric} mean: {cv_results[f'test_{metric}'].mean():.3f}, std:{cv_results[f'test_{metric}'].std():.3f}, min: {cv_results[f'test_{metric}'].min():.3f}, max: {cv_results[f'test_{metric}'].max():.3f}")
                # append the results to the over_sampling_results list
                over_sampling_results.append({algorithm: {metric: (cv_results[f'test_{metric}'].mean(), cv_results[f'test_{metric}'].std(), cv_results[f'test_{metric}'].min(), cv_results[f'test_{metric}'].max())}})
        except Exception as e:
            logging.error(f'Error using the {algorithm} algorithm: {e}')
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write(f'\nError using the {algorithm} algorithm: {e}')
            continue
    # undersample the dataset using each algorithm
    logging.info('Under-sampling the dataset')
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'\nUnder-sampling the dataset')
    for algorithm in under_sampling_algorithms:
        logging.info(f'Using the {algorithm} algorithm')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'\nUsing the {algorithm} algorithm')
        # Build the pipeline
        model = make_pipeline(
            algorithm,
            DecisionTreeClassifier(random_state=42)
        )
        # Cross-validation with multiple metrics
        cv_results = cross_validate(
            model, X, y,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )
        # Print results
        for metric in scoring:
            print(f"{metric} mean: {cv_results[f'test_{metric}'].mean():.3f}, std:{cv_results[f'test_{metric}'].std():.3f}, min: {cv_results[f'test_{metric}'].min():.3f}, max: {cv_results[f'test_{metric}'].max():.3f}")
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write(f"\n{metric} mean: {cv_results[f'test_{metric}'].mean():.3f}, std:{cv_results[f'test_{metric}'].std():.3f}, min: {cv_results[f'test_{metric}'].min():.3f}, max: {cv_results[f'test_{metric}'].max():.3f}")
            # append the results to the under_sampling_results list
            under_sampling_results.append({algorithm: {metric: (cv_results[f'test_{metric}'].mean(), cv_results[f'test_{metric}'].std(), cv_results[f'test_{metric}'].min(), cv_results[f'test_{metric}'].max())}})

    for result in over_sampling_results:
        # get the metric values for each algorithm
        for key, value in result.items():
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write(f'\nOver-sampling {key}: {value}')
    for result in under_sampling_results:
        # get the metric values for each algorithm
        for key, value in result.items():
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write(f'\nUnder-sampling {key}: {value}')

    # Group metrics per algorithm for over-sampling
    over_sampling_summary = defaultdict(dict)

    for item in over_sampling_results:
        for model, metric_result in item.items():
            for metric, values in metric_result.items():
                # Format as a string for display
                formatted_value = values[0]
                over_sampling_summary[model][metric] = formatted_value

    # Convert defaultdict to regular dict if needed
    over_sampling_summary = dict(over_sampling_summary)
    # Group metrics per algorithm for under-sampling
    under_sampling_summary = defaultdict(dict)
    for item in under_sampling_results:
        for model, metric_result in item.items():
            for metric, values in metric_result.items():
                # Format as a string for display
                formatted_value = values[0]
                under_sampling_summary[model][metric] = formatted_value

    selected_over_sampling_algorithm_results = None
    selected_under_sampling_algorithm_results = None
    # check if the domain of the dataset is healthcare
    if data_info['domain'] == 'healthcare':
        logging.info('[DATA BIAS MITIGATION] Selecting the best sampling method based on the f1-score')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'\n[DATA BIAS MITIGATION] Selecting the best sampling method based on the f1-score')
        # from the over_sampling_summary list, get the element with the highest f1-score
        best_algo = max(over_sampling_summary.items(), key=lambda x: x[1]['f1'])
        over_sampling_algo, over_sampling_metrics = best_algo
        selected_over_sampling_algorithm_results = over_sampling_algo
        # from the under_sampling_summary list, get the element with the highest f1-score
        best_algo = max(under_sampling_summary.items(), key=lambda x: x[1]['f1'])
        under_sampling_algo, under_sampling_metrics = best_algo
        selected_under_sampling_algorithm_results = under_sampling_algo
        # compare the results of the over-sampling and under-sampling algorithms
        if over_sampling_metrics['f1'] > under_sampling_metrics['f1']:
            selected_sampling_method = over_sampling_algo
        else:
            selected_sampling_method = under_sampling_algo
    else:
        logging.info('[DATA BIAS MITIGATION] Selecting the best sampling method based on the accuracy')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'\n[DATA BIAS MITIGATION] Selecting the best sampling method based on the accuracy')
        # from the over_sampling_summary list, get the element with the highest accuracy
        best_algo = max(over_sampling_summary.items(), key=lambda x: x[1]['accuracy'])
        over_sampling_algo, over_sampling_metrics = best_algo
        selected_over_sampling_algorithm_results = over_sampling_algo
        # from the under_sampling_summary list, get the element with the highest accuracy
        best_algo = max(under_sampling_summary.items(), key=lambda x: x[1]['accuracy'])
        under_sampling_algo, under_sampling_metrics = best_algo
        selected_under_sampling_algorithm_results = under_sampling_algo
        # compare the results of the over-sampling and under-sampling algorithms
        if over_sampling_metrics['accuracy'] > under_sampling_metrics['accuracy']:
            selected_sampling_method = over_sampling_algo
        else:
            selected_sampling_method = under_sampling_algo

    # reapply the selected sampling method to the data
    if selected_sampling_method == selected_over_sampling_algorithm_results:
        algorithm = selected_over_sampling_algorithm_results
        X_res, y_res = over_sampling_algorithms[over_sampling_algorithms.index(algorithm)].fit_resample(X, y)
        metrics = over_sampling_results
    else:
        algorithm = selected_under_sampling_algorithm_results
        X_res, y_res = under_sampling_algorithms[under_sampling_algorithms.index(algorithm)].fit_resample(X, y)
        metrics = under_sampling_results
    # create a new dataframe with the resampled data
    data_resampled = pd.concat([X_res, y_res], axis=1)
    logging.info('[DATA BIAS MITIGATION] Selected sampling method: %s' % selected_sampling_method)
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'\n[DATA BIAS MITIGATION] Selected sampling method: %s' % selected_sampling_method)
    logging.info('[DATA BIAS MITIGATION] Resampled dataset shape %s' % Counter(y_res))
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'\n[DATA BIAS MITIGATION] Resampled dataset shape %s' % Counter(y_res))
    matching_metrics = [
    d[selected_sampling_method]
    for d in metrics
        if selected_sampling_method in d
    ]

    if matching_metrics:
        for i, m in enumerate(matching_metrics, 1):
            logging.info(f"[DATA BIAS MITIGATION] {m}")
    else:
        logging.warning(f"[DATA BIAS MITIGATION] Sampling method '{selected_sampling_method}' not found in metrics.")

    return data_resampled
