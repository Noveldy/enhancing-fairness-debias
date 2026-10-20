import time
from darts import TimeSeries
from bias_identification.datasets import load_dataset, dataset_inspection, identify_imbalance, similarity_based_imputation
from bias_mitigation.data_cleaning import fill_timeseries_missing_values, fill_missing_values_spline, fill_missing_values_linear, fill_missing_values_mean
from bias_mitigation.dataset_resampling import data_resampling
from Models.model_trainer import train_dummy_forecaster, train_dummy_classifier
import os.path
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import plotly.express as px
from sklearn.datasets import load_iris
# Creating logging object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
DETECT_OUTLIERS=False
Z_THRESH=3
PROTECTED_ATTRIBUTE_NUM = None
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# ----------------------------------------------------- Dataset bias -------------------------------------
def check_dataset_bias(data, filename, detect_outliers, z_thresh, target_variables, target_variables_descriptions, features, features_descriptions):
    # start timer
    start_time = time.time()
    # call the similarity_based_imputation function to fill missing values in the dataset
    data = similarity_based_imputation(data, detect_outliers=detect_outliers, z_thresh=z_thresh)
    # create a copy of the dataset
    data_copy = data.copy()
    # check if the .json file about this data_info exists
    if os.path.isfile(Path(f'bias_identification/generated_data_info/{filename}.json')):
        logging.info('The file exists - loading the data_info from the .json file')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write('The file exists - loading the data_info from the .json file \n')
        with open(Path(f'bias_identification/generated_data_info/{filename}.json')) as f:
            data_info = json.load(f)
    else:
        logging.info('The file does not exist - creating the data_info and saving it to a .json file')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write('The file does not exist - creating the data_info and saving it to a .json file \n')
        data_info = dataset_inspection(data=data, fileName = filename, target_variables = target_variables, target_variables_descriptions = target_variables_descriptions, features = features, features_descriptions = features_descriptions)
    # if the dataset contains protected attributes and it is not a timeseries dataset, then the identify_protected_attributes_imbalance will be called
    if data_info['protected_attributes'] and data_info['type_of_dataset'] != 'timeseries':
        logging.info('[DATA BIAS IDENTIFICATION] The dataset contains protected attributes')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write('[DATA BIAS IDENTIFICATION] The dataset contains protected attributes\n')
        data_info = identify_imbalance(data=data, data_info = data_info, fileName = filename)
        if(PROTECTED_ATTRIBUTE_NUM is not None):
            sensitive_attribute = data_info['protected_attributes'][PROTECTED_ATTRIBUTE_NUM]
        else:
            sensitive_attribute = None
        training_report = train_dummy_classifier(data=data_copy, target_variable = data_info['target_variables'][0], sensitive_attribute = sensitive_attribute)
        # check if the dataset is imbalanced with respect to the target variables or the protected attributes
        if data_info['imbalanced_target_attributes'] or data_info['imbalanced_protected_attributes']:
            if(data_info['imbalanced_target_attributes']):
                with open(Path(f'results/logs.txt'), mode="a") as log_file:
                    log_file.write(f'[DATA BIAS IDENTIFICATION] The dataset is imbalanced with respect to the target variable(s) named {data_info["target_variables"]}\n')
                logging.info(f'[DATA BIAS IDENTIFICATION] The dataset is imbalanced with respect to the target variable(s) named {data_info["target_variables"]}')
            if(data_info['imbalanced_protected_attributes']):
                with open(Path(f'results/logs.txt'), mode="a") as log_file:
                    log_file.write(f'[DATA BIAS IDENTIFICATION] The dataset is imbalanced with respect to the protected attributes: {data_info["imbalanced_protected_attributes"]}\n')
                logging.info(f'[DATA BIAS IDENTIFICATION] The dataset is imbalanced with respect to the protected attributes: {data_info["imbalanced_protected_attributes"]}')
            data_copy = data_resampling(data=data_copy, target_variable = data_info['target_variables'][0], data_info = data_info)
        else:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write('[DATA BIAS IDENTIFICATION] The dataset is not imbalanced\n')
            logging.info('The dataset is not imbalanced')
        sensitive_attribute = data_info['protected_attributes'][PROTECTED_ATTRIBUTE_NUM]
        training_report = train_dummy_classifier(data=data_copy, target_variable = data_info['target_variables'][0], sensitive_attribute=sensitive_attribute)
        # store the data_copy in a .json file
        data_copy.to_json(Path(f'Datasets/{filename}_debiased.json'), orient='records')
        # end timer
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Time taken to check dataset bias: {execution_time} seconds')
        # add execution time to the training report
        training_report['execution_time_seconds'] = execution_time
        # store the training report in a .json file in the fairness_evaluation folder
        with open(Path(f'fairness_evaluation/{filename}_{sensitive_attribute}_complete_pipeline_dummy_classifier_report.json'), mode='w') as f:
            json.dump(training_report, f, indent=4)

    # if statement to check if the dataset is a timeseries dataset
    if data_info['type_of_dataset'] == 'timeseries':
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write('The dataset is a timeseries dataset\n')
        logging.info('The dataset is a timeseries dataset')
        # convert datetime columns to datetime objects
        for col in data_info['datetime_columns']:
            data[col] = pd.to_datetime(data[col], format=DATE_FORMAT)
        # get name of the datetime column
        datetime_column = data_info['datetime_columns'][0]
        # convert datetime column to a DateTime index
        data.set_index(datetime_column, inplace=True)
        # get the frequency of the time series
        frequency = pd.infer_freq(data.index)
        # check if the frequency is none (i.e., the time series is not continuous)
        if frequency is None:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write('The time series is not continuous - missing dates detected\n')
            logging.info('The time series is not continuous - missing dates detected')
            frequency = data.index.to_series().diff().min()
        print(f'The frequency of the time series is: {frequency}')
        idx = pd.period_range(min(data.index), max(data.index), freq=frequency).to_timestamp()
        data = data.reindex(idx, fill_value=np.nan)
        print(data.isna().sum()) 
        data.reset_index(inplace=True) 
        print(data.head())
        data.set_index('index', inplace=True)
        # check if the dataset has missing values
        if data.isna().sum().sum() > 0:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write('The time series dataset has missing values\n')
            logging.info('The time series dataset has missing values')
            # calculate execution time needed to fill the missing values
            start_time = time.time()
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'Time taken to fill missing values: {execution_time} seconds')
            # store the filled missing values in a .csv file
            data.to_csv(Path(f'results/{filename}_filled_missing_values_proposed_approach.csv'))
        else:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write('The time series dataset does not have missing values\n')
            logging.info('The time series dataset does not have missing values')
        # train a dummy forecaster on the time series dataset
        series = TimeSeries.from_dataframe(data)
        train_dummy_forecaster(series)
    # load the generated .json file
    with open(Path(f'bias_identification/generated_data_info/{filename}.json')) as f:
        data_info = json.load(f)
    # load the debias_ops.log file
    with open(Path('results/logs.txt')) as f:
        debias_ops_log = f.read()
    # create a dictionary to store the data-related info
    data_bias_response = {}
    # add the path of the resampled dataset to the data_info dictionary
    data_info['debiased_dataset_path'] = str(Path(f'Datasets/{filename}_debiased.json'))
    data_bias_response['data_info'] = data_info
    data_bias_response['debias_ops_log'] = debias_ops_log
    return data_bias_response