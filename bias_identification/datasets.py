import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_datetime64_any_dtype
import numpy as np
from numpy.linalg import norm
from scipy.stats import zscore
import json
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from transformers import pipeline

from Models.model_trainer import train_dummy_classifier
from Models.sensitive_features_text_classifier import classify_sensitive_features
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
# modules for over-sampling
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
# modules for under-sampling
from imblearn.under_sampling import CondensedNearestNeighbour, AllKNN, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks

# bin size for pandas cut (to segment data)
BIN_SIZE = 5 
#set a list with protected classes according to https://www.senate.ca.gov/protected-classes
protected_classes = ['race', 'color', 'religion','sex', 'gender','gender identity', 'gender expression', 'sexual orientation',
                     'marital status', 'medical condition', 'military status', 'nationality', 'ancesty', 'disability', 'age', 'leave request']

# create a list of possible domains where a dataset might belong to
domains = ['healthcare', 'finance', 'education', 'retail', 'marketing', 'insurance', 'telecommunications', 'transportation', 'weather', 'chemistry', 'energy', 'manufacturing', 'government', 'entertainment', 'sports', 'technology', 'agriculture', 'real estate', 'hospitality', 'media', 'non-profit', 'other']

DATASETS_FOLDER = 'Datasets/'
def load_dataset(fileName, fileExtension):
    if(fileExtension == '.csv'):
        data = pd.read_csv(Path(DATASETS_FOLDER+fileName+fileExtension))
    
    return data

#function to get basic info about the dataset. More specifically, protected attributes, data types, type of dataset
def dataset_inspection(data, fileName, target_variables = None, target_variables_descriptions = None, features = None, features_descriptions = None):
    protected_attributes = []
    numeric_cols = []
    object_cols = []
    datetime_cols = []
    type_of_dataset = None
    sensitivity_flag = 0
    # check if the features, targets etc are provided
    if(features is not None and features_descriptions is not None and target_variables is not None and target_variables_descriptions is not None):
        cols_names_list = features + target_variables
        cols_descriptions_list = features_descriptions + target_variables_descriptions
        # TODO Check that the rest of the code is working
        features_sensitivity = classify_sensitive_features(features_descriptions=cols_descriptions_list, features=cols_names_list)
        print(features_sensitivity)
        # Iterate over the dictionary
        for key, value in features_sensitivity.items():
            if value == 'Sensitive':
                # append the key to the protected attributes
                protected_attributes.append(key)
        sensitivity_flag = 1

    for col in data.columns: 
        #check if protected atributes are present in the dataset
        if((str(col).lower() in protected_classes or any(item in str(col).lower() for item in protected_classes)) and sensitivity_flag == 0):
            protected_attributes.append(str(col))
            print(str(col) +' is a protected class')
        #check the type of every column
        if(is_numeric_dtype(data[col])):
            numeric_cols.append(col)
        elif(is_object_dtype(data[col])):
            object_cols.append(col)
        elif(is_datetime64_any_dtype(col)):
            datetime_cols.append(col)
        
    #check the object cols because they might be datetime
    for potentially_date_col in object_cols:
        try:
            data[potentially_date_col]= pd.to_datetime(data[potentially_date_col])
            datetime_cols.append(potentially_date_col)
            #since added as a datetime, it is removed from the objects
            object_cols.remove(potentially_date_col)
        except:
            pass
    print(f'Numeric columns are: {numeric_cols}, datetime columns are: {datetime_cols} and object columns are {object_cols}.')
    #determine type of dataset
    number_of_numeric_features = len(numeric_cols)
    number_of_datetime_features = len(datetime_cols)
    number_of_object_features = len(object_cols)
    timeseries_flag = 0
    if(number_of_numeric_features > 0  and (number_of_datetime_features == 0 and number_of_object_features == 0)):
        type_of_dataset = 'numeric'
    if(number_of_datetime_features > 0):
        # get the frequency of each datetime column
        for datetime_col in datetime_cols:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
            # check the frequency of the datetime column
            frequency = pd.infer_freq(data[datetime_col])
            if frequency is not None :
                print(f'The frequency of {datetime_col} is: {frequency}')
                type_of_dataset = 'timeseries'
                timeseries_flag = 1
            else:
                print(f'The frequency of {datetime_col} could not be inferred.')
                type_of_dataset = 'timeseries'
                timeseries_flag = 1
    if(number_of_numeric_features == 0 and number_of_object_features > 0 and timeseries_flag == 0):
        type_of_dataset = 'categorical'
    if(number_of_numeric_features > 0 and number_of_object_features > 0 and timeseries_flag == 0):
        type_of_dataset = 'mixed'
    sequence_to_classify = ' '.join(str(x) for x in data.columns.tolist())
    candidate_labels = domains
    domain_classification_output = classifier(sequence_to_classify, candidate_labels,  multi_label=False)


    #create dictionary to add data-related info
    data_info = {}
    data_info['protected_attributes'] = protected_attributes
    data_info['numeric_columns'] = numeric_cols
    data_info['datetime_columns'] = datetime_cols
    data_info['object_columns'] = object_cols
    data_info['type_of_dataset'] = type_of_dataset
    data_info['domain'] = domain_classification_output['labels'][0]
    data_info['target_variables'] = target_variables
    # Convert and write JSON object to file
    with open(Path(f"bias_identification/generated_data_info/{fileName}.json"), "w") as outfile: 
        json.dump(data_info, outfile, indent=4)
    return data_info

# function to identify if the dataset is imbalanced with respect to the protected attributes
def identify_imbalance(data, data_info, fileName):
    protected_attributes=data_info['protected_attributes']
    data_info['imbalanced_protected_attributes'] = []
    data_info['imbalanced_target_attributes'] = []
    #check if the protected attributes are imbalanced
    for protected_attribute in protected_attributes:
        # check if the protected attribute in the data is numeric
        if(is_numeric_dtype(data[protected_attribute])):
            data[protected_attribute] = pd.cut(data[protected_attribute], bins=BIN_SIZE)
        print(f'Checking for imbalance in {protected_attribute} protected attribute')
        print(data[protected_attribute].value_counts())
        print(data[protected_attribute].value_counts(normalize=True))
        minority_protected_class_percentage = round(data[protected_attribute].value_counts(normalize=True).min()*100, 2)
        #get the name of the minority class
        minority_class_index = str(data[protected_attribute].value_counts(normalize=True).idxmin())
        if(minority_protected_class_percentage> 20 and minority_protected_class_percentage < 40):
            data_info['imbalanced_protected_attributes'].append({protected_attribute: 'mildly imbalanced', 'minority_class_percentage': minority_protected_class_percentage, 'minority_class_index': minority_class_index})
        elif (minority_protected_class_percentage >= 1 and minority_protected_class_percentage <= 20):
            data_info['imbalanced_protected_attributes'].append({protected_attribute: 'moderately imbalanced', 'minority_class_percentage': minority_protected_class_percentage, 'minority_class_index': minority_class_index})
        elif (minority_protected_class_percentage < 1):
            data_info['imbalanced_protected_attributes'].append({protected_attribute: 'extremely imbalanced', 'minority_class_percentage': minority_protected_class_percentage, 'minority_class_index': minority_class_index})
        print('-----------------------------------')
    if(data_info["target_variables"] != 'None'):
        for target_variable in data_info["target_variables"]:
            # check if the target variable is imbalanced
            print(f'Checking for imbalance in the target ({target_variable}) variable')
            # calculate percentage of data that belongs to the minority class
            minority_target_class_percentage = round(data[target_variable].value_counts(normalize=True).min()*100, 2)
            if(minority_target_class_percentage> 20 and minority_target_class_percentage < 40):
                    data_info['imbalanced_target_attributes'].append({target_variable: 'mildly imbalanced', 'minority_class_percentage': minority_target_class_percentage})
            elif (minority_target_class_percentage >= 1 and minority_target_class_percentage <= 20):
                data_info['imbalanced_target_attributes'].append({target_variable: 'moderately imbalanced', 'minority_class_percentage': minority_target_class_percentage})
            elif (minority_target_class_percentage < 1):
                data_info['imbalanced_target_attributes'].append({target_variable: 'extremely imbalanced', 'minority_class_percentage': minority_target_class_percentage})
    else:
        print('No target variable is set')

    # update data_info file
    with open(Path(f"bias_identification/generated_data_info/{fileName}.json"), "w") as outfile: 
        json.dump(data_info, outfile, indent=4)
    return data_info


# function to fill missing values in the dataset using cosine similarity of records
def similarity_based_imputation(df, detect_outliers=False, z_thresh=3):
    # if detect_outliers = True, Detect outliers and mark as NaN
    if detect_outliers:
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'[DATA BIAS IDENTIFICATION] Detecting outliers...\n')
        logging.info(f'[DATA BIAS IDENTIFICATION] Detecting outliers...')
        df_num = df.select_dtypes(include=[np.number]).astype(float)
        z_scores = df_num.apply(zscore, nan_policy='omit')
        outliers = (np.abs(z_scores) > z_thresh)
        # count the number of outliers
        num_outliers = outliers.sum().sum()
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'[DATA BIAS IDENTIFICATION] Number of outliers detected is: {num_outliers}\n')
        logging.info(f'[DATA BIAS IDENTIFICATION] Number of outliers detected is: {num_outliers}')
        df = df.copy()
        df[df_num.columns] = df_num.astype(float).mask(outliers)
    # check if the df has any missing values
    if df.isnull().sum().sum() == 0:
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'[DATA BIAS IDENTIFICATION] No missing values found in the dataset.\n')
        logging.info(f'[DATA BIAS IDENTIFICATION] No missing values found in the dataset.')
        return df
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'[DATA BIAS IDENTIFICATION] {df.isnull().sum().sum()} missing values detected. Proceeding with imputation...\n')
    logging.info(f'[DATA BIAS IDENTIFICATION] {df.isnull().sum().sum()} missing values detected. Proceeding with imputation...')
    # one-hot encode categorical variables TODO change it to map categorical variables to integers
    # get non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    one_hot_encoded_cols = {}
    # iterate over non-numeric columns
    for col in non_numeric_cols:
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'[DATA BIAS MITIGATION] One-hot encoding column: {col}\n')
        logging.info(f'[DATA BIAS MITIGATION] One-hot encoding column: {col}')
        # get unique values in the column
        unique_values = df[col].unique()
        # map unique values to integers
        value_map = {val: i for i, val in enumerate(unique_values)}
        # the mapping of nan should be nan
        value_map[np.nan] = np.nan  # ensure NaN is preserved
        df[col] = df[col].map(value_map)
        # keep track of the mapping to revert it later if needed
        one_hot_encoded_cols[col] = value_map

    # Cosine similarity using shared non-NaN values
    def cosine_sim_with_nan_handling(row1, row2):
        mask = ~row1.isna() & ~row2.isna()
        if mask.sum() == 0:
            return np.nan
        v1, v2 = row1[mask].values, row2[mask].values
        return np.dot(v1, v2) / (norm(v1) * norm(v2)) if norm(v1) and norm(v2) else np.nan

    # Create similarity matrix
    n = len(df)
    # Create version with NaNs filled temporarily (for similarity only)
    df_filled = df.fillna(df.mean())  # Or column means if 0 is not meaningful

    # Fit NearestNeighbors
    model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    model.fit(df_filled)

    # Get indices of neighbors (excluding self)
    distances, neighbor_indices = model.kneighbors(df_filled)
    df_imputed = df.copy()
    donor_log = pd.DataFrame('', index=df.index, columns=df.columns)
    for i in range(len(df)):
        missing_cols = df.loc[i][df.loc[i].isna()].index.tolist()
        if not missing_cols:
            continue  # nothing to fill
        neighbors = neighbor_indices[i]
        for col in missing_cols:
            value_filled = False
            for neighbor_idx in neighbors:
                if neighbor_idx == i:
                    continue  # skip self
                donor_value = df.at[neighbor_idx, col]
                if pd.notna(donor_value):
                    df_imputed.at[i, col] = donor_value
                    donor_log.at[i, col] = f'row_{neighbor_idx}'
                    value_filled = True
                    break
            # Fallback if no neighbor had value
            if not value_filled:
                fallback = df[col].mean(skipna=True)
                df_imputed.at[i, col] = fallback
                donor_log.at[i, col] = 'fallback_mean'

    # store donor log
    with open(Path(f'results/logs.txt'), mode="a") as log_file:
        log_file.write(f'[DATA BIAS MITIGATION] All missing values filled.\n')
    logging.info(f'[DATA BIAS MITIGATION] All missing values filled.')
    
    # revert one-hot encoding
    for col, value_map in one_hot_encoded_cols.items():
        # create a reverse mapping
        reverse_map = {v: k for k, v in value_map.items()}
        # perform the reverse mapping
        df_imputed[col] = df_imputed[col].map(reverse_map)

    return df_imputed