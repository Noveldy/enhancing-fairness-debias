import logging
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

# function to fill missing values of a time series dataset with mean imputation, median imputation, forward fill, backward fill, linear and spline interpolation
def fill_timeseries_missing_values(data,freq,plot_feature,columns,futureDateIndex=0):
    # create a dataframe
    missing_dates_df = pd.DataFrame()                            
    # iterate over data per column
    for col in data.columns:
        print('Current column is: ', col)
        # get the dates of the records with missing values
        missing_dates = data[data[col].isna()].index
        # get the first and the last date of the missing dates
        first_date = missing_dates[0]
        last_date = missing_dates[-1]
        # print(missing_dates)
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'The first date of the missing dates is: {first_date}\n')
        logging.info(f'The first date of the missing dates is: {first_date}')
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'The last date of the missing dates is: {last_date}\n')
        logging.info(f'The last date of the missing dates is: {last_date}')
        # get the number of missing values
        number_of_missing_values = data[col].isna().sum()
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'The number of missing values in the column {col} is: {number_of_missing_values}\n')
        logging.info(f'The number of missing values in the column {col} is: {number_of_missing_values}')
        # get the first and the last date of the missing dates
        first_date = missing_dates[0]
        last_date = missing_dates[-1]
        # get last non-missing value before the missing dates
        last_non_missing_value = data.loc[first_date - pd.Timedelta(1, unit='d'), col]
        # get the second to last non-missing value before the missing dates
        second_to_last_non_missing_value = data.loc[first_date - pd.Timedelta(2, unit='d'), col]
        # get the first non-missing value after the missing dates
        first_non_missing_value = data.loc[last_date + pd.Timedelta(1, unit='d'), col]
        # get the second non-missing value after the missing dates
        second_non_missing_value = data.loc[last_date + pd.Timedelta(2, unit='d'), col]
        similar_data = data.loc[(data[col] == last_non_missing_value) & (data.index > last_date)]
        # check if the similar data is empty
        if similar_data.empty:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write(f'The similar data for column {col} is empty\n')
            logging.info(f'The similar data for column {col} is empty')
            similar_data = data.loc[(data.index > last_date)]
        if similar_data.shape[0] < number_of_missing_values:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write(f'The similar data for column {col} is less than the number of missing values\n')
            logging.info(f'The similar data for column {col} is less than the number of missing values')
            similar_data = data.loc[(data.index > last_date)]
        # get the first date of the similar data
        first_date_similar_data = similar_data.index[futureDateIndex]
        with open(Path(f'results/logs.txt'), mode="a") as log_file:
            log_file.write(f'The first date of the similar data for column {col} is: {first_date_similar_data}\n')
        logging.info(f'Number of timestamps with the same value is: {similar_data.shape[0]}')
        not_nan_values = data[col].dropna()
        non_missing_values_sample = data.loc[(data.index >= first_date_similar_data)]
        # print(non_missing_values_sample)
        # keep first number_of_missing_values values from the non_missing_values_sample
        non_missing_values_sample = non_missing_values_sample[:number_of_missing_values]
        missing_dates_df[col] = non_missing_values_sample[col].values
        missing_dates_df.index = missing_dates
        # print(f'The missing dates dataframe is: {missing_dates_df}')
    # append the missing_dates_df to the data
    data = pd.concat([data, missing_dates_df])
    # # drop rows with missing values
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    fig = px.line(data, x=data.index, y=plot_feature) 
    fig.write_html(Path(f'plots/filled_missing_values_{futureDateIndex}.html'))
    return data

# function to fill missing values of a time series dataset with spline interpolation
def fill_missing_values_spline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in a time series dataset using spline interpolation.

    Args:
        data (pd.DataFrame): The input time series data.

    Returns:
        pd.DataFrame: The data with filled missing values.
    """
    for column in data.columns:
        data[column] = data[column].interpolate(method='spline', order=3)
    return data

# function to fill missing values of a time series dataset with linear interpolation
def fill_missing_values_linear(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in a time series dataset using linear interpolation.

    Args:
        data (pd.DataFrame): The input time series data.

    Returns:
        pd.DataFrame: The data with filled missing values.
    """
    for column in data.columns:
        data[column] = data[column].interpolate(method='linear')
    return data

# function to fill the missing dates of a time series dataset
def fill_missing_values_mean(data: pd.DataFrame) -> pd.DataFrame:
    # fill missing values with the mean
    data.fillna(data.mean(), inplace=True)
    return data
