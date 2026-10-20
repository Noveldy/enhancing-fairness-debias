import pandas as pd
from pathlib import Path

# read the air_quality.csv dataset
# data = pd.read_csv(Path('Datasets/air_quality.csv'))
data = pd.read_csv(Path('Datasets/individual_household_electric_power_consumption.csv'))
# merge the date and time columns into a single datetime column
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
# drop the original date and time columns
data = data.drop(columns=['Date', 'Time'])
# add datetime column to the beginning of the dataframe
data = data[['datetime'] + [col for col in data.columns if col != 'datetime']]
# store the modified dataframe to a new csv file
# data.to_csv(Path('Datasets/air_quality_modified.csv'), index=False)
# print(data.head())
# print(data.info())
# parse object columns to numeric, coercing errors to NaN
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')
# fill NaN values with the mean of the column
data.fillna(data.mean(), inplace=True)
# print(data.head())
# print(data.info())
# keep only the records from 2009-06-15 07:35:00 up to 2010-11-26 21:02:00
data = data[(data['datetime'] >= '2009-06-15 07:35:00') & (data['datetime'] <= '2010-11-26 21:02:00')]
# keep first 43800 records
data = data.head(43800)
# get datetime range of the modified dataframe
print(f"Datetime range of the modified dataframe: {data['datetime'].min()} to {data['datetime'].max()}")
data.to_csv(Path('Datasets/individual_household_electric_power_consumption_modified.csv'), index=False)



# drop records starting from 2004-11-08 00:00:00 up to 2004-12-31 23:00:00
# data = data[(data['datetime'] < '2004-11-08') | (data['datetime'] > '2004-12-31')]
data = data[(data['datetime'] < '2009-06-18') | (data['datetime'] > '2009-06-25')]
# store the modified dataframe with missing values to a new csv file
# data.to_csv(Path('Datasets/air_quality_modified_with_missing.csv'), index=False)
data.to_csv(Path('Datasets/individual_household_electric_power_consumption_with_missing.csv'), index=False)