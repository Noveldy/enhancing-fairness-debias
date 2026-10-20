from ucimlrepo import fetch_ucirepo
from pathlib import Path
  
# fetch adult dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets
print(X.head())
print(y.head())
# merge features and targets into a single dataframe
adult_df = X.join(y)
# store the dataset as a CSV file
adult_df.to_csv(Path(__file__).parent / 'adult.csv', index=False)
print(f"Dataset saved to {Path(__file__).parent / 'adult.csv'}")

# fetch banking dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets
print(X.head())
print(y.head())
# merge features and targets into a single dataframe
bank_marketing_df = X.join(y)
# store the dataset as a CSV file
bank_marketing_df.to_csv(Path(__file__).parent / 'bank_marketing.csv', index=False)
print(f"Dataset saved to {Path(__file__).parent / 'bank_marketing.csv'}")

# fetch air quality dataset 
air_quality = fetch_ucirepo(id=360)

# data (as pandas dataframes) 
X = air_quality.data.features 
print(X.head())
# store the dataset as a CSV file
X.to_csv(Path(__file__).parent / 'air_quality.csv', index=False)
print(f"Dataset saved to {Path(__file__).parent / 'air_quality.csv'}")

# fetch individual_household_electric_power_consumption dataset 
individual_household_electric_power_consumption = fetch_ucirepo(id=235)

# data (as pandas dataframes) 
X = individual_household_electric_power_consumption.data.features 
print(X.head())
# store the dataset as a CSV file
X.to_csv(Path(__file__).parent / 'individual_household_electric_power_consumption.csv', index=False)
print(f"Dataset saved to {Path(__file__).parent / 'individual_household_electric_power_consumption.csv'}")