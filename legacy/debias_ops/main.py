import time
from darts import TimeSeries
from bias_identification.datasets import load_dataset, dataset_inspection, identify_imbalance, similarity_based_imputation
from bias_identification.algorithms import train_onnx_model, load_onnx_model
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
# ----------------------------------------------------- Algorithm bias -------------------------------------
# define the trained model path
# MODEL_PATH = Path('Models/trained_models/LinearRegression_boston.onnx')
# TRAINED_MODEL_DATASET = pd.read_csv('Datasets/boston.csv')
# TARGET_VARIABLES = ['MEDV']
# TARGET_VARIABLES_DESCRIPTIONS = ['Median value of owner-occupied homes in $1000s']
# FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# FEATURES_DESCRIPTIONS = ['capital crime rate by town', 'proportion of residential land zoned for lots over 25,000 sq. ft.',
# 'proportion of non-retail business acres per town', 'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)',
# 'nitric oxides concentration (parts per 10 million)','average number of rooms per dwelling', 
# 'proportion of owner-occupied units built prior to 1940', 'weighted distances to five Boston employment centres',
# 'accessibility to radial highways', 'full-value property-tax rate per $10,000', 
# 'pupil-teacher ratio by town', '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
# 'percentage of lower status of the population.']
# TASK = 'Regression'
# ----------------------------------------------------- Algorithm bias -------------------------------------

# define variables for time series
# PLOT_FEATURE = 'meantemp'
# COLUMNS = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
# DATE_FORMAT = '%Y-%m-%d' #'%Y-%m-%d %H:%M:%S'
# Create and configure logger
# logging.basicConfig(filename="debias_ops.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')
# Creating logging object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

# ----------------------------------------------------- Dataset bias -------------------------------------
# fileName = 'DailyDelhiClimateMissing' #time series dataset
# FILENAME = 'diabetes_missing' #imbalanced dataset with protected attributes
DETECT_OUTLIERS=False
Z_THRESH=3
PROTECTED_ATTRIBUTE_NUM = None
# DATA = load_dataset(fileName=FILENAME, fileExtension='.csv')
# FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# FEATURES_DESCRIPTIONS = ["Number of times pregnant","Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
# "Diastolic blood pressure (mm Hg)","Triceps skin fold thickness (mm)","2-Hour serum insulin (mu U/ml)",
# "Body mass index (weight in kg/(height in m)^2)","Diabetes pedigree function","Age"]
# TARGET_VARIABLES = ['Outcome']
# TARGET_VARIABLES_DESCRIPTIONS = ['Class variable (0 or 1) 268 of 768 are 1, the others are 0']
#  ----------------------------------------------- ADULT DATASET -------------------------------------
# FILENAME = 'adult'
# DATA = load_dataset(fileName=FILENAME, fileExtension='.csv')
# FEATURES = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
# FEATURES_DESCRIPTIONS = ['The age of the individual','The work class of the individual','Final weight','The education level','The education level as a number','The marital status','The occupation of the individual','The relationship of the individual','The race of the individual. Can be black, white etc.','The sex of idnividual. Can be either male or female.','Capital Gain','Capital Loss','Hours per week','The native country of the individual.']
# TARGET_VARIABLES = ['income']
# TARGET_VARIABLES_DESCRIPTIONS = ['The income of the individual. Can be either <=50K or >50K.']
# PROTECTED_ATTRIBUTE_NUM = 6 # corresponds to sex
# PROTECTED_ATTRIBUTE_NUM = 5 # corresponds to race 
# -----------------------------------------------------------------------------------------------------
#  ----------------------------------------------- BANK MARKETING DATASET -------------------------------------
# FILENAME = 'bank_marketing'
# DATA = load_dataset(fileName=FILENAME, fileExtension='.csv')
# FEATURES = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
# FEATURES_DESCRIPTIONS = ['Age of the client', 'Type of job (e.g. admin technician etc.)', 'Marital status (e.g. married single etc.)', 'Education level (e.g. primary secondary tertiary)', 'Whether the client has credit in default', 'Average yearly balance of the client', 'Whether the client has a housing loan', 'Whether the client has a personal loan', 'Type of communication used to contact the client', 'Last contact month of year', 'Last contact day of week', 'Duration of last contact in seconds', 'Number of contacts performed during this campaign for this client', 'Number of days since the client was last contacted from a previous campaign (-1 means client was not previously contacted)', 'Outcome of the previous marketing campaign']
# TARGET_VARIABLES = ['y']
# TARGET_VARIABLES_DESCRIPTIONS = ['Whether the client subscribed to a term deposit (yes or no)']
# PROTECTED_ATTRIBUTE_NUM = 1 # corresponds to marital
# -----------------------------------------------------------------------------------------------------
#  ----------------------------------------------- AIR QUALITY DATASET -------------------------------------
# FILENAME = 'air_quality_modified_with_missing'
# DATA = load_dataset(fileName=FILENAME, fileExtension='.csv')
# FEATURES = ['datetime', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
# FEATURES_DESCRIPTIONS = ['The date and time of the measurement.',
# 'True hourly averaged concentration CO in mg/m^3.',
# 'True hourly averaged concentration CO in mg/m^3.',
# 'Hourly averaged sensor response (nominally CO targeted).',
# 'True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3.',
# 'True hourly averaged Benzene concentration in microg/m^3.',
# 'Hourly averaged sensor response (nominally NMHC targeted).',
# 'True hourly averaged NOx concentration in ppb.',
# 'Hourly averaged sensor response (nominally NOx targeted).',
# 'True hourly averaged NO2 concentration in microg/m^3.',
# 'Hourly averaged sensor response (nominally NO2 targeted).',
# 'Hourly averaged sensor response (nominally O3 targeted).',
# 'Temperature in °C.',
# 'Relative Humidity.',
# 'Absolute Humidity.'
# ]
# TARGET_VARIABLES = []
# TARGET_VARIABLES_DESCRIPTIONS = []
# PLOT_FEATURE = 'AH'
# COLUMNS = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
# DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# #  ------------------------------------------------------------------------------------------------------------
#  ----------------------------------------------- INDIVIDUAL HOUSEHOLD ELECTRIC POWER CONSUMPTION DATASET -------------------------------------
# FILENAME = 'individual_household_electric_power_consumption_with_missing'
# DATA = load_dataset(fileName=FILENAME, fileExtension='.csv')
# FEATURES = ['datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
# FEATURES_DESCRIPTIONS = ['The date and time of the measurement.',
# 'Global active power in kilowatts.',
# 'Global reactive power in kilowatts.',
# 'Voltage in volts.',
# 'Global intensity in amperes.',
# 'Sub metering 1 in watt-hours.',
# 'Sub metering 2 in watt-hours.',
# 'Sub metering 3 in watt-hours.'
# ]
# TARGET_VARIABLES = []
# TARGET_VARIABLES_DESCRIPTIONS = []
# PLOT_FEATURE = 'Global_active_power'
# COLUMNS = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
# DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
#  ------------------------------------------------------------------------------------------------------------
# define the target variable of the dataset
target_variable = 'None'
# ----------------------------------------------------- Dataset bias -------------------------------------
def check_dataset_bias(data, filename, detect_outliers, z_thresh, target_variables, target_variables_descriptions, features, features_descriptions):
    # ---------------- USED FOR TESTING PURPOSES ONLY ---------------- #
    # # no cleaning testing
    # sensitive_attribute = 'sex'
    # # sensitive_attribute = 'race'
    # # sensitive_attribute = 'marital'
    # # start no cleaning and imbalance timer
    # start_time_no_cleaning_imbalance = time.time()
    # # adult dataset and sex as protected attribute
    # # training_report = train_dummy_classifier(data=data, target_variable = 'income', sensitive_attribute=sensitive_attribute)
    # # bank marketing dataset and marital status as protected attribute
    # training_report = train_dummy_classifier(data=data, target_variable = 'y', sensitive_attribute=sensitive_attribute)
    # # end no cleaning and imbalance timer
    # end_time_no_cleaning_imbalance = time.time()
    # execution_time_no_cleaning_imbalance = end_time_no_cleaning_imbalance - start_time_no_cleaning_imbalance
    # print(f'Time taken to train the dummy classifier without cleaning and imbalance handling: {execution_time_no_cleaning_imbalance} seconds')
    # # add execution time to the training report
    # training_report['execution_time_seconds'] = execution_time_no_cleaning_imbalance
    # # store the training report in a .json file in the fairness_evaluation folder
    # with open(Path(f'fairness_evaluation/{filename}_{sensitive_attribute}_no_pipeline_dummy_classifier_report.json'), mode='w') as f:
    #     json.dump(training_report, f, indent=4)
    # return
    # --------------------------------------------------------------------- #

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
        # with open(Path(f'fairness_evaluation/{filename}_{sensitive_attribute}_no_cleaning_dummy_classifier_report.json'), mode='w') as f:
        #     json.dump(training_report, f, indent=4)

    # if statement to check if the dataset is a timeseries dataset
    if data_info['type_of_dataset'] == 'timeseries':
        # TODO clean the timeseries dataset as well and then store it to a file
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
        fig = px.line(data, x='index', y=PLOT_FEATURE)
        fig.write_html(Path(f'plots/{filename}.html'))
        data.set_index('index', inplace=True)
        # check if the dataset has missing values
        if data.isna().sum().sum() > 0:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write('The time series dataset has missing values\n')
            logging.info('The time series dataset has missing values')
            # calculate execution time needed to fill the missing values
            start_time = time.time()
            # fill missing values in the time series dataset
            data = fill_timeseries_missing_values(data = data, freq = frequency, plot_feature = PLOT_FEATURE, columns = COLUMNS, futureDateIndex=0)
            # data = fill_missing_values_mean(data)
            # data = fill_missing_values_linear(data)
            # data = fill_missing_values_spline(data)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'Time taken to fill missing values: {execution_time} seconds')
            # store the filled missing values in a .csv file
            data.to_csv(Path(f'results/{filename}_filled_missing_values_proposed_approach.csv'))
        else:
            with open(Path(f'results/logs.txt'), mode="a") as log_file:
                log_file.write('The time series dataset does not have missing values\n')
            logging.info('The time series dataset does not have missing values')
        # drop missing values in the time series dataset
        # data.dropna(inplace=True)
        # drop the first 50 rows
        # data = data[50:]
        # train a dummy forecaster on the time series dataset
        series = TimeSeries.from_dataframe(data[COLUMNS])
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

# function to check algoithm bias
def check_algorithm_bias(model_path, dataset, target_variables, target_variables_descriptions,features, features_descriptions, task):
    load_onnx_model(model_path, dataset, target_variables, target_variables_descriptions,features, features_descriptions, task)

check_dataset_bias(data = DATA, filename = FILENAME, detect_outliers=DETECT_OUTLIERS, z_thresh=Z_THRESH, target_variables = TARGET_VARIABLES, target_variables_descriptions = TARGET_VARIABLES_DESCRIPTIONS, features = FEATURES, features_descriptions = FEATURES_DESCRIPTIONS)
# check_algorithm_bias(MODEL_PATH, TRAINED_MODEL_DATASET, TARGET_VARIABLES, TARGET_VARIABLES_DESCRIPTIONS, FEATURES, FEATURES_DESCRIPTIONS, TASK)