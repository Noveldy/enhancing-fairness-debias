# load all the datasets located in the 'Time-Series' folder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pandas as pd
import numpy as np
from darts.datasets import AirPassengersDataset, AusBeerDataset, ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset, EnergyDataset,ExchangeRateDataset, GasRateCO2Dataset,HeartRateDataset,ILINetDataset, IceCreamHeaterDataset,MonthlyMilkDataset, SunspotsDataset, TaxiNewYorkDataset, TaylorDataset, TemperatureDataset, TrafficDataset, USGasolineDataset,WeatherDataset,WineDataset, WoolyDataset
# provlimatika datasets: australian_tourism, uber_tlc_hourly kai zurich_electricity_consumption
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
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import kaleido
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# function to train a dummy LSTM model and play around with different optimizers
def train_dummy_forecaster():
    # load the datasets
    datasets = [AirPassengersDataset().load(), AusBeerDataset().load(), ETTh1Dataset().load(), ETTh2Dataset().load(), ETTm1Dataset().load(), ETTm2Dataset().load(), ElectricityDataset().load(), EnergyDataset().load(),ExchangeRateDataset().load(), GasRateCO2Dataset().load(),HeartRateDataset().load(),ILINetDataset().load(), IceCreamHeaterDataset().load(),MonthlyMilkDataset().load(), SunspotsDataset().load(), TaxiNewYorkDataset().load(), TaylorDataset().load(), TemperatureDataset().load(), TrafficDataset().load(), USGasolineDataset().load(), WeatherDataset().load(),WineDataset().load(), WoolyDataset().load()]
    # define dataset names
    dataset_names = ['AirPassengers','AusBeer','ETTh1','ETTh2','ETTm1','ETTm2','Electricity','Energy','ExchangeRate','GasRateCO2','HeartRate','ILINet','IceCreamHeater','MonthlyMilk','Sunspots','TaxiNewYork','Taylor','Temperature','Traffic','USGasoline','Weather','Wine','Wooly']
    # define learning rates
    learning_rates = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    # define optimizers
    optimizers = ['SGD','Adam','RMSprop','Rprop','Adadelta','Adagrad','AdamW','Adamax','LBFGS']
    # optimizers = ['NAdam','ASGD']
    # define early stopping callback
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=3,
        min_delta=0.05,
        mode='min',
    )
    dataset_name_index = 0
    for series in datasets:
        # check if dataset_name_index is equal to 0
        if dataset_name_index == 0:
            f = open(Path('results/timeseries_rnn_training_info.csv'), 'a')
            f.write('Dataset,Number of rows,Frequency,Number of columns,MAPE,SMAPE,MAE,Learning rate,Optimizer,Training time\n')
            f.close()
        # find length of dataset
        dataset_length = len(series) - 1
        dataset_frequency = series.freq
        number_of_columns = len(series.columns)
        print(f'Dataset: {dataset_names[dataset_name_index]}')
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
            input_chunk_length=12,
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
                # append the results to the timeseries_rnn_training_info.csv file
                f = open(Path('results/timeseries_rnn_training_info.csv'), 'a')
                f.write(f'{dataset_names[dataset_name_index]},{dataset_length},{dataset_frequency},{number_of_columns},{mape_error},{smape_error},{mae_error},{learning_rate},{optimizer},{training_time}\n')
                f.close()
        dataset_name_index += 1

# train_dummy_forecaster()

# function to train a dummy LSTM model and play around with different optimizers
def train_dummy_forecaster_for_rest_of_datasets():
    # load the datasets
    datasets = [EnergyDataset().load(),ILINetDataset().load(), TemperatureDataset().load(), WeatherDataset().load()]
    # define dataset names
    dataset_names = ['Energy','ILINet','Temperature','Weather']
    # define learning rates
    learning_rates = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    # define optimizers
    optimizers = ['SGD','Adam','RMSprop','Rprop','Adadelta','Adagrad','AdamW','Adamax','LBFGS']
    # optimizers = ['NAdam','ASGD']
    # define early stopping callback
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=3,
        min_delta=0.05,
        mode='min',
    )
    dataset_name_index = 0
    for series in datasets:
        # check if dataset_name_index is equal to 0
        if dataset_name_index == 0:
            f = open(Path('results/rest_of_timeseries_rnn_training_info.csv'), 'a')
            f.write('Dataset,Number of rows,Frequency,Number of columns,MAPE,SMAPE,MAE,Learning rate,Optimizer,Training time\n')
            f.close()
        # drop columns with missing values or stationary columns (values do not change) based on dataset name
        if dataset_name_index == 0:
            series = series.drop_columns(['generation fossil coal-derived gas','generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead','generation fossil oil shale','generation fossil peat','generation marine','generation wind offshore'])
        elif dataset_name_index == 1:
            series = series.drop_columns(['AGE 50-64'])
        elif dataset_name_index == 3:
            series = series.drop_columns(['rain (mm)', 'raining (s)', 'SWDR (W/m²)', 'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)'])

        series = fill_missing_values(series, fill='auto')
        
        # find length of dataset
        dataset_length = len(series) - 1
        dataset_frequency = series.freq
        number_of_columns = len(series.columns)
        print(f'Dataset: {dataset_names[dataset_name_index]}')
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
            input_chunk_length=12,
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
                # append the results to the timeseries_rnn_training_info.csv file
                f = open(Path('results/rest_of_timeseries_rnn_training_info.csv'), 'a')
                f.write(f'{dataset_names[dataset_name_index]},{dataset_length},{dataset_frequency},{number_of_columns},{mape_error},{smape_error},{mae_error},{learning_rate},{optimizer},{training_time}\n')
                f.close()
        dataset_name_index += 1

# load the timeseries_rnn_training_info.csv file
experiment_results = pd.read_csv(Path('results/timeseries_rnn_training_info.csv'))
datasets = experiment_results['Dataset'].unique()
optimizers = experiment_results['Optimizer'].unique()
learning_rates = experiment_results['Learning rate'].unique()
smape_optimizer_df = pd.DataFrame(columns=['Dataset', 'Number of Rows', 'Number of Columns', 'Optimizer','Learning rate','Training Time','SMAPE'])
training_time_optimizer_df = pd.DataFrame(columns=['Dataset', 'Number of Rows', 'Number of Columns', 'Optimizer','Learning rate', 'SMAPE', 'Training Time'])
# convert Training time column to numeric
experiment_results['Training time'] = pd.to_numeric(experiment_results['Training time'], errors='coerce')
# set colors
color_map = {
    'ASGD': 'RoyalBlue',
    'Adadelta': 'MediumSlateBlue',
    'Adagrad': 'CadetBlue',
    'Adamax': 'CornflowerBlue',
    'AdamW': 'DarkBlue',
    'Adam': 'DodgerBlue',
    'LBFGS': 'LightSkyBlue',
    'Nadam': 'Cyan',
    'RMSprop': 'DarkGreen',
    'Rprop': 'DarkCyan',
    'SGD': 'LightSeaGreen'
}
# set optimizer order
optimizer_order = [
    'ASGD',
    'Adadelta',
    'Adagrad',
    'Adamax',
    'AdamW',
    'Adam',
    'LBFGS',
    'Nadam',
    'RMSprop',
    'Rprop',
    'SGD'
]
# visualization based on SMAPE metric
for dataset in datasets:
    subset = experiment_results[(experiment_results['Dataset'] == dataset)]
    # rename NAdam to Nadam
    subset['Optimizer'] = subset['Optimizer'].replace({'NAdam': 'Nadam'})
    fig = make_subplots(rows=3, cols=3)
    subset = experiment_results[(experiment_results['Dataset'] == dataset)]
    # rename NAdam to Nadam
    subset['Optimizer'] = subset['Optimizer'].replace({'NAdam': 'Nadam'})
    # sort subset by learning rate and training time
    subset = subset.sort_values(by=['Learning rate', 'SMAPE'])
    # get the learning rates
    learning_rates = subset['Learning rate'].unique()

    i = 1
    j = 1
    for learning_rate in learning_rates:
        lr_subset = subset[subset['Learning rate'] == learning_rate]
        # Create bar_colors list based on lr_subset['Optimizer']
        bar_colors = [color_map.get(opt, 'gray') for opt in lr_subset['Optimizer']]
        fig.add_trace(go.Bar(x=lr_subset['Optimizer'], y=lr_subset['SMAPE'], marker=dict(color=bar_colors), showlegend=False), row=i, col=j)
        fig.update_xaxes(
            categoryorder='array',
            categoryarray=optimizer_order,
            row=i, col=j
        )
        fig.update_xaxes(showticklabels=False)
        # fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(title_text="")
        # fig.update_yaxes(title_text="")

        # Add subplot title as learning rate
        fig.add_annotation(
            text=f"Learning rate: {learning_rate}",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=1.05,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            row=i,
            col=j,
            font=dict(size=12)
        )
        # set y axis label
        # fig.update_yaxes(title_text='Training time (seconds)', row=i, col=j)
        if( j == 3 ):
            i += 1
            j = 1
        else:
            j += 1
        if(i == 3 and j == 1):
            j += 1
    for opt, color in color_map.items():
        fig.add_trace(
            go.Bar(
                x=[None],  # invisible
                y=[None],
                name=opt,           # legend label
                marker=dict(color=color),
                showlegend=True
            )
        )
    fig.add_annotation(
        text="Optimizer",       # Global x-axis label
        xref="paper",
        yref="paper",
        x=0.5,                  # center
        y=-0.1,                 # below the figure
        showarrow=False,
        font=dict(size=16),
        xanchor="center",
        yanchor="top"
    )
    fig.add_annotation(
        text="SMAPE",   # Global y-axis label
        xref="paper",
        yref="paper",
        x=-0.08,                          # left of figure
        y=0.5,                            # center vertically
        showarrow=False,
        textangle=-90,                    # rotate text
        font=dict(size=16),
        xanchor="center",
        yanchor="middle"
    )
    for anno in fig['layout']['annotations']:
        anno['font']['size'] = 10  # smaller font for subplot titles

    fig.update_layout(
        title=dict(
            text="Comparison of SMAPE for Different Optimizers <br> and Diverse Learning Rates",
            x=0.5,               # center horizontally
            xanchor='center',
            yanchor='top',
            font=dict(size=18)    # adjust font size
        )
    )
    fig.write_image(f'plots/optimizers - SMAPE/{dataset}_smape_optimizers.png', scale=6)
    # get minimum smape value and the corresponding optimizer and learning rate
    min_smape = subset['SMAPE'].min()
    min_smape_row = subset[subset['SMAPE'] == min_smape]
    training_time = min_smape_row['Training time'].values[0]
    number_of_rows = min_smape_row['Number of rows'].values[0]
    number_of_columns = min_smape_row['Number of columns'].values[0]
    best_optimizer = min_smape_row['Optimizer'].values[0]
    best_learning_rate = min_smape_row['Learning rate'].values[0]
    # print(f'Dataset: {dataset} (size: {number_of_rows}), Best optimizer: {best_optimizer}, Best learning rate: {best_learning_rate}, Minimum SMAPE: {min_smape}')
    smape_optimizer_df = pd.concat([smape_optimizer_df, pd.DataFrame({'Dataset': dataset, 'Number of Rows': number_of_rows, 'Number of Columns': number_of_columns, 'Optimizer': best_optimizer, 'Learning rate': best_learning_rate, 'Training Time': training_time, 'SMAPE': min_smape}, index=[0])], ignore_index=True)



# visualization based on training time
for dataset in datasets:
    fig = make_subplots(rows=3, cols=3)
    subset = experiment_results[(experiment_results['Dataset'] == dataset)]
    # rename NAdam to Nadam
    subset['Optimizer'] = subset['Optimizer'].replace({'NAdam': 'Nadam'})
    # sort subset by learning rate and training time
    subset = subset.sort_values(by=['Learning rate', 'Training time'])
    # get the learning rates
    learning_rates = subset['Learning rate'].unique()

    i = 1
    j = 1
    for learning_rate in learning_rates:
        lr_subset = subset[subset['Learning rate'] == learning_rate]
        # Create bar_colors list based on lr_subset['Optimizer']
        bar_colors = [color_map.get(opt, 'gray') for opt in lr_subset['Optimizer']]
        # fig.add_trace(px.bar(lr_subset, x="Optimizer", y="Training time", title=f'Dataset: {dataset} - Learning rate: {learning_rate}', color='Optimizer', text='Training time'), row=i, col=j)
        fig.add_trace(go.Bar(x=lr_subset['Optimizer'], y=lr_subset['Training time'], marker=dict(color=bar_colors), showlegend=False), row=i, col=j)
        fig.update_xaxes(
            categoryorder='array',
            categoryarray=optimizer_order,
            row=i, col=j
        )
        fig.update_xaxes(showticklabels=False)
        # fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(title_text="")
        # fig.update_yaxes(title_text="")

        # Add subplot title as learning rate
        fig.add_annotation(
            text=f"Learning rate: {learning_rate}",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=1.05,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            row=i,
            col=j,
            font=dict(size=12)
        )
        # set y axis label
        # fig.update_yaxes(title_text='Training time (seconds)', row=i, col=j)
        if( j == 3 ):
            i += 1
            j = 1
        else:
            j += 1
        if(i == 3 and j == 1):
            j += 1
    for opt, color in color_map.items():
        fig.add_trace(
            go.Bar(
                x=[None],  # invisible
                y=[None],
                name=opt,           # legend label
                marker=dict(color=color),
                showlegend=True
            )
        )
    fig.add_annotation(
        text="Optimizer",       # Global x-axis label
        xref="paper",
        yref="paper",
        x=0.5,                  # center
        y=-0.1,                 # below the figure
        showarrow=False,
        font=dict(size=16),
        xanchor="center",
        yanchor="top"
    )
    fig.add_annotation(
        text="Training time (seconds)",   # Global y-axis label
        xref="paper",
        yref="paper",
        x=-0.08,                          # left of figure
        y=0.5,                            # center vertically
        showarrow=False,
        textangle=-90,                    # rotate text
        font=dict(size=16),
        xanchor="center",
        yanchor="middle"
    )
    for anno in fig['layout']['annotations']:
        anno['font']['size'] = 10  # smaller font for subplot titles

    fig.update_layout(
        title=dict(
            text="Comparison of Training Time for Different Optimizers <br> and Diverse Learning Rates",
            x=0.5,               # center horizontally
            xanchor='center',
            yanchor='top',
            font=dict(size=18)    # adjust font size
        )
    )


    fig.write_image(f'plots/optimizers - Training time/{dataset}_training_time_optimizers.png', scale=6)
    # fig = px.line(subset, x="Learning rate", y="Training time", title=f'Dataset: {dataset}', color='Optimizer', markers=True)
    # # set y axis label
    # fig.update_yaxes(title_text='Training time (seconds)')
    # fig.write_image(f'plots/optimizers - Training time/{dataset}_training_time_optimizers.png')
    # get minimum training time value and the corresponding optimizer and learning rate
    min_training_time = subset['Training time'].min()
    min_training_time_row = subset[subset['Training time'] == min_training_time]
    min_smape = min_training_time_row['SMAPE'].values[0]
    number_of_rows = min_training_time_row['Number of rows'].values[0]
    number_of_columns = min_training_time_row['Number of columns'].values[0]
    best_optimizer = min_training_time_row['Optimizer'].values[0]
    best_learning_rate = min_training_time_row['Learning rate'].values[0]
    # print(f'Dataset: {dataset} (size: {number_of_rows}), Best optimizer: {best_optimizer}, Best learning rate: {best_learning_rate}, Minimum training time: {min_training_time}')
    training_time_optimizer_df = pd.concat([training_time_optimizer_df, pd.DataFrame({'Dataset': dataset, 'Number of Rows': number_of_rows, 'Number of Columns': number_of_columns, 'Optimizer': best_optimizer, 'Learning rate': best_learning_rate, 'SMAPE': min_smape, 'Training Time': min_training_time}, index=[0])], ignore_index=True)
# sort the smape_optimizer_df and training_time_optimizer_df dataframes by Number of Rows in ascending order
smape_optimizer_df = smape_optimizer_df.sort_values(by=['Number of Rows'], ascending=True)
training_time_optimizer_df = training_time_optimizer_df.sort_values(by=['Number of Rows'], ascending=True)
# save the smape_optimizer_df dataframe to a csv file
smape_optimizer_df.to_csv(Path('results/smape_optimizers.csv'), index=False)
# save the training_time_optimizer_df dataframe to a csv file
training_time_optimizer_df.to_csv(Path('results/training_time_optimizers.csv'), index=False)

fig = go.Figure()
fig.add_trace(go.Bar(x=training_time_optimizer_df['Dataset'], y=training_time_optimizer_df['SMAPE'], text=training_time_optimizer_df['Optimizer'], name='Fastest Optimizer', marker_color='darkcyan', textposition='outside', textfont=dict(color=['darkcyan']*len(training_time_optimizer_df))))
fig.add_trace(go.Bar(x=smape_optimizer_df['Dataset'], y=smape_optimizer_df['SMAPE'], text=smape_optimizer_df['Optimizer'], name='Most Accurate Optimizer', marker_color='darkblue', textposition='outside', textfont=dict(color=['darkblue']*len(smape_optimizer_df))))
fig.update_layout(barmode='overlay', title_text='Visualization of the optimizers that minimize the SMAPE and <br>the ones that minimize the training time per dataset', xaxis_title='Dataset Name', yaxis_title='SMAPE')
# fig.show()
# store figure as png image
fig.write_image(f'plots/optimizers - Overview/smape_comparison_between_fastest_and_most_accurate_optimizers.png', scale=6)