import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pathlib

from pathlib import Path


from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

pd.set_option('display.max_columns', None)
logging.basicConfig(level=logging.ERROR)

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Disable logging messages unless there is an error
set_log_level("CRITICAL")
set_random_seed(0)
N = 48


def build_cluster_model_and_forecast(cluster_id, date_string, local=True):
    if local:
        #prefix = str(Path(__file__).parent) + "/data/"
        #prefix = str(pathlib.Path.cwd()) + "/data/"
        prefix = ""
        print(prefix)
    else:
        prefix = "gs://"

    # Convert the string to a date object
    forecast_date = date_string

    file_name = prefix + "raw_solar_data/Cluster_" + str(cluster_id) + "_Avg_Data.csv"
    logging.info(f"Reading File {file_name}")
    df = pd.read_csv(file_name, index_col=0)

    # Time series column must be named “ds” and contain datetime values.
    # Value column must be named “y” and contain float values.
    df = df.rename(columns={"time_index": "ds", "ghi": "y"})

    # remove dates after forecast_date
    # development data for model training & validation
    df_dev = df[df["ds"] < forecast_date]

    # future columns for which values are independent & known
    future_columns = ["solar_zenith_angle", "asymmetry"]

    # setup dataframe that will be forecasted
    df_to_pred = df[df["ds"] <= forecast_date]
    y_true = df[df["ds"] > forecast_date]["y"].values[:48]
    

    # columns that will be set to Nan as future value is not known
    columns_to_nan = df.columns.tolist()
    columns_to_nan.remove("ds")
    for c in future_columns:
        columns_to_nan.remove(c)

    df_to_pred.loc[df_to_pred["ds"] == forecast_date, columns_to_nan] = np.nan

    # Model with lagged and future regressors
    logging.info("Train Model on Historical Cluster Data")
    m = NeuralProphet(
      growth="off",
      yearly_seasonality=False,
      weekly_seasonality=False,
      daily_seasonality=False,
      n_forecasts=48,
      n_lags= 3 * 24 * 2,
      learning_rate=0.01,
    )
    m.set_plotting_backend("plotly-static")

    # Set up Future regressors
    for c in future_columns:
        m.add_future_regressor(c)

    # Set up Lagged regressors
    lag_columns = df.columns.tolist()
    lag_columns.remove("ds")
    lag_columns.remove("y")
    for c in future_columns:
        lag_columns.remove(c)
    for c in lag_columns:
        m.add_lagged_regressor(c, normalize="standardize",  n_lags=48)

    df_train, df_val = m.split_df(df_dev, freq="30min", valid_p=0.1)

    metrics = m.fit(df_train, freq='30min', validation_df=df_val)
    print(metrics.tail(1))

    forecast = m.predict(df_to_pred)
    # Visualize the forecast
    logging.info("Visualize the forecast")
    m.plot(forecast)
    m.plot(forecast[-500:])

    # Get y_pred values from the forecast df
    logging.info("Get y_pred values from the forecast df and calculate metrics")
    last = forecast.tail(48)
    y_pred = []
    MAE_sum = 0
    sMAPE_sum = 0

    for i in range(N):
        pred = max(0,last[f"yhat{i+1}"].values[i])
        true = y_true[i]
        MAE_sum += abs(pred - true)
        sMAPE_sum += (abs(pred - true) + 0.0001)/ ((abs(pred) + abs(true)+ 0.0001))
        y_pred.append(pred)

    MAE = MAE_sum / N
    logging.info(f"MAE: {MAE}")

    sMAPE = sMAPE_sum * 2 / N
    logging.info(f"sMAPE: {sMAPE}")

    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    # Calculate RMSE
    rmse = np.sqrt(mse)
    logging.info(f"RMSE: {rmse}")

    results = last['ds'].to_frame()
    results['y_pred'] = y_pred
    results['y_true'] = y_true
    results.reset_index(inplace=True)
    results.drop(columns=['index'], inplace=True)
    # Write forecast results to csv file
    output_file = prefix + "forecast_results/Cluster_" + str(cluster_id) + "_Avg_Forecast_" + date_string + ".csv"
    logging.info(f"Write forecast results to csv file {output_file}")
    results.to_csv(output_file, index=False)

    # Write metrics to file
    metric = pd.DataFrame()
    metric['date'] = date_string
    metric['MAE'] = [MAE]
    metric['sMAPE'] = [sMAPE]
    metric['RMSE'] = [rmse]
    metric['Model Hubor Loss'] = metrics.iloc[-1]['Loss']
    output_file = prefix + "forecast_results/Cluster_" + str(cluster_id) + "_Avg_Forecast_" + date_string + "_metrics.csv"
    logging.info(f"Write metrics to file: {output_file}")
    metric.to_csv(output_file, index=False)

# Defining main function
def main():
    date_string = "2022-12-13"
    cluster_id_lst = [3,6]
    for cluster_id in cluster_id_lst:
        build_cluster_model_and_forecast(cluster_id, date_string, local=False)


if __name__=="__main__":
    main()
