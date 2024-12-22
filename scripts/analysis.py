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


def cluster_points_analysis(cluster_id, date_string, local=True):
    if local:
        #prefix = str(Path(__file__).parent) + "/data/"
        prefix = "" #str(pathlib.Path.cwd()) + "/data/"
        print(prefix)
    else:
        prefix = "gs://"


    # get cluster forecast from file
    file_name = prefix + "forecast_results/Cluster_" + str(cluster_id) + "_Avg_Forecast_" + date_string + ".csv"
    logging.info(f"Reading File {file_name}")
    cluster_df = pd.read_csv(file_name)
    cluster_y_pred = cluster_df['y_pred'].values

    # combine cluster and point results
    cluster_df = cluster_df.rename(columns={"y_pred": "y_pred_cluster", "y_true": "y_true_cluster"})
    metric = pd.DataFrame()
    metric['Point_ID'] = [1,2]

    # Convert the date string to a date object
    forecast_date = date_string

    # Get Points True Values
    for point_id in [1,2]:
        file_name = prefix + "raw_solar_data/Cluster_" + str(cluster_id) + "_Point_" + str(point_id) + "_Data.csv"
        logging.info(f"Reading File {file_name}")
        df = pd.read_csv(file_name)

        # Time series column must be named “ds” and contain datetime values.
        # Value column must be named “y” and contain float values.
        df = df.rename(columns={"time_index": "ds", "ghi": "y"})

        # grab y_true
        y_true = df[df["ds"] > forecast_date]["y"].values[:48]

        column_name = 'y_true_point_' + str(point_id)
        cluster_df[column_name] = y_true

        # calculate metrics
        logging.info(f"Calculate metrics for Point {point_id}")
        y_pred = cluster_y_pred
        MAE_sum = 0
        sMAPE_sum = 0

        for i in range(N):
          pred = max(0,y_pred[i])
          true = y_true[i]
          MAE_sum += abs(pred - true)
          sMAPE_sum += (abs(pred - true) + 0.0001)/ ((abs(pred) + abs(true)+ 0.0001))

        MAE = MAE_sum / N
        logging.info(f"MAE: {MAE}")

        sMAPE = sMAPE_sum * 2 / N
        logging.info(f"sMAPE: {sMAPE}")

        # Calculate MSE
        mse = mean_squared_error(y_true, y_pred)
        # Calculate RMSE
        rmse = np.sqrt(mse)
        logging.info(f"RMSE: {rmse}")

        # Write metrics to file
        metric.loc[metric['Point_ID'] == point_id, 'date'] = date_string
        metric.loc[metric['Point_ID'] == point_id, 'MAE'] = MAE
        metric.loc[metric['Point_ID'] == point_id, 'sMAPE'] = sMAPE
        metric.loc[metric['Point_ID'] == point_id, 'RMSE'] = rmse


    output_file = prefix + "forecast_results/Combined_Cluster" + str(cluster_id) + "_Points_Forecast_" + date_string + ".csv"
    logging.info(f"Write results to file: {output_file}")
    cluster_df.to_csv(output_file, index=False)

    output_file = prefix  + "forecast_results/Cluster" + str(cluster_id) + "_Points_Forecast_" + date_string + "_metrics.csv"
    logging.info(f"Write metrics to file: {output_file}")
    metric.to_csv(output_file, index=False)


# Defining main function
def main():
    date_string = "2022-12-13"
    cluster_id_lst = [3,6]
    for cluster_id in cluster_id_lst:
        cluster_points_analysis(cluster_id, date_string, local=False)


if __name__=="__main__":
    main()
