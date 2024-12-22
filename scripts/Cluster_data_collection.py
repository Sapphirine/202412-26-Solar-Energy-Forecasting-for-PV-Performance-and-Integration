import pandas as pd
import fetch_data
import numpy as np

# Function to get the average GHI and weather time series features for all locations for a chosen cluster and year
def get_cluster_data(subset_index, fn):
    data = pd.DataFrame() 
    data['time_index'] = pd.to_datetime(fn['time_index'][...].astype(str)).tz_convert('US/Eastern')
    datasets = list(fn)

    for d in datasets:
        dset = fn[d]
        tseries = np.mean(dset[:, subset_index], 1)
        data[d] = tseries
    return data

# Function to get the GHI and weather time series features for a point in a chosen cluster and year
def get_point_data(pt_index, fn):
    data = pd.DataFrame() 
    data['time_index'] = pd.to_datetime(fn['time_index'][...].astype(str)).tz_convert('US/Eastern')
    datasets = list(fn)

    for d in datasets:
        dset = fn[d]
        tseries = dset[:, pt_index]
        data[d] = tseries
    return data

if __name__ ==  '__main__':

    # Read the CSV file with clustered locations
    clustered_locs = pd.read_csv("gs://raw_solar_data/Cluster_locs_labelled.csv")

    # Get locations in clusters 3 and 6
    cluster3 = clustered_locs[clustered_locs['Cluster']==3]
    cluster6 = clustered_locs[clustered_locs['Cluster']==6]

    # List of location indices for clusters 3 and 6
    cluster3_idx = list(cluster3.index)
    cluster6_idx = list(cluster6.index)

    # Load data from NRSDB database for 2022
    file_name1 = "/nrel/nsrdb/current/nsrdb_2022.h5" 
    f1 = fetch_data.fetch_hsds(file_name1)

    # Get averaged cluster data for year 2022
    cluster3_data_2022 = get_cluster_data(cluster3_idx, f1)
    cluster6_data_2022 = get_cluster_data(cluster6_idx, f1)

    # Get data for 2 points each from cluster 3 and cluster 6 for year 2022
    cluster3_point1_2022 = get_point_data(1194557, f1)
    cluster3_point2_2022 = get_point_data(1247832, f1)
    cluster6_point1_2022 = get_point_data(1197245, f1)
    cluster6_point2_2022 = get_point_data(1241414, f1)

    # Load data from NRSDB database for 2023
    file_name2 = "/nrel/nsrdb/GOES/aggregated/v4.0.0/nsrdb_2023.h5" 
    f2 = fetch_data.fetch_hsds(file_name2)

    # Get the average cluster data for cluster 3 and cluster 6 for year 2023
    cluster3_data_2023 = get_cluster_data(cluster3_idx, f2)
    cluster6_data_2023 = get_cluster_data(cluster6_idx, f2)

    # Get data for 2 points each from cluster 3 and cluster 6 for year 2023
    cluster3_point1_2023 = get_point_data(1194557, f2)
    cluster3_point2_2023 = get_point_data(1247832, f2)
    cluster6_point1_2023 = get_point_data(1197245, f2)
    cluster6_point2_2023 = get_point_data(1241414, f2)

    # Join all 2022 and 2023 data for clusters 3 and 6 

    cluster3_point1 = pd.concat([cluster3_point1_2022, cluster3_point1_2023], axis=0).reset_index().drop('index', axis=1)
    cluster3_point2 = pd.concat([cluster3_point2_2022, cluster3_point2_2023], axis=0).reset_index().drop('index', axis=1)
    cluster6_point1 = pd.concat([cluster6_point1_2022, cluster6_point1_2023], axis=0).reset_index().drop('index', axis=1)
    cluster6_point2 = pd.concat([cluster6_point2_2022, cluster6_point2_2023], axis=0).reset_index().drop('index', axis=1)

    cluster3_data = pd.concat([cluster3_data_2022, cluster3_data_2023], axis=0).reset_index().drop('index', axis=1)
    cluster6_data = pd.concat([cluster6_data_2022, cluster6_data_2023], axis=0).reset_index().drop('index', axis=1)

    # Save the data to GCP bucket as csv file

    cluster3_data.to_csv("gs://raw_solar_data/Cluster_3_Avg_Data.csv")
    cluster6_data.to_csv("gs://raw_solar_data/Cluster_6_Avg_Data.csv")
    cluster3_point1.to_csv("gs://raw_solar_data/Cluster_3_Point_1_Data.csv")
    cluster3_point2.to_csv("gs://raw_solar_data/Cluster_3_Point_2_Data.csv")
    cluster6_point1.to_csv("gs://raw_solar_data/Cluster_6_Point_1_Data.csv")
    cluster6_point2.to_csv("gs://raw_solar_data/Cluster_6_Point_2_Data.csv")