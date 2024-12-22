import spark_functions
from pyspark.sql import SparkSession
import fetch_data
import pandas as pd
from sklearn.cluster import KMeans

if __name__  == '__main__':

    spark = SparkSession.builder \
        .appName("H5pyd to PySpark") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.task.cpus", "2") \
        .getOrCreate()

    # Get the subset indices for the target region for year 2022
    file_name = "/nrel/nsrdb/current/nsrdb_2022.h5"  
    subset_indices = fetch_data.get_subset_indices("/nrel/nsrdb/current/nsrdb_2022.h5") 

    # Get and convert h5pyd file to PySpark DataFrame for GHI and air temperature data
    chunk_size = 500  # Adjust the chunk size according to memory limits
    ghi_df = spark_functions.h5pyd_to_spark(file_name, "ghi", subset_indices, chunk_size)
    air_temp_df = spark_functions.h5pyd_to_spark(file_name, "air_temperature", subset_indices, chunk_size)

    # Fetch 2022 NRSDB data
    f = fetch_data.fetch_hsds(file_name)

    # Get the latitude, longitude and elevation for all locations in the dataset
    meta = pd.DataFrame(f['meta'][...])
    coords = meta[['latitude', 'longitude', 'elevation']]

    # Contains latitude, longitude and elevation features
    nj_state = coords.loc[subset_indices,:]
    feature_df = nj_state.copy()

    # Contains latitude, longitude, elevation + Average monthly GHI features
    feature_df = spark_functions.add_monthly_averages_features(feature_df, ghi_df, 'ghi')

    # Contains latitude, longitude, elevation + Average monthly GHI features + Average monthly air temperature features
    feature_df = spark_functions.add_monthly_averages_features(feature_df, air_temp_df, 'air_temp')

    # K-Means++ CLustering with k = 9 clusters
    kmeans = KMeans(init = "k-means++", n_clusters = 9, random_state=7, n_init="auto")
    kmeans.fit(feature_df)

    # Assign cluster labels to locations
    clustered_locs = nj_state.assign(Cluster = kmeans.labels_)

    # Save Clustered dataframe as CSV file
    clustered_locs.to_csv("gs://raw_solar_data/Cluster_locs_labelled.csv")