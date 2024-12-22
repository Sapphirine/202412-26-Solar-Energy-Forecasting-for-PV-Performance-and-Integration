from pyspark.sql import SparkSession
import h5pyd
from pyspark.sql.types import StructType, StructField, FloatType
import h5pyd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np
import fetch_data

def h5pyd_to_spark(file_name, dataset_name, subset_indices, chunk_size=500):
    """
    Load H5pyd dataset into PySpark DataFrame by processing column chunks.
    :param file_name: HDF5 file path or HSDS file name.
    :param dataset_name: Dataset name inside the file.
    :param subset_indices: List or array of column indices to fetch from the dataset.
    :param chunk_size: Number of columns to process in each chunk.
    :return: PySpark DataFrame.
    """
    try:
        # Open HDF5 file using h5pyd
        h5file = fetch_data.fetch_hsds(file_name)
        dataset = h5file[dataset_name]
        
        # Determine the number of columns in the subset, not the full dataset
        total_columns_in_subset = len(subset_indices)
        

        # Initialize an empty PySpark DataFrame
        spark_df = None
        
        # Process the subset of columns in chunks
        count = 1
        for start_idx in range(0, total_columns_in_subset, chunk_size):
            end_idx = min(start_idx + chunk_size, total_columns_in_subset)
            chunk_columns = subset_indices[start_idx:end_idx]
            print(f"Processing chunk {count}...")
            
            # Load a chunk of the dataset (subset of columns)
            chunk = dataset[:, chunk_columns]
            
            # Convert the chunk to a list of rows (tuples) and cast to Python floats
            rows = [tuple(float(x) for x in row) for row in chunk]
            
            # Define schema dynamically for this subset
            schema = StructType([StructField(f"{i}", FloatType(), True) for i in chunk_columns])
            
            # Create a DataFrame for the current chunk of columns
            chunk_df = spark.createDataFrame(rows, schema=schema)
            
            # Combine the chunk DataFrame with the main DataFrame
            if spark_df is None:
                spark_df = chunk_df
            else:
                # Join new chunk with the existing DataFrame
                spark_df = spark_df.join(chunk_df)
                
            count += 1

        print(f"Dataset '{dataset_name}' with selected columns loaded into PySpark DataFrame.")
        return spark_df
    except Exception as e:
        print(f"Error converting H5pyd dataset to PySpark DataFrame: {e}")
        return None
    
def add_row_indices(df):
    """
    Add row index to input PySpark DataFrame for row number based operations
    """
    window_spec = Window.orderBy(F.monotonically_increasing_id())
    return df.withColumn("row_index", F.row_number().over(window_spec))
    

# Calculate monthly averages based on row indices
def add_monthly_averages_features(base_ftr_df, add_ftr_df, var_name):
    """
    Get the feature dataframe with monthly averages of GHI and air temperature in addition to spatial features
    :param base_ftr_df: Pandas DataFrame consisting of the main feature datafrane to be updated
    :param add_ftr_df: PySpark DataFrame consisting of GHI or air temperature time series for all locations in chosen region
    :param var_name: Name of the variable in the add_ftr_df DataFrame ("ghi" / "air_temp")
    :return: Pandas DataFrame with all the spatial and temporal features
    """
    
    months = {
    "Jan": (0, 31 * 48),
    "Feb": (31 * 48, 59 * 48),
    "Mar": (59 * 48, 90 * 48),
    "Apr": (90 * 48, 121 * 48),
    "May": (121 * 48, 151 * 48),
    "Jun": (151 * 48, 181 * 48),
    "Jul": (181 * 48, 212 * 48),
    "Aug": (212 * 48, 243 * 48),
    "Sep": (243 * 48, 273 * 48),
    "Oct": (273 * 48, 304 * 48),
    "Nov": (304 * 48, 334 * 48),
    "Dec": (334 * 48, 365 * 48)
    }

    # Add row indices to the DataFrame
    df_with_indices = add_row_indices(add_ftr_df)
    
    for month, (start, end) in months.items():
        # Filter rows for the current month's time range
        monthly_data = df_with_indices.filter((F.col("row_index") >= start) & (F.col("row_index") < end))
        
        # Calculate the average for each column (excluding 'row_index')
        averages = monthly_data.drop("row_index").select(
            *[F.avg(F.col(col)) for col in add_ftr_df.columns]
        )
        
        # Store the averages for the current month
        base_ftr_df[f'{var_name}_{month}'] = list(averages.collect()[0])
    
    return base_ftr_df