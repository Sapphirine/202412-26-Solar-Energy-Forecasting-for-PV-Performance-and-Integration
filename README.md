# 202412-26-Solar-Energy-Forecasting-for-PV-Performance-and-Integration

The goal of this project is to build a model that leverages the seasonality of weather and solar irradiance data in order to create a day-ahead forecast for solar power output. This structure is inspired by the real-world market structures in place today with a day-ahead, security-constrained unit commitment optimization in a bilateral bidding system. This is accomplished using publicly available data sets, isolating & clustering by geolocation, day-ahead [NeuralProphet](https://neuralprophet.com/) forecast modeling, and statistical analysis of methods and results. 

## Data Source

The data used in this project to create clusters and train models is made publicly available by the National Renewable Energy Laboratory (NREL) National Solar Radiation Database ([NSRDB](https://nsrdb.nrel.gov/)) . The NRSDB data is stored as an HDF5 file on Amazon Web Services (AWS) cloud storage. This data is available at 30-minute intervals for 2,000,000+ locations across the United States and across a 25-year time period from 1998 to 2023.

## Data Fetching and Clustering

 `scripts/fetch_data.py` : Module containing the following functions:

- fetch_hsds() : Fetch the file from the chosen endpoint using h5pyd library.
- get_subset_indices() : Gets the subset of location indices corresponding to latitude and longitude ranges. 

`scripts/spark_functions.py` : Module containing the following functions:

- h5pyd_to_spark() : Load H5pyd dataset into PySpark DataFrame by processing column chunks.
- add_row_indices() : Add row index to input PySpark DataFrame for row number based operations
- add_monthly_averages_features() : Get the feature dataframe with monthly averages of GHI and air temperature in addition to spatial features

`scripts/Clustering.py` : Perform Clustering and get the clustered locations

`scripts/Cluster_data_collection.py` : Get all the time series features for the cluster average and for 2 points, in Clusters 3 and 6.


## Run Model & Analysis

`scripts/model.py`: Train Model on Cluster using data in `data/raw_solar_data`

`scripts/analysis.py`: Compare the Model's forecasted results with the cluster's constituent points' true values. 

Forecasted results and metrics are written to `data/forecast_results` in the local dir if the `local` variable in the main scripts is set to True. When false, the scripts fetch data from the cloud storage buckets. 

### Build

```
pip install -r requirements.txt
```

### Run in Python Environment

Run `Clustering.py` followed by `Cluster_data_collection.py` to get the data stored in `data/raw_solar_data` , which is to be used for training the forecasting model.

Update `main.py` with the forecast date which can be any date in 2022 and 2023. 
```python
date_string = "2022-12-13"
```

Run
```
python3.8 Clustering.py
python3.8 Cluster_data_collection.py
python3.8 main.py
```

### Run in AirFlow

After setting up Apache Airflow environment and starting the scheduler, clone this repo in the `airflow/dags` dir. 
Make sure to install all dependencies by running the above `pip install` command. 

Run
```
airflow db init
```


Update `main_dag.py` with the forecast date which can be any date in 2022 and 2023. 
```python
FORECAST_DATE = "2022-12-25"
```

## Visualization

`visualizations/Clustering_viz.html` : To visualize clustering  results
URL : https://raw.githack.com/Sapphirine/202412-26-Solar-Energy-Forecasting-for-PV-Performance-and-Integration/refs/heads/main/visualizations/Clustering_viz.html

`visualizations/dashboard.html` : To visualize forecast results for Cluster 3 & 6 along with their respective Points. This html renders the data from a GCS bucket but the url can be updated to the corresponding local files in dir `data/forecast_results`

## Jupyter Notebooks

`notebooks/Clustering+Data_processing.ipynb` :  Fetches raw data for a US state, performs clustering & data pre-processing

`notebooks/NeuralProphet_Model_Exploration.ipynb` : Explores NeuralProphet Forecasting Models through hyperparameter tuning

### Youtube Presentation Link

https://youtu.be/pr9uRsQJTNw

