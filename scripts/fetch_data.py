import h5pyd
import pandas as pd

def fetch_hsds(file, api_key='3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf', hsds_url="https://developer.nrel.gov/api/hsds",username=None, password=None):
    """
    Fetches distributed HDF5 data using h5pyd HSDS API
    
    Parameters:
    - file: Path to file to be fetched
    - api_key: HSDS API key. 
    - hsds_url: The URL of the HSDS API endpoint; Default is NREL's API endpoint 
    
    Returns:
    - h5pyd dataset object 
    """

    return h5pyd.File(file, 'r', endpoint=hsds_url, username=username, password=password)

def get_subset_indices(file_name, lat_low = 38.93, lat_upper = 41.36, long_low = -75.56, long_upper = -73.9):

    """
    Returns subset of location indices corresponding to latitude and longitude ranges. Default ranges correspond to the state of New Jersey.

    Parameters:
    - lat_low : Lower limit of latitude range
    - lat_upper : Upper limit of latitude range
    - long_low : Lower limit of longitude range
    - long_upper : Upper limit of longitude range
    
    Returns:
    - Python list of indices
    """
    # Fetch data from NRSDB for year 2022 
    f = fetch_hsds(file_name)

    # Get the latitude, longitude and elevation for all locations in the dataset
    meta = pd.DataFrame(f['meta'][...])
    coords = meta[['latitude', 'longitude', 'elevation']]

    # Get the locations that come under the coordinate range; Default is the state of New Jersey
    nj_state = coords[
        (coords['latitude'] >= 38.93) &
        (coords['latitude'] <= 41.36) &
        (coords['longitude'] >= -75.56) &
        (coords['longitude'] <= -73.9)
    ]

    # Get the indices of the reporting locations in the coordinate range (New Jersey)
    subset_indices = list(nj_state.index) 
    return subset_indices