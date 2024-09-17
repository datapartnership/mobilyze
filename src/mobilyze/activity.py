import dask.dataframe as dd
import dask_geopandas
import h3
import mobilkit
import pandas as pd
from dask.dataframe import DataFrame
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def compute_ping_interval(ddf: DataFrame):
    def _ping_interval(df: pd.DataFrame):
        df.sort_values(by=["uid", "datetime"], inplace=True)
        df["ping_interval"] = df.groupby("uid")["datetime"].diff()
        df["ping_interval_minutes"] = df["ping_interval"].dt.total_seconds() / 60

    return (
        ddf.groupby(["uid"])
        .apply(_ping_interval)
        .map_partitions(lambda d: d.reset_index(drop=True))
    ).compute()


def compute_activity(
    ddf: DataFrame,
    start="2022-01-01",
    end="2023-12-31",
) -> pd.DataFrame:
    """
    Compute activity levels based on GPS mobility data, returning both z-scores and
    percentage change relative to a baseline period. Activity is measured as the
    number of unique devices (`uid`) detected in each spatial area (`hex_id`) daily.

    The z-score standardizes activity data, showing how much the activity deviates
    from the baseline mean, expressed in standard deviations. Percentage change
    indicates the relative increase or decrease in activity compared to the baseline.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        A Dask DataFrame containing mobility data. The DataFrame must include at
        least the following columns:
        - `datetime`: Timestamps of device detections.
        - `hex_id`: Spatial index for the detection location.
        - `uid`: Unique identifier for each device.

    start : str, optional
        The start date of the baseline period in "YYYY-MM-DD" format. The default is
        "2022-01-01".

    end : str, optional
        The end date of the baseline period in "YYYY-MM-DD" format. The default is
        "2023-12-31".


    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the calculated activity metrics with the
        following columns:
        - `hex_id`: The spatial area identifier.
        - `date`: Date of the activity.
        - `nunique`: Number of unique devices detected per `hex_id` and day.
        - `weekday`: Day of the week (0=Monday, 6=Sunday).
        - `nunique.mean`: Mean baseline device count for each `hex_id` and weekday.
        - `nunique.std`: Standard deviation of baseline device counts.
        - `n_baseline`: Baseline mean device count.
        - `n_difference`: Difference between the current and baseline device counts.
        - `percent_change`: Percentage change in device counts relative to the baseline.
        - `z_score`: Standardized activity level relative to the baseline.
    """
    # Convert datetime to date and group by hex_id and date, counting unique devices
    activity = (
        ddf.assign(date=lambda x: dd.to_datetime(ddf["datetime"].dt.date))
        .groupby(["hex_id", "date"])["uid"]
        .nunique()
        .to_frame("nunique")
        .reset_index()
        .compute()
    )

    # Add weekday column for future standardization
    activity["weekday"] = activity["date"].dt.weekday

    # Define baseline period and calculate mean and standard deviation for each hex_id and weekday
    baseline = activity[activity["date"].between(start, end)]
    mean = baseline.groupby(["hex_id", "weekday"]).agg({"nunique": ["mean", "std"]})
    mean.columns = mean.columns.map(".".join)

    # Initialize scalers for each `hex_id`
    scalers = {}
    for hex_id in tqdm(baseline["hex_id"].unique()):
        scaler = StandardScaler()
        scaler.fit(baseline[baseline["hex_id"] == hex_id][["nunique"]])
        scalers[hex_id] = scaler

    # Merge activity data with baseline mean and standard deviation
    activity = pd.merge(activity, mean, on=["hex_id", "weekday"], how="left")

    # Calculate z-scores using the pre-trained scalers
    for hex_id, scaler in tqdm(scalers.items()):
        predicate = activity["hex_id"] == hex_id
        score = scaler.transform(activity[predicate][["nunique"]])
        activity.loc[predicate, "z_score"] = score

    # Compute baseline, difference, and percent change
    activity["n_baseline"] = activity["nunique.mean"]
    activity["n_difference"] = activity["nunique"] - activity["n_baseline"]
    activity["percent_change"] = 100 * (
        activity["nunique"] / (activity["n_baseline"]) - 1
    )

    return (
        activity[
            [
                "date",
                "hex_id",
                "weekday",
                "nunique",
                "nunique.mean",
                "nunique.std",
                "n_baseline",
                "n_difference",
                "percent_change",
                "z_score",
            ]
        ]
        .sort_values(["date", "hex_id"], ascending=True)
        .set_index(["date", "hex_id"])
    )


def categorize(row):
    if pd.notna(row["healthcare"]) or row["amenity"] in (
        "doctors",
        "dentist",
        "clinic",
        "hospital",
        "pharmacy",
    ):
        return "health facilities"
    elif row["amenity"] in ("kindergarten", "school", "college", "university") or row[
        "building"
    ] in ("kindergarten", "school", "college", "university"):
        return "education"


def compute_stops(
    ddf: DataFrame,
    stay_locations_kwds={"minutes_for_a_stop": 5.0, "spatial_radius_km": 0.25},
    resolution: int = 7,
) -> DataFrame:
    """
    Calculate stop locations from a trajectory DataFrame using spatial and temporal parameters.

    The function takes a DataFrame containing GPS trajectory data and calculates stop locations
    where an entity remains within a spatial radius for a given duration. The resulting DataFrame
    contains additional information about the stops, such as entry and exit times, as well as
    H3 hexagonal grid IDs representing the stop locations.

    Parameters
    ----------
    ddf : DataFrame
        Input Dask DataFrame containing GPS trajectory data. It must have the columns 'hex_id',
        'latitude', and 'longitude'.

    stay_locations_kwds : dict, optional
        A dictionary of parameters controlling the stop detection logic. The default is
        `{"minutes_for_a_stop": 5.0, "spatial_radius_km": 0.25}`, where:
        - "minutes_for_a_stop": Minimum duration (in minutes) to consider a location as a stop.
        - "spatial_radius_km": Spatial radius (in kilometers) to define the area around a point
          to consider it as a stop.

    resolution : int, optional
        Resolution level for H3 hexagonal grid representation of stops. Higher values increase
        the resolution, meaning smaller hexagons. Default is 7.

    Returns
    -------
    DataFrame
        A Dask DataFrame with the detected stops, containing columns:
        - "datetime": The timestamp of entering the stop.
        - "leaving_datetime": The timestamp of leaving the stop.
        - "date": The date of the stop.
        - "hex_id": The H3 hexagonal grid ID at the specified resolution level, representing the
          stop location.
        - "geometry": Geopandas geometry of the stop points.
    """
    # Assign columns before using mobilkit
    ddf["tile_ID"] = ddf["hex_id"]
    ddf["lat"] = ddf["latitude"]
    ddf["lng"] = ddf["longitude"]

    # Detect stops based on spatial and temporal thresholds
    STOPS = mobilkit.spatial.findStops(
        ddf,
        stay_locations_kwds=stay_locations_kwds,
    )
    # Calculate H3 hex ID for each stop based on latitude and longitude
    STOPS["hex_id"] = STOPS.apply(
        lambda row: h3.geo_to_h3(row["lat"], row["lng"], resolution=resolution),
        meta=("hex_id", "string"),
        axis=1,
    )

    # Convert datetime columns to proper format
    STOPS["datetime"] = dd.to_datetime(STOPS["datetime"])
    STOPS["leaving_datetime"] = dd.to_datetime(STOPS["leaving_datetime"])
    STOPS["date"] = STOPS["datetime"].dt.date.astype("string")
    STOPS["hex_id"] = STOPS["hex_id"].astype("string")

    # Convert to a GeoDataFrame
    STOPS = dask_geopandas.from_dask_dataframe(
        STOPS,
        geometry=dask_geopandas.points_from_xy(STOPS, "lng", "lat"),
    ).set_crs("EPSG:4326")

    return STOPS
