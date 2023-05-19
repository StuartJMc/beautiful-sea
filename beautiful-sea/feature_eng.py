"""This module contains all functions for adding additional features"""
import functools as ft

import numpy as np
import pandas as pd
import requests
import xarray as xr


def add_time_features(df, dt_col="dt"):
    """Adds cyclical and linear time features. These could be used for modelling.

    Args:
        df (pd.DataFrame): dataframe containing datetime column
    """
    df = df.copy()
    df["date"] = df[dt_col].dt.date
    df["year"] = df[dt_col].dt.year
    df["year_month"] = df[dt_col].dt.strftime("%Y-%m")

    df["weekofyear_sin"] = np.sin(
        df[dt_col].dt.strftime("%W").astype(int) * (2 * np.pi / 52)
    )
    df["weekofyear_cos"] = np.cos(
        df[dt_col].dt.strftime("%W").astype(int) * (2 * np.pi / 52)
    )
    df["month_sin"] = np.sin(df[dt_col].dt.month * (2 * np.pi / 12))
    df["month_cos"] = np.cos(df[dt_col].dt.month * (2 * np.pi / 12))

    return df


def create_ampa_features(df_ampa: pd.DataFrame) -> pd.DataFrame:
    """Add features from AMPA data for modelling. This may need adjusting for other sources.

    Args:
        df (pd.DataFrame): Cleaned AMPA dataset

    Returns:
        pd.DataFrame: AMPA dataset with engineered features
    """
    df_tide = (
        df_ampa.groupby(["dt", "zone", "supratidal_or_middle_intertidal"])
        .tide.agg(
            tide_max="max",
            tide_min="min",
            tide_mean="mean",
        )
        .reset_index()
    )

    # EDA showed that max and min don't vary by this group, so no need to include all
    df_water_temp = (
        df_ampa.groupby(["dt", "zone", "supratidal_or_middle_intertidal"])
        .water_temperature.agg(
            water_temp_max="max",
        )
        .reset_index()
    )

    df_features = df_tide.merge(
        df_water_temp, on=["dt", "zone", "supratidal_or_middle_intertidal"]
    )

    return df_features


def get_weather_data(
    lat=38.70, lon=-9.42, start_date="2011-01-01", end_date="2020-12-31"
):
    """Get weather data from meteo API. URL and API query provided below for replication"""
    # API URL
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Parameters for near cascais
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,precipitation_sum,precipitation_hours",
        "timezone": "Europe/London",
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
    else:
        print("Error: Failed to fetch data from the API.")

    df = df.daily.apply(pd.Series).T

    df["date"] = pd.to_datetime(df["time"]).dt.date
    df.drop(columns=["time"], inplace=True)
    return df


def load_uea_netcdf_file(
    filepath, lat=37.5, lon=-7.5, start_date="2010-01-01", end_date="2021-12-31"
):
    """
    Loads a netCDF file containing University of East Anglia Climate data into a pandas DataFrame.

    Data is in 5 x 5 degree grid cells. So the below source can be used to find the suitable coords.

    Source -> https://www.uea.ac.uk/groups-and-centres/climatic-research-unit/data
    Acknowledgement->  Morice et al. (2021) and Osborn et al. (2021) https://crudata.uea.ac.uk/cru/data/temperature/?_ga=2.111816964.1851588793.1684357938-1486011184.1682889046#sciref

    Parameters:
        - filepath: The path to the netCDF file to load.

    Returns:
        A pandas DataFrame containing the data from the netCDF file.
    """
    # Load the netCDF file using xarray
    ds = xr.open_dataset(filepath)

    # Convert the data to a pandas DataFrame
    df = ds.to_dataframe()

    # Reset the index to use the time and other dimensions as columns
    df.reset_index(inplace=True)

    # Optional: Rename columns to remove the `__` separator
    df.columns = [col.replace("__", "") for col in df.columns]

    # Filter the rows that are in the 5 by 5 grid centred on -7.5 longitude and 37.5 latitude
    mask = (df.longitude == lon) & (df.latitude == lat)
    df = df[mask].drop_duplicates(subset=["time"])

    # drop columns that aren't needed
    df.drop(
        columns=[
            "latitude",
            "longitude",
            "time_bnds",
            "latitude_bnds",
            "longitude_bnds",
            "bnds",
        ],
        inplace=True,
    )

    if "realization" in df.columns:
        df.drop(columns=["realization", "realization_bnds"], inplace=True)

    # filter for time range
    df = df[(df.time >= start_date) & (df.time <= end_date)]

    # convert to year month for koin
    df["year_month"] = df["time"].dt.strftime("%Y-%m")

    df.drop(columns=["time"], inplace=True)

    return df


def get_uea_data(
    uea_data_folder="../data/crudata",
    lat=37.5,
    lon=-7.5,
    start_date="2010-01-01",
    end_date="2021-12-31",
):
    """Fetches and transformes UEA data for modelling

    Source -> https://www.uea.ac.uk/groups-and-centres/climatic-research-unit/data
    Acknowledgement->  Morice et al. (2021) and Osborn et al. (2021) https://crudata.uea.ac.uk/cru/data/temperature/?_ga=2.111816964.1851588793.1684357938-1486011184.1682889046#sciref

    Args:
        uea_data_folder (str, optional): Local path to uea data. Defaults to "../data/crudata".

    Returns:
        _type_: _description_
    """
    # get university of east anglia data
    df_hadsst = load_uea_netcdf_file(
        uea_data_folder + "/HadSST.4.0.1.0_median.nc",
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )
    df_hadsst.rename(columns={"tos": "sea_surface_temp_anomaly"}, inplace=True)

    df_crutem = load_uea_netcdf_file(
        uea_data_folder + "/CRUTEM.5.0.1.0.alt.anomalies.nc",
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )
    df_crutem.rename(columns={"tas": "land_air_temp_anomaly"}, inplace=True)

    df_hadcrut = load_uea_netcdf_file(
        uea_data_folder + "/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc",
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )
    df_hadcrut.rename(
        columns={"tas_mean": "sea_air_combined_temp_anomaly"}, inplace=True
    )

    # join all tables together (at month level)
    dfs = [df_hadsst, df_hadcrut, df_crutem]
    df = ft.reduce(
        lambda left, right: pd.merge(left, right, on="year_month", how="left"), dfs
    )

    return df


def fetch_copernicus_data(file_path: str) -> pd.DataFrame:
    """function reads netCDF files from Copernicus into pandas dataframes,
    aggregates to the date level, and outputs a dataframe"""
    # read in .nc file
    DS1 = xr.open_dataset(file_path)

    # convert to dataframe
    df = DS1.to_dataframe()

    # reset the index
    df = df.reset_index(level=[0, 1, 2, 3])

    # aggregate to date level (mean taken of all points nearby)
    df = df.groupby(by=["time"]).mean().reset_index()

    # removal of redundant columns
    df = df.drop(labels=["latitude", "longitude", "depth"], axis=1)

    return df


def prep_copernicus_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """function combines copernicus marine data at the date level"""

    df = df1.merge(df2, how="outer", on="time")
    df.rename(columns={"time": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def get_copernicus_data(copernicus_data_folder="../data/copernicus"):
    """Fetches and transforms copernicus data for modelling."""
    # get copernicus data
    df_copernicus1 = fetch_copernicus_data(
        copernicus_data_folder + "/cmems_mod_glo_phy_my_0.083_P1D-m_1683577362247.nc"
    )
    df_copernicus2 = fetch_copernicus_data(
        copernicus_data_folder
        + "/cmems_mod_ibi_bgc_my_0.083deg-3D_P1D-m_1683666099015.nc"
    )

    # combine copernicus data
    df_copernicus = prep_copernicus_data(df_copernicus1, df_copernicus2)

    return df_copernicus


def fetch_AQI_data(file_path: str) -> pd.DataFrame:
    """Function reads csv files from WAQI into pandas dataframes, removes whitespace from column names,
    replaces whitespace fields with NaN, and outputs a dataframe"""

    # read in data
    df = pd.read_csv(file_path)

    # rename columns (remove whitespace at start of names)
    df.rename(
        columns={
            " pm25": "pm25",
            " pm10": "pm10",
            " o3": "o3",
            " no2": "no2",
            " so2": "so2",
        },
        inplace=True,
    )

    # replace whitespace fields with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    return df


def prep_AQI_files(
    df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame
) -> pd.DataFrame:
    """Function takes three nearby AQI dataframes, cleans and aggregates to date level."""
    # concatenate three dataframes
    df = pd.concat([df1, df2, df3])

    # convert date field to datetime type
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # convert remaining columns to float type
    df["pm25"] = pd.to_numeric(df["pm25"])
    df["pm10"] = pd.to_numeric(df["pm10"])
    df["o3"] = pd.to_numeric(df["o3"])
    df["no2"] = pd.to_numeric(df["no2"])
    df["so2"] = pd.to_numeric(df["so2"])

    # aggregate to date level by taking average of three files
    df = df.groupby(by=["date"]).mean().reset_index()

    return df


def get_AQI_data(AQI_data_folder="../data/AQI"):
    """Fetches and transforms AQI data for modelling."""
    # get AQI data
    df_AQI_1 = fetch_AQI_data(
        AQI_data_folder + "/entrecampos,-lisboa, portugal-air-quality.csv"
    )
    df_AQI_2 = fetch_AQI_data(
        AQI_data_folder + "/mem-martins, sintra, portugal-air-quality.csv"
    )
    df_AQI_3 = fetch_AQI_data(
        AQI_data_folder + "/olivais,-lisboa, portugal-air-quality.csv"
    )

    # combine AQI data
    df_AQI = prep_AQI_files(df_AQI_1, df_AQI_2, df_AQI_3)

    return df_AQI


def get_ohi_data(filepath="../data/OHI/scores.csv"):
    """
    Get Ocean Health Index data, filter the data and prep for modelling

    Args:
        df_ohi (pd.DataFrame): input data

    Returns:
        pd.DataFrame: output data
    """
    # Load and convert csv data into pandas dataframe
    data = []
    col = []
    checkcol = False
    with open(filepath) as f:
        for val in f.readlines():
            val = val.replace("\n", "")
            val = val.split(",")
            if checkcol is False:
                col = val
                checkcol = True
            else:
                data.append(val)
    df = pd.DataFrame(data=data, columns=col)

    # Filter data to show only portugal data
    df_portugal = df[df.region_id == "183"]

    # Drop irrelevant columns
    df_portugal = df_portugal.drop(["region_name", "region_id"], axis=1)

    # Rename columns for readability
    df_portugal = df_portugal.rename(columns={"scenario": "year", "value": "score"})

    # Prep and further filter the data for biodiversity, habitat and species goals only
    goal_list = ["Biodiversity", "Habitat (subgoal)", "Species condition (subgoal)"]
    dimension_list = ["status", "future"]
    filtered_df = df_portugal[df_portugal["long_goal"].isin(goal_list)]
    filtered_df = filtered_df[df_portugal["dimension"].isin(dimension_list)]

    filtered_df["metric"] = (
        filtered_df.long_goal.str.split(" \(").str[0].str.lower()
        + "_"
        + filtered_df.dimension
    )
    filtered_df = filtered_df.drop(["long_goal", "dimension"], axis=1)

    # Convert the date column to a datetime format
    filtered_df["year"] = pd.to_datetime(filtered_df.year, format="%Y").dt.year
    filtered_df["score"] = pd.to_numeric(filtered_df.score)

    filtered_df = filtered_df.pivot_table(
        index="year", columns="metric", values="score", fill_value=0
    ).reset_index()

    return filtered_df
