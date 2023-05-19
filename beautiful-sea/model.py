import calendar
import random
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pmdarima as pm
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


def aggregate_by_date(df, features, label, date_features=["date"]):
    """Aggregate feaature table to date level for time series analysis. This is a requirment for the models being used

    Args:
        df (pd.DataFrame): Main feature table (at more granular levels than just date)
        features (list of str): external features
        label (str): target label column
        non_features (list of str): any features that aren't features but are needed for segementation/visulisation
        date_features (list of str), default ['date']: date features (aside from date) that are needed for features

    Returns:
        _type_: _description_
    """
    all_columns = [label] + features + date_features

    # map column to aggregation
    feature_agg = {
        feature: "first" for feature in features
    }  # all features are date level at the most granular
    label_agg = {
        label: "mean"
    }  # mean, so that different sampling frequencies don't affect the label
    all_agg = {**feature_agg, **label_agg}
    df_date_level = df.copy()[all_columns].groupby("date").agg(all_agg)

    return df_date_level


def resample_features(data, features, label, freq="M", show_info_loss=False):
    """
    Resample multiple features in a DataFrame to a specified frequency level,
    allowing for different aggregation methods for each feature.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the features.
    freq (str): The desired frequency for resampling (default: 'M' for monthly).
    features (list): The list of features to resample.
    label (list): Label to resample.
    show_info_loss (bool): Whether to print information about the loss of data

    Returns:
    - resampled_data (pd.DataFrame): The resampled DataFrame with features at the
        specified frequency level. The data is interpolated to fill the gaps.
    """
    # Convert the index to DatetimeIndex if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Define the aggregation methods for each feature type
    min_agg = {feature: "min" for feature in features if "min" in feature.lower()}
    max_agg = {feature: "max" for feature in features if "max" in feature.lower()}
    mean_feature_agg = {
        feature: "mean"
        for feature in features
        if feature not in min_agg.keys() and feature not in max_agg.keys()
    }
    label_agg = {label: "mean"}
    all_agg = {**min_agg, **max_agg, **mean_feature_agg, **label_agg}

    # Initialize an empty DataFrame to store the resampled data
    resampled_data = pd.DataFrame()

    # Iterate over each feature and perform resampling
    for feature, method in all_agg.items():
        if feature not in data.columns:
            continue  # skip

        # Perform resampling with the specified aggregation method
        resampled_feature = getattr(data[feature].resample(freq), method)()

        # Add the resampled feature to the output DataFrame
        resampled_data[feature] = resampled_feature

    if show_info_loss:
        print(
            f"{sum(data[label].notnull())} non null {label} labels in original, {sum(resampled_data[label].notnull())} non null {label} labels in resampled_data"
        )

    # Interpolate resample data to fill in gaps (e.g months where no sampling occured)
    interpolated_data = resampled_data.interpolate(
        method="linear", limit_direction="both"
    )

    return interpolated_data


def add_lagged_features(df, feature_columns, lag_values):
    """
    Add lagged versions of specified feature columns to a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to add lagged features to.
        feature_columns (list): List of column names for the features to lag.
        lag_values (list): List of lag values to create for each feature.

    Returns:
        pandas.DataFrame: The DataFrame with added lagged features.
    """
    # copy to avoid fragmentation
    warnings.filterwarnings("ignore")
    for column in feature_columns:
        for lag in lag_values:
            lagged_column = f"{column}_lag{lag}"
            df[lagged_column] = df[column].copy().shift(lag)
    warnings.filterwarnings("default")
    return df.copy()


def create_time_series_table(
    df,
    features,
    label,
    date_features=["date"],
    freq="M",
    lag_values=[1, 2, 3],
    return_features=True,
):
    """_summary_

    Args:
        df (pd.DataFrame): Main feature table (Potentially at more granular levels than just date)
        features (list of str): external features
        label (str):  target label column
        date_features (list of str), default ['date']: date features (aside from date) that are needed for features.
        freq (str, optional): Frequency to resample. Defaults to 'M'.
        lag_values (list of int, optional): Lag to be applied to each feature. Defaults to [1,2,3]
    """

    # select date range where the first label is present
    min_date = df.date[df[label].notnull()].min()
    max_date = df.date[df[label].notnull()].max()

    # reduce to date level index
    df_date_level = aggregate_by_date(df, features, label, date_features)

    # resample to [freq] (e.g make sure features are evenly space per month ('M'))
    df_resampled = resample_features(df_date_level, features, label, freq)

    # add lagged features (to hold information about how past features inpact the label)
    df_lag = add_lagged_features(df_resampled.copy(), features, [1, 2, 3]).copy()

    # note the lagged features
    lagged_features = [x for x in df_lag.columns if "lag" in x]
    features = lagged_features + features

    # only keep the date range where the label is present (not null)
    df_final = df_lag[(df_lag.index.date >= min_date) & (df_lag.index.date <= max_date)]

    if return_features:
        return df_final, features
    else:
        return df_final


def split_dataset_by_date(dataset, label, features, split_date, last_date=None):
    """
    Split a dataset into training and testing sets based on a specified date.

    Args:
        dataset (pandas.DataFrame): The dataset to split.
        label(str): The name of the target variable in the dataset.
        split_date (str or pd.Timestamp): The split date to separate the data.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    date = pd.to_datetime(split_date)

    if last_date is not None:
        df = dataset[pd.to_datetime(dataset.index) <= last_date].copy()
    else:
        df = dataset.copy()

    # Split the dataset based on the split date
    df_train = df[pd.to_datetime(df.index) < date].copy()
    df_test = df[pd.to_datetime(df.index) >= date].copy()

    # Separate the target variable
    X_train = df_train[features]
    X_test = df_test[features]
    y_train = df_train[label]
    y_test = df_test[label]

    return X_train, X_test, y_train, y_test


def plot_forecast(
    train, forecast, actual=None, error_bounds=None, title="Time Series Forecast"
):
    """
    Plot the training time series, predicted/actual forecast, and error bounds (optional).

    Args:
        train (array-like): Training time series data.
        forecast (array-like): Forecasted values.
        actual (array-like, optional): Actual values (if available).
        error_bounds (tuple, optional): Tuple containing upper and lower error bounds.

    Returns:
        None (displays the plot)
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=train.index, y=train, label="Training Data")

    forecast_start = len(train)
    forecast_end = forecast_start + len(forecast)

    if actual is not None:
        sns.lineplot(x=actual.index, y=actual, label="Actual")

    sns.lineplot(x=actual.index, y=forecast, label="Forecast")

    if error_bounds is not None:
        upper_bound, lower_bound = error_bounds
        plt.fill_between(
            range(forecast_start, forecast_end),
            lower_bound,
            upper_bound,
            alpha=0.2,
            label="Error Bounds",
        )

    plt.xlabel("Date")
    plt.ylabel(train.name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


class ConstantValueModel:
    """
    Last Value Model: Predicts future values by repeating the last observed value.
    """

    def __init__(self, endog, method):
        self.endog = endog
        self.method = method
        self.constant_value = None

    def fit(self):
        if self.method == "mean":
            self.constant_value = self.endog.mean()
        elif self.method == "median":
            self.constant_value = self.endog.median()
        elif self.method == "last":
            self.constant_value = self.endog[-1]

    def forecast(self, future_dates):
        return np.array([self.constant_value] * len(future_dates))


class MonthlyAverageModel:
    """
    Monthly Average Model: Predicts future values by taking the average of the corresponding month's historical values.
    Assumes that the input series has monthly frequency.
    """

    def __init__(self, endog):
        self.endog = endog
        self.monthly_average = None

    def fit(self):
        self.endog = pd.Series(self.endog)
        self.endog.index = pd.to_datetime(self.endog.index)
        self.monthly_average = self.endog.groupby(self.endog.index.month).mean()
        self.nobs = len(self.endog)

    def forecast(self, future_dates):
        preds = []
        for month in pd.to_datetime(future_dates).month:
            preds = preds + [self.monthly_average[month]]
        return np.array(preds)


def evaluate_predictions(results_df, y_pred, y_test, model_name):
    """Calculate metrics and appends to a results df

    Args:
        results_df (pd.DataFrame): columns=['Model','RMSE','MAE','MAPE']
        y_pred (array type): Results of a model prediction
        y_test (array type): Actual value for test period
        model_name (str): descriptive name

    Returns:
        results_df: updated results df
    """

    # Calculate the MAE and MAPE metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Print the metrics to the console
    print("MAE:", mae)
    print("MAPE:", mape)
    print("RMSE:", rmse)

    results_df.loc[results_df.shape[0]] = [model_name, rmse, mae, mape]

    # Return the updated results
    return results_df


def visualize_monthly_trend(y):
    """Plots trend per month (for trend and seasonality analysis)

    Args:
        y (pd.Series float): A time series (indexed by date)
    """
    data = y.to_frame()
    # Extract month and year from the date
    data["month"] = data.index.month
    data["year"] = data.index.year

    # Group the data by month and year
    grouped_data = data.groupby(["month", "year"]).mean().reset_index()

    # Get the unique months
    months = sorted(data["month"].unique())

    # Set up the plot grid
    fig, axes = plt.subplots(1, len(months), figsize=(15, 3), sharey=True)

    # Iterate over each month and plot the trends
    for i, month in enumerate(months):
        month_data = grouped_data[grouped_data["month"] == month]
        ax = axes[i]

        # Plot the monthly trends
        ax.plot(month_data["year"], month_data[y.name])

        # Calculate and plot the mean line across all years
        mean_value = grouped_data[grouped_data["month"] == month][y.name].mean()
        ax.axhline(mean_value, color="red", linestyle="--")

        # Set the x-axis label, y-axis label, and title for each subplot
        ax.set_xlabel("Year")
        if i == 0:
            ax.set_ylabel(f"Mean {y.name}")
        else:
            ax.set_ylabel("")
        ax.set_title(calendar.month_name[month])

        # Rotate x-axis tick labels
        ax.tick_params(axis="x", rotation=90)

        # Format x-axis tick labels as integers
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(f"Monthly Trend for {y.name}")

    plt.tight_layout()

    # Display the plot
    plt.show()


class MarineTimeSeriesAnalysis:
    """Class for pregressing through time series analysis, fitting SARIMAX models and understanding the impact of external features"""

    def __init__(
        self,
        df,
        label,
        features,
        date_features,
        split_date,
        last_date=None,
        freq="M",
        lag_values=[1, 2, 3],
    ):
        self.df = df
        self.features = features
        self.label = label
        self.split_date = split_date
        self.last_date = last_date

        # seperate out the train and test data
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_dataset(
            self.df,
            self.label,
            self.features,
            date_features,
            self.split_date,
            self.last_date,
            freq,
            lag_values,
        )

        self.model = None
        self.y_pred = None
        self.best_params = None

        self.exog_model = None
        self.y_pred_exog = None

    def create_dataset(
        self,
        df,
        label,
        features,
        date_features,
        split_date,
        last_date=None,
        freq="M",
        lag_values=[1, 2, 3],
    ):
        """Wrapper for create_time_series_table and split_dataset_by_date, already defined."""
        # combine the date aggregation with the feature split
        df_date_level, features = create_time_series_table(
            df,
            features=features,
            label=label,
            date_features=date_features,
            freq=freq,
            lag_values=lag_values,
        )

        return split_dataset_by_date(
            df_date_level, label, features, split_date, last_date=last_date
        )

    def train_auto_sarima(self):
        """Use pmdarima to find the best params for a SARIMAX model. Then train a SARIMAX type with these params and save the results object."""
        model = pm.auto_arima(self.y_train, seasonal=True, m=12)

        # extract best params
        self.best_params = model.get_params()
        self.y_pred = model.predict(self.y_test.shape[0])

        # save a fitted SARIMAX type model with params
        sarima_model = SARIMAX(self.y_train, **self.best_params)
        self.model = sarima_model.fit(disp=False)

        return self.model

    def forecast(self, model_type="SARIMA"):
        """Forecast the test data using the fitted SARIMAX model.

        model_type(str): SARIMA or SARIMAX (with exogenous features)
        """
        results_df = pd.DataFrame(columns=["model_name", "rmse", "mae", "mape"])
        if model_type == "SARIMA":
            results = evaluate_predictions(
                results_df, self.y_pred, self.y_test, "SARIMA From AutoArima"
            )
            plot_forecast(
                self.y_train,
                self.y_pred,
                actual=self.y_test,
                error_bounds=None,
                title="Time Series Forecast",
            )
        elif model_type == "SARIMAX":
            results = evaluate_predictions(
                results_df, self.y_pred_exog, self.y_test, "SARIMAX From AutoArima"
            )
            plot_forecast(
                self.y_train,
                self.y_pred_exog,
                actual=self.y_test,
                error_bounds=None,
                title="Time Series Forecast",
            )

    def plot_seasonality(self):
        """Plot the trend per month of the time series."""
        visualize_monthly_trend(self.y_train)

    def plot_seasonal_decomposition(self):
        """Plot the seasonal decomposition of the time series."""
        result = seasonal_decompose(
            self.y_train, model="additive", period=12, two_sided=False
        )
        result.plot()

    def residual_analysis(self):
        """Compare the residuals of the SARIMAX model to potential explainer variables. Return Visual and ranked correlation plot."""
        resids = self.model.resid

        correlations = []

        # Iterate over each external variable
        for feature in self.features:
            # Calculate the correlation between residuals and the current feature
            correlation = pd.Series(resids).corr(
                self.X_train[feature].fillna(self.X_train[feature].mean())
            )

            # Append the correlation to the DataFrame
            correlations.append({"Feature": feature, "Correlation": correlation})

        # Create the correlations DataFrame
        correlations_df = pd.DataFrame(correlations)

        # Sort the DataFrame by correlation in descending order
        correlations_df = correlations_df.sort_values("Correlation", ascending=False)

        plt.figure(figsize=(6, 8))
        sns.barplot(
            y=correlations_df["Feature"], x=correlations_df["Correlation"], orient="h"
        )
        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.title("Correlations of External Variables with Residuals")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        return correlations_df

    def train_exog_sarima(self, exogs):
        """Train a SARIMAX model with the best params and the given exogenous features."""
        exog_model = SARIMAX(
            endog=self.y_train, exog=self.X_train[exogs], **self.best_params
        )
        self.exog_model = exog_model.fit(disp=False)
        self.y_pred_exog = self.exog_model.forecast(
            self.y_test.shape[0], exog=self.X_test[exogs]
        )
        return self.exog_model
