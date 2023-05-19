import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def forecast_future_values(historic_data, num_months, end_value):
    """Synthesizes future values for a time series by projecting a lniear change in mean, resampling from corresponding historic months and adding in noise.

    Args:
        historic_data (pd.Series): Historic time series of a variable
        num_months (int): Number of months to forecast forward
        end_value (float): The mean value to project the variable to (from the historic mean)

    Returns:
        Synthetic forecasts of a variable at num_months in the future.
    """
    # Generate the array of dates at a monthly frequency
    dates = pd.date_range(start=historic_data.index[-1], periods=num_months, freq="MS")

    # The step size to add to the historic mean each month
    value_increment = (end_value - historic_data.mean()) / num_months

    # Create an array with the index for each forcast month and value the number of steps
    dates = pd.date_range(start=historic_data.index[-1], periods=num_months, freq="MS")
    positions = np.arange(1, len(dates) + 1)
    position_series = pd.Series(positions, index=dates)

    # Multiply step number by increment to get the projected mean for each month
    increment_series = position_series * value_increment

    # Initialize an empty dataframe for the forecasted values
    forecasted_data = pd.Series(name=historic_data.name, index=dates)

    # Generate forecasts for each future month by selecting from historic values.
    # Adding the mean change and noise (noise is required as SARIMA models typically smooth out predictions and this is required for visulisations)
    for date in dates:
        # get values at this month in history
        month = date.month
        month_values = historic_data[historic_data.index.month == month]

        # Select a random month value
        random_value = random.choice(month_values)

        # Add noise to the historic mean value
        noise = np.random.normal(loc=0, scale=5 * historic_data.std())
        forecasted_value = random_value + noise + increment_series[date]

        # Add the forecasted value to the dataframe
        forecasted_data.loc[date] = forecasted_value

    return forecasted_data


def create_forcasted_df(df, num_months, forecast_mapping={}):
    """
    Create a dataframe of forecasted values for each column in the input dataframe.

    Args:
        df (DataFrame): Input dataframe.
        num_months (int): Number of months to forecast.
        end_value (float): Value to forecast to.

    Returns:
        DataFrame of forecasted values.
    """
    forecasted_df = pd.DataFrame()
    for column in df.columns:
        # if a mapping is provided use this value
        if forecast_mapping[column] is not None:
            forecasted_df[column] = forecast_future_values(
                df[column], num_months, end_value=forecast_mapping[column]
            )
        # other wise use the historic mean (no real drift)
        else:
            forecasted_df[column] = forecast_future_values(
                df[column], num_months, end_value=df[column].mean()
            )

    return forecasted_df


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

    sns.lineplot(x=forecast.index, y=forecast, label="Forecast")

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
    plt.ylabel(y_train.name)
    plt.title(title)
    plt.legend()
    plt.grid(True)


def model_scenarios(model, X_train, forecast_steps, variable_ranges):
    df_scenarios = pd.DataFrame()
    # for each variable being simulated
    for variable in variable_ranges.keys():
        # get the range of values to simulate
        variable_range = variable_ranges[variable]

        # for each of the projected mean values
        for target in variable_range:
            # create empty mappings for the other variables
            forecast_mapping = {key: None for key in X_train.columns}
            # map this variable to the target value
            forecast_mapping[variable] = target

            exog_vars = list(forecast_mapping.keys())

            # get the future values for the exogenous variables
            X_future = create_forcasted_df(
                X_train[exog_vars], forecast_steps, forecast_mapping
            )

            # train the model passed in
            trained_model = model.fit(disp=False)

            # make predictions using the forecasted data
            y_pred = trained_model.forecast(forecast_steps, exog=X_future)
            df_scenario = y_pred.to_frame().reset_index()
            df_scenario.rename(columns={"index": "date"}, inplace=True)
            df_scenario["variable"] = variable
            df_scenario["projected_value"] = target
            # add to the other scenarios
            df_scenarios = pd.concat([df_scenarios, df_scenario], axis=0)

    return df_scenarios
