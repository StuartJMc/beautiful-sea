import numpy as np
import pandas as pd


def data_summary(df):
    """
    Perform exploratory data analysis on a given dataset.

    Args:
    df: Pandas DataFrame.

    Returns:
    None
    """

    # Set the maximum number of columns and rows to None
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    # Preview the first 5 rows of the dataset
    print("First 5 rows of the dataset:")
    display(df.head())

    # Check the shape of the dataset
    print("Shape of the dataset:", df.shape)

    # Check the data types of each column
    print("Data types of each column:")
    display(df.dtypes)

    # Check for missing values
    print("Missing values per column:")
    display(df.isnull().sum())

    # Descriptive statistics
    print("Descriptive statistics of numerical columns:")
    display(df.describe())

    # Set the maximum number of columns and rows to None
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
