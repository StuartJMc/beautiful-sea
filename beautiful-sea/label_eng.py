"""This module contains functions for creating target variable for species predictions"""
import functools as ft

import numpy as np
import pandas as pd


def group_by_sample_date(
    df: pd.DataFrame, target_col: str, group: list[str], agg_col: str = "occurrence"
) -> pd.DataFrame:
    """Custom groupby to first group by sample id then by chosen grouping.

    Justification: As number of samples per date is not consistent, we need to group by sample id first to get the total number of species per sample. Then take the mean sample per date/area

    Args:
        df : DataFrame to be grouped.
        target_col : Name of aggregated column passed out.
        group: A list of the grouping hierarchy (after sample_id grouping)
        agg_col : Value to be aggregated. Defaults to "occurrence".

    Returns:
        grouped dataframe.
    """
    df_sample_grouped = (
        df.groupby(["sample_id"] + group)[agg_col]
        .sum()
        .rename(target_col)
        .reset_index()
    )
    df_date_grouped = df_sample_grouped.groupby(group)[target_col].mean().reset_index()

    return df_date_grouped


def sessil_to_count_convertion(df: pd.DataFrame) -> pd.Series:
    """Represent Percentage coverage of sessil species as a count.

    Justification: to aggregate sessil and mobil abundance values together. This reflects the conversion in the ampa report.

    Source->https://www.researchgate.net/publication/262458621_Human_Disturbance_in_a_Tropical_Rocky_Shore_Reduces_Species_Diversity
    """
    # calculate the totals and number of samples for both mobil and sessil
    n_sessil = df[df["is_mobil"] == False]["occurrence"].sum()
    n_sessil_samples = df[df["is_mobil"] == False]["sample_id"].nunique()
    n_mobil = df[df["is_mobil"] == True]["occurrence"].sum()
    n_mobil_samples = df[df["is_mobil"] == True]["sample_id"].nunique()

    # only need to do this if both sessil and mobil species are present
    if (n_mobil > 0) and (n_sessil > 0):
        mobil_sessil_ratio = (n_sessil / n_sessil_samples) / (n_mobil / n_mobil_samples)

        adjusted_col = df.apply(
            lambda row: row["occurrence"] * mobil_sessil_ratio
            if row["is_mobil"] == False
            else row["occurrence"],
            axis=1,
        )

        return adjusted_col

    else:
        return df["occurrence"]


def calculate_biodiversity_metric(group: pd.DataFrame) -> pd.Series:
    """Calculate shannon diversity index and shannon equitability index.

    Justification: To have a metric that reflects the diversity of the overall community. This is a standard metric used in ecology.

    info-> https://www.statology.org/shannon-diversity-index/#:~:text=The%20Shannon%20Diversity%20Index%20(sometimes,i%20*%20ln(pi)

    Args:
        group : grouped dataframe

    Returns:
        results: pandas series containing shannon diversity index and shannon equitability index
    """
    df = group.copy()[group["occurrence"] > 0]

    if df["species"].nunique() > 1:
        # Convert the sessil species to count
        df["occurrence"] = sessil_to_count_convertion(df)

        # Calculate the proportion of each species in the total count
        proportions = df.groupby("species").occurrence.sum() / df.occurrence.sum()

        # Calculate the Shannon diversity index
        shannon_index = -sum(proportions * np.log(proportions))

        # Calculate the weighted diversity score by multiplying the Shannon index with coverage
        shannon_equitability_index = shannon_index / np.log(df["species"].nunique())
        return pd.Series(
            {
                "shannon_index": shannon_index,
                "shannon_equitability_index": shannon_equitability_index,
            }
        )
        # return pd.Series(shannon_index,shannon_equitability_index)
    else:
        return pd.Series({"shannon_index": 0, "shannon_equitability_index": 0})


def create_target_table(df_all):
    """Creates target variables aggregated at the date, zone and tidal level.

    Params:
        - df_all: cleaned features table

    Returns:
        - df_targets: table with target columns for each datetime, zone and tidal level
    """
    # the level of the final aggregation
    dt_group = ["dt", "zone", "supratidal_or_middle_intertidal"]

    df_sessile_to_count = df_all[df_all.is_mobil == False]
    df_mobile_to_count = df_all[df_all.is_mobil == True]

    # calculate target related to invasive species
    # assumption: treat invasive == unknown as non-invasive
    df_sessile_to_count.loc[:, "invasive"] = (
        df_sessile_to_count.loc[:, "invasive"]
        .apply(lambda x: x.replace("unknown", "No"))
        .copy()
    )
    df_total_by_invasive = group_by_sample_date(
        df_sessile_to_count,
        target_col="total_occurrence",
        group=dt_group + ["invasive"],
    )
    # pivot to get one column for invasive and one column for non-invasive
    target_invasive = pd.pivot_table(
        df_total_by_invasive,
        values="total_occurrence",
        index=dt_group,
        columns=["invasive"],
    )
    target_invasive.rename(
        columns={"No": "total_non_invasive_sessile", "Yes": "total_invasive_sessile"},
        inplace=True,
    )

    # calculated targets related to endangered status
    # assumption: defined our own endangered status groups (see data dict)
    df_total_by_endangered = group_by_sample_date(
        df_mobile_to_count,
        target_col="total_occurrence",
        group=dt_group + ["endangered"],
    )
    # pivot to get one column for each endangered status
    target_endangered = pd.pivot_table(
        df_total_by_endangered,
        values="total_occurrence",
        index=dt_group,
        columns=["endangered"],
    )
    target_endangered.rename(
        columns={
            "endangered": "total_endangered_mobile",
            "not_endangered": "total_not_endangered_mobile",
            "potentially_endangered": "total_pot_endangered_mobile",
        },
        inplace=True,
    )

    # As the unique values can't be combined again, multiple aggregations are needed at dt/zpne/supra_medium and just date
    # avoid divide by zero error
    df_non_zero = df_all[df_all.occurrence > 0]
    # create biodiversity index
    # idea: using the shannon index to calculate the biodiversity(as in ampa paper)
    shannon_dt_group = (
        df_non_zero.groupby(dt_group)
        .apply(
            lambda x: calculate_biodiversity_metric(
                x[["species", "occurrence", "is_mobil", "sample_id"]]
            )
        )
        .rename(
            columns={
                "shannon_index": "shannon_index_dt_z_sm",
                "shannon_equitability_index": "shannon_equitability_index_dt_z_sm",
            }
        )
    ).reset_index()

    # groupby date
    shannon_date = df_non_zero.groupby("date").apply(
        lambda x: calculate_biodiversity_metric(
            x[["species", "occurrence", "is_mobil", "sample_id"]]
        )
    )

    # join all target columns together at dt/zone/supratidal_or_middle_intertidal level
    dfs = [
        target_invasive,
        target_endangered,
        shannon_dt_group,
    ]
    df_targets_dt = ft.reduce(
        lambda left, right: pd.merge(left, right, how="outer", on=dt_group),
        dfs,
    )
    df_targets_dt["date"] = df_targets_dt["dt"].dt.date

    # join all target columns together at date level
    dfs = [df_targets_dt, shannon_date]
    df_targets_all = ft.reduce(
        lambda left, right: pd.merge(left, right, how="outer", on="date"),
        dfs,
    )

    # drop rows with no dt
    df_targets_all.dropna(subset=["dt"], inplace=True)

    return df_targets_all
