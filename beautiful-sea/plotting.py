import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_lineplot(
    df,
    ylabel="occurrence",
    xlabel="dt",
    grouping=["dt", "is_mobil"],
    agg="mean",
    hue=None,
):
    # create lineplot

    df_grp = (
        df.groupby(grouping).occurrence.agg(agg).reset_index().sort_values(by=xlabel)
    )

    df_grp_mobil = df_grp[df_grp.is_mobil == True]
    df_grp_sessil = df_grp[df_grp.is_mobil == False]

    # create subplots
    fig, axs = plt.subplots(ncols=2, figsize=(18, 6))

    # plot first Seaborn plot in the first subplot
    sns.lineplot(data=df_grp_mobil, x=xlabel, y=ylabel, ax=axs[0], hue=hue)
    axs[0].set_title(f"{agg.capitalize()} Mobil Species over time")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(f"{ylabel} (num of species in sample)")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)

    # plot second Seaborn plot in the second subplot
    sns.lineplot(data=df_grp_sessil, x=xlabel, y=ylabel, ax=axs[1], hue=hue)
    axs[1].set_title(f"{agg.capitalize()} Sessil Species over time")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(f"{ylabel} (% coverage of sample)")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)

    # show plot
    plt.show()
