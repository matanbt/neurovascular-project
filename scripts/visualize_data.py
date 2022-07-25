import os.path
import pandas as pd

import plotly.express as px
import plotly.figure_factory as ff

from src.datamodules.components.nv_fetcher import NVDatasetFetcher


def plot_neurons(
        fetcher: NVDatasetFetcher,
        show_first_only=True,
        plot_line=True,
        visualization_path="",
        show_fig=False
):
    """

    Args:
        fetcher: initialized dataset fetcher
        show_first_only: shows only the first graph
        plot_line: whether to plot lines or scatter
        visualization_path: path to save html (or empty string to not save)
        show_fig: whether to end the function with fig.show() or not

    Returns: the built figure
    """
    df = fetcher.get_neurons_df()

    fig = px.line(df) if plot_line else px.scatter(df)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title="Neuronal-Activity by Time",
        legend_title_text="Neurons List",
        xaxis_title='Time',
        yaxis_title='Neuronal Activity',
    )

    # Epilogue:
    if show_first_only:
        # hides all graphs except for the first
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                                         if trace.name != "neuron_0" else ())
    if visualization_path:
        fig.write_html(os.path.join(visualization_path, "neuro_activity_vis.html"))
    if show_fig:
        fig.show()

    return fig


def plot_vessels(
        fetcher: NVDatasetFetcher,
        show_first_only=True,
        plot_line=True,
        visualization_path="",
        show_fig=False
):
    """
    Same as above...
    """
    df = fetcher.get_vessels_df()

    fig = px.line(df) if plot_line else px.scatter(df)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title="Vascular-Activity by Time",
        legend_title_text="Vessels List",
        xaxis_title='Time',
        yaxis_title='Vascular Activity',
    )

    # Epilogue:
    if show_first_only:
        # hides all graphs except for the first
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                                         if trace.name != "vessel_0" else ())
    if visualization_path:
        fig.write_html(os.path.join(visualization_path, "vascu_activity_vis.html"))
    if show_fig:
        fig.show()
    return fig


def plot_coords(
        fetcher: NVDatasetFetcher,
        visualization_path="",
        show_fig=False
):
    df = fetcher.get_coords_df()

    # Create plot:
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="type")
    fig.update_layout(
        title="Coordinates Map",
        legend_title_text="Legend",
    )
    fig.update_traces(marker=dict(size=5))

    # Epilogue:
    if visualization_path:
        fig.write_html(os.path.join(visualization_path, "coordinates_vis.html"))
    if show_fig:
        fig.show()
    return fig


def plot_correlation_of_mean_activity(
    fetcher: NVDatasetFetcher,
    visualization_path="",
    show_fig=False
):
    """
    Plots the pair of dots neuronal and vascular mean-activity at each time-stamp
    """
    df_neuro = fetcher.get_neurons_df()
    df_neuro['mean_activity'] = df_neuro.mean(axis=1)
    df_vascu = fetcher.get_vessels_df()
    df_vascu['mean_activity'] = df_vascu.mean(axis=1)

    fig = px.scatter(x=df_neuro['mean_activity'], y=df_vascu['mean_activity'])
    fig.update_layout(
        title=f"'Correlation' Plot - activity means [at each time stamp] "
              f"\n Pearson Correlation: {df_neuro['mean_activity'].corr(df_vascu['mean_activity'])}",
        xaxis_title='Neuronal Activity Mean (at each time stamp)',
        yaxis_title='Vascular Activity Mean (at each time stamp)',
    )

    # Epilogue:
    if visualization_path:
        fig.write_html(os.path.join(visualization_path, "mean_activity_corr_vis.html"))
    if show_fig:
        fig.show()
    return fig


def plot_dist_neuro(
    fetcher: NVDatasetFetcher,
    visualization_path="",
    show_fig=False,
):
    df_neuro = fetcher.get_neurons_df()

    fig = ff.create_distplot(df_neuro.values.T, df_neuro.columns,
                             show_hist=False, bin_size=.2)
    fig.update_layout(
        title="Neuronal Activity - Values Distribution (of each neuron separately)",
        legend_title_text="Neurons List",
        xaxis_title='Neuronal Activity (value)',
        yaxis_title='Density',
    )

    # Epilogue:
    if visualization_path:
        fig.write_html(os.path.join(visualization_path, "neuro_dist_vis.html"))
    if show_fig:
        fig.show()
    return fig


def plot_dist_vascu(
    fetcher: NVDatasetFetcher,
    visualization_path="",
    show_fig=False,
):
    df_vascu = fetcher.get_vessels_df()

    fig = ff.create_distplot(df_vascu.values.T, df_vascu.columns,
                             show_hist=False, bin_size=.2)
    fig.update_layout(
        title="Vascular Activity - Values Distribution (of each vessel separately)",
        legend_title_text="Vessels List",
        xaxis_title='Vascular Activity (value)',
        yaxis_title='Density',
    )

    # Epilogue:
    if visualization_path:
        fig.write_html(os.path.join(visualization_path, "vascu_dist_vis.html"))
    if show_fig:
        fig.show()
    return fig


def calc_correlation():
    """ calculate the 50*425 correlations (all pair of neuron-vessel) and plot and correlations vals (probably heatmap)"""
    df_neuro = fetcher.get_neurons_df()
    df_vascu = fetcher.get_vessels_df()
    pass  # TODO


if __name__ == '__main__':
    data_dir = "./data"
    dataset_name = "2021_02_01_18_45_51_neurovascular_full_dataset"
    # path to save visualization artifacts:
    visualization_path = os.path.join(data_dir, dataset_name, "visualizations")

    fetcher = NVDatasetFetcher(dataset_name=dataset_name)

    # Save plots (HTMLs) to path:
    plot_neurons(fetcher, visualization_path=visualization_path)
    plot_vessels(fetcher, visualization_path=visualization_path)
    plot_coords(fetcher, visualization_path=visualization_path)
    # plot_correlation_of_mean_activity(fetcher)
    plot_dist_neuro(fetcher, visualization_path=visualization_path)
    plot_dist_vascu(fetcher, visualization_path=visualization_path)