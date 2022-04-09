"""
Visualization of the predicted (vascular activity) data (compared to the actual data)
"""

import pandas as pd
import plotly.express as px


def vis_pred(y_true, y_pred, time_vector=None, test_size=None):
    """
    Given vascular activity and predicted vascular activity,
    plots the two
    """
    df = get_true_and_pred_to_df(y_true[:, :50], y_pred[:, :50], time_vector)

    fig = px.line(df)
    fig.update_xaxes(rangeslider_visible=True)
    # TODO add vertical lines for train, test, etc
    fig.update_layout(
        title="Vascular-Activity (Actual and Predicted) by Time",
        legend_title_text="Vessels List",
        xaxis_title='Time',
        yaxis_title='Vascular Activity',
    )

    # hides all graphs except for the first
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                                     if trace.name != "vessel_true_0" else ())

    fig.show()


def get_true_and_pred_to_df(y_true, y_pred, time_vector):
    """ Get a dataframe based on blood-vessels time-series"""
    y_true = y_true.T
    y_pred = y_pred.T

    # Build dict to feed dataframe
    data = {}
    for i in range(len(y_true)):
        data[f"vessel_true_{i}"] = y_true[i]
        data[f"vessel_pred_{i}"] = y_pred[i]

    # Build dataframe
    df = pd.DataFrame(data)
    # df.index = time_vector

    return df