"""
Visualization of the predicted (vascular activity) data (compared to the actual data)
"""

import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# TODO decorator for plot functions (implement in src.utils package)

class ResultsSummarizer:
    """
    Class for creating summary and visualizations from our models results and predictions
    """
    def __init__(self,
                 x, x_train, x_test,  y, y_train, y_test, y_pred, y_pred_train, y_pred_test
                 ):
        """ initialize the summarizer with all the actual / predicted data """
        self.x = x
        self.x_train = x_train
        self.x_test = x_test

        # Y stands for vascular-activity actual/predicted
        self.y = y
        self.y_train = y_train
        self.y_test = y_test

        self.y_pred = y_pred
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test

    def plot_vascular_pred(self):
        """
        Given vascular activity and predicted vascular activity, plots them
        """
        # we only take the first 50 vessels (to not overload plotly...)
        df = self.get_true_and_pred_to_df(self.y[:, :50], self.y_pred[:, :50])

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

    def plot_mse_per_vessel(self):
        pass

    @staticmethod
    def get_true_and_pred_to_df(y_true, y_pred, time_vector=None):
        """ returns a dataframe based on blood-vessels time-series """
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