"""
Linear Regression Baseline
"""
import itertools
from typing import List

import numpy as np
import pandas as pd

from src.baseline.more_models import NVXGBLinearRegressionModel, PersistModel
from src.baseline.results_summarizer import ResultsSummarizer
from src.datamodules.components.nv_datasets import NVDataset_Classic

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TEST_SIZE = 700


class NVLinearRegressionModel:
    def __init__(self,
                 dataset: NVDataset_Classic,
                 test_size=TEST_SIZE):
        """ classic linear regression model """
        self.y_pred = None
        self.y_pred_test = None
        self.y_pred_train = None
        self.model = None

        x, y = dataset.to_numpy()
        self.x, self.y = x, y

        # split the dataset
        self.x_train, self.x_test = x[:-test_size], x[-test_size:]
        self.y_train, self.y_test = y[:-test_size], y[-test_size:]

        self.test_size = test_size

    def fit(self):
        """ train the model on the training set, then predict """
        alphas = 10.0 ** np.arange(-10, 10, 1)  # possible regularization alphas
        self.model = RidgeCV(alphas=alphas)
        self.model.fit(self.x_train, self.y_train)

        self.y_pred = self.model.predict(self.x)
        self.y_pred_train = self.model.predict(self.x_train)
        self.y_pred_test = self.model.predict(self.x_test)

        return self

    def get_model_hparams(self):
        return {
            'alpha': self.model.alpha_,
            'test_size': self.test_size
        }

    def evaluate(self):
        # evaluate
        mse_train = mean_squared_error(self.y_train, self.y_pred_train)
        mse_test = mean_squared_error(self.y_test, self.y_pred_test)
        mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
        mae_test = mean_absolute_error(self.y_test, self.y_pred_test)
        r2_train = r2_score(self.y_train, self.y_pred_train)
        r2_test = r2_score(self.y_test, self.y_pred_test)

        # print
        print(f">> Chosen Alpha: {self.model.alpha_}")
        print(f">> Training: MSE={mse_train}, R^2={r2_train}, MAE={mae_train}")
        print(f">> Testing: MSE={mse_test}, R^2={r2_test}, MAE={mae_test}")

        return {
            'mse_train': mse_train,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
        }

    def get_split_data(self):
        """ returns dict with splits used by model, and the model predictions """
        return {
            'x': self.x,
            'x_train': self.x_train,
            'x_test': self.x_test,
            'y': self.y,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'y_pred_train': self.y_pred_train,
            'y_pred_test': self.y_pred_test,
            'y_pred': self.y_pred,
        }


def tune_dataset_parameters(save_to_csv=True) -> pd.DataFrame:
    """
    Runs the model with multiple dataset-parameters,
    returns Dataframe with the experiments results, for further analysis
    """
    experiments_table = []

    # Possible window lengths
    poss_window_lens = list(range(1, 22, 2))
    poss_window_lens_w_zero = [0] + poss_window_lens
    poss_poly_degrees = [None]#, 2, 3]
    all_combinations = itertools.product(poss_window_lens,
                                         poss_window_lens_w_zero,
                                         poss_window_lens_w_zero,
                                         poss_poly_degrees)

    # Iterate on all possible window lengths and evaluate
    for neuro_back, neuro_forward, vascu_back, poly_degree in all_combinations:
        # create the dataset
        data_hparams = dict(
            window_len_neuro_back=neuro_back,
            window_len_neuro_forward=neuro_forward,
            window_len_vascu_back=vascu_back
        )
        dataset = NVDataset_Classic(**data_hparams)

        # train and evaluate the model
        regr_model = NVLinearRegressionModel(dataset=dataset, test_size=TEST_SIZE)
        regr_model.fit()
        model_results = regr_model.evaluate()

        # Add hyper parameters and results to the current experiment's row
        experiments_table.append(dict())
        experiments_table[-1].update(dataset.get_data_hparams())
        experiments_table[-1].update(regr_model.get_model_hparams())
        experiments_table[-1].update(model_results)

    experiments_df = pd.DataFrame(experiments_table)
    if save_to_csv:
        experiments_df.to_csv("./results/linear_regression_hparams_tuning.csv")
    return experiments_df


def main():
    """ runs the regular pipeline of the model """
    # create the dataset with its hparams
    dataset = NVDataset_Classic(
        window_len_neuro_back=5,
        window_len_neuro_forward=2,
        window_len_vascu_back=1,
        window_len_y=1,
        scale_method=None,
        poly_degree=None,
        destroy_data=False  # control-group (shuffles the time-series)
    )

    # run and evaluate the model
    regr_model = NVLinearRegressionModel(dataset=dataset, test_size=TEST_SIZE)  # simple baseline

    # More model (comment out to run these instead)
    # regr_model = NVXGBLinearRegressionModel(dataset=dataset, test_size=TEST_SIZE)  # XGB-Regressor
    # regr_model = PersistModel(dataset=dataset, test_size=TEST_SIZE)  # control group (naive predictor)

    regr_model.fit()
    regr_model.evaluate()

    # Plot the actual vs predicted vascular activity
    summarizer = ResultsSummarizer(**regr_model.get_split_data())
    summarizer.plot_vascular_pred()


if __name__ == '__main__':
    main()
    # df = tune_dataset_parameters()


