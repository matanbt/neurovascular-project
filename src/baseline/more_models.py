import itertools
from typing import List

import numpy as np
from src.datamodules.components.nv_datasets import NVDataset_Base, NVDataset_Classic, NVDataset_Tabular

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


"""
------------ Multiple Regressor ------------
    The model is built of multiple regressor, each is a special regressor for a different blood-vessel
"""
class NVMultiLinearRegressionModel:
    def __init__(self,
                 dataset: NVDataset_Classic,
                 test_size):
        """ fitting a linear regression model to EACH blood vessel (sort of ensemble model) """

        x, y = dataset.to_numpy()
        self.x, self.y = x, y

        # list of regressors (one for each blood vessel)
        self.models: List[RidgeCV] = [None] * self.y.shape[1]

        # split the dataset
        self.x_train, self.x_test = x[:-test_size], x[-test_size:]
        self.y_train, self.y_test = y[:-test_size], y[-test_size:]

        self.y_pred = np.empty_like(self.y)
        self.y_pred_test = np.empty_like(self.y_test)
        self.y_pred_train = np.empty_like(self.y_train)

        self.test_size = test_size

    def fit(self):
        """ train the model on the training set, then predict """
        alphas = 10.0 ** np.arange(-10, 10, 1)  # possible regularization alphas
        for i in range(self.y.shape[1]):
            self.models[i] = RidgeCV(alphas=alphas)
            self.models[i].fit(self.x_train, self.y_train[:, i])

            self.y_pred[:, i] = self.models[i].predict(self.x)
            self.y_pred_train[:, i] = self.models[i].predict(self.x_train)
            self.y_pred_test[:, i] = self.models[i].predict(self.x_test)

        return self

    def get_model_hparams(self):
        return {
            'alpha': [model.alpha_ for model in self.models],
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
        print(f">> HParams: {self.get_model_hparams()}")
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


"""
------------ XGBoost Regressor ------------
    Utilizing the random forest gradient-boost algorithm of XGBoost
    # TODO [WORK IN PROGRESS]
"""
from xgboost import XGBRegressor, cv, DMatrix
class NVXGBLinearRegressionModel:
    def __init__(self,
                 dataset: NVDataset_Tabular,
                 train_size,
                 test_size,
                 lr=0.01,
                 max_depth=6,
                 n_estimators=1000):
        """ classic linear regression model """
        self.y_pred = None
        self.y_pred_test = None
        self.y_pred_train = None
        self.model = None
        self.lr = lr
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        x = dataset.fetcher.get_neurons_df()
        y = dataset.fetcher.get_vessels_df()
        self.x, self.y = x, y

        # split the dataset
        self.x_train, self.x_test = x[:train_size], x[-test_size:]
        self.y_train, self.y_test = y[:train_size], y[-test_size:]

        self.test_size = test_size

    def fit(self):
        """ train the model on the training set, then predict """
        from sklearn.multioutput import MultiOutputRegressor
        self.model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror',
                                                       max_depth=self.max_depth,
                                                       learning_rate=self.lr,
                                                       tree_method="hist",
                                                       n_estimators=self.n_estimators,
                                                       eval_metric=mean_absolute_error,))

        self.model.fit(self.x_train, self.y_train)

        self.y_pred = self.model.predict(self.x)
        self.y_pred_train = self.model.predict(self.x_train)
        self.y_pred_test = self.model.predict(self.x_test)

        return self

    def get_model_hparams(self):
        return {
            'test_size': self.test_size,
            'lr': self.lr,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators
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


"""
Persistence Model to be used as a control experiment
(simply predict the previous vascular activity)
"""
# ------------ Stupid Models ------------
class PersistModel:
    """ """
    def __init__(self,
                 dataset: NVDataset_Classic,
                 test_size):
        """ classic linear regression model """
        self.y_pred = None
        self.y_pred_test = None
        self.y_pred_train = None

        x, y = dataset.to_numpy()
        self.x, self.y = x, y

        # split the dataset
        self.x_train, self.x_test = x[:-test_size], x[-test_size:]
        self.y_train, self.y_test = y[:-test_size], y[-test_size:]

        self.test_size = test_size

    def fit(self):
        """ well... it does fit somehow"""
        self.y_pred = self.x[:, -426:]  # we simply 'hijack' the previous blood-vessel activity
        self.y_pred_train = self.x_train[:, -426:]
        self.y_pred_test = self.x_test[:, -426:]

        return self

    def get_model_hparams(self):
        return {
            'stupid_fitting_method': 'taking the previously known vascular activity',
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


"""
Another Control Model (to experiment):
Return the mean activity of each blood-vessel, through all the time
"""
class MeanModel:
    """ """
    def __init__(self,
                 dataset: NVDataset_Classic,
                 test_size,
                 mean_on_training_only=True):
        """ classic linear regression model """
        self.y_pred = None
        self.y_pred_test = None
        self.y_pred_train = None

        self.vascu_activity_array = dataset.fetcher.vascu_activity_array

        x, y = dataset.to_numpy()
        self.x, self.y = x, y

        # split the dataset
        self.x_train, self.x_test = x[:-test_size], x[-test_size:]
        self.y_train, self.y_test = y[:-test_size], y[-test_size:]

        self.test_size = test_size

        # means of the vascular activity:
        training_vascu_mean = self.vascu_activity_array[:, :-test_size].mean(axis=1)
        overall_vascu_mean = self.vascu_activity_array.mean(axis=1)

        self.chosen_mean = training_vascu_mean if mean_on_training_only else overall_vascu_mean

    def fit(self):
        """ well... it does fit somehow"""
        self.y_pred = np.tile(self.chosen_mean, (self.x.shape[0], 1))   # we simply 'hijack' the previous blood-vessel activity
        self.y_pred_train = np.tile(self.chosen_mean, (self.x_train.shape[0], 1))
        self.y_pred_test = np.tile(self.chosen_mean, (self.x_test.shape[0], 1))

        return self

    def get_model_hparams(self):
        return {
            'stupid_fitting_method': 'taking the previously known vascular activity',
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
