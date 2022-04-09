"""
Linear Regression Baseline
"""
import numpy as np

from visualize_prediction import vis_pred
from src.datamodules.components.neurovascu_dataset_utils import NVDataset_Classic
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

TEST_SIZE = 700

def run_model(x, y):
    # Split dataset
    x_train, x_test = x[:-TEST_SIZE], x[-TEST_SIZE:]
    y_train, y_test = y[:-TEST_SIZE], y[-TEST_SIZE:]

    # train & predict
    alphas = 10.0 ** np.arange(-10, 10, 1)  # possible regularization alphas
    model = RidgeCV(alphas=alphas)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # evaluate
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # print
    print(f">> Chosen Alpha: {model.alpha_}")
    print(f">> Training: MSE={mse_train}, R^2={r2_train}")
    print(f">> Testing: MSE={mse_test}, R^2={r2_test}")

    return model


if __name__ == '__main__':
    dataset = NVDataset_Classic(
        window_len=3,
        include_feature_blood=True,
    )
    x, y = dataset.to_numpy()

    # fit the model
    model = run_model(x, y)

    # Plot the actual vs predicted vascular activity
    vis_pred(y, model.predict(x),
             time_vector=dataset.time_vector,
             test_size=TEST_SIZE)


