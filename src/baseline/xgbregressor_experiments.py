import itertools

import pandas as pd

from src.baseline.more_models import NVXGBLinearRegressionModel
from src.datamodules.components.nv_datasets import NVDataset_Tabular

TEST_SIZE = 800


def tune_hyperparameters(save_to_csv=True) -> pd.DataFrame:
    """
    Runs the model with multiple dataset-parameters,
    returns Dataframe with the experiments results, for further analysis
    """
    experiments_table = []

    # Possible window lengths
    ds_names = ["2021_02_01_18_45_51_neurovascular_full_dataset",
                "2021_02_01_19_19_39_neurovascular_full_dataset"]

    window_sizes = [1, 10, 50]
    lrs = [0.001, 0.0001, 0.00001]
    estimator_nums = [500, 1000, 2000]
    max_depths = [5, 10]

    all_combinations = itertools.product(ds_names, window_sizes, window_sizes, lrs, estimator_nums, max_depths)
    # Iterate on all possible window lengths and evaluate
    for ds, neuro_back, neuro_forward, lr, n_estimators, max_depth in all_combinations:
        # loading the full dataset might fail, so we catch the exception
        try:
            dataset = NVDataset_Tabular(
                dataset_name=ds,
                window_len_neuro_back=neuro_back,
                window_len_neuro_forward=neuro_forward,
                destroy_data=False  # control-group (shuffles the time-series)
            )
        except:
            continue

        model = NVXGBLinearRegressionModel(dataset=dataset, test_size=TEST_SIZE, lr=lr, max_depth=max_depth,
                                           n_estimators=n_estimators)

        model.fit()
        model_results = model.evaluate()
        experiments_table.append(dict())
        experiments_table[-1].update(dataset.get_data_hparams())
        experiments_table[-1].update(model.get_model_hparams())
        experiments_table[-1].update(model_results)

        experiments_df = pd.DataFrame(experiments_table)

        if save_to_csv:
            experiments_df.to_csv("xgbregressor_hparams_tuning.csv")

    return experiments_df


if __name__ == "__main__":
    tune_hyperparameters()
