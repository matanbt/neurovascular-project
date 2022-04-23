"""
Classes for feature engineering a raw neuro-vascular-dataset
    - All extends torch's Dataset class
    - All consume the raw dataset from the Fetcher class
    - Each class is different in the way it defines the X,y pairs
"""
import math

import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import PolynomialFeatures

from src.datamodules.components.nv_fetcher import NVDatasetFetcher
from src.utils import get_logger

log = get_logger(__name__)


class NVDataset_Classic(Dataset):
    """
        Classic feature-engineering for the NV data, defines:
        X - concat of: neurons back-window, neurons forward-window, blood back-window (by flag)
        y - blood activity of time-stamp (of shape `vessels count`)
    """

    def __init__(self,
                 # Dataset source:
                 data_dir="data/",
                 dataset_name="2021_02_01_neurovascular_datasets",

                 # Dataset hyper parameters:
                 window_len_neuro_back: int = 5,     # must be positive
                 window_len_neuro_forward: int = 5,  # `0` means no neuro-forward-window
                 window_len_vascu_back: int = 5,     # `0` means no vascular-window
                 aggregate_window="flatten",
                 poly_degree: int = None,     # `None` means no polynomial featuring
                 destroy_data: bool = False   # shuffles the data to make it "bad"
                 ):
        self.dataset_name = dataset_name
        self.fetcher = NVDatasetFetcher(data_dir=data_dir, dataset_name=dataset_name)

        # back and forward window size
        self.window_len_neuro_back = window_len_neuro_back
        self.window_len_neuro_forward = window_len_neuro_forward
        self.window_len_vascu_back = window_len_vascu_back
        self.max_window_len = max(window_len_neuro_back, window_len_neuro_forward, window_len_vascu_back)

        # Polynomial featuring
        self.poly_degree = poly_degree
        self.poly_transform = None
        if poly_degree is not None:
            self.poly_transform = PolynomialFeatures(degree=poly_degree)

        # method to aggregate the windows (we'll flatten and concat by default)
        self.aggregate_window = aggregate_window

        # destroy the data by shuffling it
        if destroy_data:
            self.fetcher.destroy_data()

        # Validate dataset parameters
        self.validate_params()

        # Calculate X,y sizes:
        self.neuro_window_size = self.window_len_neuro_back * self.fetcher.metadata["neurons_count"]
        self.neuro_window_size += self.window_len_neuro_forward * self.fetcher.metadata["neurons_count"]
        self.vascu_window_size = self.window_len_vascu_back * self.fetcher.metadata["blood_vessels_count"]

        # size-of-x-before any transformation (such as polynomial features)
        self.x_size_before = self.neuro_window_size + self.vascu_window_size
        self.x_size = self.x_size_before  # by default be keep the 'before' size
        if self.poly_transform:
            self.x_size = math.comb((self.x_size_before +1) + 2 - 1 , 2)

        self.y_size = self.fetcher.metadata["blood_vessels_count"]

        # the effective time vector, truncating the window edges
        self.time_vector = self.fetcher.time_vector_array[self.max_window_len: -self.max_window_len]

    def __len__(self):
        return self.fetcher.metadata['timeseries_len'] - self.max_window_len * 2

    def __getitem__(self, idx):
        """
        Defines the X and y of each timestamp (=idx)
        """
        assert 0 <= idx < len(self), f"Expected index in [0, {len(self) - 1}] but got {idx}"

        idx += self.max_window_len  # adds window offset to get the actual time-stamp

        x = np.zeros(self.x_size_before)
        y = np.zeros(self.y_size)

        neuro_wind_start = idx - self.window_len_neuro_back
        neuro_wind_end = idx + self.window_len_neuro_forward
        x[:self.neuro_window_size] = self.fetcher.neuro_activity_array[:, neuro_wind_start: neuro_wind_end].flatten()

        if self.vascu_window_size > 0:
            vascu_wind_start = idx - self.window_len_vascu_back
            vascu_wind_end = idx
            x[self.neuro_window_size:] = self.fetcher.vascu_activity_array[:, vascu_wind_start: vascu_wind_end].flatten()

        if self.poly_transform is not None:
            x = np.expand_dims(x, axis=0)  # reshaping is essential for fit_transform
            x = self.poly_transform.fit_transform(x)
            x = np.squeeze(x, axis=0)

        y = self.fetcher.vascu_activity_array[:, idx]

        return x, y

    def validate_params(self):
        """
            Some basic invariants we want to hold, after the dataset parameters initialization
        """
        assert self.window_len_neuro_back > 0, \
            "Expected positive back-window length for neuronal activity"

        aggregate_window_methods = ['flatten']  # TODO add 'mean', 'sum', etc
        assert self.aggregate_window in aggregate_window_methods, \
            f"Expected aggregate_window arg ({self.aggregate_window}) to " \
            f"be in {aggregate_window_methods}"

    def get_data_hparams(self):
        """ returns the hyper parameters that defines this dataset instance """
        return {
            'dataset_name': self.dataset_name,
            'window_len_neuro_back': self.window_len_neuro_back,
            'window_len_neuro_forward': self.window_len_neuro_forward,
            'window_len_vascu_back': self.window_len_vascu_back,
            'poly_degree': self.poly_degree,
            'aggregate_window': self.aggregate_window
        }

    def to_numpy(self):
        """
        Returns numpy matrix of our crafted dataset
        (useful for sklearn models training)
        """
        x_all = np.zeros((len(self), len(self[0][0])))
        y_all = np.zeros((len(self), len(self[0][1])))
        for i in range(len(self)):
            x, y = self[i]
            x_all[i], y_all[i] = np.array(x), np.array(y)
        return x_all, y_all
