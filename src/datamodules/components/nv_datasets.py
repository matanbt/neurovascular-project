"""
Classes for feature engineering a raw neuro-vascular-dataset
    - All extends torch's Dataset class
    - All consume the raw dataset from the Fetcher class
    - Each class is different in the way it defines the X,y pairs
"""
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import PolynomialFeatures

from src.datamodules.components.nv_fetcher import NVDatasetFetcher
from src.utils import get_logger

log = get_logger(__name__)


class NVDataset_Base(Dataset):
    """
    Base class for NVDatasets
    """
    def __init__(self,
                 data_dir: str = "data/",
                 dataset_name: str = "2021_02_01_18_45_51_neurovascular_partial_dataset",
                 scaling_method=None, destroy_data=False, **kwargs):
        self.dataset_name = dataset_name
        self.fetcher = NVDatasetFetcher(data_dir=data_dir, dataset_name=dataset_name)

        # destroy the data by shuffling it
        if destroy_data:
            self.fetcher.destroy_data()

        # Important arrays:
        self.neuro_activity_array = self.fetcher.neuro_activity_array
        self.vascu_activity_array = self.fetcher.vascu_activity_array

        # Scale the data
        self.scaling_method = scaling_method
        self._scale_data()

        # Attributes
        self.distances = None
        self.mean_vascular_activity = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def validate_params(self):
        """
            Some basic invariants we want to hold, after the dataset parameters initialization
        """
        pass

    def get_extras(self):
        """ Defines data to pass to the model BEFORE training """
        return {}

    def get_data_hparams(self):
        """ returns the hyper parameters that defines this dataset instance """
        return {}

    def _get_distances(self, normalize=True) -> torch.Tensor:
        if self.distances is not None:
            return self.distances

        # define the distance metric that will use:
        def calc_distance(p1, p2) -> int:
            return np.linalg.norm(p1 - p2)

        # each entry (i,j) is the distance between blood-vessel i to neuron j
        vascu_coord_array = self.fetcher.vascu_coord_array
        neuro_coord_array = self.fetcher.neuro_coord_array
        self.distances = torch.zeros((vascu_coord_array.shape[0], neuro_coord_array.shape[0]))

        for i in range(vascu_coord_array.shape[0]):
            for j in range(neuro_coord_array.shape[0]):
                self.distances[i, j] = calc_distance(vascu_coord_array[i], neuro_coord_array[j])

        # Normalize Distances:
        if normalize:
            self.distances = (self.distances - self.distances.mean()) / self.distances.std()

        return self.distances

    def _get_mean_vascular_activity(self, end_calc_idx=2200):
        """ returns the mean vascular activity of the first part of the data """
        if self.mean_vascular_activity is not None:
            return self.mean_vascular_activity
        self.mean_vascular_activity = self.vascu_activity_array[:, :end_calc_idx].mean(axis=1)
        return self.mean_vascular_activity

    def _scale_data(self):
        """ """
        if self.scaling_method is None:
            return
        if self.scaling_method == 'neuro_mean_removal':
            self.neuro_activity_array -= np.expand_dims(self.neuro_activity_array.mean(axis=1), axis=1)
        if self.scaling_method == 'neuro_normalize':  # this option is a HIGHLY RECOMMENDED in new datasets
            self.neuro_activity_array -= np.expand_dims(self.neuro_activity_array.mean(axis=1), axis=1)
            self.neuro_activity_array /= np.expand_dims(self.neuro_activity_array.std(axis=1), axis=1)


class NVDataset_Classic(NVDataset_Base):
    """
        Classic feature-engineering for the NV data, defines:
        X - concat of: neurons back-window, neurons forward-window, blood back-window (by flag)
        y - blood activity of time-stamp (of shape `vessels count`)
    """

    def __init__(self,
                 # Dataset source:
                 data_dir: str = "data/",
                 dataset_name: str = "2021_02_01_18_45_51_neurovascular_partial_dataset",

                 # Dataset hyper parameters:
                 window_len_neuro_back: int = 5,
                 window_len_neuro_forward: int = 5,
                 window_len_vascu_back: int = 5,
                 window_len_y: int = 1,
                 scaling_method: str = None,
                 aggregate_window="flatten",
                 poly_degree: int = None,
                 destroy_data: bool = False
                 ):
        """
        Args:
            data_dir: path for the dir we keep the data.
            dataset_name: name of the raw dataset we build this instance upon.
            window_len_neuro_back: length of the neuronal window of the 'past' ( must be positive).
            window_len_neuro_forward: length of the neuronal window of the 'future' (`0` means no neuro-forward-window).
            window_len_vascu_back: length of the vascular window of the 'past' (`0` means no vascular-window).
            window_len_y: length of the vascular window to *predict* (regularly we assign `1`)
            scaling_method: default to `None` i.e. no scaling.
            aggregate_window: method to aggregate each window (by default we simply flatten the window)  # TODO (# TODO add 'mean', 'sum', etc)
            poly_degree: adding polynomial features, defaults to `None` i.e. not adding these (WARNING: This yields high RAM consumption)
            destroy_data: shuffles the data to make it "bad" (for control experiments)
        """
        super().__init__(data_dir=data_dir, dataset_name=dataset_name,
                         destroy_data=destroy_data, scaling_method=scaling_method)

        # back and forward window size
        self.window_len_neuro_back = window_len_neuro_back
        self.window_len_neuro_forward = window_len_neuro_forward
        self.window_len_vascu_back = window_len_vascu_back
        self.window_len_y = window_len_y
        self.max_window_len = max(window_len_neuro_back, window_len_neuro_forward, window_len_vascu_back, window_len_y)

        # Polynomial featuring
        self.poly_degree = poly_degree
        self.poly_transform = None
        if poly_degree is not None:
            self.poly_transform = PolynomialFeatures(degree=poly_degree)

        # method to aggregate the windows (we'll flatten and concat by default)
        self.aggregate_window = aggregate_window

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
            self.x_size = math.comb((self.x_size_before + 1) + 2 - 1, 2)

        # calculate size of y
        self.y_size = self.fetcher.metadata["blood_vessels_count"] * self.window_len_y

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

        # Build X:
        neuro_wind_start = idx - self.window_len_neuro_back
        neuro_wind_end = idx + self.window_len_neuro_forward
        x[:self.neuro_window_size] = self.neuro_activity_array[:, neuro_wind_start: neuro_wind_end].flatten()

        if self.vascu_window_size > 0:
            vascu_wind_start = idx - self.window_len_vascu_back
            vascu_wind_end = idx
            x[self.neuro_window_size:] = self.vascu_activity_array[:, vascu_wind_start: vascu_wind_end].flatten()

        if self.poly_transform is not None:
            x = np.expand_dims(x, axis=0)  # reshaping is essential for fit_transform
            x = self.poly_transform.fit_transform(x)
            x = np.squeeze(x, axis=0)

        # Build Y:
        y = self.vascu_activity_array[:, idx: idx + self.window_len_y].flatten()

        return x, y

    def validate_params(self):
        """
            Some basic invariants we want to hold, after the dataset parameters initialization
        """
        assert self.window_len_neuro_back > 0, \
            "Expected positive back-window length for neuronal activity"

        aggregate_window_methods = ['flatten']  # + 'mean', 'sum'
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


class NVDataset_EHRF(NVDataset_Base):
    """
        Feature-engineering for the EHRF with distances as extra data, defines:
        X - neuronal activity window, with shape preserved (each row is a different neuron)
        y - blood activity of a time-stamp (of shape `vessels count`)
    """

    def __init__(self,
                 # Dataset source:
                 data_dir: str = "data/",
                 dataset_name: str = "2021_02_01_18_45_51_neurovascular_partial_dataset",

                 # Dataset hyper parameters:
                 window_len_neuro_back: int = 5,
                 window_len_neuro_forward: int = 2,
                 scaling_method: str = None,  # maybe scaling with Sigmoid will help?
                 destroy_data: bool = False
                 ):
        """
        Args:
            data_dir: path for the dir we keep the data.
            dataset_name: name of the raw dataset we build this instance upon.
            window_len_neuro_back: length of the neuronal window of the 'past' ( must be positive).
            window_len_neuro_forward: length of the neuronal window of the 'future' (`0` means no neuro-forward-window).
            scaling_method: How to scale the data (usually just the neurons), None means no scaling
            destroy_data: shuffles the data to make it "bad" (for control experiments)
        """
        super().__init__(data_dir=data_dir, dataset_name=dataset_name,
                         destroy_data=destroy_data, scaling_method=scaling_method)

        # back and forward window size
        self.window_len_neuro_back = window_len_neuro_back
        self.window_len_neuro_forward = window_len_neuro_forward
        self.max_window_len = max(window_len_neuro_back, window_len_neuro_forward)

        # Validate dataset parameters
        self.validate_params()

        # Calculate X,y sizes:
        self.single_neuro_window_size = self.window_len_neuro_back + self.window_len_neuro_forward
        self.neuro_window_size = self.single_neuro_window_size * self.fetcher.metadata["neurons_count"]

        # x_size is actually the neuron-window length
        self.x_size = (self.fetcher.metadata["neurons_count"], self.single_neuro_window_size)

        # calculate size of y
        self.y_size = self.fetcher.metadata["blood_vessels_count"]

        # the effective time vector, truncating the window edges
        self.time_vector = self.fetcher.time_vector_array[self.max_window_len: -self.max_window_len]
        self.idx_vector = np.arange(len(self.fetcher.time_vector_array))[self.max_window_len: -self.max_window_len]
        self.true_vector = self.vascu_activity_array.T[self.max_window_len: -self.max_window_len]

        self.distances = None
        self._get_distances()

    def __len__(self):
        return self.fetcher.metadata['timeseries_len'] - self.max_window_len * 2

    def __getitem__(self, idx):
        """
        Defines the X and y of each timestamp (=idx)
        """
        assert 0 <= idx < len(self), f"Expected index in [0, {len(self) - 1}] but got {idx}"

        idx += self.max_window_len  # adds window offset to get the actual time-stamp

        x = np.zeros(self.x_size)
        y = np.zeros(self.y_size)

        # Build X (neuronal activity entry):
        neuro_wind_start = idx - self.window_len_neuro_back
        neuro_wind_end = idx + self.window_len_neuro_forward
        x = self.neuro_activity_array[:, neuro_wind_start: neuro_wind_end]

        # Build Y:
        y = self.vascu_activity_array[:, idx]

        return x, y

    def validate_params(self):
        """
            Some basic invariants we want to hold, after the dataset parameters initialization
        """
        assert self.window_len_neuro_back > 0, \
            "Expected positive back-window length for neuronal activity"
        assert self.scaling_method in [None, "neuro_mean_removal", "neuro_normalize"]

    def get_data_hparams(self):
        """ returns the hyper parameters that defines this dataset instance """
        return {
            'dataset_name': self.dataset_name,
            'window_len_neuro_back': self.window_len_neuro_back,
            'window_len_neuro_forward': self.window_len_neuro_forward,
        }

    def get_extras(self):
        """ Defines data to pass to the model BEFORE training """
        return {
            'distances': self._get_distances(),
            'mean_vascular_activity': self._get_mean_vascular_activity()
        }

class NVDataset_RNN(NVDataset_Base):
    """
        Feature-engineering for the EHRF with distances as extra data, defines:
        X - neuronal activity window, with shape preserved (each row is a different neuron)
        y - blood activity of a time-stamp (of shape `vessels count`)
    """

    def __init__(self,
                 # Dataset source:
                 data_dir: str = "data/",
                 dataset_name: str = "2021_02_01_18_45_51_neurovascular_partial_dataset",

                 # Dataset hyper parameters:
                 window_len_vascu_back: int = 50,
                 window_len_neuro_back: int = 5,
                 window_len_neuro_forward: int = 2,
                 scaling_method: str = None,  # maybe scaling with Sigmoid will help?
                 destroy_data: bool = False
                 ):
        """
        Args:
            data_dir: path for the dir we keep the data.
            dataset_name: name of the raw dataset we build this instance upon.
            window_len_neuro_back: length of the neuronal window of the 'past' ( must be positive).
            window_len_neuro_forward: length of the neuronal window of the 'future' (`0` means no neuro-forward-window).
            scaling_method: How to scale the data (usually just the neurons), None means no scaling
            destroy_data: shuffles the data to make it "bad" (for control experiments)
        """
        super().__init__(data_dir=data_dir, dataset_name=dataset_name,
                         destroy_data=destroy_data, scaling_method=scaling_method)

        # back and forward window size
        self.window_len_vascu_back = window_len_vascu_back
        self.window_len_neuro_back = window_len_neuro_back
        self.window_len_neuro_forward = window_len_neuro_forward
        self.max_window_len = max(window_len_neuro_back, window_len_neuro_forward, window_len_vascu_back)

        # Validate dataset parameters
        self.validate_params()

        # Calculate X,y sizes:
        self.single_neuro_window_size = self.window_len_neuro_back + self.window_len_neuro_forward
        self.neuro_window_size = self.single_neuro_window_size * self.fetcher.metadata["neurons_count"]

        # x_size is actually the neuron-window length
        self.x_size = {
            'x_neuro_size': (self.fetcher.metadata["neurons_count"], self.single_neuro_window_size),
            'x_vascu_size': (self.fetcher.metadata["blood_vessels_count"], self.window_len_vascu_back)
        }

        # calculate size of y
        self.y_size = self.fetcher.metadata["blood_vessels_count"]

        # the effective time vector, truncating the window edges
        self.time_vector = self.fetcher.time_vector_array[self.max_window_len: -self.max_window_len]
        self.idx_vector = np.arange(len(self.fetcher.time_vector_array))[self.max_window_len: -self.max_window_len]
        self.true_vector = self.vascu_activity_array.T[self.max_window_len: -self.max_window_len]

        self.distances = None
        self._get_distances()

    def __len__(self):
        return self.fetcher.metadata['timeseries_len'] - self.max_window_len * 2

    def __getitem__(self, idx):
        """
        Defines the X and y of each timestamp (=idx)
        """
        assert 0 <= idx < len(self), f"Expected index in [0, {len(self) - 1}] but got {idx}"

        idx += self.max_window_len  # adds window offset to get the actual time-stamp

        # Build X (neuronal activity entry):
        vascu_wind_start = idx - self.window_len_vascu_back
        neuro_wind_start = idx - self.window_len_neuro_back
        neuro_wind_end = idx + self.window_len_neuro_forward
        x_neuro = self.neuro_activity_array[:, neuro_wind_start: neuro_wind_end]
        x_vascu = self.vascu_activity_array[:, vascu_wind_start: idx]

        # Build Y:
        y = self.vascu_activity_array[:, idx]

        return x_neuro, x_vascu, y

    def validate_params(self):
        """
            Some basic invariants we want to hold, after the dataset parameters initialization
        """
        assert self.window_len_neuro_back > 0, \
            "Expected positive back-window length for neuronal activity"
        assert self.scaling_method in [None, "neuro_mean_removal", "neuro_normalize"]

    def get_data_hparams(self):
        """ returns the hyper parameters that defines this dataset instance """
        return {
            'dataset_name': self.dataset_name,
            'window_len_vascu_back': self.window_len_vascu_back,
            'window_len_neuro_back': self.window_len_neuro_back,
            'window_len_neuro_forward': self.window_len_neuro_forward,
        }

    def get_extras(self):
        """ Defines data to pass to the model BEFORE training """
        return {}


class NVDataset_Tabular(NVDataset_Base):
    """
        Feature-engineering for Tabular-vectorization, with specific blood-vessels
        X - neuronal activity window, blood-vessels idx and average activity
        y - blood activity of a *specific* blood-vessel in a time-stamp
    """

    def __init__(self,
                 # Dataset source:
                 data_dir: str = "data/",
                 dataset_name: str = "2021_02_01_18_45_51_neurovascular_partial_dataset",

                 # Dataset hyper parameters (Tabular Features):
                 include_vascular_idx: bool = True,
                 include_vascular_avg: bool = True,
                 include_vascular_distances: bool = True,
                 window_len_neuro_back: int = 5,
                 window_len_neuro_forward: int = 2,
                 neuro_aggr_method: str = 'flatten',
                 scaling_method: str = None,

                 destroy_data: bool = False
                 ):
        """
        Args:
            data_dir: path for the dir we keep the data.
            dataset_name: name of the raw dataset we build this instance upon.

            include_vascular_idx: Whether to include the (integer) index of the blood vessel in the vector
            include_vascular_avg: Whether to include the (training) activity-mean of the blood vessel in the vector
            include_vascular_distances: Whether to include the distances of the blood-vessels from the neurons.
            window_len_neuro_back: length of the neuronal window of the 'past' ( must be positive).
            window_len_neuro_forward: length of the neuronal window of the 'future' (`0` means no neuro-forward-window).
            neuro_aggr_method: How to aggregate the neuronal activity (on the time dimension)

            destroy_data: shuffles the data to make it "bad" (for control experiments)
        """
        super().__init__(data_dir=data_dir, dataset_name=dataset_name,
                         destroy_data=destroy_data, scaling_method=scaling_method)

        self.neuro_aggr_method = neuro_aggr_method
        self.include_vascular_idx = include_vascular_idx
        self.include_vascular_avg = include_vascular_avg
        self.include_vascular_distances = include_vascular_distances
        self.scaling_method = scaling_method

        # back and forward window size
        self.window_len_neuro_back = window_len_neuro_back
        self.window_len_neuro_forward = window_len_neuro_forward
        self.max_window_len = max(window_len_neuro_back, window_len_neuro_forward)

        # Validate dataset parameters
        self.validate_params()

        # Calculate X,y sizes:
        self.single_neuro_window_size = self.window_len_neuro_back + self.window_len_neuro_forward
        self.neurons_window_size = self.single_neuro_window_size * self.fetcher.metadata["neurons_count"]

        # x_size is actually the neuron-window length
        self.x_size = int(self.include_vascular_idx) + int(self.include_vascular_avg)
        self.x_size += int(include_vascular_distances) * self.fetcher.metadata["neurons_count"]
        if self.neuro_aggr_method == 'flatten':
            self.x_size += self.neurons_window_size
        elif self.neuro_aggr_method in ('mean', 'sum'):
            self.x_size += self.fetcher.metadata["neurons_count"]

        # calculate size of y
        self.y_size = 1  # the specific activity of blood vessel in time `t`

        # the effective time vector, truncating the window edges
        self.time_vector = self.fetcher.time_vector_array[self.max_window_len: -self.max_window_len]
        self.idx_vector = np.arange(len(self.fetcher.time_vector_array))[self.max_window_len: -self.max_window_len]
        self.true_vector = self.vascu_activity_array.T[self.max_window_len: -self.max_window_len]

        self._get_distances()

    def __len__(self):
        return len(self.time_vector) * self.fetcher.metadata['blood_vessels_count']

    def __getitem__(self, idx):
        """
        Defines the X and y of each idx
        """
        assert 0 <= idx < len(self), f"Expected index in [0, {len(self) - 1}] but got {idx}"

        timestamp_idx, blood_vessel_idx = self.idx_to_vessel_timestamp(idx)
        timestamp_idx += self.max_window_len  # adds window offset to get the actual time-stamp

        x = np.zeros(self.x_size)
        y = np.zeros(self.y_size)

        # Build X (neuronal activity entry):
        i = 0  # index for inserting tabular data to X
        # 1. Metadata:
        if self.include_vascular_idx:
            x[i] = blood_vessel_idx
            i += 1
        if self.include_vascular_avg:
            x[i] = self._get_mean_vascular_activity()[blood_vessel_idx]
            i += 1
        if self.include_vascular_distances:
            distances_from_vessel = self._get_distances()[blood_vessel_idx]
            x[i: i + len(distances_from_vessel)] = distances_from_vessel
            i += len(distances_from_vessel)
        # 2. (Neuronal) Activity Data
        neuro_wind_start = timestamp_idx - self.window_len_neuro_back
        neuro_wind_end = timestamp_idx + self.window_len_neuro_forward
        neuro_wind = self.neuro_activity_array[:, neuro_wind_start: neuro_wind_end]
        # Aggregate the activity in the desired way:
        if self.neuro_aggr_method == 'flatten':
            neuro_wind = neuro_wind.flatten()
        elif self.neuro_aggr_method == 'sum':
            neuro_wind = neuro_wind.sum(axis=1)
        elif self.neuro_aggr_method == 'mean':
            neuro_wind = neuro_wind.mean(axis=1)
        x[i: i + len(neuro_wind)] = neuro_wind
        i += len(neuro_wind)

        # Build Y:
        y[0] = self.vascu_activity_array[blood_vessel_idx, timestamp_idx]

        return x, y

    def idx_to_vessel_timestamp(self, idx):
        """
        Maps an index to a tuple of specific blood-vessel index and timestamp
        """
        timestamp_idx = idx // self.fetcher.metadata['blood_vessels_count']
        blood_vessel_idx = idx % self.fetcher.metadata['blood_vessels_count']

        return timestamp_idx, blood_vessel_idx

    def validate_params(self):
        """
            Some basic invariants we want to hold, after the dataset parameters initialization
        """
        assert self.window_len_neuro_back > 0, \
            "Expected positive back-window length for neuronal activity"
        assert self.neuro_aggr_method in ['flatten', 'mean', 'sum']

    def get_data_hparams(self):
        """ returns the hyper parameters that defines this dataset instance """
        return {
            'dataset_name': self.dataset_name,
            'window_len_neuro_back': self.window_len_neuro_back,
            'window_len_neuro_forward': self.window_len_neuro_forward,
        }

    def get_extras(self):
        """ Defines data to pass to the model BEFORE training """
        return {
            'distances': self._get_distances(),
            'mean_vascular_activity': self._get_mean_vascular_activity()
        }
