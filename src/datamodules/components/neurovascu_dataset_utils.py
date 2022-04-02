import os.path
import zarr
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# -------- PreProcessing the raw dataset --------
class NVDatasetFetcher:
    """
        class for fetching and preliminary pre-processing features from a neuro-vascular dataset
    """
    def __init__(self,
                 dataset_path="data/2021_02_01_neurovascular_datasets"):
        # 1. Define paths to dataset's component
        neuro_path = os.path.join(dataset_path, "neuronal.zarr")
        vascu_path = os.path.join(dataset_path, "vascular.zarr")
        neuro_coordinates_path = os.path.join(dataset_path, "coordinates", "neuronal_coordinates.csv")
        vascu_coordinates_path = os.path.join(dataset_path, "coordinates", "vascular_coordinates.csv")

        # 2. Load data components
        neuro_timeseries = zarr.load(neuro_path)
        vascu_timeseries = zarr.load(vascu_path)
        neuro_coord_df = pd.read_csv(neuro_coordinates_path)
        vascu_coord_df = pd.read_csv(vascu_coordinates_path)

        # 3. Fetch what we want (as numpy arrays)
        self.time_vector_array = neuro_timeseries.get("time_vector")
        self.neuro_activity_array = neuro_timeseries.get("neuronal_dff")  # matrix of [neurons X timeunits]
        self.vascu_activity_array = vascu_timeseries.get("time_varying_vascular_diameter").T  # matrix of [blood X timeunits]
        self.neuro_coord_array = self._get_coords(neuro_coord_df)
        self.vascu_coord_array = self._get_coords(vascu_coord_df)
        # TODO: consider adding more features from the dataset?

        # 4. Validate dimensions
        self.validate_dims()

        # 5. Deduce metadata
        self.metadata = {
            "timeseries_len": self.time_vector_array.shape[0],
            "neurons_count": self.neuro_activity_array.shape[0],
            "blood_vessels_count": self.vascu_activity_array.shape[0],
            # TODO: enrich if needed
        }

        # 6. Further preprocessing - smoothing
        # TODO

    @staticmethod
    def _get_coords(coord_df):
        """
            returns a (np) matrix with scaled coordinates
        """
        dims_count = 3  # we're in 3d space
        result = np.zeros((len(coord_df), dims_count))

        result[:, 0] = ((coord_df['X1 [μm]'] + coord_df['X2 [μm]']) / 2).to_numpy()
        result[:, 1] = ((coord_df['Y1 [μm]'] + coord_df['Y2 [μm]']) / 2).to_numpy()
        result[:, 2] = coord_df['Z [μm]']

        return result

    def validate_dims(self):
        """
            Sanity checks for the dimensions of the loaded data
        """
        assert (self.time_vector_array.shape[0] ==
                self.neuro_activity_array.shape[1] ==
                self.vascu_activity_array.shape[1]), "All activities must share the amount of time units"
        assert (self.neuro_activity_array.shape[0] ==
                self.neuro_coord_array.shape[0]), "Amount of neurons must be in sync"
        assert (self.vascu_activity_array.shape[0] ==
                self.vascu_activity_array.shape[0]), "Amount of blood-vessels must be in sync"


# -------- Feature Engineering Variations on the NV-dataset --------
class NVDataset_Classic(Dataset):
    """
        Classic feature-engineering for the NV data, defines:
        X - concat of: neurons back-window, neurons forward-window, blood back-window (by flag)
        y - blood activity of time-stamp (of shape `vessels count`)
    """
    def __init__(self,
                 dataset_path="data/2021_02_01_neurovascular_datasets",
                 window_len=5,
                 include_feature_blood=True,
                 aggregate_window="flatten"):
        self.fetcher = NVDatasetFetcher(dataset_path=dataset_path)

        # back and forward window size
        self.window_len = window_len

        # Flags for feature engineering:
        # whether to include blood-back-window in X
        self.include_feature_blood = include_feature_blood
        # method to aggregate the windows (we'll flatten and concat by default)
        aggregate_window_methods = ['flatten']  # TODO add 'mean', 'sum', etc
        assert aggregate_window in aggregate_window_methods, \
            f"Expected aggregate_window arg to be in {aggregate_window_methods}"
        self.aggregate_window = aggregate_window

        # get metadata

    def __len__(self):
        return self.fetcher.metadata['timeseries_len'] - self.window_len * 2

    def __getitem__(self, idx):
        """
        Defines the X and y of each timestamp
        """
        assert 0 <= idx < len(self), f"Expected index in [0, {len(self) - 1}] but got {idx}"
        idx += self.window_len  # adds window offset for real time-stamp

        neuro_window_size = self.window_len * self.fetcher.metadata["neurons_count"] * 2
        blood_window_size = self.window_len * self.fetcher.metadata["blood_vessels_count"] \
                            * self.include_feature_blood
        x = np.zeros(neuro_window_size + blood_window_size)
        y = np.zeros(self.fetcher.metadata["blood_vessels_count"])

        x[:neuro_window_size] = self.fetcher.neuro_activity_array[:, (idx - self.window_len) : (idx + self.window_len)].flatten()
        x[neuro_window_size:] = self.fetcher.vascu_activity_array[:, (idx - self.window_len) : idx].flatten()

        y = self.fetcher.vascu_activity_array[:, idx]

        return x, y


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