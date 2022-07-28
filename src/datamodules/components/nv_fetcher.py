"""
PreProcessing a raw neuro-vascular dataset
"""
import os.path
import zarr
import pandas as pd
import numpy as np
from numpy import ma

from src.utils import get_logger

log = get_logger(__name__)


class NVDatasetFetcher:
    """
        class for fetching (single) raw neuro-vascular-dataset,
        and *preliminary* pre-processing the data, filling missing data, etc.
    """
    def __init__(self,
                 data_dir="data/",
                 dataset_name="2021_02_01_18_45_51_neurovascular_partial_dataset"):
        # 1. Define paths to dataset's component
        dataset_path = os.path.join(data_dir, dataset_name)
        neuro_path = os.path.join(dataset_path, "neuronal.zarr")
        vascu_path = os.path.join(dataset_path, "vascular.zarr")
        neuro_coordinates_path = os.path.join(dataset_path, "coordinates", "neuronal_coordinates.csv")
        vascu_coordinates_path = os.path.join(dataset_path, "coordinates", "vascular_coordinates.csv")

        # 2. Load data components
        neuro_timeseries = zarr.load(neuro_path)
        vascu_timeseries = zarr.load(vascu_path)
        neuro_coord_df = pd.read_csv(neuro_coordinates_path)
        vascu_coord_df = pd.read_csv(vascu_coordinates_path)

        # 3. Get names by dataset
        map_names_by_dataset = {
            'neuro_activity_name': {
                '2021_02_01_18_45_51_neurovascular_full_dataset': 'neuronal_dynamics_501',
                '2021_02_01_19_19_39_neurovascular_full_dataset': 'neuronal_dynamics_466',
                '2021_02_01_18_45_51_neurovascular_partial_dataset': 'neuronal_dff',
            }
        }

        # 4. Fetch what we want (as numpy arrays)
        self.neuro_activity_array = neuro_timeseries.get(map_names_by_dataset['neuro_activity_name'][dataset_name])  # matrix of [neurons X timestamps]
        self.vascu_activity_array = vascu_timeseries.get("time_varying_vascular_diameter").T  # matrix of [blood X timestamps]
        self.time_vector_array = np.arange(self.vascu_activity_array.shape[-1])  # vector of [timestamps]
        self.neuro_coord_array = self._get_coords(neuro_coord_df)
        self.vascu_coord_array = self._get_coords(vascu_coord_df)

        # 5. Ad-hoc datasets fixes
        if dataset_name in ("2021_02_01_18_45_51_neurovascular_full_dataset", "2021_02_01_19_19_39_neurovascular_full_dataset"):
            self.neuro_activity_array = self.neuro_activity_array.T
        if dataset_name in ("2021_02_01_18_45_51_neurovascular_full_dataset", "2021_02_01_18_45_51_neurovascular_partial_dataset", "2021_02_01_19_19_39_neurovascular_full_dataset"):
            # This last row appears twice, we shall remove the second appearance
            self.vascu_activity_array = self.vascu_activity_array[:-1, :]

        # 6. Handle missing values
        self.vascu_activity_array = self.fill_missing_values(self.vascu_activity_array)

        # 7. Validate stuff
        self.validate_dims()
        self.validate_no_nans()

        # 8. Deduce metadata
        self.metadata = {
            "timeseries_len": self.time_vector_array.shape[0],
            "neurons_count": self.neuro_activity_array.shape[0],
            "blood_vessels_count": self.vascu_activity_array.shape[0],
            # TODO: enrich if needed
        }

        # 9. Further preprocessing - smoothing
        # TODO

    @staticmethod
    def _get_coords(coord_df: pd.DataFrame) -> np.ndarray:
        """
            returns a (np) matrix with scaled coordinates
        """
        dims_count = 3  # we're in 3d space
        result = np.zeros((len(coord_df), dims_count))

        result[:, 0] = ((coord_df['X1 [μm]'] + coord_df['X2 [μm]']) / 2).to_numpy()
        result[:, 1] = ((coord_df['Y1 [μm]'] + coord_df['Y2 [μm]']) / 2).to_numpy()
        result[:, 2] = coord_df['Z [μm]']

        return result

    @staticmethod
    def fill_missing_values(matrix_to_fix):
        """
            takes care of any `nan` in the given matrix, caused by missing data
            [ We currently take care of it by averaging the entire time-series of
              each vessels / neuron, but we can actually do it with a smaller window (TODO?) ]
        """
        nan_count = np.argwhere(np.isnan(matrix_to_fix)).shape[0]
        log.info(f"Filling {nan_count} missing values with all-time average of the element (neuron / vessels).")

        # Summing each row as the fill-nan value
        # Ref: https://stackoverflow.com/a/40209161/3476618
        matrix_fixed = np.where(np.isnan(matrix_to_fix),
                                ma.array(matrix_to_fix, mask=np.isnan(matrix_to_fix)).mean(axis=1)[:, np.newaxis],
                                matrix_to_fix)
        return matrix_fixed

    def destroy_data(self):
        """
            shuffles the raw data, making it stupid
            * DON'T USE IT IF YOU WANT TO LEARN SOMETHING *
            This dataset can be used as our 'control group', and be compared with the real dataset
        """
        # TODO - shuffle more gently with Lior's idea: ' Shift the temporal trace of each vessel segment and neuron by some random large number of frames'

        # shuffles each array of raw data naively
        for neuro in self.neuro_activity_array:
            np.random.shuffle(neuro)
        for vessel in self.vascu_activity_array:
            np.random.shuffle(vessel)

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

    def validate_no_nans(self):
        """
            Validates the data excludes `nan`s (that were not taken care of)
        """
        # * Useful for debugging: np.argwhere(np.isnan(my_naned_array))
        assert (not np.isnan(self.vascu_activity_array).any() and
                not np.isnan(self.neuro_activity_array).any() and
                not np.isnan(self.vascu_coord_array).any() and
                not np.isnan(self.neuro_coord_array).any() and
                not np.isnan(self.time_vector_array).any()
                ), "Expected data to be clear of nan (= missing values)"

    def get_neurons_df(self):
        """ Get a dataframe based on neurons time-series"""
        # Build dict to feed dataframe
        data = {}
        for i in range(len(self.neuro_activity_array)):
            data[f"neuron_{i}"] = self.neuro_activity_array[i]

        # Build dataframe
        df = pd.DataFrame(data)
        df.index = self.time_vector_array

        return df

    def get_vessels_df(self):
        """ Get a dataframe based on blood-vessels time-series"""
        # Build dict to feed dataframe
        data = {}
        for i in range(len(self.vascu_activity_array)):
            data[f"vessel_{i}"] = self.vascu_activity_array[i]

        # Build dataframe
        df = pd.DataFrame(data)
        df.index = self.time_vector_array

        return df

    def get_coords_df(self):
        """ Get a dataframe based on (both) neurons and blood-vessels coordinates"""
        data = []
        indices = []
        for i, coord in enumerate(self.neuro_coord_array):
            indices.append(f"neuron_{i}")
            data.append({
                "x":coord[0], "y": coord[1], "z": coord[2],
                "type": "Neuron"
            })
        for i, coord in enumerate(self.vascu_coord_array):
            indices.append(f"vessel_{i}")
            data.append({
                "x":coord[0], "y": coord[1], "z": coord[2],
                "type": "Blood-Vessel"
            })

        df = pd.DataFrame(data, index=indices)
        return df
