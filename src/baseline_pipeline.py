"""
    Pipeline for simple models, dedicated *specifically* for our task (neuro-vascular interface).
"""
import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig

import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from src import utils

log = utils.get_logger(__name__)

# ---------------- Pipeline Functions ----------------
def get_dataset(dataset_path):
    """
    Args:
        dataset_path: full path to dataset dir, contains `neuronal.zarr` and `vascular.zarr`.

    Returns: tuple of (neuronal_activity, vacular_activity, time_vector, ?? More Stuff ??)
    """
    pass


DATA_PROC_VARIATIONS = ['classic']
def preprocess_data(data_process_variation,
                    neuronal_activity, vacular_activity, *args) -> tuple[list, list]:
    """
    Args:
        data_process_variation: the name of variation of preprocessing (from `DATA_PROC_VARIATIONS`)
                                (this will influence: smoothing method, feature engineering, etc)
        neuronal_activity, vacular_activity: from dataset
    Returns: the freshly crafted (X,y)

    """
    assert data_process_variation in DATA_PROC_VARIATIONS, \
        f"Expected configuration variation to be from {DATA_PROC_VARIATIONS}"

    # TODO
    # preprocess the data by the given `data_process_variation`

    # return X,y
    pass


MODEL_NAMES = ['linear_regression']
def run_model(model_name,
              x_train, y_train, x_test, y_test,
              logger) -> dict:
    """
    Trains and evaluates the given model, returns a dictionary with ALL the results
    """
    # TODO
    return {
        'train/acc': None,
        'train/loss': None,
        'test/acc': None,
        'test/loss': None,
    }
# ---------------- [END] Pipeline Functions ----------------

def run_pipeline(config: DictConfig) -> Optional[float]:
    """Contains the "simple-models" training pipeline.
        (That is not-deep non-torch models)

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        # TODO set any desired seed here.
        pass

    # Init and process data
    log.info(f"Fetch dataset: {config.logic.dataeset_name}")
    dataset = get_dataset(os.path.join(config.data_dir, config.logic.dataeset_name))

    log.info(f"Fetching and Preprocessing dataset: Variation-{config.logic.data_process_variation}")
    x, y = preprocess_data(config.data_dir, *dataset)

    # Split dataset to training and test
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=config.data_split_ratio,
                                                        random_state=None)

    # Hack for Init WANDB logger [ not good coding :( ]
    wandb_logger = None
    if "logger" in config:
        if 'wandb' in config.logger:
            log.info(f"Instantiating W&B logger")
            wandb_logger = wandb.init(
                project="baselines", # project name in wandb
                name=config.logger.wandb.name,
                job_type=config.logger.wandb.job_type,
                config=config.logic
            )


    # Run model
    log.info(f"Training and evaluating model: {config.logic.model_name}")
    results = run_model(config.model_name, x_train, y_train, x_test, y_test)

    # Print and log results
    log.info(f"------------------------")
    log.info(f"Results: {results}")
    log.info(f"------------------------")

    if wandb_logger:
        wandb.log(results)

        # Make sure everything closed properly
        log.info("Finalizing!")
        wandb.finish()

    # Return metric score for hyperparameter optimization
    return results['test/acc']
