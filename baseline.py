"""
Runs training and evaluations of baseline models
(That is, simple non-deep ML methods, that in particular do not use Torch)
"""
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="baseline.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.baseline_pipeline import run_pipeline

    # Applies optional utilities
    utils.extras(config)

    # Train & Eval (simple) model
    return run_pipeline(config)


if __name__ == "__main__":
    main()
