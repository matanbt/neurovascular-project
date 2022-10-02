<div align="center">

# Neuro-Vascular Interface : ML Project
<img width="450" src="docs/img/brain-img.jpg">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description
Exploring the neuro-vascular interface, with the goal of finding the HRF function that connects the two.

This repo contains the data preprocessing logic and the models we created to estimate the HRF. It also includes infra to run experiments in scale.

[Paper Link](docs/neurovascu-ml-paper.pdf)

[Experiments](https://wandb.ai/neurovascular-ml/neurovascular-ml-server-experiments)

## How to run

### Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
# Using python3.8 is a must.
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# [OPTIONAL] if using WandB, insert your API key using: 
wandb login

# [OPTIONAL] if using CometML, insert your API key using:
export COMET_API_TOKEN={YOUR-API-KEY-FROM-COMET.ML}
```

### Train model (i.e. Reproduction of Experiments)
Train a model, with chosen experiment (= model variation) configuration from [configs/experiment/](configs/experiment/).

```bash
python train.py experiment=experiment_name
```

**Main Experiments List:**
All the experiment are elaborated in our project's booklet, and may be reproduced using the following configs:
- `experiment_mean_baseline.yaml`, `experiment_persistence_model.yaml`: Control models, used to serve as primitive comparable baselines. 
- `experiment_lin_regr_vascu.yaml`: Naive Linear Regression baseline, but including the vascular activity.
- `experiment_lin_regr.yaml`: Linear Regression baseline, from neuronal activity only.
- `experiment_lin_net.yaml`: Deep Linear Network model, from neuronal activity only.
- `experiment_ehrf.yaml`: A carefully engineered learnable function from neuronal activity.
- `experiment_dual_rnn.yaml`: LSTM based time-series model, processes both vascular and neuronal activity (each in another LSTM), to predict the vascular activity.
- `experiment_rnn.yaml`: LSTM based time-series model, procceses neuronal activity *only* in an LSTM, to predict the vascular activity.

**Main Datasets List**
- `2021_02_01_18_45_51_neurovascular_full_dataset`: recording 19.3 minutes long (~35K timestamps).
- `2021_02_01_18_45_51_neurovascular_partial_dataset`: the first 30 seconds of `2021_02_01_18_45_51_neurovascular_full_dataset` (~3K timestamps).
- `2021_02_01_19_19_39_neurovascular_full_dataset`: recording 26.6 minutes long (~48K timestamps).

### Utilize existing model as a predictor
Use a trained model to fully predict certain dataset, and save it to CSVs.
1. 
```bash
python train.py experiment=experiment_name train=False test=False model.generate_pred_csv=True model.predictor_ckpt_path={PATH_TO_CKPT_ARTIFACT}
```
where `PATH_TO_CKPT_ARTIFACT` is the path the `*.ckpt` file of the model.
2. You may use the notebook `notebooks/explore-results.ipynp` to analyze and visualize the results.
### Override configurations
You can override any parameter from command line like this. Some useful examples:
- `trainer.gpus=1`: train with single GPU (`0` for CPU).
- `logger=csv`: change the logger to be a simple _CSV_ file.
- `trainer.max_epochs=20`: Change the max epochs amount to 20.

### Run the baseline model (Sklearn based, non-deep model)
```bash
python -m src.baseline.baseline
```

### Utils
- **Visualize dataset:** You can visualize a dataset (of similar form to the given datasets) by utilizing `scripts/visualize_data.py`

## Dirs and Files (with most logic)
- `./data`: all datasets used for our models.
- `./notebooks`: all notebooks used for exploring the data and POCing models.
- `./configs`: configuration yaml files, used to run the project.
- `./src`: source code used for the project (from data extraction, feature engineering, models logic to training and testing).
  - `datamodules/`: package for feature engineering and data preprocessing.
    - `neurovascu_datamodule.py`: lightning's datamodule that wraps our datasets.
    - `components/nv_fetcher.py`: module for fetching raw data from neuro-vascular dataset.
    - `components/nv_datasets.py`: module for various datasets of neuro-vascular, created using feature-engineering.
  - `baseline/`: package that contain our first and simplest model
    - `baseline.py`: contains the classic linear-regression model.
    - `models.py`: contains some additional models, including naive one for control. 
  - `models`: package that contain all our Deep Learning models.
  - `utils/handmade_metrics.py`: implementation of the special metrics we applied on the models during training / testing.