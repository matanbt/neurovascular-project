<div align="center">

# Neuro-Vascular Interface : ML Project
<img width="450" src="docs/img/brain-img.jpg">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)
</div>

## Description
Exploring the neuro-vascular interface, with the goal of finding the HRF function that connects the two.

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
```

### Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

**Main Experiments List:**
- `experiment_lin_regr_vascu.yaml`: Naive Linear Regression baseline, but including the vascular activity.
- `experiment_lin_regr.yaml`: Linear Regression baseline, from neuronal activity only.
- `experiment_ehrf.yaml`: A carefully engineered learnable function from neuronal activity.

### Override configurations
You can override any parameter from command line like this. Some useful examples:
- `trainer.gpus=1`: train with single GPU (`0` for CPU).
- `logger=csv`: change the logger to be a simple _CSV_ file.
- `trainer.max_epochs=20`: Change the max epochs amount to 20.

### Run the baseline model (non-Deep model)
```bash
python src/baseline/baseline.py
```

## Dirs and Files (with most logic)
- `./data`: all datasets used for our models.
- `./notebooks`: all notebooks used for exploring the data and POCing models.
- `./src/datamodules`: package for feature engineering and data preprocessing.
  - `mnist_datamodule.py`: lightning's datamodule for our dataset.
  - `components/nv_fetcher.py`: module for fetching raw data from neuro-vascular dataset.
  - `components/nv_datasets.py`: module for various datasets of neuro-vascular, created using feature-engineering.
- `./src/baseline`: package that contain our first and simplest model
  - `baseline.py`: contains the classic linear-regression model.
  - `models.py`: contains some additional models, including naive one for control. 
- `./src/models`: package that contain all our Deep Learning models.
  - `linear_regression_module.py`: reproducing the results from regression in `baseline.py` just in NN.