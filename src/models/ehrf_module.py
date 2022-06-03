from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics import MeanSquaredError, MeanAbsoluteError

import plotly.express as px


class EHRFModule(LightningModule):
    def __init__(
        self,
        x_size,
        y_size,
        latent_dim: int = 1,

        # Model's Hyper-parameters:
        hidden_layers: int = 2,            # Hidden layers of the latent transform
        hidden_layer_dim: int = 100,       # Dims of ^
        hidden_dropout: float = 0,         # Dropout layer to the latent transform
        with_vascular_mean: bool = False,  # Whether to insert vascular mean before prediction (should help)
        with_1st_latent_dim: bool = True,

        # Average Baseline config
        predict_with_mean_vascular_only: bool = False,  # Naive model predicts the mean activity

        lr: float = 0.001,
        weight_decay: float = 0.0005,

        **kwargs
    ):
        super().__init__()

        # this line allows accessing init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # More data:
        self.x_size, self.y_size = x_size, y_size
        self.neuron_count, self.neuro_window_size, self.vessels_count = x_size[0], x_size[1], y_size
        self.latent_dim = latent_dim
        self.with_vascular_mean = with_vascular_mean
        self.with_1st_latent_dim = with_1st_latent_dim
        self.distances = None  # will be added later
        self.mean_vascular_activity = None  # will be added later

        # Build the NN transformation to latent space
        to_latent_space_layers = [
            nn.Linear(self.neuron_count * self.neuro_window_size,
                      hidden_layer_dim if hidden_layers > 0 else (self.neuron_count * self.latent_dim)),
        ]
        for i in range(hidden_layers):
            to_latent_space_layers.append(nn.ReLU())
            if i == hidden_layers - 1:
                to_latent_space_layers.append(nn.Dropout(p=hidden_dropout))
                to_latent_space_layers.append(nn.Linear(hidden_layer_dim, self.neuron_count * self.latent_dim))
            else:
                to_latent_space_layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
        self.to_latent_space = nn.Sequential(*to_latent_space_layers)
        self.to_latent_space.double()  # our data is passed in float64 (i.e. double)
        self.to_latent_space.apply(self.init_weights)

        # Build the latent space to vascular activity transformation
        self.to_vascular = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.neuron_count * (self.latent_dim - 1), self.y_size),
            nn.ReLU(),
            nn.Linear(self.y_size, self.y_size),
        )
        self.to_vascular.double()  # our data is passed in float64 (i.e. double)
        self.to_vascular.apply(self.init_weights)

        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        # for logging best so far validation accuracy
        self.val_mse_best = MinMetric()

        # debug flags:
        self.show_weight_heatmap = False

    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def set_extras(self, extras):
        """ Set extras from dataset (MUST be called before training) """
        self.distances = extras['distances']
        self.distances = self.distances.to(device=self.device).double()
        self.mean_vascular_activity = torch.Tensor(extras['mean_vascular_activity'])
        self.mean_vascular_activity = self.mean_vascular_activity.to(device=self.device).double()

    def forward(self, batch_x: torch.Tensor):
        if isinstance(batch_x, list):
            # Lightning's prediction API calls `forward` with an (X,y) pair, so we extract the X
            batch_x = batch_x[0]

        vascu_pred = torch.zeros(batch_x.shape[0], self.y_size,
                                 device=self.device, dtype=torch.double)
        self.distances = self.distances.to(device=self.device)
        self.mean_vascular_activity = self.mean_vascular_activity.to(device=self.device)

        flattened_x = torch.flatten(batch_x, start_dim=1)
        latent_space = self.to_latent_space(flattened_x)

        # From latent to vascular activity
        latent_space = latent_space.view(batch_x.shape[0], batch_x.shape[1], self.latent_dim)
        if self.latent_dim > 1:
            # 1. Predict based on the last latent dims
            vascu_pred = self.to_vascular(latent_space[:, :, 1:].flatten(start_dim=1))
        # 2. Predict based on the first latent dim, with special function
        for i in range(batch_x.shape[0]):
            if self.with_vascular_mean:
                # adding vascular activity mean as a bias
                vascu_pred[i] += self.mean_vascular_activity
            #  each neuron is weighted by distance from blood vessel
            if self.with_1st_latent_dim and not self.hparams.predict_with_mean_vascular_only:
                vascu_pred[i] += (torch.exp(-self.distances) * latent_space[i, :, 0]).sum(dim=1)
            elif self.hparams.predict_with_mean_vascular_only:
                # HACK for predicting mean only for naive model configuration
                vascu_pred[i] += (torch.exp(-self.distances) * latent_space[i, :, 0]).sum(dim=1) * 0

        return vascu_pred

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = logits
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        mse = self.train_mse(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        mse = self.val_mse(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        mse = self.val_mse.compute()  # get val accuracy from current epoch
        self.val_mse_best.update(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_mse(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        if self.show_weight_heatmap:
            print(">> [1st Layer's] Weight: \n", self.to_latent_space[0].weight)
            print(">> [1st Layer's] Bias: \n", self.to_latent_space[0].bias)
            fig = px.imshow(self.to_latent_space[0].weight.to('cpu').detach().numpy())
            fig.show()

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_mse.reset()
        self.test_mse.reset()
        self.val_mse.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    @staticmethod
    def init_weights(m):
        """ Initialize the given layer """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
