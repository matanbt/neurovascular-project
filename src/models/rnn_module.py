from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics import MeanSquaredError, MeanAbsoluteError

import plotly.express as px

from src import utils

log = utils.get_logger(__name__)


class RNNHRFModule(LightningModule):
    def __init__(
        self,
        x_size,
        y_size,

        # LSTM Model's Hyper-parameters:
        rnn_model_type: str = 'LSTM',
        rnn_dropout: float = 0.3,  # dropout between RNN layers
        rnn_hidden_dim: int = 500,
        rnn_layers_count: int = 1,
        rnn_bidirectional: bool = False,

        # Regressor Model's Hyper-parameters:
        regressor_hidden_layers_list: list = (500, ),  # dims for the hidden FC layers of regressor
        regressor_hidden_layers_dropout: float = 0.4,

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
        self.distances = None  # will be added later
        self.mean_vascular_activity = None  # will be added later

        RNN_Model = self._get_rnn_model(rnn_model_type)

        # Build RNN as feature extractor (alternative)
        rnn_layers = [
            RNN_Model(input_size=self.neuron_count,
                      hidden_size=rnn_hidden_dim,
                      num_layers=rnn_layers_count,
                      dropout=rnn_dropout,
                      bidirectional=rnn_bidirectional,
                      batch_first=True)
            ]
        self.feature_extractor = nn.Sequential(*rnn_layers)

        # Build the regressor:
        fc_layers = []
        for hidden_dim in regressor_hidden_layers_list:
            fc_layers.extend([
                nn.LazyLinear(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=regressor_hidden_layers_dropout)
            ])
        fc_layers.append(  # append with last layer
            nn.LazyLinear(self.y_size)
        )
        self.regressor = nn.Sequential(*fc_layers)

        # Init weights
        self.feature_extractor.double()  # our data is passed in float64 (i.e. double)
        self.feature_extractor.apply(self.init_weights)
        log.info(self.feature_extractor)

        self.regressor.double()  # our data is passed in float64 (i.e. double)
        # Initializing the regressor weight is done after the mock forward pass...

        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        # TODO NRMSE

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

    def mock_forward_pass(self):
        """ Runs a mock forward pass, in order to initialize dimensions (we also initialize what we need afterwards)"""
        with torch.no_grad():
            self.forward(torch.rand(1, *self.x_size))

        # Stuff we can do only after the first forward pass:
        self.regressor.apply(self.init_weights)
        log.info(self.regressor)

    def _get_rnn_model(self, rnn_model_type: str ) -> torch.nn.Module:
        """ Map user-string to RNN module """
        # List of possible rnn models:
        rnn_models = {
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        possible_rnn_models = list(rnn_models.keys())
        assert rnn_model_type in possible_rnn_models, \
            f'Expected RNN model to be from {possible_rnn_models} but got {rnn_model_type}'
        return rnn_models[rnn_model_type]

    def forward(self, batch_x: torch.Tensor):
        if self.distances is not None:
            self.distances = self.distances.to(device=self.device)
        if self.mean_vascular_activity is not None:
            self.mean_vascular_activity = self.mean_vascular_activity.to(device=self.device)

        if isinstance(batch_x, list):
            # Lightning's prediction API calls `forward` with an (X,y) pair, so we extract the X
            batch_x = batch_x[0]

        batch_x = batch_x.double()
        batch_x = torch.transpose(batch_x, -2, -1)  # LSTM will run on each timestamp
        features_vector, _ = self.feature_extractor(batch_x)
        features_vector = features_vector[:, -1, :]  # get last hidden state
        vascu_pred = self.regressor(features_vector)

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
            print(">> [1st Layer's] Weight: \n", self.regressor[-1].weight)
            print(">> [1st Layer's] Bias: \n", self.regressor[-1].bias)
            fig = px.imshow(self.regressor[0].weight.to('cpu').detach().numpy())
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
