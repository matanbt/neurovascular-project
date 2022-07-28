from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from src.utils.handmade_metrics import MeanBestKMSE, NormalizedRootMeanSquaredError
from src.models.components.fully_connected_net import FullyConnectedNet
from src.models.components.res_connection import ResConnection
import plotly.express as px

from src import utils

log = utils.get_logger(__name__)


class VResNetModule(LightningModule):
    def __init__(
        self,
        x_size,
        y_size,
        vessels_count,

        # Model's Hyper-parameters:
        with_res_connections: bool = True,
        with_vascular_mean: bool = False,  # whether to insert the vascular mean to the residuals sum
        with_batchnorm: bool = True,
        with_droupout: float = 0,

        start_hidden_layers_count: int = 0,  # count of the first hidden layers
        end_hidden_layers_count: int = 0,  # count of the last hidden layers

        res_block_layers_count: int = 3,
        res_blocks_count: int = 5,  # how many res-blocks?

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
        self.flatten_x_size = x_size[0] * x_size[1]
        self.vessels_count = vessels_count
        self.distances = None  # will be added later
        self.mean_vascular_activity = None  # will be added later
        start_hidden_layers_count = int(start_hidden_layers_count)
        res_block_layers_count = int(res_block_layers_count)
        end_hidden_layers_count = int(end_hidden_layers_count)

        # Build the first section (from neuronal-dim to vascular-dim):
        self.start_section = nn.Sequential(nn.Flatten(start_dim=1),
                                           FullyConnectedNet(self.flatten_x_size, self.y_size,
                                                             hidden_layers_count=start_hidden_layers_count,
                                                             hidden_layers_unified_dim=self.y_size,
                                                             with_batch_norm_hidden=with_batchnorm))

        # Build the residuals chain
        res_blocks_list = []
        for _ in range(res_blocks_count):
            block = FullyConnectedNet(y_size, y_size,
                                      hidden_layers_count=res_block_layers_count,
                                      hidden_layers_unified_dim=y_size,
                                      with_batch_norm_hidden=with_batchnorm,
                                      with_dropout_hidden=with_droupout)
            if with_res_connections:
                block = ResConnection(block)
            res_blocks_list.append(block)
        self.res_blocks_section = nn.Sequential(*res_blocks_list)

        # Build the ending section
        self.end_section = FullyConnectedNet(self.y_size, self.y_size,
                                             hidden_layers_count=end_hidden_layers_count,
                                             hidden_layers_unified_dim=self.y_size,
                                             with_batch_norm_hidden=with_batchnorm)

        # our data is passed in float64 (i.e. double)
        self.start_section.double()
        self.res_blocks_section.double()
        self.end_section.double()

        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        # "handmade" metrics:
        self.train_nrmse = NormalizedRootMeanSquaredError(vessels_count=self.vessels_count)
        self.val_nrmse = NormalizedRootMeanSquaredError(vessels_count=self.vessels_count)
        self.train_mbkmse = MeanBestKMSE(vessels_count=self.vessels_count)
        self.val_mbkmse = MeanBestKMSE(vessels_count=self.vessels_count)

        # for logging best so far validation accuracy
        self.val_mse_best = MinMetric()
        self.val_nrmse_best = MinMetric()
        self.val_mbkmse_best = MinMetric()

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

        out = self.start_section(batch_x)
        if self.hparams.with_vascular_mean:
            out = out + self.mean_vascular_activity
        out = self.res_blocks_section(out)
        out = self.end_section(out)

        return out

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
        nrmse = self.train_nrmse(preds, targets)
        mbkmse = self.train_mbkmse(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/nrmse", nrmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mbkmse", mbkmse, on_step=False, on_epoch=True, prog_bar=True)

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
        nrmse = self.val_nrmse(preds, targets)
        mbkmse = self.val_mbkmse(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/nrmse", nrmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mbkmse", mbkmse, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        mse = self.val_mse.compute()  # get val accuracy from current epoch
        self.val_mse_best.update(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), on_epoch=True, prog_bar=True)

        # log other minimum metrics as well:
        self.val_nrmse_best.update(self.val_nrmse.compute())
        self.log("val/nrmse_best", self.val_nrmse_best.compute(), on_epoch=True, prog_bar=True)
        self.val_mbkmse_best.update(self.val_mbkmse.compute())
        self.log("val/mbkmse_best", self.val_mbkmse_best.compute(), on_epoch=True, prog_bar=True)

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
        self.train_nrmse.reset()
        self.val_nrmse.reset()
        self.train_mbkmse.reset()
        self.val_mbkmse.reset()

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
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)
