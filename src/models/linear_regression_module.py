from typing import Any, List

import torch
from torch.nn import Sequential, Linear, Dropout, MSELoss
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanSquaredError, MeanAbsoluteError
from src.utils.handmade_metrics import MeanBestKMSE, NormalizedRootMeanSquaredError


class LinearRegressionModule(LightningModule):
    def __init__(
        self,
        x_size,
        y_size,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        # this line allows accessing init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.net = torch.nn.Linear(x_size, y_size)
        self.net = Sequential(
            Linear(x_size, y_size),
            Dropout(p=dropout)
        )
        self.net.double()  # our data is passed in float64 (i.e. double)

        def init_weights(m):
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        # Weight initialization
        # torch.nn.init.xavier_uniform_(self.net.weight)
        self.net.apply(init_weights)

        # loss function
        self.criterion = MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        # "handmade" metrics:
        self.train_nrmse = NormalizedRootMeanSquaredError(vessels_count=self.vessels_count)
        self.val_nrmse = NormalizedRootMeanSquaredError(vessels_count=self.vessels_count)
        self.test_nrmse = NormalizedRootMeanSquaredError(vessels_count=self.vessels_count)
        self.train_mbkmse = MeanBestKMSE(vessels_count=self.vessels_count)
        self.val_mbkmse = MeanBestKMSE(vessels_count=self.vessels_count)
        self.test_mbkmse = MeanBestKMSE(vessels_count=self.vessels_count)

        # for logging best so far validation accuracy
        self.val_mse_best = MinMetric()
        self.val_nrmse_best = MinMetric()
        self.val_mbkmse_best = MinMetric()

        # flags:
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

    def forward(self, x: torch.Tensor):
        return self.net(x)

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
        mse = self.test_mse(preds, targets)
        nrmse = self.test_nrmse(preds, targets)
        mbkmse = self.test_mbkmse(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", mse, on_step=False, on_epoch=True)
        self.log("test/nrmse", nrmse, on_step=False, on_epoch=True)
        self.log("test/mbkmse", mbkmse, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass


    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_mse.reset()
        self.test_mse.reset()
        self.val_mse.reset()
        self.train_nrmse.reset()
        self.val_nrmse.reset()
        self.test_nrmse.reset()
        self.train_mbkmse.reset()
        self.val_mbkmse.reset()
        self.test_mbkmse.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
