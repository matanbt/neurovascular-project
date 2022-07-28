"""
Custom metrics dedicated to Neuro-Vascular interface task
(Or any regression-task that deals with loss of multiple MSEs on distinguishable objects)
"""
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


class MeanSquaredErrorPerVessel(Metric):
    """
        Base class to metrics that depends on calculating the MSE for each blood vessel separately.
        Motivation: Detect models that are distinguishably good on subset of blood-vessels.
    """
    is_differentiable = True
    higher_is_better = False
    sum_squared_error_per_vessel: Tensor
    sum_target_per_vessel: Tensor
    total: Tensor

    def __init__(self, vessels_count=425, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error_per_vessel", default=torch.zeros(vessels_count), dist_reduce_fx="sum")
        self.add_state("sum_target_per_vessel", default=torch.zeros(vessels_count), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # triggered on __call__
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        # calculate the sum of the square-differences of each blood vessel
        sum_squared_error_per_vessel = (preds - target).pow(2).sum(dim=0)
        # calculate the sum of the target of each blood vessel (will be used to calculate the mean)
        sum_target_per_vessel = target.sum(dim=0)
        # count the obs that summed from target for each blood vessel (again, for mean calculation)
        n_obs_per_vessel = target[:, 0].numel()  # obs are the same for all vessels

        self.sum_squared_error_per_vessel += sum_squared_error_per_vessel
        self.sum_target_per_vessel += sum_target_per_vessel
        self.total += n_obs_per_vessel

    def get_current_mse_per_vessel(self) -> Tensor:
        """ returns the MSE calculated separately on each vessel """
        mse_per_vessel = self.sum_squared_error_per_vessel / self.total
        return mse_per_vessel

    def get_current_mean_vascular_per_vessel(self) -> Tensor:
        """ returns the mean vascular activity per blood-vessel"""
        return self.sum_target_per_vessel / self.total

    def get_nrmse_per_vessel(self):
        rmse_per_vessel = self.get_current_mse_per_vessel().sqrt()
        # normalize RMSE with mean vascular activity
        nrmse_per_vessel = rmse_per_vessel / self.get_current_mean_vascular_per_vessel()

        return nrmse_per_vessel

    def compute(self) -> Tensor:
        raise NotImplementedError("MeanSquaredErrorPerVessel must be inherited, and override `compute()` method")


class MeanBestKMSE(MeanSquaredErrorPerVessel):
    """
        Calculates the mean of the Top-K best NRMSEs of blood vessels
        This should detect a model that is specifically good on a subset of vessels
    """

    def __init__(self, k: int = 50, **kwargs):
        """
            Args:
                k: defines the amount of top NRMSE to take
        """
        super().__init__(**kwargs)
        self.k = k

    def compute(self) -> Tensor:
        # mse_per_vessel = self.get_current_mse_per_vessel()  # commented out as we currently calculate with the NRMSE
        nrmse_per_vessel = self.get_nrmse_per_vessel()
        sorted_idx = nrmse_per_vessel.argsort()
        # Extract the K-best NRMSEs and take a mean
        mean_best_mses = nrmse_per_vessel[sorted_idx[:self.k]].mean()

        return mean_best_mses


class NormalizedRootMeanSquaredError(MeanSquaredErrorPerVessel):
    """
        Computes the MSE, *but* divided with the mean of the target.
        This way we give less significance to big blood vessels, as they are most likely to yield big MSE as well
        \frac{\sqrt{\frac{\sum_{i}\left(y_{i}-\hat{y_{i}}\right)^{2}}{N}}}{\overline{\boldsymbol{y}}}
    """

    def __init__(self, k: int = 50, **kwargs):
        """
            Args:
                k: defines the amount of top MSE to take
        """
        super().__init__(**kwargs)
        self.k = k

    def compute(self) -> Tensor:
        nrmse_per_vessel = self.get_nrmse_per_vessel()

        # we return the mean of the NRMSE
        return nrmse_per_vessel.mean()
