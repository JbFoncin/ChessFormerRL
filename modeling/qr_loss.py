"""
This module contains the QR-DQN loss
"""
import torch as t
from torch.nn import Module


class QRLoss(Module):
    """
    This is the quantile regression loss.
    It estimates the quantiles of the Q function.
    """
    def __init__(self, nb_quantiles=100, kappa=0.01):
        """
        Args:
            nb_quantiles (int): number of quantiles. Default to 100.
            kappa (float): error threshold. Absolute errors greater than
                           this threshold are processed with MSE, above with MAE
        """
        super().__init__()
        quantiles = t.tensor([i / (nb_quantiles + 1) for i in range(1, nb_quantiles + 1)],
                             dtype=t.float32)
        self.register_buffer('quantiles', quantiles)
        self.kappa = kappa

    def forward(self, choice_quantile, ground_truth, weights=None):
        """
        Args:
            choice_quantile (t.Tensor): model output for an action,
                                        size = batch_size * nb_quantiles
            ground_truth (t.Tensor): Q function estimation + reward, size = batch_size * nb_quantiles
            weights (t.Tensor): weigt for each batch element. size = batch_size
        """
        error = choice_quantile - ground_truth
        batch_size, n_quantiles = error.size()
        error = error.unsqueeze(-1).expand(batch_size, n_quantiles, n_quantiles)
        huber_error = t.where(error.abs() <= self.kappa,
                              0.5 * error.pow(2),
                              self.kappa * (error.abs() - 0.5 * self.kappa))
        
        # We detach error in the following line as abs(self.quantile_tau -(error.detach() < 0).float()) is
        # used as weight on the huberloss
        quantiles = self.quantiles.expand(batch_size, n_quantiles, n_quantiles)
        huber_error_weighted = t.abs(quantiles - (error.detach() < 0).float()) * huber_error 
        
        batch_error = huber_error_weighted.mean(-1).mean(-1)
        
        if weights is not None:
            batch_error = batch_error * weights
        
        return batch_error.mean(0)
        
        