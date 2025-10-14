from typing import Type, Union

import pandas as pd
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from asf.predictors.utils.datasets import RegressionDataset


@torch.jit.script
def lognorm_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    s = y_pred[:, 0]
    s = torch.reshape(s, [-1, 1])

    scale = y_pred[:, 1]
    scale = torch.reshape(scale, [-1, 1])
    log_scale = torch.log(scale)
    log_true = torch.log(y_true)

    # Compute logged lh (removed constants)
    help1 = log_true - log_scale
    help1 = 0.5 * torch.pow(help1 / s, 2)

    # add terms (not multiplying them)
    lh = -torch.log(s) - log_true - help1

    return -lh


@torch.jit.script
def invgauss_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    mu = y_pred[:, 0]
    mu = torch.reshape(mu, [-1, 1])

    scale = y_pred[:, 1]
    scale = torch.reshape(scale, [-1, 1])

    tmp_true = torch.zeros_like(y_true)
    y_true = tmp_true + y_true

    # Compute logged lh (removed constants)
    help1 = 0.5 * torch.log(scale)
    help2 = 3.0 / 2.0 * torch.log(y_true)

    tmp = y_true / scale

    help3 = torch.pow(tmp - mu, 2)
    lower = 2 * tmp * torch.pow(mu, 2)
    help3 = help3 / lower

    # add terms (not multiplying them)
    lh = help1 - help2 - help3

    return -lh


@torch.jit.script
def exp_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    scale = y_pred[:, 0]
    scale = torch.reshape(scale, [-1, 1])
    scale = 1 / scale

    log_scale = torch.log(scale)

    # Compute logged lh (removed constants)
    lh = log_scale - y_true * scale

    return -lh


# def weibull_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
#     shape = y_pred[:, 0]
#     shape = torch.reshape(shape, [-1, 1])

#     scale = y_pred[:, 1]
#     scale = torch.reshape(scale, [-1, 1])


#     return -lh


class DistNet:
    def __init__(
        self,
        model: Type[Module],
        optimizer: Type[Optimizer],
        loss_function=lognorm_loss,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, list],
        y: Union[pd.Series, list],
    ):
        dataset = RegressionDataset(X, y)

        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(10):
            for input, target in loader:
                input = input.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input)
                loss = self.loss_function(outputs, target)
                loss.backward()
                self.optimizer.step()

    def predict(self, X: Union[pd.DataFrame, pd.Series, list]) -> torch.Tensor:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions
