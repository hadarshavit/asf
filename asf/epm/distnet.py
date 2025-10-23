import logging
from typing import Type, Union

import pandas as pd
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from asf.predictors.utils.losses import lognorm_loss
from asf.predictors.utils.datasets import RegressionDataset
from asf.predictors.utils.mlp import get_mlp
from asf.predictors.utils.mlp import ExpActivation


class DistNet:
    def __init__(
        self,
        model: Type[Module] = None,
        optimizer: Type[Optimizer] = None,
        loss_function=lognorm_loss,
        n_feats: int = 1,
        n_loss_params: int = 2,
        epochs: int = 10,
        batch_size: int = 16,
        device=torch.device("cpu"),
    ):
        if model is None:
            model = get_mlp(
                input_size=n_feats,
                output_size=n_loss_params,
                hidden_sizes=[16, 16],
                compile=True,
                output_activation=ExpActivation(),
            )
            optimizer = torch.optim.Adam(model.parameters())

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, list],
        y: Union[pd.Series, list],
        create_graph: bool = False,
    ):
        dataset = RegressionDataset(X, y)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_epoch_loss = 0.0
            for input, target in loader:
                input = input.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input)
                loss = self.loss_function(target, outputs)
                total_epoch_loss += loss.sum().item()
                loss.backward(create_graph=create_graph)
                self.optimizer.step()
            self.logger.debug(
                f"Epoch {epoch}, Total Loss: {total_epoch_loss / len(dataset)}"
            )

    def predict(self, X: Union[pd.DataFrame, pd.Series, list]) -> torch.Tensor:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions
