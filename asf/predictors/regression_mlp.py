"""
Regression MLP predictor using PyTorch.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.utils.configurable import ConfigurableMixin

try:
    import torch

    TORCH_AVAILABLE = True
    from asf.predictors.utils.datasets import RegressionDataset
    from asf.predictors.utils.mlp import get_mlp
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ConfigSpace import (
        Float,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class RegressionMLP(AbstractPredictor, ConfigurableMixin):
    """
    A regression-based predictor using a Multi-Layer Perceptron (MLP).
    """

    PREFIX: str = "regression_mlp"

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        loss: torch.nn.modules.loss._Loss | None = None,
        optimizer: type[torch.optim.Optimizer] | None = None,
        batch_size: int = 128,
        epochs: int = 2000,
        seed: int = 42,
        device: str = "cpu",
        compile_model: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        torch.manual_seed(seed)

        self.model = model
        self.device = device
        self.loss = loss or torch.nn.MSELoss()
        self.batch_size = batch_size
        self.optimizer = optimizer or torch.optim.Adam
        self.epochs = epochs
        self.compile_model = compile_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _get_dataloader(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
    ) -> torch.utils.data.DataLoader:
        dataset = RegressionDataset(features, performance)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def fit(
        self,
        X: Any,
        Y: Any,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The features for each instance.
        Y : pd.DataFrame
            The performance of each algorithm on each instance.
        sample_weight : np.ndarray or None, default=None
            Sample weights. Currently not supported.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        RegressionMLP
            The fitted model.

        Raises
        ------
        AssertionError
            If sample_weight is provided.
        """
        sample_weight = kwargs.get("sample_weight")
        assert sample_weight is None, "Sample weights are not supported."

        if self.model is None:
            input_size = X.shape[1] if hasattr(X, "shape") else len(X.columns)
            self.model = get_mlp(input_size=input_size, output_size=1)

        self.model.to(self.device)  # type: ignore[attr-defined]

        if self.compile_model:
            self.model = torch.compile(self.model)

        features_imputed = pd.DataFrame(
            SimpleImputer().fit_transform(X.values),
            index=X.index,
            columns=X.columns,
        )
        dataloader = self._get_dataloader(features_imputed, Y)

        optimizer = self.optimizer(
            self.model.parameters(),  # type: ignore[attr-defined]
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.model.train()  # type: ignore[attr-defined]
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(dataloader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                X_batch = X_batch.float()
                y_batch = y_batch.unsqueeze(-1).float()
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss(y_pred, y_batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

        return None

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : pd.DataFrame
            The features to predict on.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()  # type: ignore[attr-defined]

        features_tensor = torch.from_numpy(X.values).to(self.device).float()
        predictions = self.model(features_tensor).detach().cpu().numpy().squeeze(1)

        return predictions

    def save(self, file_path: str) -> None:
        """
        Save the model to a file.
        """
        torch.save(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> AbstractPredictor:
        """
        Load the model from a file.
        """
        return torch.load(file_path)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for RegressionMLP.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("batch_size", (32, 256), log=True, default=128),
            Integer("epochs", (200, 2000), log=True, default=500),
            Float("learning_rate", (1e-4, 1e-1), log=True, default=1e-3),
            Float("weight_decay", (1e-6, 1e-2), log=True, default=1e-5),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[RegressionMLP]:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(RegressionMLP, **config)
