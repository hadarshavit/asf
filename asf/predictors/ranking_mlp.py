from __future__ import annotations

from typing import Callable, Any

import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
    from asf.predictors.utils.datasets import RankingDataset
    from asf.predictors.utils.losses import bpr_loss
    from asf.predictors.utils.mlp import get_mlp
except Exception:
    TORCH_AVAILABLE = False

from asf.predictors.abstract_predictor import AbstractPredictor

from asf.utils.configurable import ConfigurableMixin
from functools import partial

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
        Float,
        Categorical,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

import logging


class RankingMLP(ConfigurableMixin, AbstractPredictor):
    """
    A ranking-based predictor using a Multi-Layer Percetron (MLP).

    This class implements a ranking model that uses an MLP to predict
    the performance of algorithms based on input features.
    """

    def __init__(
        self,
        model: Any | None = None,
        input_size: int | None = None,
        loss: Callable | None = None,
        optimizer: Callable[..., Any] | None = None,
        batch_size: int = 128,
        epochs: int = 500,
        seed: int = 42,
        device: str = "cpu",
        compile: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        assert model is not None or input_size is not None, (
            "Either model or input_size must be provided."
        )

        torch.manual_seed(seed)

        if model is None:
            assert input_size is not None
            self.model = get_mlp(input_size=input_size, output_size=1)
        else:
            self.model = model

        self.model.to(device)
        self.device = device

        self.loss = loss or bpr_loss
        self.batch_size = batch_size
        self.optimizer = optimizer or torch.optim.Adam
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if compile:
            self.model = torch.compile(self.model)

    def _get_dataloader(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame,
    ) -> Any:
        dataset = RankingDataset(features, performance, algorithm_features)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def fit(
        self,
        X: Any,
        Y: Any,
        **kwargs: Any,
    ) -> None:
        # Extract algorithm_features from kwargs
        algorithm_features = kwargs.get("algorithm_features")
        if algorithm_features is None:
            raise ValueError(
                "algorithm_features must be provided in kwargs for RankingMLP.fit"
            )

        dataloader = self._get_dataloader(X, Y, algorithm_features)

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i, ((Xc, Xs, Xl), (yc, ys, yl)) in enumerate(dataloader):
                Xc, Xs, Xl = (
                    Xc.to(self.device),
                    Xs.to(self.device),
                    Xl.to(self.device),
                )
                yc, ys, yl = (
                    yc.to(self.device),
                    ys.to(self.device),
                    yl.to(self.device),
                )

                yc = yc.float().unsqueeze(1)
                ys = ys.float().unsqueeze(1)
                yl = yl.float().unsqueeze(1)

                optimizer.zero_grad()

                y_pred = self.model(Xc)
                y_pred_s = self.model(Xs)
                y_pred_l = self.model(Xl)

                loss = self.loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            logging.debug(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

        return None

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.model.eval()

        features_tensor = torch.from_numpy(X.values).to(self.device).float()
        predictions = self.model(features_tensor).detach().numpy()

        return predictions

    def save(self, file_path: str) -> None:
        torch.save(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> AbstractPredictor:
        return torch.load(file_path)

    PREFIX = "ranking_mlp"

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for RankingMLP."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("batch_size", (32, 256), log=True, default=128),
            Integer("epochs", (50, 1000), log=True, default=500),
            Float("learning_rate", (1e-4, 1e-1), log=True, default=1e-3),
            Float("weight_decay", (1e-6, 1e-2), log=True, default=1e-5),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(RankingMLP, **config)
