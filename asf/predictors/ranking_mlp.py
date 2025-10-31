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

import logging


class RankingMLP(AbstractPredictor):
    """
    A ranking-based predictor using a Multi-Layer Percetron (MLP).

    This class implements a ranking model that uses an MLP to predict
    the performance of algorithms based on input features.
    """

    def __init__(
        self,
        model: Any | None = None,
        input_size: int | None = None,
        loss: Callable = bpr_loss,
        optimizer: Callable[..., Any] | None = None,
        batch_size: int = 128,
        epochs: int = 500,
        seed: int = 42,
        device: str = "cpu",
        compile: bool = True,
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
            self.model = get_mlp(input_size=input_size, output_size=1)
        else:
            self.model = model

        self.model.to(device)
        self.device = device

        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = optimizer or torch.optim.Adam
        self.epochs = epochs

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
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame,
    ) -> "RankingMLP":
        dataloader = self._get_dataloader(features, performance, algorithm_features)

        optimizer = self.optimizer(self.model.parameters())
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

        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        self.model.eval()

        features = torch.from_numpy(features.values).to(self.device).float()
        predictions = self.model(features).detach().numpy()

        return predictions

    def save(self, file_path: str) -> None:
        torch.save(self.model, file_path)

    def load(self, file_path: str) -> None:
        self.model = torch.load(file_path)
