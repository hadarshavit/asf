from __future__ import annotations

try:
    import torch

    TORCH_AVAILABLE = True
    from asf.predictors.utils.datasets import RegressionDataset
    from asf.predictors.utils.mlp import get_mlp
except ImportError:
    TORCH_AVAILABLE = False

import pandas as pd
from sklearn.impute import SimpleImputer

from asf.predictors.abstract_predictor import AbstractPredictor


class RegressionMLP(AbstractPredictor):
    def __init__(
        self,
        model: object | None = None,
        loss: object | None = None,
        optimizer: object | None = None,
        batch_size: int = 128,
        epochs: int = 2000,
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

        torch.manual_seed(seed)

        self.model = model
        self.device = device

        self.loss = loss or torch.nn.MSELoss()
        self.batch_size = batch_size
        self.optimizer = optimizer or torch.optim.Adam
        self.epochs = epochs
        self.compile = compile

    def _get_dataloader(
        self, features: pd.DataFrame, performance: pd.DataFrame
    ) -> object:
        dataset = RegressionDataset(features, performance)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, sample_weight=None
    ) -> "RegressionMLP":
        assert sample_weight is None, "Sample weights are not supported."

        if self.model is None:
            self.model = get_mlp(input_size=features.shape[1], output_size=1)

        self.model.to(self.device)

        if self.compile:
            self.model = torch.compile(self.model)

        features = pd.DataFrame(
            SimpleImputer().fit_transform(features.values),
            index=features.index,
            columns=features.columns,
        )
        dataloader = self._get_dataloader(features, performance)

        optimizer = self.optimizer(self.model.parameters())
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                X = X.float()
                y = y.unsqueeze(-1)
                optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.loss(y_pred, y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        self.model.eval()

        features = torch.from_numpy(features.values).to(self.device).float()
        predictions = self.model(features).detach().numpy().squeeze(1)

        return predictions

    def save(self, file_path: str) -> None:
        torch.save(self.model, file_path)

    def load(self, file_path: str) -> None:
        self.model = torch.load(file_path)
