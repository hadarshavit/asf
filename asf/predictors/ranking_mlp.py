from typing import Callable, Union

import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer

    TORCH_AVAILABLE = True
    from asf.predictors.utils.datasets import RankingDataset
    from asf.predictors.utils.losses import bpr_loss
    from asf.predictors.utils.mlp import get_mlp
except ImportError:
    TORCH_AVAILABLE = False


from asf.predictors.abstract_predictor import AbstractPredictor

import logging

if TORCH_AVAILABLE:

    class RankingMLP(AbstractPredictor):
        """
        A ranking-based predictor using a Multi-Layer Perceptron (MLP).

        This class implements a ranking model that uses an MLP to predict
        the performance of algorithms based on input features.
        """

        def __init__(
            self,
            model: Union[torch.nn.Module, None] = None,
            input_size: Union[int, None] = None,
            loss: Callable = bpr_loss,
            optimizer: Callable[..., Optimizer] = torch.optim.Adam,
            batch_size: int = 128,
            epochs: int = 500,
            seed: int = 42,
            device: str = "cpu",
            compile: bool = True,
            **kwargs,
        ):
            """
            Initializes the RankingMLP with the given parameters.

            Args:
                model (torch.nn.Module | None): The pre-defined PyTorch model to use. If None, a new MLP is created.
                input_size (int | None): The input size for the MLP. Required if `model` is None.
                loss (Callable): The loss function to use. Defaults to `bpr_loss`.
                optimizer (Callable[..., torch.optim.Optimizer]): The optimizer class to use. Defaults to `torch.optim.Adam`.
                batch_size (int): The batch size for training. Defaults to 128.
                epochs (int): The number of training epochs. Defaults to 500.
                seed (int): The random seed for reproducibility. Defaults to 42.
                device (str): The device to use for training (e.g., "cpu" or "cuda"). Defaults to "cpu".
                compile (bool): Whether to compile the model using `torch.compile`. Defaults to True.
                **kwargs: Additional arguments for the parent class.
            """
            super().__init__(**kwargs)
            assert TORCH_AVAILABLE, "PyTorch is not available. Please install it."

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
            self.optimizer = optimizer
            self.epochs = epochs

            if compile:
                self.model = torch.compile(self.model)

        def _get_dataloader(
            self,
            features: pd.DataFrame,
            performance: pd.DataFrame,
            algorithm_features: pd.DataFrame,
        ) -> DataLoader:
            """
            Creates a DataLoader for the given features and performance data.

            Args:
                features (pd.DataFrame): DataFrame containing the feature data.
                performance (pd.DataFrame): DataFrame containing the performance data.
                algorithm_features (pd.DataFrame): DataFrame containing algorithm-specific features.

            Returns:
                torch.utils.data.DataLoader: A DataLoader for the dataset.
            """
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
            """
            Fits the model to the given feature and performance data.

            Args:
                features (pd.DataFrame): DataFrame containing the feature data.
                performance (pd.DataFrame): DataFrame containing the performance data.
                algorithm_features (pd.DataFrame): DataFrame containing algorithm-specific features.

            Returns:
                RankingMLP: The fitted model.
            """
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
            """
            Predicts the performance of algorithms for the given features.

            Args:
                features (pd.DataFrame): DataFrame containing the feature data.

            Returns:
                pd.DataFrame: DataFrame containing the predicted performance data.
            """
            self.model.eval()

            features = torch.from_numpy(features.values).to(self.device).float()
            predictions = self.model(features).detach().numpy()

            return predictions

        def save(self, file_path: str) -> None:
            """
            Saves the model to the specified file path.

            Args:
                file_path (str): The path to save the model.
            """
            torch.save(self.model, file_path)

        def load(self, file_path: str) -> None:
            """
            Loads the model from the specified file path.

            Args:
                file_path (str): The path to load the model from.
            """
            self.model = torch.load(file_path)
else:

    class RankingMLP(AbstractPredictor):
        """
        A placeholder for RankingMLP when PyTorch is not available.
        This class raises an ImportError if any method is called.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not available. Please install it to use RankingMLP."
            )
