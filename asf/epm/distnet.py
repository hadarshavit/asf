import logging
from typing import Type, Union

import pandas as pd
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from asf.predictors.utils.datasets import RegressionDataset
from asf.predictors.utils.losses import lognorm_loss
from asf.epm.epm import AbstractEPM
from asf.predictors.utils.mlp import ExpActivation, get_mlp
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from asf.preprocessing.performance_scaling import DummyNormalization
import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    Categorical,
    Float,
    Integer,
)


class DistNet(AbstractEPM):
    def __init__(
        self,
        model: Type[Module] = None,
        optimizer: Type[Optimizer] = torch.optim.RAdam,
        loss_function=lognorm_loss,
        n_loss_params: int = 2,
        epochs: int = 10,
        gradient_clip: float = 1e-2,
        batch_size: int = 16,
        lr_scheduler: Type[LRScheduler] = CosineAnnealingLR,
        device=torch.device("cpu"),
        optimizer_kwargs: dict | None = None,
        **kwargs,
    ):
        # DistNet operates directly on positive runtimes; keep targets untransformed by default
        super().__init__(normalization_class=DummyNormalization, **kwargs)
        self.n_loss_params = n_loss_params
        self.model = model
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.lr_scheduler = lr_scheduler
        self.logger = logging.getLogger(__name__)

    def _fit(
        self,
        X: Union[pd.DataFrame, pd.Series, list],
        y: Union[pd.Series, list],
        weight: list | None = None,
    ):
        assert weight is None, "Sample weights are not supported in DistNet."

        if self.model is None:
            self.model = get_mlp(
                input_size=X.shape[1],
                output_size=self.n_loss_params,
                hidden_sizes=[16, 16],
                compile=True,
                dropout=0.5,
                output_activation=ExpActivation(),
            )

        self.model.to(self.device)

        self.optimizer = self.optimizer(
            self.model.parameters(), **self.optimizer_kwargs
        )

        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.concatenate([[x for i in range(100)] for x in X])
        y = y.flatten()

        dataset = RegressionDataset(X, y)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.lr_scheduler is not None:
            self.lr_scheduler = self.lr_scheduler(
                self.optimizer, T_max=self.epochs, eta_min=1e-5
            )

        last_losses = []
        import time

        start = time.time()
        for epoch in range(self.epochs):
            total_epoch_loss = 0.0
            self.model.train()
            for input, target in loader:
                input = input.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input)
                loss = self.loss_function(target, outputs)
                total_epoch_loss += loss.item() * input.size(0)
                loss.backward()

                if torch.isnan(loss):
                    return

                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            avg_epoch_loss = total_epoch_loss / len(dataset)
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs}, Train NLLH: {avg_epoch_loss:.6f}, LR {self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 'N/A'}"
            )

            last_losses.append(avg_epoch_loss)
            if len(last_losses) > 4 and np.std(last_losses[-4:]) < 1e-8:
                self.logger.info(
                    "Early stopping triggered due to no improvement in training loss."
                )
                break

            end = time.time()
            if end - start > 3600:
                self.logger.info("Early stopping triggered due to time limit exceeded.")
                break

    def predict(self, X: Union[pd.DataFrame, pd.Series, list]) -> torch.Tensor:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions

    @staticmethod
    def get_configuration_space():
        """Configuration space for DistNet hyperparameters.

        Returns a ConfigSpace with common architectural and optimization hyperparameters.
        """
        assert ConfigurationSpace is not None, (
            "ConfigSpace must be installed to use the tuner"
        )

        cs = ConfigurationSpace()

        cs.add(
            [
                Integer("hidden_layers", (1, 3), default=2),
                Categorical(
                    "hidden_size", [8, 16, 32, 64, 128], default=16, ordered=True
                ),
                Float("dropout", (0.0, 0.5), default=0.0),
                Categorical("activation", ["tanh", "relu", "gelu"], default="tanh"),
                Categorical("use_batchnorm", [False, True], default=False),
                Categorical("optimizer", ["adam", "radam", "sgd"], default="radam"),
                Float("lr", (1e-4, 1e-1), log=True, default=1e-2),
                Float("weight_decay", (1e-6, 1e-2), log=True, default=1e-2),
                Float("momentum", (0.0, 0.95), default=0.9),  # used for SGD only
                Categorical(
                    "batch_size", [16, 32, 64, 128, 256], default=16, ordered=True
                ),
                Float("gradient_clip", (1e-3, 10), log=True, default=1e-2),
                Categorical("scheduler", ["none", "cosine"], default="cosine"),
            ]
        )

        return cs

    @staticmethod
    def get_from_configuration(
        *,
        input_size: int,
        config: dict,
        **kwargs,
    ) -> "DistNet":
        """Helper to instantiate DistNet given a sampled configuration."""
        hidden_layers = int(config.get("hidden_layers", 2))
        hidden_size = int(config.get("hidden_size", 16))
        hidden_sizes = [hidden_size] * hidden_layers

        activation_name = config.get("activation", "tanh")
        activation_cls = torch.nn.Tanh if activation_name == "tanh" else torch.nn.ReLU

        dropout = float(config.get("dropout", 0.0))
        use_batchnorm = bool(config.get("use_batchnorm", False))

        # Model
        model = get_mlp(
            input_size=input_size,
            output_size=2,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            output_activation=ExpActivation(),
            compile=False,
            activation_cls=activation_cls,
            use_batchnorm=use_batchnorm,
        )

        # Optimizer and kwargs
        opt_name = config.get("optimizer", "radam")
        lr = float(config.get("lr", 1e-2))
        weight_decay = float(config.get("weight_decay", 0.0))

        if opt_name == "adam":
            opt_cls = torch.optim.Adam
            opt_kwargs = {"lr": lr, "weight_decay": weight_decay}
        elif opt_name == "sgd":
            opt_cls = torch.optim.SGD
            momentum = float(config.get("momentum", 0.9))
            opt_kwargs = {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": True,
            }
        else:
            opt_cls = torch.optim.RAdam
            opt_kwargs = {"lr": lr, "weight_decay": weight_decay}

        # Scheduler
        scheduler_name = config.get("scheduler", "cosine")
        scheduler = CosineAnnealingLR if scheduler_name == "cosine" else None

        # Training params
        epochs = int(config.get("epochs", 200))
        batch_size = int(config.get("batch_size", 16))
        gradient_clip = float(config.get("gradient_clip", 1e-2))

        dn_kwargs = {
            "model": model,
            "optimizer": opt_cls,
            "optimizer_kwargs": opt_kwargs,
            "lr_scheduler": scheduler,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_clip": gradient_clip,
            **kwargs,
        }

        return DistNet(**dn_kwargs)
