from functools import partial
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
    InCondition
)


class DistNet:
    def __init__(
        self,
        model: Type[Module] = None,
        optimizer: Type[Optimizer] = torch.optim.SGD,
        loss_function=lognorm_loss,
        n_loss_params: int = 2,
        epochs: int = 200,
        gradient_clip: float = 1e-2,
        batch_size: int = 16,
        lr_scheduler: Type[LRScheduler] = None,
        device=torch.device("cpu"),
        optimizer_kwargs: dict | None = None,
        **kwargs,
    ):
        # DistNet operates directly on positive runtimes; keep targets untransformed by default
        self.n_loss_params = n_loss_params
        self.model = model
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self._is_trained = False  # Track if model has trained weights
        self.gradient_clip = gradient_clip
        self.lr_scheduler = lr_scheduler
        self.logger = logging.getLogger(__name__)

    def fit(
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
                dropout=0.0,
                output_activation=ExpActivation(),
            )

        self.model.to(self.device)

        self.optimizer = self.optimizer(
            self.model.parameters(), **self.optimizer_kwargs
        )

        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X = np.concatenate([[x for i in range(y.shape[1])] for x in X]).astype(
            np.float32, copy=False
        )
        y = y.flatten().astype(np.float32, copy=False)

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
            self.logger.debug(
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
        
        # Mark model as trained after fitting completes
        self._is_trained = True

    def is_trained(self) -> bool:
        """Check if the model has trained weights loaded.
        
        Returns
        -------
        bool
            True if model was loaded from a valid checkpoint with trained weights,
            False if model has random/untrained weights.
        """
        return getattr(self, '_is_trained', False)
    
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
                Categorical("activation", ["relu", "tanh", "gelu"], default="relu"),
                Categorical("use_batchnorm", [True, False], default=True),
                Categorical("optimizer", ["sgd", "adam", "radam"], default="sgd"),
                Float("lr", (1e-4, 1e-1), log=True, default=1e-2),
                Float("weight_decay", (1e-6, 1e-2), log=True, default=1e-2),
                Float("momentum", (0.0, 0.95), default=0.9),  # used for SGD only
                Categorical(
                    "batch_size", [16, 32, 64, 128, 256], default=16, ordered=True
                ),
                Float("gradient_clip", (1e-3, 10), log=True, default=1e-2),
                Categorical("scheduler", ["none", "cosine"], default="none"),
            ]
        )

        cs.add(InCondition(cs["momentum"], cs["optimizer"], ["sgd"]))

        return cs

    @staticmethod
    def get_from_configuration(
        *,
        input_size: int,
        output_size: int,
        configuration: dict,
        **kwargs,
    ) -> "DistNet":
        """Helper to instantiate DistNet given a sampled configuration."""
        hidden_layers = int(configuration.get("hidden_layers", 2))
        hidden_size = int(configuration.get("hidden_size", 16))
        hidden_sizes = [hidden_size] * hidden_layers

        activation_name = configuration.get("activation", "tanh")
        activation_cls = torch.nn.Tanh if activation_name == "tanh" else torch.nn.ReLU

        dropout = float(configuration.get("dropout", 0.0))
        use_batchnorm = bool(configuration.get("use_batchnorm", False))

        # Model
        model = get_mlp(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            output_activation=ExpActivation(),
            compile=False,
            activation_cls=activation_cls,
            use_batchnorm=use_batchnorm,
        )

        # Optimizer and kwargs
        opt_name = configuration.get("optimizer", "radam")
        lr = float(configuration.get("lr", 1e-2))
        weight_decay = float(configuration.get("weight_decay", 0.0))

        if opt_name == "adam":
            opt_cls = torch.optim.Adam
            opt_kwargs = {"lr": lr, "weight_decay": weight_decay}
        elif opt_name == "sgd":
            opt_cls = torch.optim.SGD
            momentum = float(configuration.get("momentum", 0.9))
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
        scheduler_name = configuration.get("scheduler", "cosine")
        scheduler = CosineAnnealingLR if scheduler_name == "cosine" else None

        # Training params
        epochs = int(configuration.get("epochs", 200))
        batch_size = int(configuration.get("batch_size", 16))
        gradient_clip = float(configuration.get("gradient_clip", 1e-2))

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

        return partial(DistNet, **dn_kwargs)
    
    def save(self, path: str):
        """Saves the DistNet model and configuration to the specified path."""
        # Extract input size from the model structure
        input_size = None
        if self.model is not None:
            # Try to find the first linear layer to get input size
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    input_size = module.in_features
                    break
        
        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model is not None else None,
            'model_config': {
                'input_size': input_size,
                'output_size': self.n_loss_params,
            },
            'n_loss_params': self.n_loss_params,
            # Don't try to pickle TorchScript functions directly; save name and module for reconstruction
            'loss_function_name': getattr(self.loss_function, '__name__', None),
            'loss_function_module': getattr(self.loss_function, '__module__', None),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'gradient_clip': self.gradient_clip,
            'optimizer': self.optimizer,
            'optimizer_kwargs': self.optimizer_kwargs,
            'lr_scheduler': self.lr_scheduler,
            'device': self.device,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            # Remove incomplete file if created and raise a clear error
            try:
                import os

                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            raise RuntimeError(f"Failed to save DistNet checkpoint to {path}: {e}") from e
    
    @classmethod
    def load(cls, path: str, loss_function=None, n_loss_params=None, input_size=None):
        """Loads the DistNet model from the specified path.
        
        Backward compatible: handles both old format (state_dict only) and new format (full config).
        
        Parameters
        ----------
        path : str
            Path to the saved model
        loss_function : callable, optional
            Required for old format models or corrupted checkpoints. Loss function to use.
        n_loss_params : int, optional
            Required for old format models or corrupted checkpoints. Number of loss parameters.
        input_size : int, optional
            Required for old format models or corrupted checkpoints. Input feature size.
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
        except RuntimeError as e:
            # torch.load failed - file is corrupted or not a valid checkpoint
            if loss_function is None or n_loss_params is None or input_size is None:
                raise RuntimeError(
                    f"Failed to load checkpoint from {path}: {e}. "
                    "The file appears to be corrupted or incomplete. "
                    "Please provide loss_function, n_loss_params, and input_size parameters "
                    "to create a new model, or retrain and save the model properly."
                ) from e
            
            # Create a new model from scratch since checkpoint is corrupted
            logging.warning(
                f"Checkpoint at {path} is corrupted or incomplete. "
                f"Creating untrained model. This model will NOT produce meaningful predictions. "
                f"Please retrain the model."
            )
            instance = cls(
                loss_function=loss_function,
                n_loss_params=n_loss_params,
            )
            
            # Build a new model from scratch
            instance.model = get_mlp(
                input_size=input_size,
                output_size=n_loss_params,
                hidden_sizes=[16, 16],
                compile=True,
                dropout=0.0,
                output_activation=ExpActivation(),
            )
            instance._is_trained = False  # Mark as untrained
            
            return instance
        
        # Check if this is the new format with 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: state_dict with config
            # Reconstruct loss function if stored as name/module, otherwise try direct callable
            loss_fn = None
            if 'loss_function' in checkpoint and callable(checkpoint['loss_function']):
                loss_fn = checkpoint['loss_function']
            else:
                # Try reconstructing by name from the losses module
                lf_name = checkpoint.get('loss_function_name') or checkpoint.get('loss_function')
                lf_module = checkpoint.get('loss_function_module')
                if lf_name:
                    try:
                        from asf.predictors.utils import losses as _losses

                        loss_fn = getattr(_losses, lf_name)
                    except Exception:
                        loss_fn = None

            instance = cls(
                loss_function=loss_fn,
                n_loss_params=checkpoint['n_loss_params'],
                epochs=checkpoint.get('epochs', 200),
                batch_size=checkpoint.get('batch_size', 16),
                gradient_clip=checkpoint.get('gradient_clip', 1e-2),
                optimizer=checkpoint.get('optimizer', torch.optim.SGD),
                optimizer_kwargs=checkpoint.get('optimizer_kwargs', {}),
                lr_scheduler=checkpoint.get('lr_scheduler'),
                device=checkpoint.get('device', torch.device('cpu')),
            )
            
            # Build the model architecture from config
            model_config = checkpoint.get('model_config', {})
            if model_config.get('input_size') is not None:
                instance.model = get_mlp(
                    input_size=model_config['input_size'],
                    output_size=model_config['output_size'],
                    hidden_sizes=[16, 16],
                    compile=True,
                    dropout=0.0,
                    output_activation=ExpActivation(),
                )
                # Load the state_dict
                if checkpoint['model_state_dict'] is not None:
                    instance.model.load_state_dict(checkpoint['model_state_dict'])
                    instance._is_trained = True
                else:
                    instance._is_trained = False
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Old format: trying to save entire model (likely corrupted)
            # This format is problematic and needs manual recovery
            if loss_function is None or n_loss_params is None or input_size is None:
                raise RuntimeError(
                    f"Model at {path} was saved with old format and appears corrupted. "
                    "Please provide loss_function, n_loss_params, and input_size parameters "
                    "to attempt recovery, or retrain and save the model with the new version."
                )
            
            instance = cls(
                loss_function=loss_function,
                n_loss_params=n_loss_params,
            )
            
            # Try to use the saved model if it exists
            if checkpoint.get('model') is not None:
                instance.model = checkpoint['model']
                instance._is_trained = True
            else:
                # Build a new model from scratch
                instance.model = get_mlp(
                    input_size=input_size,
                    output_size=n_loss_params,
                    hidden_sizes=[16, 16],
                    compile=True,
                    dropout=0.0,
                    output_activation=ExpActivation(),
                )
                instance._is_trained = False
        else:
            # Old format: only state_dict saved
            if loss_function is None or n_loss_params is None or input_size is None:
                raise RuntimeError(
                    f"Model at {path} was saved with old format. "
                    "Please provide loss_function, n_loss_params, and input_size parameters, "
                    "or retrain and save the model with the new version."
                )
            
            # Create instance with provided parameters
            instance = cls(
                loss_function=loss_function,
                n_loss_params=n_loss_params,
            )
            
            # Build the model architecture
            instance.model = get_mlp(
                input_size=input_size,
                output_size=n_loss_params,
                hidden_sizes=[16, 16],
                compile=True,
                dropout=0.0,
                output_activation=ExpActivation(),
            )
            
            # Load the old state_dict
            instance.model.load_state_dict(checkpoint)
            instance._is_trained = True
        
        return instance
