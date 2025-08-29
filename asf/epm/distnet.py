"""
DistNet implementation for algorithm runtime distribution prediction.

Based on "Neural Networks for Predicting Algorithm Runtime Distributions" 
by Eggensperger et al. (https://arxiv.org/abs/1709.07615).

This module implements DistNet as a predictor for runtime distribution estimation
in the ASF framework.
"""

from typing import Any, Optional, Dict, Tuple
from functools import partial
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
    from ConfigSpace.hyperparameters import Hyperparameter
    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from asf.predictors.abstract_predictor import AbstractPredictor


if TORCH_AVAILABLE:

    class DistNetArchitecture(nn.Module):
        """
        Neural network architecture for predicting algorithm runtime distributions.
        
        Predicts parameters of log-normal distribution for algorithm runtimes.
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_sizes: list = [128, 64, 32],
            dropout: float = 0.1,
            activation: str = 'relu',
        ):
            super().__init__()
            
            # Build the network layers
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'elu':
                    layers.append(nn.ELU())
                    
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                    
                prev_size = hidden_size
            
            self.backbone = nn.Sequential(*layers)
            
            # Output heads for log-normal distribution parameters
            # Predict log(mean) and log(variance) for numerical stability
            self.log_mean_head = nn.Linear(prev_size, 1)
            self.log_var_head = nn.Linear(prev_size, 1)
            
        def forward(self, x):
            """Forward pass predicting log-normal distribution parameters."""
            features = self.backbone(x)
            
            # Predict log-space parameters for numerical stability
            log_mean = self.log_mean_head(features)
            log_var = self.log_var_head(features)
            
            return log_mean, log_var


    class DistNet(AbstractPredictor):
        """
        DistNet predictor for algorithm runtime distribution prediction.
        
        Predicts parameters of log-normal distributions for algorithm runtimes,
        providing uncertainty quantification for runtime predictions.
        """
        
        def __init__(
            self,
            hidden_sizes: list = [128, 64, 32],
            dropout: float = 0.1,
            activation: str = 'relu',
            learning_rate: float = 0.001,
            batch_size: int = 64,
            epochs: int = 100,
            device: str = 'cpu',
            seed: int = 42,
            early_stopping_patience: int = 10,
            **kwargs
        ):
            """
            Initialize DistNet predictor for runtime prediction.
            
            Parameters
            ----------
            hidden_sizes : list
                Sizes of hidden layers
            dropout : float
                Dropout probability
            activation : str
                Activation function ('relu', 'tanh', 'elu')
            learning_rate : float
                Learning rate for optimizer
            batch_size : int
                Training batch size
            epochs : int
                Number of training epochs
            device : str
                Device to use ('cpu' or 'cuda')
            seed : int
                Random seed for reproducibility
            early_stopping_patience : int
                Patience for early stopping
            """
            super().__init__(**kwargs)
            
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Please install it with: pip install torch")
            
            self.hidden_sizes = hidden_sizes
            self.dropout = dropout
            self.activation = activation
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.device = device
            self.seed = seed
            self.early_stopping_patience = early_stopping_patience
            
            # Set random seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            self.model = None
            self.optimizer = None
            self.scaler = None
            self.feature_scaler = None
            
        def _create_model(self, input_size: int):
            """Create the DistNet model."""
            self.model = DistNetArchitecture(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                dropout=self.dropout,
                activation=self.activation
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            
        def _compute_loss(self, log_mean_pred, log_var_pred, log_runtime_targets):
            """
            Compute negative log-likelihood loss for log-normal distribution.
            
            For log-normal distribution with parameters μ and σ²:
            log p(x|μ,σ²) = -0.5*log(2π) - 0.5*log(σ²) - 0.5*(log(x)-μ)²/σ²
            
            We predict log(μ) and log(σ²) for numerical stability.
            """
            # Convert predictions back to distribution parameters
            mu = log_mean_pred  # This is already log(mean) of the underlying normal
            log_sigma_sq = log_var_pred  # This is log(variance) of the underlying normal
            
            # Negative log-likelihood for log-normal distribution
            # Note: log_runtime_targets are already log-transformed runtimes
            nll = 0.5 * (np.log(2 * np.pi) + log_sigma_sq + 
                        (log_runtime_targets - mu) ** 2 / torch.exp(log_sigma_sq))
            
            return torch.mean(nll)
            
        def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None) -> "DistNet":
            """
            Fit the DistNet model to runtime data.
            
            Parameters
            ----------
            X : pd.DataFrame
                Feature matrix (algorithm/instance features)
            y : pd.DataFrame
                Runtime values (will be log-transformed)
            sample_weight : optional
                Sample weights (not supported)
                
            Returns
            -------
            self : DistNet
                Fitted predictor
            """
            if sample_weight is not None:
                warnings.warn("Sample weights are not supported by DistNet and will be ignored")
                
            # Preprocess features
            self.feature_scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.feature_scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
            # Handle missing values
            self.scaler = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_scaled),
                index=X_scaled.index,
                columns=X_scaled.columns
            )
            
            # Log-transform runtime targets (typical for runtime distributions)
            # Add small epsilon to handle zero runtimes
            y_log = np.log(np.maximum(y.values.flatten(), 1e-6))
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            y_tensor = torch.FloatTensor(y_log.reshape(-1, 1)).to(self.device)
            
            # Create model
            if self.model is None:
                self._create_model(X_processed.shape[1])
                
            # Training loop
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
            
            best_loss = float('inf')
            patience_counter = 0
            
            self.model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()
                    log_mean_pred, log_var_pred = self.model(batch_X)
                    loss = self._compute_loss(log_mean_pred, log_var_pred, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.early_stopping_patience:
                    break
                    
            return self
            
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            """
            Predict runtime (point estimates).
            
            Parameters
            ----------
            X : pd.DataFrame
                Feature matrix
                
            Returns
            -------
            predictions : np.ndarray
                Predicted runtime values (median of log-normal distribution)
            """
            if self.model is None:
                raise ValueError("Model must be fitted before making predictions")
                
            # Preprocess features
            X_scaled = pd.DataFrame(
                self.feature_scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
            X_processed = pd.DataFrame(
                self.scaler.transform(X_scaled),
                index=X_scaled.index,
                columns=X_scaled.columns
            )
            
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                log_mean_pred, log_var_pred = self.model(X_tensor)
                
                # For log-normal distribution, median = exp(μ) where μ is the mean of log-space
                median_runtime = torch.exp(log_mean_pred).cpu().numpy().flatten()
                
            return median_runtime
            
        def predict_distribution(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
            """
            Predict runtime distribution parameters.
            
            Parameters
            ----------
            X : pd.DataFrame
                Feature matrix
                
            Returns
            -------
            results : dict
                Dictionary containing 'mean', 'variance', 'median', 'std'
            """
            if self.model is None:
                raise ValueError("Model must be fitted before making predictions")
                
            # Preprocess features
            X_scaled = pd.DataFrame(
                self.feature_scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
            X_processed = pd.DataFrame(
                self.scaler.transform(X_scaled),
                index=X_scaled.index,
                columns=X_scaled.columns
            )
            
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                log_mean_pred, log_var_pred = self.model(X_tensor)
                
                # Convert to numpy
                mu = log_mean_pred.cpu().numpy().flatten()  # Mean in log-space
                log_sigma_sq = log_var_pred.cpu().numpy().flatten()  # Log variance in log-space
                sigma_sq = np.exp(log_sigma_sq)  # Variance in log-space
                
                # For log-normal distribution with log-space parameters μ and σ²:
                # Mean = exp(μ + σ²/2)
                # Variance = exp(2μ + σ²) * (exp(σ²) - 1)
                # Median = exp(μ)
                
                runtime_mean = np.exp(mu + sigma_sq / 2)
                runtime_variance = np.exp(2 * mu + sigma_sq) * (np.exp(sigma_sq) - 1)
                runtime_median = np.exp(mu)
                runtime_std = np.sqrt(runtime_variance)
                
            return {
                'mean': runtime_mean,
                'variance': runtime_variance,
                'median': runtime_median,
                'std': runtime_std,
                'log_mean': mu,
                'log_variance': sigma_sq
            }
            
        def save(self, file_path: str) -> None:
            """Save the model to a file."""
            if self.model is None:
                raise ValueError("No model to save")
                
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'config': {
                    'hidden_sizes': self.hidden_sizes,
                    'dropout': self.dropout,
                    'activation': self.activation,
                    'learning_rate': self.learning_rate,
                    'input_size': self.model.backbone[0].in_features,
                }
            }, file_path)
            
        def load(self, file_path: str) -> None:
            """Load the model from a file."""
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            
            # Restore configuration
            config = checkpoint['config']
            for key, value in config.items():
                if key != 'input_size':
                    setattr(self, key, value)
                    
            # Create and load model
            self._create_model(config['input_size'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler = checkpoint['scaler']
            self.feature_scaler = checkpoint['feature_scaler']
            
        if CONFIGSPACE_AVAILABLE:
            @staticmethod
            def get_configuration_space(
                cs: Optional[ConfigurationSpace] = None,
                pre_prefix: str = "",
                parent_param: Optional[Hyperparameter] = None,
                parent_value: Optional[str] = None,
            ) -> ConfigurationSpace:
                """Get configuration space for hyperparameter optimization."""
                if cs is None:
                    cs = ConfigurationSpace()
                    
                prefix = f"{pre_prefix}:DistNet" if pre_prefix else "DistNet"
                
                # Network architecture parameters
                hidden_size_1 = Integer(f"{prefix}:hidden_size_1", (32, 256), default=128, log=True)
                hidden_size_2 = Integer(f"{prefix}:hidden_size_2", (16, 128), default=64, log=True)
                hidden_size_3 = Integer(f"{prefix}:hidden_size_3", (8, 64), default=32, log=True)
                
                dropout = Float(f"{prefix}:dropout", (0.0, 0.3), default=0.1)
                activation = Categorical(f"{prefix}:activation", ['relu', 'tanh', 'elu'], default='relu')
                
                # Training parameters
                learning_rate = Float(f"{prefix}:learning_rate", (1e-5, 1e-2), default=1e-3, log=True)
                batch_size = Integer(f"{prefix}:batch_size", (16, 128), default=64, log=True)
                
                params = [
                    hidden_size_1, hidden_size_2, hidden_size_3,
                    dropout, activation, learning_rate, batch_size
                ]
                
                if parent_param is not None:
                    from ConfigSpace import EqualsCondition
                    conditions = [
                        EqualsCondition(param, parent_param, parent_value)
                        for param in params
                    ]
                    cs.add_hyperparameters(params)
                    cs.add_conditions(conditions)
                else:
                    cs.add_hyperparameters(params)
                    
                return cs
                
            @staticmethod
            def get_from_configuration(
                configuration: Dict[str, Any], 
                pre_prefix: str = "", 
                **kwargs
            ) -> partial:
                """Create DistNet instance from configuration."""
                prefix = f"{pre_prefix}:DistNet" if pre_prefix else "DistNet"
                
                config_params = {
                    'hidden_sizes': [
                        configuration[f"{prefix}:hidden_size_1"],
                        configuration[f"{prefix}:hidden_size_2"],
                        configuration[f"{prefix}:hidden_size_3"]
                    ],
                    'dropout': configuration[f"{prefix}:dropout"],
                    'activation': configuration[f"{prefix}:activation"],
                    'learning_rate': configuration[f"{prefix}:learning_rate"],
                    'batch_size': configuration[f"{prefix}:batch_size"],
                    **kwargs
                }
                
                return partial(DistNet, **config_params)

else:
    # Placeholder when PyTorch is not available
    class DistNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it with: pip install torch")
            
        def fit(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it with: pip install torch")
            
        def predict(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it with: pip install torch")
            
        def save(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it with: pip install torch")
            
        def load(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it with: pip install torch")