"""
DistNet implementation for deep probability estimation.

Based on the paper "Deep Probability Estimation" (https://arxiv.org/abs/1709.07615).
This module implements DistNet as a predictor for the ASF framework.
"""

from typing import Any, Optional, Dict
from functools import partial

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

from asf.predictors.abstract_predictor import AbstractPredictor


if TORCH_AVAILABLE:

    class DistNetArchitecture(nn.Module):
        """
        Deep probability estimation network architecture.
        
        This implements a neural network designed for probability estimation
        with multiple output distributions and uncertainty quantification.
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_sizes: list = [256, 128, 64],
            output_size: int = 1,
            dropout: float = 0.2,
            activation: str = 'relu',
            use_batch_norm: bool = True,
            num_components: int = 5,  # For mixture distributions
        ):
            super().__init__()
            
            self.input_size = input_size
            self.output_size = output_size
            self.num_components = num_components
            
            # Build the main network
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                
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
            
            # Output heads for probability estimation
            # Mean prediction
            self.mean_head = nn.Linear(prev_size, output_size)
            
            # Variance prediction (log scale for numerical stability)
            self.log_var_head = nn.Linear(prev_size, output_size)
            
            # Mixture weights (for mixture of experts)
            self.mixture_weights = nn.Linear(prev_size, num_components)
            
            # Component means and variances
            self.component_means = nn.Linear(prev_size, num_components * output_size)
            self.component_log_vars = nn.Linear(prev_size, num_components * output_size)
            
        def forward(self, x):
            """Forward pass through the network."""
            features = self.backbone(x)
            
            # Basic predictions
            mean = self.mean_head(features)
            log_var = self.log_var_head(features)
            
            # Mixture components
            mixture_weights = F.softmax(self.mixture_weights(features), dim=-1)
            component_means = self.component_means(features).view(-1, self.num_components, self.output_size)
            component_log_vars = self.component_log_vars(features).view(-1, self.num_components, self.output_size)
            
            return {
                'mean': mean,
                'log_var': log_var,
                'mixture_weights': mixture_weights,
                'component_means': component_means,
                'component_log_vars': component_log_vars
            }


    class DistNet(AbstractPredictor):
        """
        DistNet predictor implementing deep probability estimation.
        
        This class provides a neural network approach to probability estimation
        with uncertainty quantification and mixture distributions.
        """
        
        def __init__(
            self,
            hidden_sizes: list = [256, 128, 64],
            dropout: float = 0.2,
            activation: str = 'relu',
            use_batch_norm: bool = True,
            num_components: int = 5,
            learning_rate: float = 0.001,
            batch_size: int = 64,
            epochs: int = 100,
            device: str = 'cpu',
            seed: int = 42,
            early_stopping_patience: int = 10,
            mixture_loss_weight: float = 0.1,
            **kwargs
        ):
            """
            Initialize DistNet predictor.
            
            Parameters
            ----------
            hidden_sizes : list
                Sizes of hidden layers
            dropout : float
                Dropout probability
            activation : str
                Activation function ('relu', 'tanh', 'elu')
            use_batch_norm : bool
                Whether to use batch normalization
            num_components : int
                Number of mixture components
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
            mixture_loss_weight : float
                Weight for mixture loss component
            """
            super().__init__(**kwargs)
            
            assert TORCH_AVAILABLE, "PyTorch is not available. Please install it."
            
            self.hidden_sizes = hidden_sizes
            self.dropout = dropout
            self.activation = activation
            self.use_batch_norm = use_batch_norm
            self.num_components = num_components
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.device = device
            self.seed = seed
            self.early_stopping_patience = early_stopping_patience
            self.mixture_loss_weight = mixture_loss_weight
            
            # Set random seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            self.model = None
            self.optimizer = None
            self.scaler = None
            
        def _create_model(self, input_size: int, output_size: int = 1):
            """Create the DistNet model."""
            self.model = DistNetArchitecture(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=output_size,
                dropout=self.dropout,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                num_components=self.num_components
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            
        def _compute_loss(self, outputs, targets):
            """Compute the loss for training."""
            mean_pred = outputs['mean']
            log_var_pred = outputs['log_var']
            
            # Basic Gaussian negative log-likelihood
            var_pred = torch.exp(log_var_pred)
            nll_loss = 0.5 * (torch.log(2 * np.pi * var_pred) + 
                             (targets - mean_pred) ** 2 / var_pred)
            nll_loss = torch.mean(nll_loss)
            
            # Mixture loss
            mixture_weights = outputs['mixture_weights']
            component_means = outputs['component_means']
            component_log_vars = outputs['component_log_vars']
            component_vars = torch.exp(component_log_vars)
            
            # Compute likelihood for each component
            targets_expanded = targets.unsqueeze(1).expand(-1, self.num_components, -1)
            component_nll = 0.5 * (torch.log(2 * np.pi * component_vars) + 
                                  (targets_expanded - component_means) ** 2 / component_vars)
            
            # Weighted mixture likelihood
            component_likelihood = torch.exp(-component_nll)
            mixture_likelihood = torch.sum(mixture_weights.unsqueeze(-1) * component_likelihood, dim=1)
            mixture_loss = -torch.mean(torch.log(mixture_likelihood + 1e-8))
            
            # Total loss
            total_loss = nll_loss + self.mixture_loss_weight * mixture_loss
            
            return total_loss
            
        def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None) -> "DistNet":
            """
            Fit the DistNet model.
            
            Parameters
            ----------
            X : pd.DataFrame
                Feature matrix
            y : pd.DataFrame
                Target values
            sample_weight : optional
                Sample weights (not supported)
                
            Returns
            -------
            self : DistNet
                Fitted predictor
            """
            if sample_weight is not None:
                raise ValueError("Sample weights are not supported by DistNet")
                
            # Preprocess features
            self.scaler = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            y_tensor = torch.FloatTensor(y.values.reshape(-1, 1)).to(self.device)
            
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
                    outputs = self.model(batch_X)
                    loss = self._compute_loss(outputs, batch_y)
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
            Predict using the fitted model.
            
            Parameters
            ----------
            X : pd.DataFrame
                Feature matrix
                
            Returns
            -------
            predictions : np.ndarray
                Predicted values (mean predictions)
            """
            if self.model is None:
                raise ValueError("Model must be fitted before making predictions")
                
            # Preprocess features
            X_processed = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
            
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = outputs['mean'].cpu().numpy().flatten()
                
            return predictions
            
        def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
            """
            Predict with uncertainty quantification.
            
            Parameters
            ----------
            X : pd.DataFrame
                Feature matrix
                
            Returns
            -------
            results : dict
                Dictionary containing 'mean', 'variance', and 'mixture_weights'
            """
            if self.model is None:
                raise ValueError("Model must be fitted before making predictions")
                
            # Preprocess features
            X_processed = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
            
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                
            return {
                'mean': outputs['mean'].cpu().numpy().flatten(),
                'variance': torch.exp(outputs['log_var']).cpu().numpy().flatten(),
                'mixture_weights': outputs['mixture_weights'].cpu().numpy(),
                'component_means': outputs['component_means'].cpu().numpy(),
                'component_variances': torch.exp(outputs['component_log_vars']).cpu().numpy()
            }
            
        def save(self, file_path: str) -> None:
            """Save the model to a file."""
            if self.model is None:
                raise ValueError("No model to save")
                
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler,
                'config': {
                    'hidden_sizes': self.hidden_sizes,
                    'dropout': self.dropout,
                    'activation': self.activation,
                    'use_batch_norm': self.use_batch_norm,
                    'num_components': self.num_components,
                    'learning_rate': self.learning_rate,
                }
            }, file_path)
            
        def load(self, file_path: str) -> None:
            """Load the model from a file."""
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            
            # Restore configuration
            config = checkpoint['config']
            for key, value in config.items():
                setattr(self, key, value)
                
            # Create and load model
            if self.model is None:
                # We need input size, but it's not saved. This is a limitation.
                raise ValueError("Cannot load model without knowing input size. "
                               "Consider saving input size in the checkpoint.")
                
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler = checkpoint['scaler']
            
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
                hidden_size_1 = Integer(f"{prefix}:hidden_size_1", (64, 512), default=256, log=True)
                hidden_size_2 = Integer(f"{prefix}:hidden_size_2", (32, 256), default=128, log=True)
                hidden_size_3 = Integer(f"{prefix}:hidden_size_3", (16, 128), default=64, log=True)
                
                dropout = Float(f"{prefix}:dropout", (0.0, 0.5), default=0.2)
                activation = Categorical(f"{prefix}:activation", ['relu', 'tanh', 'elu'], default='relu')
                use_batch_norm = Categorical(f"{prefix}:use_batch_norm", [True, False], default=True)
                num_components = Integer(f"{prefix}:num_components", (2, 10), default=5)
                
                # Training parameters
                learning_rate = Float(f"{prefix}:learning_rate", (1e-5, 1e-1), default=1e-3, log=True)
                batch_size = Integer(f"{prefix}:batch_size", (16, 256), default=64, log=True)
                mixture_loss_weight = Float(f"{prefix}:mixture_loss_weight", (0.01, 1.0), default=0.1, log=True)
                
                params = [
                    hidden_size_1, hidden_size_2, hidden_size_3,
                    dropout, activation, use_batch_norm, num_components,
                    learning_rate, batch_size, mixture_loss_weight
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
                    'use_batch_norm': configuration[f"{prefix}:use_batch_norm"],
                    'num_components': configuration[f"{prefix}:num_components"],
                    'learning_rate': configuration[f"{prefix}:learning_rate"],
                    'batch_size': configuration[f"{prefix}:batch_size"],
                    'mixture_loss_weight': configuration[f"{prefix}:mixture_loss_weight"],
                    **kwargs
                }
                
                return partial(DistNet, **config_params)

else:
    # Placeholder when PyTorch is not available
    class DistNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it to use DistNet.")
            
        def fit(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it to use DistNet.")
            
        def predict(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it to use DistNet.")
            
        def save(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it to use DistNet.")
            
        def load(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install it to use DistNet.")