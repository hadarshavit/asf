from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from asf.predictors.utils.losses import lognorm_loss
from asf.predictors.utils.mlp import ExpActivation

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Constant,
        EqualsCondition,
        Float,
        Integer,
        Categorical
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class XGBDistNet:
    PREFIX = "xgb_distnet"

    def __init__(
        self,
        loss_function=lognorm_loss,
        n_loss_params: int = 2,
        batch_size: int | None = 1000,
        output_activation=ExpActivation(),
        **kwargs
    ):
        self.loss_function = loss_function
        self.n_loss_params = n_loss_params
        self.batch_size = batch_size
        self.output_activation = output_activation
        self.kwargs = kwargs

    def objective(self, data, preds):
        if self.n_loss_params == 1:
            preds = preds.reshape(-1, 1)

        if self.batch_size is None:
            # Compute gradients and Hessians w.r.t. raw margins (pre-activation)
            raw = torch.from_numpy(preds).requires_grad_(True).float()
            target = torch.from_numpy(data.reshape(-1, 1)).float()
            activated = self.output_activation(raw)
            activated = torch.clamp(activated, min=1e-12)

            loss = self.loss_function(target, activated)

            # Gradient wrt raw margins
            grad_raw = torch.autograd.grad(loss, inputs=raw, create_graph=True)[0]

            # Diagonal Hessian wrt raw margins
            if grad_raw.dim() == 1:
                hess_raw = torch.autograd.grad(
                    grad_raw.sum(), inputs=raw, retain_graph=True
                )[0]
            else:
                h_cols = []
                for d in range(grad_raw.shape[1]):
                    gcol_sum = grad_raw[:, d].sum()
                    h_col_full = torch.autograd.grad(
                        gcol_sum, inputs=raw, retain_graph=True
                    )[0]
                    h_cols.append(h_col_full[:, d])
                hess_raw = torch.stack(h_cols, dim=1)

            # Return 2D shapes (n_samples, n_targets) as required by XGBoost 2.1+
            grad = grad_raw.detach().numpy()
            hess = hess_raw.detach().numpy()

            return grad, hess
        else:
            n_samples = len(data)
            grad = torch.zeros_like(torch.from_numpy(preds))
            hess = torch.zeros_like(torch.from_numpy(preds))

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)

                # raw margins for this batch
                batch_raw = (
                    torch.from_numpy(preds[start_idx:end_idx])
                    .requires_grad_(True)
                    .float()
                )
                batch_activated = self.output_activation(batch_raw)
                batch_activated = torch.clamp(batch_activated, min=1e-12)
                batch_target = torch.from_numpy(
                    data[start_idx:end_idx].reshape(-1, 1)
                ).float()
                batch_loss = self.loss_function(batch_target, batch_activated)

                batch_grad_raw = torch.autograd.grad(
                    batch_loss, inputs=batch_raw, create_graph=True
                )[0]

                if batch_grad_raw.dim() == 1:
                    batch_hess_raw = torch.autograd.grad(
                        batch_grad_raw.sum(), inputs=batch_raw, retain_graph=True
                    )[0]
                else:
                    h_cols = []
                    for d in range(batch_grad_raw.shape[1]):
                        gcol_sum = batch_grad_raw[:, d].sum()
                        h_col_full = torch.autograd.grad(
                            gcol_sum, inputs=batch_raw, retain_graph=True
                        )[0]
                        h_cols.append(h_col_full[:, d])
                    batch_hess_raw = torch.stack(h_cols, dim=1)

                grad[start_idx:end_idx] = batch_grad_raw.detach()
                hess[start_idx:end_idx] = batch_hess_raw.detach()

            # Return 2D arrays
            grad = grad.numpy()
            hess = hess.numpy()

            return grad, hess

    def fit(
        self,
        X: pd.DataFrame | pd.Series | list,
        y: pd.Series | list,
    ):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        X = np.concatenate([[x for i in range(100)] for x in X])
        y = y.flatten()
        self.model = xgb.XGBRegressor(
            objective=self.objective,
            eval_metric=self.loss_function,
            num_target=self.n_loss_params,
            **self.kwargs,
        )

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame | pd.Series | list) -> torch.Tensor:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        predictions = self.model.predict(X)
        # Ensure strictly positive, numerically stable distribution parameters
        preds_tensor = self.output_activation(torch.from_numpy(predictions))
        preds_tensor = torch.clamp(preds_tensor, min=1e-12)
        predictions = preds_tensor.numpy()

        print(predictions.shape)
        print(self.n_loss_params)
        if self.n_loss_params == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions
    
    def save(self, path: str):
        """Save the XGBDistNet model to a file.

        Parameters
        ----------
        path : str
            The file path where the model will be saved.
        """
        self.model.save_model(path)

    def load(self, path: str):
        """Load the XGBDistNet model from a file.

        Parameters
        ----------
        path : str
            The file path from which the model will be loaded.
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the XGBoost regressor.

        Parameters
        ----------
        cs : ConfigurationSpace, optional
        The configuration space to add the parameters to. If None, a new ConfigurationSpace will be created.

        Returns
        -------
        ConfigurationSpace
        The configuration space with the XGBoost parameters.
        """
        if cs is None:
            cs = ConfigurationSpace(name="XGBoostRegressor")

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBDistNet.PREFIX}"
        else:
            prefix = XGBDistNet.PREFIX

        booster = Constant(f"{prefix}:booster", "gbtree")
        n_estimators = Integer(
            f"{prefix}:n_estimators",
            (10, 2000),
            log=True,
            default=100,
        )
        max_depth = Integer(
            f"{prefix}:max_depth",
            (1, 20),
            log=False,
            default=13,
        )
        min_child_weight = Integer(
            f"{prefix}:min_child_weight",
            (1, 100),
            log=True,
            default=39,
        )
        colsample_bytree = Float(
            f"{prefix}:colsample_bytree",
            (0.0, 1.0),
            log=False,
            default=0.2545374925231651,
        )
        colsample_bylevel = Float(
            f"{prefix}:colsample_bylevel",
            (0.0, 1.0),
            log=False,
            default=0.6909224923784677,
        )
        lambda_param = Float(
            f"{prefix}:lambda",
            (0.001, 1000),
            log=True,
            default=31.393252465064943,
        )
        alpha = Float(
            f"{prefix}:alpha",
            (0.001, 1000),
            log=True,
            default=0.24167936088332426,
        )
        learning_rate = Float(
            f"{prefix}:learning_rate",
            (0.001, 0.1),
            log=True,
            default=0.008237525103357958,
        )
        multi_strategy = Categorical(
            f"{prefix}:multi_strategy", ["one_output_per_tree", "multi_output_tree"]
        )

        params = [
            booster,
            n_estimators,
            max_depth,
            min_child_weight,
            colsample_bytree,
            colsample_bylevel,
            lambda_param,
            alpha,
            learning_rate,
            multi_strategy
        ]
        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)

        return cs

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any],
        pre_prefix: str = "",
        input_size: int = None,
        **kwargs,
    ) -> Callable[..., "XGBDistNet"]:
        """
        Create an XGBoostRegressorWrapper from a configuration.

        Parameters
        ----------
        configuration : dict
        The configuration dictionary.
        additional_params : dict, optional
        Additional parameters to include in the configuration.

        Returns
        -------
        Callable[..., XGBoostRegressorWrapper]
        A callable that initializes the wrapper with the given configuration.
        """
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{XGBDistNet.PREFIX}"
        else:
            prefix = XGBDistNet.PREFIX

        xgb_params = {
            "booster": configuration[f"{prefix}:booster"],
            "n_estimators": configuration[f"{prefix}:n_estimators"],
            "max_depth": configuration[f"{prefix}:max_depth"],
            "min_child_weight": configuration[f"{prefix}:min_child_weight"],
            "colsample_bytree": configuration[f"{prefix}:colsample_bytree"],
            "colsample_bylevel": configuration[f"{prefix}:colsample_bylevel"],
            "lambda": configuration[f"{prefix}:lambda"],
            "alpha": configuration[f"{prefix}:alpha"],
            "learning_rate": configuration[f"{prefix}:learning_rate"],
            "multi_strategy": configuration[f"{prefix}:multi_strategy"],
            **kwargs,
        }

        return partial(XGBDistNet, **xgb_params)
