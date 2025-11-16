from asf.predictors.utils.losses import lognorm_loss
import xgboost as xgb
import pandas as pd
import torch
from asf.predictors.utils.mlp import ExpActivation
import numpy as np


class XGBDistNet:
    def __init__(
        self,
        xgb_kwargs: dict = {},
        loss_function=lognorm_loss,
        n_loss_params: int = 2,
        batch_size: int | None = 1000,
        output_activation=ExpActivation(),
    ):
        self.xgb_kwargs = xgb_kwargs
        self.loss_function = loss_function
        self.n_loss_params = n_loss_params
        self.batch_size = batch_size
        self.output_activation = output_activation

    def objective(self, data, preds):
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
        X = np.concatenate([[x for i in range(100)] for x in X])
        y = y.flatten()
        self.model = xgb.XGBRegressor(
            objective=self.objective,
            eval_metric=self.loss_function,
            num_target=self.n_loss_params,
            **self.xgb_kwargs,
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
        return predictions
