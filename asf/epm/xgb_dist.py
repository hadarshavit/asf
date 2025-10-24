from asf.predictors.utils.losses import lognorm_loss
import xgboost as xgb
import pandas as pd
import torch


class XGBDistNet:
    def __init__(
        self,
        xgb_kwargs: dict = {},
        loss_function=lognorm_loss,
        n_loss_params: int = 2,
        batch_size: int | None = 1000,
    ):
        self.xgb_kwargs = xgb_kwargs
        self.loss_function = loss_function
        self.n_loss_params = n_loss_params
        self.batch_size = batch_size

    def objective(self, data, preds):
        if self.batch_size is None:
            preds_tensor = torch.from_numpy(preds).requires_grad_(True).float()
            target = torch.from_numpy(data.reshape(-1, 1)).float()
            loss = self.loss_function(target, preds_tensor)

            grad = torch.autograd.grad(loss, inputs=preds_tensor, create_graph=True)[0]

            if grad.dim() == 1:
                hess = torch.autograd.grad(
                    grad.sum(), inputs=preds_tensor, retain_graph=True
                )[0]
            else:
                h_cols = []
                for d in range(grad.shape[1]):
                    gcol_sum = grad[:, d].sum()
                    h_col_full = torch.autograd.grad(
                        gcol_sum, inputs=preds_tensor, retain_graph=True
                    )[0]
                    h_cols.append(h_col_full[:, d])
                hess = torch.stack(h_cols, dim=1)

            grad = grad.detach().numpy().flatten()
            hess = hess.detach().numpy().flatten()

            return grad, hess

        n_samples = len(data)
        grad = torch.zeros_like(torch.from_numpy(preds))
        hess = torch.zeros_like(torch.from_numpy(preds))

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)

            batch_preds = torch.from_numpy(preds[start_idx:end_idx]).requires_grad_(
                True
            )
            batch_target = torch.from_numpy(data[start_idx:end_idx].reshape(-1, 1))
            batch_loss = self.loss_function(batch_target, batch_preds)

            batch_grad = torch.autograd.grad(
                batch_loss, inputs=batch_preds, create_graph=True
            )[0]

            if batch_grad.dim() == 1:
                batch_hess = torch.autograd.grad(
                    batch_grad.sum(), inputs=batch_preds, retain_graph=True
                )[0]
            else:
                h_cols = []
                for d in range(batch_grad.shape[1]):
                    gcol_sum = batch_grad[:, d].sum()
                    h_col_full = torch.autograd.grad(
                        gcol_sum, inputs=batch_preds, retain_graph=True
                    )[0]
                    h_cols.append(h_col_full[:, d])
                batch_hess = torch.stack(h_cols, dim=1)

            grad[start_idx:end_idx] = batch_grad.detach()
            hess[start_idx:end_idx] = batch_hess.detach()

        grad = grad.numpy().flatten()
        hess = hess.numpy().flatten()

        return grad, hess

    def fit(
        self,
        X: pd.DataFrame | pd.Series | list,
        y: pd.Series | list,
    ):
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

        return predictions
