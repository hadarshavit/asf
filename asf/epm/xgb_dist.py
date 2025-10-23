from asf.predictors.utils.losses import lognorm_loss
import xgboost as xgb
import pandas as pd
import torch


class XGBDistNet:
    def __init__(
        self,
        xgb_kwargs: dict = {},
        loss_function=lognorm_loss,
        n_feats: int = 1,
        n_loss_params: int = 2,
    ):
        self.xgb_kwargs = xgb_kwargs
        self.loss_function = loss_function
        self.n_feats = n_feats
        self.n_loss_params = n_loss_params

    def objective(self, data, preds):
        preds = torch.from_numpy(preds).requires_grad_(True)
        target = torch.from_numpy(data.reshape(-1, 1))
        loss = self.loss_function(target, preds)

        grad = torch.autograd.grad(loss, inputs=preds, create_graph=True)[0]
        hess = torch.autograd.grad(grad.sum(), inputs=preds, retain_graph=True)[0]

        grad = grad.detach().numpy().flatten()
        hess = hess.detach().numpy().flatten()

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
