"""
A generic wrapper for scikit-learn models.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.utils.random_state import random_state_kwargs


class SklearnWrapper(AbstractPredictor):
    """
    A generic wrapper for scikit-learn models.

    This class allows scikit-learn models to be used with the ASF framework.

    Parameters
    ----------
    model_class : type[BaseEstimator]
        A scikit-learn model class.
    init_params : dict[str, Any], optional
        Initialization parameters for the scikit-learn model (default is {}).
    """

    def __init__(
        self,
        model_class: Any,
        init_params: dict[str, Any] | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(random_state=random_state)
        params = dict(init_params) if isinstance(init_params, dict) else {}
        params.update(kwargs)

        # Filter out parameters that are for the selector/pipeline and not the model
        model_params = {
            k: v
            for k, v in params.items()
            if k not in ["budget", "maximize", "n_algorithms"]
        }

        model_params = {
            **random_state_kwargs(model_class, random_state),
            **model_params,
        }
        self.model_class: Any = model_class(**model_params)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose constructor parameters for scikit-learn cloning."""
        params = self.model_class.get_params(deep=deep)
        if type(self) is SklearnWrapper:
            params = {"model_class": type(self.model_class), **params}
        return params

    def set_params(self, **params: Any) -> SklearnWrapper:
        """Update the wrapped estimator through sklearn's parameter API."""
        if type(self) is SklearnWrapper and "model_class" in params:
            self.model_class = params.pop("model_class")()
        self.model_class.set_params(**params)
        self.random_state = getattr(self.model_class, "random_state", self.random_state)
        return self

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        Y : np.ndarray
            Target values of shape (n_samples,).
        sample_weight : np.ndarray or None, default=None
            Sample weights of shape (n_samples,).
        **kwargs : Any
            Additional keyword arguments for the scikit-learn model's `fit` method.
        """
        X_in = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        Y_in = Y.to_numpy() if isinstance(Y, (pd.DataFrame, pd.Series)) else Y
        sw_in = (
            sample_weight.to_numpy()
            if isinstance(sample_weight, (pd.DataFrame, pd.Series))
            else sample_weight
        )
        if sw_in is not None:
            kwargs["sample_weight"] = sw_in
        self.model_class.fit(X_in, Y_in, **kwargs)

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on of shape (n_samples, n_features).
        **kwargs : Any
            Additional keyword arguments for the scikit-learn model's `predict` method.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        X_in = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        return self.model_class.predict(X_in, **kwargs)

    def save(self, file_path: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> SklearnWrapper:
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        SklearnWrapper
            The loaded model.
        """
        return joblib.load(file_path)
