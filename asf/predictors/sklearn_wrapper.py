"""
A generic wrapper for scikit-learn models.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from asf.predictors.abstract_predictor import AbstractPredictor


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
    ):
        super().__init__()
        self.model_class: Any = model_class(**(init_params or {}))

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
        self.model_class.fit(X, Y, sample_weight=sample_weight, **kwargs)

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
        return self.model_class.predict(X, **kwargs)

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
