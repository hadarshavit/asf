from sklearn.base import ClassifierMixin
from asf.predictors.abstract_predictor import AbstractPredictor
from typing import Any
import numpy as np


class SklearnWrapper(AbstractPredictor):
    """
    A generic wrapper for scikit-learn models.

    This class allows scikit-learn models to be used with the ASF framework.

    Methods
    -------
    fit(X, Y, sample_weight=None, **kwargs)
        Fit the model to the data.
    predict(X, **kwargs)
        Predict using the model.
    save(file_path)
        Save the model to a file.
    load(file_path)
        Load the model from a file.
    """

    def __init__(self, model_class: ClassifierMixin, init_params: dict[str, Any] = {}):
        """
        Initialize the wrapper with a scikit-learn model.

        Parameters
        ----------
        model_class : ClassifierMixin
            A scikit-learn model class.
        init_params : dict, optional
            Initialization parameters for the scikit-learn model (default is an empty dictionary).
        """
        self.model_class = model_class(**init_params)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray = None,
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
        sample_weight : np.ndarray, optional
            Sample weights of shape (n_samples,) (default is None).
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
        import joblib

        joblib.dump(self, file_path)

    def load(self, file_path: str) -> "SklearnWrapper":
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
        import joblib

        return joblib.load(file_path)
