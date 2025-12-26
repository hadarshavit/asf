from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    """
    Abstract base class for all predictors.

    Provides a framework for fitting, predicting, and managing model persistence.

    Methods
    -------
    fit(X, Y, **kwargs)
        Fit the model to the data.
    predict(X, **kwargs)
        Predict using the model.
    save(file_path)
        Save the model to a file.
    load(file_path)
        Load the model from a file.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the predictor.
        """
        pass

    @abstractmethod
    def fit(self, X: Any, Y: Any, **kwargs: Any) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : Any
            Training data.
        Y : Any
            Target values.
        kwargs : Any
            Additional arguments for fitting the model.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Predict using the model.

        Parameters
        ----------
        X : Any
            Data to predict on.
        kwargs : Any
            Additional arguments for prediction.

        Returns
        -------
        Any
            Predicted values.
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        file_path : str
            Path to the file where the model will be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> AbstractPredictor:
        """
        Load the model from a file.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        AbstractPredictor
            The loaded model.
        """
        pass
