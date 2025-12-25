from __future__ import annotations

from typing import Any

try:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    """
    Abstract base class for all predictors.

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
    get_configuration_space(cs)
        Get the configuration space for the predictor.
    get_from_configuration(configuration)
        Get a predictor instance from a configuration.
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

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: Any | None = None,
    ) -> ConfigurationSpace:
        """
        Get the configuration space for the predictor.

        Parameters
        ----------
        cs : ConfigurationSpace or None, default=None
            The configuration space to add the parameters to.
            If None, a new configuration space will be created.
        pre_prefix : str, default=""
            Prefix for all hyperparameters.
        parent_param : Hyperparameter or None, default=None
            Parent hyperparameter for conditions.
        parent_value : Any or None, default=None
            Value of the parent hyperparameter for conditions.

        Returns
        -------
        ConfigurationSpace
            The configuration space for the predictor.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        NotImplementedError
            If the method is not implemented for the predictor.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        raise NotImplementedError(
            "get_configuration_space() is not implemented for this predictor"
        )

    @staticmethod
    def get_from_configuration(
        configuration: dict[str, Any], pre_prefix: str = "", **kwargs: Any
    ) -> AbstractPredictor:
        """
        Get a predictor instance from a configuration.

        Parameters
        ----------
        configuration : dict[str, Any]
            The configuration to create the predictor from.
        pre_prefix : str, default=""
            Prefix used in the configuration.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        AbstractPredictor
            The predictor instance.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        NotImplementedError
            If the method is not implemented for the predictor.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        raise NotImplementedError(
            "get_from_configuration() is not implemented for this predictor"
        )
