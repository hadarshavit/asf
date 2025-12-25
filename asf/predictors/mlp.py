"""
Multi-Layer Perceptron (MLP) wrappers from scikit-learn.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from sklearn.neural_network import MLPClassifier, MLPRegressor

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (
        Float,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class MLPClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the MLPClassifier from scikit-learn.
    """

    PREFIX: str = "mlp_classifier"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the MLPClassifierWrapper.

        Parameters
        ----------
        init_params : dict[str, Any] or None, default=None
            Initial parameters for the MLPClassifier.
        """
        super().__init__(MLPClassifier, init_params or {})

    def fit(
        self,
        X: Any,
        Y: Any,
        sample_weight: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        sample_weight : array-like or None, default=None
            Sample weights. Not supported for MLPClassifier.
        **kwargs : Any
            Additional arguments for the fit method.

        Raises
        ------
        AssertionError
            If sample_weight is provided.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPClassifier"
        )
        self.model_class.fit(X, Y, **kwargs)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for the MLP Classifier.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            (hyperparameters, conditions, forbiddens)
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        depth = Integer("depth", (1, 3), default=3, log=False)
        width = Integer("width", (16, 1024), default=64, log=True)
        batch_size = Integer(
            "batch_size",
            (256, 1024),
            default=256,
            log=True,
        )
        alpha = Float(
            "alpha",
            (10**-8, 1),
            default=10**-3,
            log=True,
        )
        learning_rate_init = Float(
            "learning_rate_init",
            (10**-5, 1),
            default=10**-3,
            log=True,
        )

        params = [depth, width, batch_size, alpha, learning_rate_init]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[MLPClassifierWrapper]:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        hidden_layers = [clean_config["width"]] * clean_config["depth"]

        if "activation" not in kwargs:
            kwargs["activation"] = "relu"
        if "solver" not in kwargs:
            kwargs["solver"] = "adam"

        mlp_params = {
            "hidden_layer_sizes": tuple(hidden_layers),
            "batch_size": clean_config["batch_size"],
            "alpha": clean_config["alpha"],
            "learning_rate_init": clean_config["learning_rate_init"],
            **kwargs,
        }

        return partial(MLPClassifierWrapper, init_params=mlp_params)


class MLPRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the MLPRegressor from scikit-learn.
    """

    PREFIX: str = "mlp_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the MLPRegressorWrapper.

        Parameters
        ----------
        init_params : dict[str, Any] or None, default=None
            Initial parameters for the MLPRegressor.
        """
        super().__init__(MLPRegressor, init_params or {})

    def fit(
        self,
        X: Any,
        Y: Any,
        sample_weight: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        Y : array-like
            Target values.
        sample_weight : array-like or None, default=None
            Sample weights. Not supported for MLPRegressor.
        **kwargs : Any
            Additional arguments for the fit method.

        Raises
        ------
        AssertionError
            If sample_weight is provided.
        """
        assert sample_weight is None, (
            "Sample weights are not supported for MLPRegressor"
        )
        self.model_class.fit(X, Y, **kwargs)

    @staticmethod
    def _define_hyperparameters(
        dataset_size: str = "large",
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for the MLP Regressor.

        Parameters
        ----------
        dataset_size : str, default="large"
            The size of the dataset ('small', 'medium', or 'large').
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            (hyperparameters, conditions, forbiddens)
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        depth = Integer("depth", (1, 3), default=3, log=False)
        width = Integer("width", (16, 1024), default=64, log=True)

        if dataset_size == "small":
            batch_size = Integer(
                "batch_size",
                (16, 256),
                default=64,
                log=True,
            )
        elif dataset_size == "medium":
            batch_size = Integer(
                "batch_size",
                (128, 512),
                default=128,
                log=True,
            )
        elif dataset_size == "large":
            batch_size = Integer(
                "batch_size",
                (256, 1024),
                default=256,
                log=True,
            )
        else:
            raise ValueError(
                f"Invalid dataset_size: {dataset_size}. Choose from 'small', 'medium', 'large'."
            )

        alpha = Float(
            "alpha",
            (10**-8, 1),
            default=10**-3,
            log=True,
        )

        learning_rate_init = Float(
            "learning_rate_init",
            (10**-5, 1),
            default=10**-3,
            log=True,
        )

        params = [depth, width, batch_size, alpha, learning_rate_init]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[MLPRegressorWrapper]:
        """
        Create a MLPRegressorWrapper partial from a clean configuration.
        """
        hidden_layers = [clean_config["width"]] * clean_config["depth"]

        if "activation" not in kwargs:
            kwargs["activation"] = "relu"
        if "solver" not in kwargs:
            kwargs["solver"] = "adam"

        mlp_params = {
            "hidden_layer_sizes": tuple(hidden_layers),
            "batch_size": clean_config["batch_size"],
            "alpha": clean_config["alpha"],
            "learning_rate_init": clean_config["learning_rate_init"],
            **kwargs,
        }

        return partial(MLPRegressorWrapper, init_params=mlp_params)
