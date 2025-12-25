"""
Ridge models wrappers.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from sklearn.linear_model import Ridge, RidgeClassifier

try:
    from ConfigSpace import (
        Categorical,
        Float,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.utils.configurable import ConfigurableMixin


class RidgeRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for sklearn.linear_model.Ridge.
    """

    PREFIX: str = "ridge_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the RidgeRegressorWrapper.

        Parameters
        ----------
        init_params : dict[str, Any] or None, default=None
            A dictionary of initialization parameters for the Ridge regressor.
        """
        super().__init__(Ridge, init_params or {})

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for the Ridge Regressor.

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

        alpha = Float(
            "alpha",
            (1e-6, 100.0),
            log=True,
            default=1.0,
        )
        fit_intercept = Categorical(
            "fit_intercept",
            [True, False],
            default=True,
        )
        solver = Categorical(
            "solver",
            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            default="auto",
        )

        params = [alpha, fit_intercept, solver]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[RidgeRegressorWrapper]:
        """
        Create a partial function from a clean (unprefixed) configuration.

        Parameters
        ----------
        clean_config : dict[str, Any]
            The clean configuration dictionary.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        partial
            A partial function for instantiating the wrapper.
        """
        params = {
            "alpha": clean_config["alpha"],
            "fit_intercept": clean_config["fit_intercept"],
            "solver": clean_config["solver"],
            **kwargs,
        }
        return partial(RidgeRegressorWrapper, init_params=params)


class RidgeClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for sklearn.linear_model.RidgeClassifier.
    """

    PREFIX: str = "ridge_classifier"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the RidgeClassifierWrapper.

        Parameters
        ----------
        init_params : dict[str, Any] or None, default=None
            A dictionary of initialization parameters for the Ridge classifier.
        """
        super().__init__(RidgeClassifier, init_params or {})

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for the Ridge Classifier.

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

        alpha = Float(
            "alpha",
            (1e-6, 100.0),
            log=True,
            default=1.0,
        )
        solver = Categorical(
            "solver",
            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            default="auto",
        )

        params = [alpha, solver]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[RidgeClassifierWrapper]:
        """
        Create a RidgeClassifierWrapper partial from a clean configuration.
        """
        params = {
            "alpha": clean_config["alpha"],
            "solver": clean_config["solver"],
            **kwargs,
        }
        return partial(RidgeClassifierWrapper, init_params=params)
