from __future__ import annotations

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Float,
        EqualsCondition,
        Categorical,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from sklearn.linear_model import SGDClassifier, SGDRegressor, Ridge

from asf.predictors.sklearn_wrapper import SklearnWrapper

from functools import partial
from typing import Any

from asf.utils.configurable import ConfigurableMixin


class LinearClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the SGDClassifier from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_classifier"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the LinearClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the SGDClassifier.
        """
        super().__init__(SGDClassifier, init_params or {})

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """
        Define hyperparameters for the Linear Classifier.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        alpha = Float(
            "alpha",
            (1e-5, 1),
            log=True,
            default=1e-3,
        )
        eta0 = Float(
            "eta0",
            (1e-5, 1),
            log=True,
            default=1e-2,
        )

        params = [
            alpha,
            eta0,
        ]

        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        linear_classifier_params = {
            "alpha": clean_config["alpha"],
            "eta0": clean_config["eta0"],
            **kwargs,
        }

        return partial(LinearClassifierWrapper, init_params=linear_classifier_params)


class LinearRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the SGDRegressor from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the LinearRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the SGDRegressor.
        """
        super().__init__(SGDRegressor, init_params or {})

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """
        Define hyperparameters for the Linear Regressor.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        alpha = Float(
            "alpha",
            (1e-5, 1),
            log=True,
            default=1e-3,
        )
        eta0 = Float(
            "eta0",
            (1e-5, 1),
            log=True,
            default=1e-2,
        )

        params = [alpha, eta0]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        linear_regressor_params = {
            "alpha": clean_config["alpha"],
            "eta0": clean_config["eta0"],
            **kwargs,
        }

        return partial(LinearRegressorWrapper, init_params=linear_regressor_params)


class RidgeRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """Wrapper around scikit-learn's Ridge regressor for ASF predictors."""

    PREFIX = "ridge_regressor"

    def __init__(self, init_params: dict[str, Any] = {}):
        super().__init__(Ridge, init_params)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """
        Define hyperparameters for the Ridge Regressor.
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
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        params = {
            "alpha": clean_config["alpha"],
            "fit_intercept": clean_config["fit_intercept"],
            "solver": clean_config["solver"],
            **kwargs,
        }
        return partial(RidgeRegressorWrapper, init_params=params)
