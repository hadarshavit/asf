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

from typing import Any

from asf.utils.configurable import ConfigurableMixin


class LinearClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the SGDClassifier from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_classifier"

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the LinearClassifierWrapper.
        """
        super().__init__(SGDClassifier, init_params=init_params, **kwargs)

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


class LinearRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the SGDRegressor from scikit-learn, providing additional functionality
    for configuration space generation and parameter extraction.
    """

    PREFIX = "linear_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        """
        Initialize the LinearRegressorWrapper.
        """
        super().__init__(SGDRegressor, init_params=init_params, **kwargs)

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


class RidgeRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """Wrapper around scikit-learn's Ridge regressor for ASF predictors."""

    PREFIX = "ridge_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(Ridge, init_params=init_params, **kwargs)

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
