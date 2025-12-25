from __future__ import annotations

from asf.predictors.sklearn_wrapper import SklearnWrapper
from typing import Any
from sklearn.linear_model import Ridge, RidgeClassifier

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Float,
        Categorical,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from functools import partial


from asf.utils.configurable import ConfigurableMixin


class RidgeRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for sklearn.linear_model.Ridge
    """

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


class RidgeClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    Wrapper for sklearn.linear_model.RidgeClassifier
    """

    PREFIX = "ridge_classifier"

    def __init__(self, init_params: dict[str, Any] = {}):
        super().__init__(RidgeClassifier, init_params)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """
        Define hyperparameters for the Ridge Classifier.
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
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        params = {
            "alpha": clean_config["alpha"],
            "solver": clean_config["solver"],
            **kwargs,
        }
        return partial(RidgeClassifierWrapper, init_params=params)
