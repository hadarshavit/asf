from __future__ import annotations

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.utils.configurable import ConfigurableMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
        Float,
        Categorical,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial
from typing import Any


class RandomForestClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the RandomForestClassifier from scikit-learn, providing
    additional functionality for configuration space management.
    """

    PREFIX = "rf_classifier"

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the RandomForestClassifierWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the RandomForestClassifier.
        """
        super().__init__(RandomForestClassifier, init_params)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for RandomForestClassifier."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("n_estimators", (16, 128), log=True, default=116),
            Integer("min_samples_split", (2, 20), log=False, default=2),
            Integer("min_samples_leaf", (1, 20), log=False, default=2),
            Float("max_features", (0.1, 1.0), log=False, default=0.17055852159745608),
            Categorical("bootstrap", items=[True, False], default=False),
        ]
        conditions = []
        forbiddens = []

        return hyperparameters, conditions, forbiddens

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        rf_params = clean_config.copy()
        rf_params.update(kwargs)

        return partial(RandomForestClassifierWrapper, init_params=rf_params)


class RandomForestRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the RandomForestRegressor from scikit-learn, providing
    additional functionality for configuration space management.
    """

    PREFIX = "rf_regressor"

    def __init__(self, init_params: dict[str, Any] = {}):
        """
        Initialize the RandomForestRegressorWrapper.

        Parameters
        ----------
        init_params : dict, optional
            A dictionary of initialization parameters for the RandomForestRegressor.
        """
        super().__init__(RandomForestRegressor, init_params)

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for RandomForestRegressor."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("n_estimators", (16, 128), log=True, default=116),
            Integer("min_samples_split", (2, 20), log=False, default=2),
            Integer("min_samples_leaf", (1, 20), log=False, default=2),
            Float("max_features", (0.1, 1.0), log=False, default=0.17055852159745608),
            Categorical("bootstrap", items=[True, False], default=False),
        ]
        conditions = []
        forbiddens = []

        return hyperparameters, conditions, forbiddens

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        rf_params = clean_config.copy()
        rf_params.update(kwargs)

        return partial(RandomForestRegressorWrapper, init_params=rf_params)
