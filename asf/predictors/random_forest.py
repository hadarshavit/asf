"""
Random Forest wrappers.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from ConfigSpace import (
        Categorical,
        Float,
        Integer,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.predictors.sklearn_wrapper import SklearnWrapper
from asf.utils.configurable import ConfigurableMixin


class RandomForestClassifierWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the RandomForestClassifier from scikit-learn.
    """

    PREFIX: str = "rf_classifier"

    def __init__(self, **kwargs: Any):
        """
        Initialize the RandomForestClassifierWrapper.

        Parameters
        ----------
        **kwargs : Any
            Parameters for the RandomForestClassifier.
        """
        super().__init__(RandomForestClassifier, **kwargs)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for RandomForestClassifier.

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

        hyperparameters = [
            Integer("n_estimators", (16, 128), log=True, default=116),
            Integer("min_samples_split", (2, 20), log=False, default=2),
            Integer("min_samples_leaf", (1, 20), log=False, default=2),
            Float("max_features", (0.1, 1.0), log=False, default=0.17055852159745608),
            Categorical("bootstrap", items=[True, False], default=False),
        ]
        return hyperparameters, [], []


class RandomForestRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the RandomForestRegressor from scikit-learn.
    """

    PREFIX: str = "rf_regressor"

    def __init__(self, **kwargs: Any):
        """
        Initialize the RandomForestRegressorWrapper.

        Parameters
        ----------
        **kwargs : Any
            Parameters for the RandomForestRegressor.
        """
        super().__init__(RandomForestRegressor, **kwargs)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
        """
        Define hyperparameters for RandomForestRegressor.

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

        hyperparameters = [
            Integer("n_estimators", (16, 128), log=True, default=116),
            Integer("min_samples_split", (2, 20), log=False, default=2),
            Integer("min_samples_leaf", (1, 20), log=False, default=2),
            Float("max_features", (0.1, 1.0), log=False, default=0.17055852159745608),
            Categorical("bootstrap", items=[True, False], default=False),
        ]
        return hyperparameters, [], []
