"""
Random Forest wrappers.
"""

from __future__ import annotations

from functools import partial
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

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the RandomForestClassifierWrapper.

        Parameters
        ----------
        init_params : dict[str, Any] or None, default=None
            A dictionary of initialization parameters for the RandomForestClassifier.
        """
        super().__init__(RandomForestClassifier, init_params or {})

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

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[RandomForestClassifierWrapper]:
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
        rf_params = clean_config.copy()
        rf_params.update(kwargs)
        return partial(RandomForestClassifierWrapper, init_params=rf_params)


class RandomForestRegressorWrapper(ConfigurableMixin, SklearnWrapper):
    """
    A wrapper for the RandomForestRegressor from scikit-learn.
    """

    PREFIX: str = "rf_regressor"

    def __init__(self, init_params: dict[str, Any] | None = None):
        """
        Initialize the RandomForestRegressorWrapper.

        Parameters
        ----------
        init_params : dict[str, Any] or None, default=None
            A dictionary of initialization parameters for the RandomForestRegressor.
        """
        super().__init__(RandomForestRegressor, init_params or {})

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

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[RandomForestRegressorWrapper]:
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
        rf_params = clean_config.copy()
        rf_params.update(kwargs)
        return partial(RandomForestRegressorWrapper, init_params=rf_params)
