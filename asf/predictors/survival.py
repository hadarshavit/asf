"""
Lightweight wrapper around sksurv's RandomSurvivalForest model.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import joblib

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.utils.configurable import ConfigurableMixin

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

try:
    from sksurv.ensemble import RandomSurvivalForest

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False


if SKSURV_AVAILABLE:

    class RandomSurvivalForestWrapper(ConfigurableMixin, AbstractPredictor):
        """
        Lightweight wrapper around ``sksurv``'s ``RandomSurvivalForest`` model.
        """

        PREFIX: str = "random_survival_forest"

        def __init__(self, init_params: dict[str, Any] | None = None) -> None:
            """
            Initialize the RandomSurvivalForestWrapper.

            Parameters
            ----------
            init_params : dict[str, Any] or None, default=None
                Initial parameters for the RandomSurvivalForest model.

            Raises
            ------
            ImportError
                If sksurv is not installed.
            """
            if not SKSURV_AVAILABLE:
                raise ImportError(
                    "sksurv is not installed. Install scikit-survival to use RandomSurvivalForestWrapper."
                )
            params = init_params or {}
            self.model = RandomSurvivalForest(**params)

        @staticmethod
        def _define_hyperparameters(
            **kwargs: Any,
        ) -> tuple[list[Hyperparameter], list[Any], list[Any]]:
            """
            Define hyperparameters for RandomSurvivalForestWrapper.

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
                Integer("n_estimators", (10, 1000), log=True, default=100),
                Integer("min_samples_split", (2, 20), default=6),
                Integer("min_samples_leaf", (1, 20), default=3),
                Float("max_features", (0.1, 1.0), default=1.0),
                Categorical("bootstrap", items=[True, False], default=True),
            ]
            return hyperparameters, [], []

        @classmethod
        def _get_from_clean_configuration(
            cls,
            clean_config: dict[str, Any],
            **kwargs: Any,
        ) -> partial:
            """
            Create a partial function from a clean (unprefixed) configuration.
            """
            config = clean_config.copy()
            config.update(kwargs)
            return partial(cls, init_params=config)

        def fit(self, X: Any, Y: Any, **kwargs: Any) -> None:
            """
            Fit the model to the data.

            Parameters
            ----------
            X : Any
                Training data.
            y : Any
                Target values.
            **kwargs : Any
                Additional arguments for the fit method.
            """
            self.model.fit(X, Y, **kwargs)

        def predict(self, X: Any, **kwargs: Any) -> Any:
            """
            Predict using the model.

            Parameters
            ----------
            X : Any
                Data to predict on.
            **kwargs : Any
                Additional arguments for the predict method.

            Returns
            -------
            Any
                Predicted values.
            """
            return self.model.predict(X, **kwargs)

        def predict_survival_function(self, X: Any, **kwargs: Any) -> Any:
            """
            Predict survival function.
            """
            return self.model.predict_survival_function(X, **kwargs)

        def save(self, file_path: str) -> None:
            """
            Save the model to a file.
            """
            joblib.dump(self, file_path)

        @classmethod
        def load(cls, file_path: str) -> AbstractPredictor:
            """
            Load the model from a file.
            """
            return joblib.load(file_path)

else:
    RandomSurvivalForestWrapper = None
