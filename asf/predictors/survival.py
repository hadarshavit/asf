"""
Lightweight wrapper around sksurv's RandomSurvivalForest model.
"""

from __future__ import annotations

import inspect
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

    _ASF_META_KWARGS = {"budget", "maximize", "n_algorithms"}

    def _filter_constructor_kwargs(
        estimator: type, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        params = inspect.signature(estimator).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return {k: v for k, v in kwargs.items() if k not in _ASF_META_KWARGS}
        return {k: v for k, v in kwargs.items() if k in params}


    class RandomSurvivalForestWrapper(ConfigurableMixin, AbstractPredictor):
        """
        Lightweight wrapper around ``sksurv``'s ``RandomSurvivalForest`` model.
        """

        PREFIX: str = "random_survival_forest"

        def __init__(self, **kwargs: Any) -> None:
            """
            Initialize the RandomSurvivalForestWrapper.
            """
            if not SKSURV_AVAILABLE:
                raise ImportError(
                    "sksurv is not installed. Install scikit-survival to use RandomSurvivalForestWrapper."
                )
            kwargs = _filter_constructor_kwargs(RandomSurvivalForest, kwargs)
            self.model = RandomSurvivalForest(**kwargs)

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
                Integer("n_estimators", (10, 100), log=True, default=100),
                Integer("min_samples_split", (1, 50), default=10),
                Integer("min_samples_leaf", (1, 50), default=10),
                Float("max_features", (0.1, 1.0), default=0.5),
                Categorical("bootstrap", items=[True, False], default=True),
            ]
            return hyperparameters, [], []

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
