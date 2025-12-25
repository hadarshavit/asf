from __future__ import annotations

from typing import Any

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.utils.configurable import ConfigurableMixin
from functools import partial

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
        Float,
        Categorical,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

try:
    from sksurv.ensemble import RandomSurvivalForest

    SKSURV_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RandomSurvivalForest = None  # type: ignore[assignment]
    SKSURV_AVAILABLE = False


if SKSURV_AVAILABLE:

    class RandomSurvivalForestWrapper(ConfigurableMixin, AbstractPredictor):
        """Lightweight wrapper around ``sksurv``'s ``RandomSurvivalForest`` model."""

        PREFIX = "random_survival_forest"

        def __init__(self, init_params: dict[str, Any] | None = None) -> None:
            if not SKSURV_AVAILABLE:
                raise ImportError(
                    "sksurv is not installed. Install scikit-survival to use RandomSurvivalForestWrapper."
                )
            params = init_params or {}
            self.model = RandomSurvivalForest(**params)  # type: ignore[misc]

        @staticmethod
        def _define_hyperparameters(**kwargs):
            """Define hyperparameters for RandomSurvivalForestWrapper."""
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
            **kwargs,
        ) -> partial:
            """
            Create a partial function from a clean (unprefixed) configuration.
            """
            config = clean_config.copy()
            config.update(kwargs)
            return partial(RandomSurvivalForestWrapper, init_params=config)

        def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
            self.model.fit(X, y, **kwargs)

        def predict(self, X: Any, **kwargs: Any) -> Any:
            return self.model.predict(X, **kwargs)

        def predict_survival_function(self, X: Any, **kwargs: Any) -> Any:
            return self.model.predict_survival_function(X, **kwargs)

        def save(self, file_path: str) -> None:
            import joblib

            joblib.dump(self.model, file_path)

        def load(self, file_path: str) -> "RandomSurvivalForestWrapper":
            import joblib

            self.model = joblib.load(file_path)
            return self

else:
    RandomSurvivalForestWrapper = None  # type: ignore[assignment]
