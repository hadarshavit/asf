from __future__ import annotations

from typing import Any, Dict, Optional

from asf.predictors.abstract_predictor import AbstractPredictor

try:
    from sksurv.ensemble import RandomSurvivalForest

    SKSURV_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RandomSurvivalForest = None  # type: ignore[assignment]
    SKSURV_AVAILABLE = False


if SKSURV_AVAILABLE:

    class RandomSurvivalForestWrapper(AbstractPredictor):
        """Lightweight wrapper around ``sksurv``'s ``RandomSurvivalForest`` model."""

        PREFIX = "random_survival_forest"

        def __init__(self, init_params: Optional[Dict[str, Any]] = None) -> None:
            if not SKSURV_AVAILABLE:
                raise ImportError(
                    "sksurv is not installed. Install scikit-survival to use RandomSurvivalForestWrapper."
                )
            params = init_params or {}
            self.model = RandomSurvivalForest(**params)  # type: ignore[misc]

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
