"""Construction of independent estimator instances."""

from typing import Any

from sklearn.base import clone


def clone_estimator(estimator: Any) -> Any:
    """Clone a configured predictor without retaining its fitted state."""
    if isinstance(estimator, type):
        raise TypeError(
            "estimator must be an instance, for example RandomForestRegressor()."
        )
    if not callable(getattr(estimator, "fit", None)) or not callable(
        getattr(estimator, "predict", None)
    ):
        raise TypeError("estimator must implement fit and predict.")
    try:
        return clone(estimator)
    except (TypeError, RuntimeError) as exc:
        raise TypeError(
            "estimator must support sklearn.base.clone. Use a scikit-learn "
            "compatible estimator with get_params, or implement __sklearn_clone__. "
            "Raw PyTorch modules are not supported."
        ) from exc
