"""Construction of independent estimator instances."""

from typing import Any

from asf.utils.random_state import clone_with_random_state


def clone_estimator(estimator: Any, random_state: int | None = None) -> Any:
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
        return clone_with_random_state(estimator, random_state)
    except (TypeError, RuntimeError) as exc:
        raise TypeError(
            "estimator must support sklearn.base.clone. Use a scikit-learn "
            "compatible estimator with get_params, or implement __sklearn_clone__. "
            "Raw PyTorch modules are not supported."
        ) from exc
