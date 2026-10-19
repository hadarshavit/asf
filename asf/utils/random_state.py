"""Pass ASF random states across third-party constructor boundaries."""

from functools import partial
from inspect import Parameter, signature
from typing import Any, Callable

from sklearn.base import clone


def random_state_kwargs(
    factory: Callable[..., Any], random_state: int | None
) -> dict[str, Any]:
    """Pass a state only to constructors that accept it; respect bound states."""
    if (
        isinstance(factory, partial)
        and factory.keywords.get("random_state") is not None
    ):
        return {}
    parameters = signature(factory).parameters
    if "random_state" in parameters or any(
        p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return {"random_state": random_state}
    return {}


def clone_with_random_state(component: Any, random_state: int | None) -> Any:
    """Clone a sklearn-compatible template and fill unset random states."""
    result = clone(component)
    if random_state is not None and hasattr(result, "get_params"):
        params = result.get_params(deep=True)
        updates = {
            name: random_state
            for name, value in params.items()
            if name.split("__")[-1] == "random_state" and value is None
        }
        if updates:
            if hasattr(result, "set_params"):
                result.set_params(**updates)
            else:
                result = type(result)(**{**params, **updates})
    return result
