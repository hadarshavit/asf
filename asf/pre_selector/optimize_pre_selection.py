"""
Optimization-based algorithm for algorithm pre-selection.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
from asf.utils.configurable import ConfigurableMixin

try:
    import scipy.optimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class OptimizePreSelection(ConfigurableMixin, AbstractPreSelector):
    """
    Optimization-based algorithm for algorithm pre-selection.

    This selector uses optimization techniques (e.g., differential evolution)
    to identify the best subset of algorithms.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    n_algorithms : int
        The number of algorithms to select.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    fmin_function : Callable or str or None, default=None
        Optimization function or method name (e.g., "SLSQP").
    **kwargs : Any
        Additional arguments passed to the parent class.
    """

    PREFIX = "optimize"

    def __init__(
        self,
        metric: Callable[[pd.DataFrame], float],
        n_algorithms: int,
        maximize: bool = False,
        fmin_function: Callable | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

        if fmin_function is None or isinstance(fmin_function, str):
            if SCIPY_AVAILABLE:
                if fmin_function is None or fmin_function == "differential_evolution":
                    self.fmin_function = scipy.optimize.differential_evolution
                else:
                    method = fmin_function
                    self.fmin_function = partial(scipy.optimize.minimize, method=method)
            else:
                raise ImportError(
                    "Scipy is not available. Please install scipy to use this feature."
                )
        else:
            self.fmin_function = fmin_function

    def fit_transform(
        self,
        performance: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        """
        Fit the pre-selector and transform the performance data.

        Parameters
        ----------
        performance : pd.DataFrame or np.ndarray
            The performance data.

        Returns
        -------
        pd.DataFrame or np.ndarray
            The performance data with only the selected algorithms.

        Raises
        ------
        ValueError
            If the number of selected algorithms is incorrect.
        """
        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],  # type: ignore[arg-type]
            )
            is_numpy = True
        else:
            performance_frame = performance
            is_numpy = False

        if self.n_algorithms is None:
            raise ValueError("n_algorithms must be set")

        def objective_function(x: np.ndarray) -> float:
            if self.n_algorithms is None:
                raise ValueError("n_algorithms must be set")
            selected_indices = x.argsort()[-self.n_algorithms :]
            selected_algorithms = performance_frame.columns[selected_indices]
            performance_with_algorithm = performance_frame[selected_algorithms]
            metric_val = self.metric(performance_with_algorithm)

            return metric_val if not self.maximize else -metric_val

        initial_guess = np.zeros(performance_frame.shape[1])
        initial_guess[: self.n_algorithms] = 1
        bounds = [(0, 1) for _ in range(performance_frame.shape[1])]

        if self.fmin_function == (
            scipy.optimize.differential_evolution if SCIPY_AVAILABLE else None
        ):
            result = self.fmin_function(
                objective_function,
                bounds=bounds,
            )
        else:
            result = self.fmin_function(
                objective_function,
                x0=initial_guess,
                bounds=bounds,
            )

        selected_indices = result.x.argsort()[-self.n_algorithms :]
        selected_algorithms = performance_frame.columns[selected_indices]
        selected_performance = performance_frame[selected_algorithms]

        if selected_performance.shape[1] != self.n_algorithms:
            raise ValueError(
                f"Selected performance has {selected_performance.shape[1]} algorithms, "
                f"but expected {self.n_algorithms}."
            )

        if is_numpy:
            selected_performance = selected_performance.values

        return selected_performance

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial:
        """Create a partial function from a clean configuration."""
        init_kwargs = {**clean_config}
        if "metric" in kwargs:
            init_kwargs["metric"] = kwargs["metric"]
        if "n_algorithms" in kwargs:
            init_kwargs["n_algorithms"] = kwargs["n_algorithms"]
        if "maximize" in kwargs:
            init_kwargs["maximize"] = kwargs["maximize"]
        return partial(cls, **init_kwargs)
