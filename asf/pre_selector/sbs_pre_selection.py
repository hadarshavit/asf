"""
Sequential Backward Selection-based pre-selector for algorithm pre-selection.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
from asf.utils.configurable import ConfigurableMixin


class SBSPreSelector(ConfigurableMixin, AbstractPreSelector):
    """
    Sequential Backward Selection-based pre-selector.

    This selector selects algorithms based on their individual aggregate performance.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    n_algorithms : int
        The number of algorithms to select.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    **kwargs : Any
        Additional arguments passed to the parent class.
    """

    PREFIX = "sbs"

    def __init__(
        self,
        metric: Callable[[pd.DataFrame], float],
        n_algorithms: int,
        maximize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

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

        # Calculate the sum of performances for each algorithm
        algorithms_performances = performance_frame.sum(axis=0)
        # Sort algorithms based on their performance
        algorithms_performances = algorithms_performances.sort_values(
            ascending=not self.maximize
        )

        # Select the top `n_algorithms`
        selected_algorithms = algorithms_performances.index[: self.n_algorithms]
        selected_algorithms = selected_algorithms.tolist()

        selected_performance = performance_frame[selected_algorithms]

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
