"""
Brute-force algorithm for algorithm pre-selection.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
from asf.utils.configurable import ConfigurableMixin


class BruteForcePreSelector(ConfigurableMixin, AbstractPreSelector):
    """
    Brute-force algorithm for algorithm pre-selection.

    This selector evaluates all possible combinations of a specified size and
    selects the subset that optimizes the given metric.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    n_algorithms : int
        The number of algorithms to select.
    maximize : bool, default=False
        Whether to maximize the metric.
    **kwargs : Any
        Additional keyword arguments passed to the parent class.
    """

    PREFIX = "brute_force"

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

        all_combinations = list(
            combinations(performance_frame.columns, self.n_algorithms)
        )
        best_combination = None
        best_performance = float("-inf") if self.maximize else float("inf")

        for combination in all_combinations:
            selected_performance = self.metric(performance_frame[list(combination)])
            if (self.maximize and selected_performance > best_performance) or (
                not self.maximize and selected_performance < best_performance
            ):
                best_performance = selected_performance
                best_combination = combination

        if best_combination is None:
            raise ValueError("No valid combination found")

        selected_performance = performance_frame[list(best_combination)]

        if is_numpy:
            selected_performance = selected_performance.to_numpy()
        else:
            selected_performance = selected_performance.reset_index(drop=True)
        return selected_performance
