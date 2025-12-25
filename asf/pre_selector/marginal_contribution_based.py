"""
Marginal contribution-based algorithm for algorithm pre-selection.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector

try:
    from ConfigSpace import Configuration, ConfigurationSpace

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class MarginalContributionBasedPreSelector(AbstractPreSelector):
    """
    Pre-selector that selects algorithms based on their marginal contribution.

    Supports two modes:
    - "backward" (default): Computes marginal contribution by measuring the impact of
      removing each algorithm from the full set.
    - "forward" (greedy forward selection): Iteratively builds the subset by adding
      the algorithm that provides the best marginal improvement at each step.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    n_algorithms : int
        The number of algorithms to select.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    mode : Literal["backward", "forward"], default="backward"
        Selection mode.
    **kwargs : Any
        Additional arguments passed to the parent class.
    """

    def __init__(
        self,
        metric: Callable[[pd.DataFrame], float],
        n_algorithms: int,
        maximize: bool = False,
        mode: Literal["backward", "forward"] = "backward",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.mode = mode

    def _is_better(self, new_score: float, old_score: float) -> bool:
        """Check if new_score is better than old_score."""
        if self.maximize:
            return new_score > old_score
        return new_score < old_score

    def _forward_selection(self, performance_frame: pd.DataFrame) -> list[str]:
        """Greedy forward selection."""
        if self.n_algorithms is None:
            raise ValueError("n_algorithms must be set")
        all_algorithms = set(performance_frame.columns)
        selected_algorithms: list[str] = []

        for _ in range(self.n_algorithms):
            best_candidate = None
            best_score = float("-inf") if self.maximize else float("inf")

            candidates = all_algorithms - set(selected_algorithms)

            for candidate in candidates:
                test_subset = selected_algorithms + [candidate]
                score = self.metric(performance_frame[test_subset])

                if self._is_better(score, best_score):
                    best_score = score
                    best_candidate = candidate

            if best_candidate is not None:
                selected_algorithms.append(best_candidate)

        return selected_algorithms

    def _backward_selection(self, performance_frame: pd.DataFrame) -> list[str]:
        """Backward marginal contribution selection."""
        if self.n_algorithms is None:
            raise ValueError("n_algorithms must be set")
        mcs: list[tuple[str, float]] = []
        total_performance = self.metric(performance_frame)
        for algorithm in performance_frame.columns:
            performance_without_algorithm = performance_frame.drop(columns=[algorithm])
            total_performance_without_algorithm = self.metric(
                performance_without_algorithm
            )
            marginal_contribution = (
                total_performance - total_performance_without_algorithm
                if self.maximize
                else total_performance_without_algorithm - total_performance
            )

            mcs.append((algorithm, marginal_contribution))
        mcs.sort(key=lambda x: x[1], reverse=True)
        selected_algorithms = [x[0] for x in mcs[: self.n_algorithms]]

        return selected_algorithms

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

        if self.mode == "forward":
            selected_algorithms = self._forward_selection(performance_frame)
        else:
            selected_algorithms = self._backward_selection(performance_frame)

        selected_performance = performance_frame[selected_algorithms]

        if is_numpy:
            selected_performance = selected_performance.values

        return selected_performance

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        cs_transform: dict[str, Any] | None = None,
        parent_param: Any | None = None,
        parent_value: Any | None = None,
        n_algorithms_max: int | None = None,
        **kwargs: Any,
    ) -> tuple[ConfigurationSpace, dict[str, Any]]:
        """
        Get the configuration space.
        """
        return AbstractPreSelector.get_configuration_space(
            cs=cs,
            cs_transform=cs_transform,
            parent_param=parent_param,
            parent_value=parent_value,
            n_algorithms_max=n_algorithms_max,
            **kwargs,
        )

    @staticmethod
    def get_from_configuration(
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, Any],
        maximize: bool = False,
        pre_selector_name: str | None = None,
        **kwargs: Any,
    ) -> MarginalContributionBasedPreSelector:
        """
        Create a MarginalContributionBasedPreSelector instance from a configuration.
        """
        n_algorithms = AbstractPreSelector.get_from_configuration(
            configuration=configuration,
            cs_transform=cs_transform,
            maximize=maximize,
            pre_selector_name=pre_selector_name,
            **kwargs,
        )
        return MarginalContributionBasedPreSelector(
            n_algorithms=n_algorithms,
            maximize=maximize,
            **kwargs,
        )
