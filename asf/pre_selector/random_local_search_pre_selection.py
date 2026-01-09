"""
Random local search algorithm for algorithm pre-selection.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector

try:
    from ConfigSpace import Configuration, ConfigurationSpace

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class RandomLocalSearchPreSelector(AbstractPreSelector):
    """
    Random local search algorithm for algorithm pre-selection.

    This selector identifies a good subset of algorithms by combining random
    sampling with local search.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    n_algorithms : int
        The number of algorithms to select.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    n_restarts : int, default=10
        Number of random restarts.
    max_iterations : int, default=100
        Maximum number of local search iterations per restart.
    seed : int or None, default=None
        Random seed for reproducibility.
    **kwargs : Any
        Additional arguments passed to the parent class.
    """

    def __init__(
        self,
        metric: Callable[[pd.DataFrame], float],
        n_algorithms: int,
        maximize: bool = False,
        n_restarts: int = 10,
        max_iterations: int = 100,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.n_restarts = n_restarts
        self.max_iterations = max_iterations
        self.seed = seed

    def _is_better(self, new_score: float, old_score: float) -> bool:
        """Check if new_score is better than old_score."""
        if self.maximize:
            return new_score > old_score
        return new_score < old_score

    def _local_search(
        self,
        current_subset: list[str],
        all_algorithms: list[str],
        performance_frame: pd.DataFrame,
        rng: np.random.Generator,
    ) -> tuple[list[str], float]:
        """Perform local search by swapping algorithms."""
        current_score = self.metric(performance_frame[current_subset])
        best_subset = current_subset.copy()
        best_score = current_score

        for _ in range(self.max_iterations):
            improved = False
            subset_indices = list(range(len(current_subset)))
            rng.shuffle(subset_indices)  # type: ignore

            not_in_subset = [a for a in all_algorithms if a not in current_subset]
            rng.shuffle(not_in_subset)  # type: ignore

            for i in subset_indices:
                for new_algo in not_in_subset:
                    new_subset = current_subset.copy()
                    new_subset[i] = new_algo

                    new_score = self.metric(performance_frame[new_subset])

                    if self._is_better(new_score, best_score):
                        best_subset = new_subset.copy()
                        best_score = new_score
                        current_subset = new_subset
                        current_score = new_score
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return best_subset, best_score

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

        all_algorithms = list(performance_frame.columns)
        n_total = len(all_algorithms)

        if self.n_algorithms >= n_total:
            if is_numpy:
                return performance_frame.values
            return performance_frame.reset_index(drop=True)

        rng = np.random.default_rng(self.seed)

        best_overall_subset = None
        best_overall_score = float("-inf") if self.maximize else float("inf")

        for _ in range(self.n_restarts):
            initial_subset = list(
                rng.choice(all_algorithms, size=self.n_algorithms, replace=False)
            )

            local_best_subset, local_best_score = self._local_search(
                initial_subset, all_algorithms, performance_frame, rng
            )

            if self._is_better(local_best_score, best_overall_score):
                best_overall_subset = local_best_subset
                best_overall_score = local_best_score

        selected_performance = performance_frame[best_overall_subset]

        if is_numpy:
            selected_performance = selected_performance.values
        else:
            selected_performance = selected_performance.reset_index(drop=True)

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
    ) -> RandomLocalSearchPreSelector:
        """
        Create a RandomLocalSearchPreSelector instance from a configuration.
        """
        n_algorithms = AbstractPreSelector.get_from_configuration(
            configuration=configuration,
            cs_transform=cs_transform,
            maximize=maximize,
            pre_selector_name=pre_selector_name,
            **kwargs,
        )
        return RandomLocalSearchPreSelector(
            n_algorithms=n_algorithms,
            maximize=maximize,
            **kwargs,
        )
