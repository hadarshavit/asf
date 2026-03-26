"""
Sequential Backward Selection-based pre-selector for algorithm pre-selection.
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


class SBSPreSelector(AbstractPreSelector):
    """
    Sequential Backward Selection-based pre-selector.

    This selector starts from the full portfolio and iteratively removes the
    algorithm whose removal yields the best subset score under the provided
    metric until only ``n_algorithms`` remain.

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
                columns=pd.Index(
                    [f"Algorithm_{i}" for i in range(performance.shape[1])]
                ),
            )
            is_numpy = True
        else:
            performance_frame = performance
            is_numpy = False

        if self.n_algorithms is None:
            raise ValueError("n_algorithms must be set")
        if self.n_algorithms <= 0:
            raise ValueError("n_algorithms must be positive")
        if self.n_algorithms > performance_frame.shape[1]:
            raise ValueError(
                "n_algorithms cannot exceed the number of available algorithms"
            )

        selected_algorithms = list(performance_frame.columns)
        while len(selected_algorithms) > self.n_algorithms:
            best_subset: list[str] | None = None
            best_score = float("-inf") if self.maximize else float("inf")

            for algorithm in selected_algorithms:
                candidate_subset = [
                    candidate
                    for candidate in selected_algorithms
                    if candidate != algorithm
                ]
                score = self.metric(performance_frame[candidate_subset])
                is_better = score > best_score if self.maximize else score < best_score
                if is_better:
                    best_score = score
                    best_subset = candidate_subset

            if best_subset is None:
                raise RuntimeError("Failed to identify a valid subset.")
            selected_algorithms = best_subset

        selected_algorithms.sort(
            key=lambda algorithm: self.metric(performance_frame[[algorithm]]),
            reverse=self.maximize,
        )

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
    ) -> SBSPreSelector:
        """
        Create a SBSPreSelector instance from a configuration.
        """
        n_algorithms = AbstractPreSelector.get_from_configuration(
            configuration=configuration,
            cs_transform=cs_transform,
            maximize=maximize,
            pre_selector_name=pre_selector_name,
            **kwargs,
        )
        return SBSPreSelector(
            n_algorithms=n_algorithms,
            maximize=maximize,
            **kwargs,
        )
