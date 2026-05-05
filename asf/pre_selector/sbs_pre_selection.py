"""
Single-best-solver ranking pre-selector for algorithm pre-selection.
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
    Single-best-solver ranking pre-selector.

    This selector scores each algorithm independently with the provided metric
    and keeps the top ``n_algorithms`` algorithms.

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
        metric: Callable[..., float],
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

        selected_algorithms = self._select_top_k(performance_frame)

        selected_performance = performance_frame[selected_algorithms]

        if is_numpy:
            selected_performance = selected_performance.values

        return selected_performance

    def _select_top_k(self, performance_frame: pd.DataFrame) -> list[Any]:
        scores = self._score_algorithms(performance_frame)
        scores = scores.sort_values(ascending=not self.maximize, kind="stable")
        return scores.index[: self.n_algorithms].tolist()

    def _score_algorithms(self, performance_frame: pd.DataFrame) -> pd.Series:
        try:
            scores = self.metric(performance_frame, batch=True)
        except TypeError:
            scores = None

        if scores is not None:
            if isinstance(scores, pd.Series):
                return scores.reindex(performance_frame.columns)

            score_values = np.asarray(scores, dtype=float)
            if score_values.shape == (performance_frame.shape[1],):
                return pd.Series(score_values, index=performance_frame.columns)

        return pd.Series(
            [
                self.metric(performance_frame[[algorithm]])
                for algorithm in performance_frame.columns
            ],
            index=performance_frame.columns,
        )

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
