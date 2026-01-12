"""
Beam search algorithm for algorithm pre-selection.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
from asf.utils.configurable import ConfigurableMixin


class BeamSearchPreSelector(ConfigurableMixin, AbstractPreSelector):
    """
    Beam search algorithm for algorithm pre-selection.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    n_algorithms : int
        The number of algorithms to select.
    maximize : bool, default=False
        Whether to maximize the metric.
    beam_width : int, default=10
        The width of the beam search.
    **kwargs : Any
        Additional keyword arguments passed to the parent class.
    """

    PREFIX = "beam_search"

    def __init__(
        self,
        metric: Callable[[pd.DataFrame], float],
        n_algorithms: int,
        maximize: bool = False,
        beam_width: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.beam_width = beam_width

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

        best_combinations = [
            ((col,), self.metric(performance_frame[[col]]))
            for col in performance_frame.columns
        ]
        best_combinations.sort(
            key=lambda x: x[1],
            reverse=self.maximize,
        )
        best_combinations = best_combinations[: self.beam_width]

        for _ in range(self.n_algorithms - 1):
            new_combinations = []

            for combination, comb_perf in best_combinations:
                for col in performance_frame.columns:
                    if col not in combination:
                        new_combination = combination + (col,)
                        selected_performance = self.metric(
                            performance_frame[list(new_combination)]
                        )
                        new_combinations.append((new_combination, selected_performance))
            new_combinations.sort(
                key=lambda x: x[1],
                reverse=self.maximize,
            )
            best_combinations = new_combinations[: self.beam_width]

        best_combination = (
            max(
                best_combinations,
                key=lambda x: x[1],
            )[0]
            if self.maximize
            else min(
                best_combinations,
                key=lambda x: x[1],
            )[0]
        )

        selected_performance = performance_frame[list(best_combination)]

        if is_numpy:
            selected_performance = selected_performance.to_numpy()
        else:
            selected_performance = selected_performance.reset_index(drop=True)
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
        if "beam_width" in kwargs:
            init_kwargs["beam_width"] = kwargs["beam_width"]
        return partial(cls, **init_kwargs)
