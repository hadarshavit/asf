"""
Beam search algorithm for algorithm pre-selection.
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


class BeamSearchPreSelector(AbstractPreSelector):
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

        Parameters
        ----------
        cs : ConfigurationSpace | None, default=None
            The configuration space to extend.
        cs_transform : dict[str, Any] | None, default=None
            The configuration space transform to extend.
        parent_param : Any | None, default=None
            The parent parameter.
        parent_value : Any | None, default=None
            The parent value.
        n_algorithms_max : int | None, default=None
            The maximum number of algorithms to select.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[ConfigurationSpace, dict[str, Any]]
            The configuration space and the configuration space transform.
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
    ) -> BeamSearchPreSelector:
        """
        Create a BeamSearchPreSelector instance from a configuration.

        Parameters
        ----------
        configuration : Configuration | dict[str, Any]
            The configuration to create the instance from.
        cs_transform : dict[str, Any]
            The configuration space transform.
        maximize : bool, default=False
            Whether to maximize the metric.
        pre_selector_name : str | None, default=None
            The name of the pre-selector.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        BeamSearchPreSelector
            A BeamSearchPreSelector instance.
        """
        n_algorithms = AbstractPreSelector.get_from_configuration(
            configuration=configuration,
            cs_transform=cs_transform,
            maximize=maximize,
            pre_selector_name=pre_selector_name,
            **kwargs,
        )
        return BeamSearchPreSelector(
            n_algorithms=n_algorithms,
            maximize=maximize,
            **kwargs,
        )
