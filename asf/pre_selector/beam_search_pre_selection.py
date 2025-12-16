from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable


class BeamSearchPreSelector(AbstractPreSelector):
    def __init__(
        self,
        metric: Callable,
        n_algorithms: int,
        maximize=False,
        beam_width=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.beam_width = beam_width

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],
            )
            numpy = True
        else:
            performance_frame = performance
            numpy = False

        best_combinations = [
            ((col,), self.metric(performance_frame[[col]]))
            for col in performance_frame.columns
        ]
        best_combinations.sort(
            key=lambda x: x[1],
            reverse=self.maximize,
        )
        best_combinations = best_combinations[
            : self.beam_width
            if self.beam_width < len(best_combinations)
            else len(best_combinations)
        ]

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

        if numpy:
            selected_performance = selected_performance.to_numpy()
        else:
            selected_performance = selected_performance.reset_index(drop=True)
        return selected_performance

    @staticmethod
    def get_configuration_space(
        cs=None,
        cs_transform=None,
        parent_param=None,
        parent_value=None,
        n_algorithms_max=None,
        **kwargs,
    ):
        """Get the configuration space for BeamSearchPreSelector."""
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
        configuration,
        cs_transform,
        maximize=False,
        pre_selector_name=None,
        **kwargs,
    ):
        """Create a BeamSearchPreSelector instance from a configuration."""
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
