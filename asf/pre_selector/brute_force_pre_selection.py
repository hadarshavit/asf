from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Callable


class BruteForcePreSelector(AbstractPreSelector):
    """
    BruteForcePreSelector selects the optimal subset of algorithms from a given set
    by exhaustively evaluating all possible combinations of a specified size.

    This pre-selector uses a user-provided metric function to evaluate each combination
    of algorithms and selects the subset that either maximizes or minimizes the metric,
    depending on the `maximize` flag.

    Attributes:
        metric (Callable): A function to evaluate the performance of a subset of algorithms.
        n_algorithms (int): The number of algorithms to select.
        maximize (bool): Whether to maximize or minimize the metric.
    """

    def __init__(self, metric: Callable, n_algorithms: int, maximize=False, **kwargs):
        """
        Initializes the MarginalContributionBasedPreSelector with the given configuration.

        Args:
            config (dict): Configuration for the pre-selector.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """
        Selects the best subset of algorithms based on the provided performance data and metric.

        This method evaluates all possible combinations of algorithms of size `n_algorithms` and selects
        the combination that optimizes the given metric (either maximizes or minimizes, depending on the
        `maximize` flag).

        Args:
            performance (pd.DataFrame | np.ndarray): A DataFrame or ndarray containing the performance
                scores of algorithms. Rows correspond to instances, columns to algorithms.

        Returns:
            pd.DataFrame | np.ndarray: The performance data of the selected subset of algorithms,
                in the same format as the input (DataFrame or ndarray).
        """

        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],
            )
            numpy = True
        else:
            performance_frame = performance
            numpy = False

        # Generate all possible combinations of algorithms

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
        """Get the configuration space for BruteForcePreSelector."""
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
        """Create a BruteForcePreSelector instance from a configuration."""
        n_algorithms = AbstractPreSelector.get_from_configuration(
            configuration=configuration,
            cs_transform=cs_transform,
            maximize=maximize,
            pre_selector_name=pre_selector_name,
            **kwargs,
        )
        return BruteForcePreSelector(
            n_algorithms=n_algorithms,
            maximize=maximize,
            **kwargs,
        )
