from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable


class SBSPreSelector(AbstractPreSelector):
    """
    SBSPreSelector (Sequential Backward Selection PreSelector) is a pre-selector that selects algorithms
    based on their marginal contribution to the overall performance. It uses a sequential backward
    selection approach to identify the best subset of algorithms.

    Attributes:
        metric (Callable): A function to evaluate the performance of the selected algorithms.
        n_algorithms (int): The number of algorithms to select.
        maximize (bool): Whether to maximize or minimize the performance metric.
    """

    def __init__(self, metric: Callable, n_algorithms: int, maximize=False, **kwargs):
        """
        Initializes the SBSPreSelector with the given configuration.

        Args:
            metric (Callable): A function to evaluate the performance of the selected algorithms.
            n_algorithms (int): The number of algorithms to select.
            maximize (bool, optional): Whether to maximize the performance metric. Defaults to False.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """
        Selects the best subset of algorithms based on the performance data using a sequential
        backward selection approach.

        Args:
            performance (pd.DataFrame | np.ndarray): A DataFrame or NumPy array containing
                the performance data of algorithms. Rows represent instances, and columns represent algorithms.

        Returns:
            pd.DataFrame | np.ndarray: A DataFrame or NumPy array containing the performance
                data of the selected algorithms.
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

        # Calculate the sum of performances for each algorithm
        algorithms_performances = performance_frame.sum(axis=0)
        # Sort algorithms based on their performance (ascending or descending based on maximize flag)
        algorithms_performances = algorithms_performances.sort_values(
            ascending=not self.maximize
        )

        # Select the top `n_algorithms` based on the sorted performance
        selected_algorithms = algorithms_performances.index[: self.n_algorithms]
        selected_algorithms = selected_algorithms.tolist()

        selected_performance = performance_frame[selected_algorithms]

        if numpy:
            selected_performance = selected_performance.values

        return selected_performance
