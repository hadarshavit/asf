from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable


class MarginalContributionBasedPreSelector(AbstractPreSelector):
    """

    Attributes:
        metric (Callable): A callable function to compute the performance metric.
        n_algorithms (int): The number of algorithms to select.
        maximize (bool): A flag indicating whether to maximize or minimize the metric.
        **kwargs: Additional arguments passed to the parent class.

    Methods:
    fit_transform(performance: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
            Selects a subset of algorithms based on their marginal contribution to the performance metric.

                performance (pd.DataFrame | np.ndarray): A DataFrame or NumPy array containing the performance
                    metrics of the algorithms.

            Returns:
                pd.DataFrame | np.ndarray: A DataFrame or NumPy array containing the performance metrics of the
                    selected algorithms.
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
            Selects a subset of algorithms based on their marginal contributions to the
            overall performance and returns the performance data for the selected algorithms.

            Parameters:
            ----------
        performance : pd.DataFrame | np.ndarray
                A DataFrame or NumPy array containing the performance metrics of algorithms.
                Each column represents an algorithm, and each row represents a performance metric.

            Returns:
            -------
        pd.DataFrame | np.ndarray
                A DataFrame or NumPy array containing the performance metrics of the selected
                algorithms. The format matches the input type (DataFrame or NumPy array).

            Notes:
            -----
            - The selection is based on the marginal contribution of each algorithm to the
              overall performance, calculated using the provided `self.metric` function.
            - The `self.maximize` attribute determines whether the metric is maximized or minimized.
            - The number of algorithms to select is determined by `self.n_algorithms`.
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

        mcs = []
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
        selected_performance = performance_frame[selected_algorithms]

        if numpy:
            selected_performance = selected_performance.values

        return selected_performance
