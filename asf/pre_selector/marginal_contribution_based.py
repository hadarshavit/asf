from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable, Literal


class MarginalContributionBasedPreSelector(AbstractPreSelector):
    """
    Pre-selector that selects algorithms based on their marginal contribution to the
    performance metric. Supports two modes:

    - "backward" (default): Computes marginal contribution by measuring the impact of
      removing each algorithm from the full set. Selects algorithms with highest contribution.

    - "forward" (greedy forward selection): Iteratively builds the subset by adding
      the algorithm that provides the best marginal improvement at each step.
      This is slower but often finds better subsets as it considers algorithm complementarity.

    Attributes:
        metric (Callable): A callable function to compute the performance metric.
        n_algorithms (int): The number of algorithms to select.
        maximize (bool): A flag indicating whether to maximize or minimize the metric.
        mode (str): Selection mode - "backward" or "forward".
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

    def __init__(
        self,
        metric: Callable,
        n_algorithms: int,
        maximize: bool = False,
        mode: Literal["backward", "forward"] = "backward",
        **kwargs,
    ):
        """
        Initializes the MarginalContributionBasedPreSelector with the given configuration.

        Args:
            metric (Callable): A callable function to compute the performance metric.
            n_algorithms (int): The number of algorithms to select.
            maximize (bool, optional): Whether to maximize the metric. Defaults to False.
            mode (str, optional): Selection mode - "backward" computes marginal contribution
                from the full set, "forward" uses greedy forward selection. Defaults to "backward".
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.mode = mode

    def _is_better(self, new_score: float, old_score: float) -> bool:
        """Check if new_score is better than old_score based on maximize flag."""
        if self.maximize:
            return new_score > old_score
        return new_score < old_score

    def _forward_selection(self, performance_frame: pd.DataFrame) -> list:
        """
        Greedy forward selection: iteratively add the algorithm that provides
        the best marginal improvement.

        Args:
            performance_frame: DataFrame with algorithm performance data.

        Returns:
            List of selected algorithm names.
        """
        all_algorithms = set(performance_frame.columns)
        selected_algorithms = []

        for _ in range(self.n_algorithms):
            best_candidate = None
            best_score = float("-inf") if self.maximize else float("inf")

            candidates = all_algorithms - set(selected_algorithms)

            for candidate in candidates:
                # Evaluate adding this candidate to the current selection
                test_subset = selected_algorithms + [candidate]
                score = self.metric(performance_frame[test_subset])

                if self._is_better(score, best_score):
                    best_score = score
                    best_candidate = candidate

            if best_candidate is not None:
                selected_algorithms.append(best_candidate)

        return selected_algorithms

    def _backward_selection(self, performance_frame: pd.DataFrame) -> list:
        """
        Original backward marginal contribution method: compute contribution
        by measuring impact of removing each algorithm from the full set.

        Args:
            performance_frame: DataFrame with algorithm performance data.

        Returns:
            List of selected algorithm names.
        """
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

        return selected_algorithms

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
            - When mode="forward", uses greedy forward selection which considers algorithm
              complementarity but is slower (O(k*n) metric evaluations vs O(n) for backward).
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

        if self.mode == "forward":
            selected_algorithms = self._forward_selection(performance_frame)
        else:
            selected_algorithms = self._backward_selection(performance_frame)

        selected_performance = performance_frame[selected_algorithms]

        if numpy:
            selected_performance = selected_performance.values

        return selected_performance
