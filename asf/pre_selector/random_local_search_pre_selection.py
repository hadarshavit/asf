from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable


class RandomLocalSearchPreSelector(AbstractPreSelector):
    """
    RandomLocalSearchPreSelector is a pre-selector that combines random sampling
    with local search to find a good subset of algorithms.

    The algorithm works as follows:
    1. Generate multiple random subsets of algorithms
    2. For each subset, perform local search by swapping algorithms to improve the metric
    3. Return the best subset found across all restarts

    This approach balances exploration (random restarts) with exploitation (local search),
    making it efficient for large search spaces while still finding good solutions.

    Attributes:
        metric (Callable): A function to evaluate the performance of the selected algorithms.
        n_algorithms (int): The number of algorithms to select.
        maximize (bool): Whether to maximize or minimize the performance metric.
        n_restarts (int): Number of random restarts (random initial subsets to try).
        max_iterations (int): Maximum number of local search iterations per restart.
        seed (int | None): Random seed for reproducibility.
    """

    def __init__(
        self,
        metric: Callable,
        n_algorithms: int,
        maximize: bool = False,
        n_restarts: int = 10,
        max_iterations: int = 100,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Initializes the RandomLocalSearchPreSelector with the given configuration.

        Args:
            metric (Callable): A function to evaluate the performance of the selected algorithms.
            n_algorithms (int): The number of algorithms to select.
            maximize (bool, optional): Whether to maximize the performance metric. Defaults to False.
            n_restarts (int, optional): Number of random restarts. Defaults to 10.
            max_iterations (int, optional): Maximum local search iterations per restart. Defaults to 100.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize
        self.n_restarts = n_restarts
        self.max_iterations = max_iterations
        self.seed = seed

    def _is_better(self, new_score: float, old_score: float) -> bool:
        """Check if new_score is better than old_score based on maximize flag."""
        if self.maximize:
            return new_score > old_score
        return new_score < old_score

    def _local_search(
        self,
        current_subset: list,
        all_algorithms: list,
        performance_frame: pd.DataFrame,
        rng: np.random.Generator,
    ) -> tuple[list, float]:
        """
        Perform local search by swapping algorithms in the current subset.

        Args:
            current_subset: Current selected algorithms.
            all_algorithms: All available algorithms.
            performance_frame: Performance data.
            rng: Random number generator.

        Returns:
            Tuple of (best_subset, best_score).
        """
        current_score = self.metric(performance_frame[current_subset])
        best_subset = current_subset.copy()
        best_score = current_score

        for _ in range(self.max_iterations):
            improved = False

            # Try swapping each algorithm in subset with each algorithm not in subset
            # Randomize order to avoid bias
            subset_indices = list(range(len(current_subset)))
            rng.shuffle(subset_indices)

            not_in_subset = [a for a in all_algorithms if a not in current_subset]
            rng.shuffle(not_in_subset)

            for i in subset_indices:
                for new_algo in not_in_subset:
                    # Create new subset by swapping
                    new_subset = current_subset.copy()
                    new_subset[i] = new_algo

                    new_score = self.metric(performance_frame[new_subset])

                    if self._is_better(new_score, best_score):
                        best_subset = new_subset.copy()
                        best_score = new_score
                        current_subset = new_subset
                        current_score = new_score
                        improved = True
                        break  # First improvement strategy

                if improved:
                    break

            # If no improvement found, local optimum reached
            if not improved:
                break

        return best_subset, best_score

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """
        Selects the best subset of algorithms using random sampling with local search.

        Args:
            performance (pd.DataFrame | np.ndarray): A DataFrame or NumPy array containing
                the performance data of algorithms. Rows represent instances, and columns
                represent algorithms.

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

        all_algorithms = list(performance_frame.columns)
        n_total = len(all_algorithms)

        # Handle edge case where n_algorithms >= total algorithms
        if self.n_algorithms >= n_total:
            if numpy:
                return performance_frame.values
            return performance_frame.reset_index(drop=True)

        rng = np.random.default_rng(self.seed)

        # Initialize best solution
        best_overall_subset = None
        best_overall_score = float("-inf") if self.maximize else float("inf")

        for _ in range(self.n_restarts):
            # Generate random initial subset
            initial_subset = list(
                rng.choice(all_algorithms, size=self.n_algorithms, replace=False)
            )

            # Perform local search from this starting point
            local_best_subset, local_best_score = self._local_search(
                initial_subset, all_algorithms, performance_frame, rng
            )

            # Update global best if this is better
            if self._is_better(local_best_score, best_overall_score):
                best_overall_subset = local_best_subset
                best_overall_score = local_best_score

        selected_performance = performance_frame[best_overall_subset]

        if numpy:
            selected_performance = selected_performance.values
        else:
            selected_performance = selected_performance.reset_index(drop=True)

        return selected_performance
