"""
Greedy Presolver - SATzilla-style pre-solver selection.

This presolver greedily selects algorithms that solve the most instances
within a given time cutoff, similar to the pre-solver approach described
in the SATzilla paper (Xu et al., 2008).
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from asf.presolving.presolver import AbstractPresolver


class GreedyPresolver(AbstractPresolver):
    """
    A greedy presolver that selects algorithms based on how many instances
    they can solve within a given time cutoff.

    This follows the SATzilla approach where pre-solvers are selected to:
    1. Solve easy instances quickly before feature computation
    2. Filter out easy instances so empirical hardness models train on harder ones

    The greedy selection picks the algorithm that solves the most unsolved
    instances within the cutoff time, then repeats until the budget is exhausted
    or no more instances can be solved.

    Attributes:
        budget (float): Total time budget for the pre-solve schedule.
        cutoff_per_solver (float): Maximum time to allocate per solver (default 5s).
        max_presolvers (int): Maximum number of pre-solvers to include.
        min_coverage (float): Minimum fraction of instances a presolver must solve
                              to be included (default 0.01 = 1%).
    """

    PREFIX = "greedy_presolver"

    def __init__(
        self,
        budget: float = 30.0,
        cutoff_per_solver: float = 5.0,
        max_presolvers: int = 3,
        min_coverage: float = 0.01,
        maximize: bool = False,
        **kwargs,
    ):
        """
        Initialize the GreedyPresolver.

        Args:
            budget: Total time budget for pre-solving (sum of all pre-solver cutoffs).
            cutoff_per_solver: Time cutoff per solver to consider instances "solved".
            max_presolvers: Maximum number of pre-solvers to select.
            min_coverage: Minimum fraction of remaining instances a presolver must
                          solve to be included in the schedule.
            maximize: If True, maximize performance values instead of minimize.
        """
        super().__init__(budget=budget, maximize=maximize)
        self.cutoff_per_solver = cutoff_per_solver
        self.max_presolvers = max_presolvers
        self.min_coverage = min_coverage
        self.schedule: List[Tuple[str, float]] = []
        self.algorithms: List[str] = []

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the greedy presolver by selecting algorithms that solve the most instances.

        Args:
            features: DataFrame of instance features (not used, but required by interface).
            performance: DataFrame of runtimes/performance (n_instances x n_algorithms).
                         Lower values are better (unless maximize=True).
        """
        perf = performance.copy()
        self.algorithms = list(perf.columns)
        n_instances = len(perf)

        # Track which instances are "solved" by the schedule
        solved_mask = np.zeros(n_instances, dtype=bool)

        self.schedule = []
        remaining_budget = self.budget

        for _ in range(self.max_presolvers):
            if remaining_budget <= 0:
                break

            # Determine actual cutoff for this iteration
            actual_cutoff = min(self.cutoff_per_solver, remaining_budget)

            best_algo = None
            best_count = 0
            best_cutoff = actual_cutoff

            # For each algorithm, count how many unsolved instances it can solve
            for algo in self.algorithms:
                algo_times = perf[algo].values

                # Count instances this algo solves within cutoff that aren't already solved
                if self.maximize:
                    # For maximize, "solved" means performance >= some threshold
                    # but typically we minimize runtime, so this branch is less common
                    can_solve = (algo_times >= actual_cutoff) & (~solved_mask)
                else:
                    # For minimize (runtime), "solved" means runtime <= cutoff
                    can_solve = (algo_times <= actual_cutoff) & (~solved_mask)

                count = np.sum(can_solve)

                if count > best_count:
                    best_count = count
                    best_algo = algo

                    # Find the minimum cutoff needed to solve these instances
                    # (optimization: don't allocate more time than needed)
                    if count > 0:
                        solved_times = algo_times[can_solve]
                        best_cutoff = float(np.max(solved_times))
                        # Round up slightly for safety
                        best_cutoff = min(actual_cutoff, best_cutoff * 1.01)

            # Check if the best algorithm meets minimum coverage threshold
            coverage = best_count / n_instances if n_instances > 0 else 0
            if best_algo is None or coverage < self.min_coverage:
                break

            # Add to schedule
            self.schedule.append((best_algo, best_cutoff))
            remaining_budget -= best_cutoff

            # Update solved mask
            algo_times = perf[best_algo].values
            if self.maximize:
                newly_solved = algo_times >= best_cutoff
            else:
                newly_solved = algo_times <= best_cutoff
            solved_mask = solved_mask | newly_solved

            # If all instances are solved, stop
            if np.all(solved_mask):
                break

    def predict(self) -> List[Tuple[str, float]]:
        """
        Return the computed pre-solve schedule.

        Returns:
            List of (algorithm_name, cutoff_time) tuples representing the schedule.
        """
        return self.schedule
