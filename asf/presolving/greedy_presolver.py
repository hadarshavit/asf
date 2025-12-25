"""
Greedy Presolver - SATzilla-style pre-solver selection.
"""

from __future__ import annotations

from typing import Any, cast
import numpy as np
import pandas as pd

from asf.presolving.presolver import AbstractPresolver


class GreedyPresolver(AbstractPresolver):
    """
    A greedy presolver that selects algorithms based on instance coverage.

    This follows the SATzilla approach where pre-solvers are selected to:
    1. Solve easy instances quickly before feature computation.
    2. Filter out easy instances so empirical hardness models train on harder ones.

    The greedy selection picks the algorithm that solves the most unsolved
    instances within the cutoff time, then repeats until the budget is exhausted
    or no more instances can be solved.

    Parameters
    ----------
    budget : float, default=30.0
        Total time budget for the pre-solve schedule.
    cutoff_per_solver : float, default=5.0
        Maximum time to allocate per solver.
    max_presolvers : int, default=3
        Maximum number of pre-solvers to include.
    min_coverage : float, default=0.01
        Minimum fraction of instances a presolver must solve to be included.
    maximize : bool, default=False
        If True, maximize performance values instead of minimize.
    **kwargs : Any
        Additional keyword arguments.
    """

    PREFIX: str = "greedy_presolver"

    def __init__(
        self,
        budget: float = 30.0,
        cutoff_per_solver: float = 5.0,
        max_presolvers: int = 3,
        min_coverage: float = 0.01,
        maximize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(budget=budget, maximize=maximize)
        self.cutoff_per_solver = cutoff_per_solver
        self.max_presolvers = max_presolvers
        self.min_coverage = min_coverage
        self.schedule: list[tuple[str, float]] = []
        self.algorithms: list[str] = []

    def fit(
        self,
        features: pd.DataFrame | np.ndarray | None,
        performance: pd.DataFrame | np.ndarray | None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the greedy presolver.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The instance features.
        performance : pd.DataFrame or np.ndarray
            The algorithm performances.
        """
        if performance is None:
            raise ValueError("GreedyPresolver requires performance data for fitting.")

        if isinstance(performance, pd.DataFrame):
            perf = performance.copy()
            self.algorithms = list(perf.columns)
        else:
            perf = pd.DataFrame(performance)
            self.algorithms = [f"a{i}" for i in range(cast(Any, performance).shape[1])]

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

                count = int(np.sum(can_solve))

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

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | None = None,
        performance: pd.DataFrame | np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
        """
        Return the computed pre-solve schedule.

        Parameters
        ----------
        features : pd.DataFrame or None, default=None
            The features for the instances.
        performance : pd.DataFrame or None, default=None
            The algorithm performances.

        Returns
        -------
        list or dict
            The presolving schedule.
        """
        if features is not None:
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features)
            return {str(inst): self.schedule for inst in features.index}
        return self.schedule
