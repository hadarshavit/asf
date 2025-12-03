"""
Submodular Presolver - Based on Streeter & Golovin (NIPS 2008).

This presolver implements the offline greedy algorithm for maximizing submodular
functions, as described in:

"An Online Algorithm for Maximizing Submodular Functions"
Daniel Golovin, Matthew Streeter (NIPS 2008)

The algorithm greedily selects actions (algorithm, time) pairs that maximize
the marginal gain per unit time, achieving a (1-1/e)-approximation for
benefit-maximization and 4-approximation for cost-minimization.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from asf.presolving.presolver import AbstractPresolver


class SubmodularPresolver(AbstractPresolver):
    """
    A presolver based on submodular function maximization.

    This follows the Streeter & Golovin (2008) approach where:
    - Each action (v, τ) represents running algorithm v for time τ
    - The objective function f(S) represents the fraction of instances solved
    - f(S) is monotone and submodular
    - Actions are greedily selected to maximize marginal gain per unit time

    The greedy algorithm achieves:
    - (1 - 1/e) ≈ 0.632 approximation for benefit-maximization
    - 4-approximation for cost-minimization

    Attributes:
        budget (float): Total time budget for the pre-solve schedule.
        time_discretization (List[float]): Discrete time values to consider for actions.
        max_actions (int): Maximum number of actions in the schedule.
        epsilon (float): Small value to avoid division by zero.
    """

    PREFIX = "submodular_presolver"

    def __init__(
        self,
        budget: float = 30.0,
        time_discretization: Optional[List[float]] = None,
        max_actions: int = 10,
        epsilon: float = 1e-9,
        maximize: bool = False,
        **kwargs,
    ):
        """
        Initialize the SubmodularPresolver.

        Args:
            budget: Total time budget T for the schedule (sum of action durations).
            time_discretization: List of time values τ to consider for each algorithm.
                                 If None, uses [1, 2, 5, 10, 20] seconds.
            max_actions: Maximum number of actions to include in the schedule.
            epsilon: Small value to prevent division by zero.
            maximize: If True, maximize performance values instead of minimize.
                      For runtime minimization problems, this should be False.
        """
        super().__init__(budget=budget, maximize=maximize)
        self.time_discretization = time_discretization or [1.0, 2.0, 5.0, 10.0, 20.0]
        self.max_actions = max_actions
        self.epsilon = epsilon
        self.schedule: List[Tuple[str, float]] = []
        self.algorithms: List[str] = []
        self._performance: Optional[pd.DataFrame] = None

    def _compute_f(
        self,
        schedule: List[Tuple[str, float]],
        performance: np.ndarray,
        algo_indices: dict,
    ) -> float:
        """
        Compute f(S) = fraction of instances "solved" by the schedule.

        For algorithm selection (runtime minimization), an instance is "solved"
        if any action (algo, τ) in the schedule has runtime ≤ τ for that instance.

        This function is monotone and submodular.

        Args:
            schedule: List of (algorithm_name, time) tuples.
            performance: Performance matrix (n_instances x n_algorithms).
            algo_indices: Dict mapping algorithm names to column indices.

        Returns:
            Fraction of instances solved (value in [0, 1]).
        """
        n_instances = performance.shape[0]
        if n_instances == 0:
            return 0.0

        solved = np.zeros(n_instances, dtype=bool)

        for algo, tau in schedule:
            if algo not in algo_indices:
                continue
            algo_idx = algo_indices[algo]
            algo_times = performance[:, algo_idx]

            if self.maximize:
                # For maximization problems, "solved" means performance >= tau
                solved = solved | (algo_times >= tau)
            else:
                # For minimization (runtime), "solved" means runtime <= tau
                solved = solved | (algo_times <= tau)

        return np.mean(solved)

    def _compute_marginal_gain(
        self,
        schedule: List[Tuple[str, float]],
        action: Tuple[str, float],
        performance: np.ndarray,
        algo_indices: dict,
    ) -> float:
        """
        Compute the marginal gain f_a(S) = f(S ⊕ <a>) - f(S).

        Args:
            schedule: Current schedule S.
            action: Action a = (algorithm, time) to add.
            performance: Performance matrix.
            algo_indices: Dict mapping algorithm names to column indices.

        Returns:
            Marginal gain from adding the action.
        """
        f_current = self._compute_f(schedule, performance, algo_indices)
        f_with_action = self._compute_f(schedule + [action], performance, algo_indices)
        return f_with_action - f_current

    def _get_candidate_actions(
        self, remaining_budget: float
    ) -> List[Tuple[str, float]]:
        """
        Generate candidate actions (algorithm, time) pairs.

        Args:
            remaining_budget: Remaining time budget.

        Returns:
            List of (algorithm, time) tuples that fit within budget.
        """
        actions = []
        for algo in self.algorithms:
            for tau in self.time_discretization:
                if tau <= remaining_budget:
                    actions.append((algo, tau))
        return actions

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the submodular presolver using the offline greedy algorithm.

        The greedy algorithm selects actions to maximize marginal gain per unit time:
            g_j = argmax_{(v,τ)} f_{(v,τ)}(G_j) / τ

        This achieves (1-1/e)-approximation for benefit maximization.

        Args:
            features: DataFrame of instance features (not used, but required by interface).
            performance: DataFrame of runtimes/performance (n_instances x n_algorithms).
                         Lower values are better (unless maximize=True).
        """
        self._performance = performance.copy()
        self.algorithms = list(performance.columns)
        n_instances = len(performance)

        if n_instances == 0:
            self.schedule = []
            return

        # Convert to numpy for efficiency
        perf_matrix = performance.values
        algo_indices = {algo: i for i, algo in enumerate(self.algorithms)}

        self.schedule = []
        remaining_budget = self.budget

        for _ in range(self.max_actions):
            if remaining_budget <= self.epsilon:
                break

            # Get candidate actions that fit within budget
            candidates = self._get_candidate_actions(remaining_budget)
            if not candidates:
                break

            best_action = None
            best_ratio = -np.inf  # marginal gain per unit time

            # Greedy selection: maximize f_a(S) / τ
            for action in candidates:
                algo, tau = action
                marginal_gain = self._compute_marginal_gain(
                    self.schedule, action, perf_matrix, algo_indices
                )
                ratio = marginal_gain / tau

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_action = action

            # Stop if no positive marginal gain
            if best_action is None or best_ratio <= self.epsilon:
                break

            # Add action to schedule
            self.schedule.append(best_action)
            remaining_budget -= best_action[1]

            # Check if we've solved all instances
            if (
                self._compute_f(self.schedule, perf_matrix, algo_indices)
                >= 1.0 - self.epsilon
            ):
                break

    def predict(self) -> List[Tuple[str, float]]:
        """
        Return the computed pre-solve schedule.

        Returns:
            List of (algorithm_name, cutoff_time) tuples representing the schedule.
        """
        return self.schedule

    def get_schedule_cost(self, performance: Optional[pd.DataFrame] = None) -> float:
        """
        Compute the expected cost (average time to solve) of the schedule.

        The cost is defined as: c(f, S) = ∫_0^∞ (1 - f(S<t>)) dt

        For discrete schedules, this simplifies to summing over time intervals.

        Args:
            performance: Performance matrix. If None, uses stored performance.

        Returns:
            Expected cost (average time to solve an instance).
        """
        if performance is None:
            performance = self._performance
        if performance is None:
            return float("inf")

        perf_matrix = performance.values
        algo_indices = {algo: i for i, algo in enumerate(performance.columns)}
        n_instances = perf_matrix.shape[0]

        if n_instances == 0:
            return 0.0

        # Compute cost by integrating (1 - f(S<t>)) over time
        # For discrete schedule, sum over action intervals
        total_cost = 0.0
        current_schedule: List[Tuple[str, float]] = []
        prev_time = 0.0

        for algo, tau in self.schedule:
            # Cost contribution from time interval [prev_time, prev_time + tau]
            f_before = self._compute_f(current_schedule, perf_matrix, algo_indices)

            # Add the action
            current_schedule.append((algo, tau))

            # For simplicity, assume linear interpolation within the interval
            # (This is an approximation; exact computation would require more detail)
            f_after = self._compute_f(current_schedule, perf_matrix, algo_indices)
            avg_unsolved = 1.0 - (f_before + f_after) / 2.0
            total_cost += avg_unsolved * tau

            prev_time += tau

        # Add cost for remaining unsolved instances (they take infinite time,
        # but we cap at budget for practical purposes)
        f_final = self._compute_f(self.schedule, perf_matrix, algo_indices)
        if f_final < 1.0:
            # Instances not solved within budget contribute budget to cost
            remaining_unsolved = 1.0 - f_final
            total_cost += remaining_unsolved * self.budget

        return total_cost
