"""
ASAPv2 presolver - Algorithm Selector and Prescheduler.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar

from asf.presolving.presolver import AbstractPresolver

try:
    from ConfigSpace import Configuration

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class ASAPv2(AbstractPresolver):
    """
    ASAPv2 presolver - Algorithm Selector and Prescheduler.

    This implementation follows the original ASAP v2 paper:
    Gonard, F., Schoenauer, M., & Sebag, M. (2016). Algorithm Selector and
    Prescheduler in the ICON challenge.

    Uses differential evolution instead of CMA-ES for optimization.

    Parameters
    ----------
    runcount_limit : float, default=100.0
        Maximum number of iterations for differential evolution.
    budget : float, default=30.0
        Total time budget (timeout) for solving.
    maximize : bool, default=False
        Whether to maximize performance (False for runtime minimization).
    size_preschedule : int, default=3
        Number of algorithms to include in the preschedule.
    max_runtime_preschedule : float, default=-1
        Maximum time for preschedule. If < 0, uses 10% of budget.
        If < 1, uses this fraction of budget. Otherwise uses the value directly.
    regularization_weight : float, default=0.0
        Weight for regularization term in objective function.
    variance_weight : float, default=0.0
        Weight for variance penalty in objective function.
    de_popsize : int, default=15
        Population size for differential evolution.
    seed : int, default=42
        Random seed for reproducibility.
    verbosity : int, default=0
        Verbosity level (0=silent, 1=basic, 2=detailed).
    """

    def __init__(
        self,
        runcount_limit: float = 100.0,
        budget: float = 30.0,
        maximize: bool = False,
        size_preschedule: int = 3,
        max_runtime_preschedule: float = -1,
        regularization_weight: float = 0.0,
        variance_weight: float = 0.0,
        de_popsize: int = 15,
        seed: int = 42,
        verbosity: int = 0,
    ) -> None:
        """
        Initialize ASAPv2 presolver.

        Parameters
        ----------
        runcount_limit : float
            Maximum number of iterations for differential evolution.
        budget : float
            Total time budget (timeout) for solving.
        maximize : bool
            Whether to maximize performance (False for runtime minimization).
        size_preschedule : int
            Number of algorithms to include in the preschedule.
        max_runtime_preschedule : float
            Maximum time for preschedule. If < 0, uses 10% of budget.
            If < 1, uses this fraction of budget. Otherwise uses the value directly.
        regularization_weight : float
            Weight for regularization term in objective function.
        variance_weight : float
            Weight for variance penalty in objective function.
        de_popsize : int
            Population size for differential evolution.
        seed : int
            Random seed for reproducibility.
        verbosity : int
            Verbosity level (0=silent, 1=basic, 2=detailed).
        """
        super().__init__(budget=budget, maximize=maximize)

        self.size_preschedule = size_preschedule
        self.regularization_weight = regularization_weight
        self.variance_weight = variance_weight
        self.de_popsize = de_popsize
        self.de_maxiter = int(runcount_limit)
        self.seed = seed
        self.verbosity = verbosity
        self.rand_st = np.random.RandomState(seed)

        # Set max runtime for preschedule
        if max_runtime_preschedule < 0:
            self.max_runtime_preschedule = 0.1 * budget
        elif max_runtime_preschedule < 1:
            self.max_runtime_preschedule = max_runtime_preschedule * budget
        else:
            self.max_runtime_preschedule = max_runtime_preschedule

        # Will be set during fit
        self.algorithms: list[str] = []
        self.numAlg: int = 0
        self.ialgos_preschedule: np.ndarray | None = (
            None  # Indices of algorithms in preschedule
        )
        self.runtimes_preschedule: np.ndarray | None = None
        self.features: pd.DataFrame | None = None
        self.performance: pd.DataFrame | None = None
        self.schedule: list[tuple[str, float]] = []

    def fit(
        self,
        features: pd.DataFrame | np.ndarray | None,
        performance: pd.DataFrame | np.ndarray | None,
        **kwargs: Any,
    ) -> None:
        """
        Train the ASAP v2 presolver.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The instance features.
        performance : pd.DataFrame or np.ndarray
            The algorithm performances.
        """
        if features is None or performance is None:
            raise ValueError(
                "ASAPv2 requires features and performance data for fitting."
            )
        # Convert to DataFrame if needed
        if isinstance(features, np.ndarray):
            features_frame = pd.DataFrame(features)
        else:
            features_frame = features

        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(performance)
        else:
            performance_frame = performance

        self.features = features_frame
        self.performance = performance_frame
        self.algorithms = list(performance_frame.columns)
        self.numAlg = len(self.algorithms)

        # Convert to numpy - apply PAR10 penalty for unsolved instances
        self.feature_train = features_frame.values
        self.performance_train = performance_frame.values.copy()

        # PAR10: instances with runtime >= budget are penalized with 10x budget
        self.performance_train[self.performance_train >= self.budget] = 10 * self.budget

        if self.verbosity > 0:
            print()
            print("+ " * 30)
            print(f"Training ASAP v2 with {len(self.algorithms)} algorithms")

        # Adjust size_preschedule to not exceed numAlg - 1
        self.size_preschedule = min(self.size_preschedule, self.numAlg - 1)

        # Step 1: Identify algorithms for preschedule
        if self.size_preschedule > 0:
            self._identify_algorithms_for_preschedule()
        else:
            self.ialgos_preschedule = np.zeros((0,), dtype=int)
            self.runtimes_preschedule = np.zeros((0,))

        # Step 2: Optimize preschedule times using differential evolution
        if self.size_preschedule > 1:
            self._optimize_preschedule_de()
        elif self.verbosity > 0 and self.size_preschedule == 1:
            print("1-D schedule optimization is not implemented.")

        # Build final schedule
        self._build_schedule()

    def _identify_algorithms_for_preschedule(self) -> None:
        """
        Identify which algorithms should be included in the preschedule.

        Uses a greedy approach to find the best combination of algorithms
        that can solve the most instances within a given time budget.
        """
        numInstances = self.performance_train.shape[0]
        numSolvers = self.numAlg

        step_size = max(5, self.max_runtime_preschedule / 100)
        # Start from step_size instead of 0 to ensure positive timesteps
        timesteps = np.arange(
            step_size, self.max_runtime_preschedule + step_size, step_size
        )
        if len(timesteps) == 0:
            timesteps = np.array([self.max_runtime_preschedule])

        best_config = {
            "algos": np.zeros((len(timesteps), self.size_preschedule), dtype=int),
            "rate_solved_instances": np.zeros(len(timesteps)),
        }

        for its, ts in enumerate(timesteps):
            if ts <= 0:
                continue
            # All solving times < ts are considered as solved within preschedule
            # Each algorithm in preschedule gets equal time
            time_per_algo = ts / self.size_preschedule
            perf = np.where(
                self.performance_train > time_per_algo, self.performance_train, 0
            )

            # Search the best combination of algorithms
            num_unsolved_instances = np.zeros((numSolvers,) * self.size_preschedule)
            i_algos = (range(numSolvers),) * self.size_preschedule

            for algos in product(*i_algos):
                perf_k_algos = np.hstack([perf[:, a].reshape((-1, 1)) for a in algos])
                num_unsolved_instances[algos] = np.count_nonzero(
                    np.min(perf_k_algos, axis=1)
                )

            best_config["rate_solved_instances"][its] = (
                numInstances - np.min(num_unsolved_instances)
            ) / numInstances
            best_config["algos"][its, :] = np.unravel_index(
                np.argmin(num_unsolved_instances), (numSolvers,) * self.size_preschedule
            )

        # Identify the best timestep
        step_lim = self._identify_end_preschedule(
            timesteps,
            best_config["rate_solved_instances"],
            self.max_runtime_preschedule,
        )

        self.ialgos_preschedule = best_config["algos"][step_lim, :]
        # Each algorithm in preschedule gets equal time initially
        time_per_algo = timesteps[step_lim] / self.size_preschedule
        self.runtimes_preschedule = np.array((time_per_algo,) * self.size_preschedule)

        if self.verbosity > 0:
            print(f"Preschedule algorithms: {self.ialgos_preschedule}")
            print(f"Initial preschedule times: {self.runtimes_preschedule}")
            print(f"Solved rate: {best_config['rate_solved_instances'][step_lim]:.3f}")

    def _identify_end_preschedule(
        self, timesteps: np.ndarray, rate: np.ndarray, max_time_schedule: float
    ) -> int:
        """
        Identify the optimal endpoint for the preschedule based on
        the trade-off between time spent and instances solved.
        """
        if len(rate) <= 1 or (rate[-1] - rate[0]) < 1e-10:
            return len(timesteps) - 1

        normalized_rate = rate / (rate[-1] - rate[0])
        criterion = (normalized_rate - normalized_rate[0]) * (timesteps - 0) + (
            normalized_rate[-1] - normalized_rate
        ) * (timesteps[-1] - timesteps)

        if timesteps[np.argmin(criterion)] < max_time_schedule / self.size_preschedule:
            return int(np.argmin(criterion))
        else:
            return len(timesteps) - 1

    def _optimize_preschedule_de(self) -> None:
        """
        Optimize preschedule time allocations using differential evolution.

        This follows the original ASAP v2 approach where:
        - The preschedule algorithms are fixed
        - A "selector algorithm" is appended (gets remaining time after preschedule)
        - The selector represents the best algorithm predicted for each instance
        """
        if self.verbosity > 0:
            print("Optimizing preschedule with differential evolution...")

        if self.runtimes_preschedule is None or self.ialgos_preschedule is None:
            return

        # Total time allocated to preschedule (selector gets remaining time)
        total_runtime_preschedule = np.sum(self.runtimes_preschedule)

        # For the original ASAP v2, the selector is the oracle (best algorithm per instance)
        # This simulates having a perfect predictor
        best_algorithm_per_instance = np.argmin(self.performance_train, axis=1)

        # Build selector column: the performance of the best algorithm for each instance
        selector_column = np.array(
            [
                self.performance_train[i, best_algorithm_per_instance[i]]
                for i in range(len(self.performance_train))
            ]
        ).reshape((-1, 1))

        # Extended performance matrix: original + selector column
        extd_performance_matrix = np.hstack([self.performance_train, selector_column])

        # Preschedule algorithm indices + selector index (numAlg)
        ialgos_preschedule_ext = np.append(self.ialgos_preschedule, self.numAlg)
        runtimes_preschedule_ext = np.append(
            self.runtimes_preschedule,
            max(self.budget - np.sum(self.runtimes_preschedule), 0.0),
        )

        def encode_runtimes(rt: np.ndarray) -> np.ndarray:
            """Encode runtime for optimization (all but last 2 elements)"""
            rt_ = rt / total_runtime_preschedule
            x_ = np.zeros((rt_.size - 2,))
            if np.sum(rt_[:-1]) <= 1.0:
                x_ = rt_[:-2]
            else:
                for irt in range(len(rt_) - 2):
                    if np.cumsum(rt_)[irt] < 1.0:
                        x_[irt] = rt_[irt]
                    else:
                        x_[irt] = 1.0 - np.sum(x_[:irt])
            return x_

        def decode_runtimes(x: np.ndarray) -> np.ndarray:
            """Decode runtime from optimization variables"""
            x_ = np.abs(x)
            rt = np.zeros((x_.size + 2,))

            if len(x_) > 0 and np.cumsum(x_)[-1] <= 1.0:
                rt[:-2] = x_ * total_runtime_preschedule
            else:
                rt[:-2] = np.where(
                    1.0 - np.cumsum(x_) >= 0,
                    x_ * total_runtime_preschedule,
                    np.maximum(np.roll(1.0 - np.cumsum(x_), 1), np.zeros(x_.shape))
                    * total_runtime_preschedule,
                )
                if len(x_) > 0 and x_[0] > 1.0:
                    rt[0] = total_runtime_preschedule

            rt[-2] = (
                max((1.0 - np.cumsum(x_)[-1]) * total_runtime_preschedule, 0)
                if len(x_) > 0
                else total_runtime_preschedule
            )
            rt[-1] = self.budget - np.sum(rt[:-1])

            return rt

        def objective_function(x_: np.ndarray) -> float:
            """Evaluate preschedule performance with regularization"""
            x = decode_runtimes(np.abs(x_))
            rts_p = x[:-1]  # Preschedule runtimes (excluding selector's remaining time)

            # Compute time to solve for all instances
            time_to_solve = self._get_time_to_solve(
                extd_performance_matrix, ialgos_preschedule_ext, x
            )

            runtime_res = np.sum(time_to_solve)

            # Regularization: penalize uneven time distribution
            reg = 0.0
            if self.regularization_weight > 0:
                reg = np.linalg.norm(rts_p / np.sum(rts_p))
                reg *= (
                    self.regularization_weight
                    * len(extd_performance_matrix)
                    * self.budget
                )

            # Variance penalty
            var_pen = 0.0
            if self.variance_weight > 0:
                partial_var = np.var(time_to_solve[time_to_solve < self.budget])
                var_pen = self.variance_weight * partial_var

            return float(runtime_res + reg + var_pen)

        # Encode initial guess
        schedule_ini = encode_runtimes(runtimes_preschedule_ext.astype(float)).reshape(
            (-1,)
        )

        if len(schedule_ini) >= 1:
            # Set up bounds
            bounds = [(0.0, 1.0) for _ in range(len(schedule_ini))]

            try:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    seed=self.seed,
                    popsize=self.de_popsize,
                    maxiter=self.de_maxiter,
                    disp=self.verbosity > 0,
                    x0=schedule_ini,
                )
                optimized_runtimes = decode_runtimes(result.x)
                # Update preschedule runtimes (excluding selector time)
                self.runtimes_preschedule = np.delete(optimized_runtimes, -1)

                if self.verbosity > 0:
                    print(f"Optimization completed. Final objective: {result.fun}")
                    print(f"Optimized preschedule times: {self.runtimes_preschedule}")

            except Exception as e:
                if self.verbosity > 0:
                    print(f"Optimization failed: {e}")
                    print("Using initial runtimes")
        else:
            # Single variable optimization
            try:
                res_optim = minimize_scalar(
                    lambda x: objective_function(np.asarray([x])),
                    bounds=(0, 1),
                    method="bounded",
                )
                optimized_runtimes = decode_runtimes(np.asarray([res_optim.x]))
                self.runtimes_preschedule = np.delete(optimized_runtimes, -1)
            except Exception as e:
                if self.verbosity > 0:
                    print(f"Scalar optimization failed: {e}")

    def _get_time_to_solve(
        self,
        performance_matrix: np.ndarray,
        i_solvers: np.ndarray,
        runtimes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute time to solve for each instance given a schedule.

        Parameters
        ----------
        performance_matrix : np.ndarray
            Performance matrix (instances x algorithms), including selector column.
        i_solvers : np.ndarray
            Indices of algorithms in the schedule.
        runtimes : np.ndarray
            Time allocated to each algorithm in the schedule.

        Returns
        -------
        np.ndarray
            Time to solve for each instance.
        """
        n_instances = performance_matrix.shape[0]
        n_steps = len(runtimes)

        time_to_solve = np.zeros((n_instances,), dtype=float)
        is_solved_per_algo = performance_matrix[:, i_solvers] < np.abs(runtimes)
        is_already_solved = np.zeros((n_instances,), dtype=bool)
        is_newly_solved = np.zeros((n_instances,), dtype=bool)

        for runid in range(n_steps):
            is_newly_solved[:] = is_solved_per_algo[:, runid] & (~is_already_solved)
            time_to_solve[is_newly_solved] += performance_matrix[:, i_solvers[runid]][
                is_newly_solved
            ]
            is_already_solved |= is_newly_solved
            time_to_solve[~is_already_solved] += np.abs(runtimes)[runid]

        # PAR10 penalty for unsolved instances
        time_to_solve[time_to_solve > self.budget] = 10.0 * self.budget
        time_to_solve[~is_already_solved] = 10.0 * self.budget

        return time_to_solve

    def _build_schedule(self) -> None:
        """Build the final schedule from preschedule algorithms"""
        schedule = []

        if (
            self.ialgos_preschedule is not None
            and self.runtimes_preschedule is not None
        ):
            for i, (alg_idx, time_alloc) in enumerate(
                zip(self.ialgos_preschedule, self.runtimes_preschedule)
            ):
                if time_alloc > 0:
                    alg_name = self.algorithms[alg_idx]
                    schedule.append((alg_name, round(float(time_alloc), 3)))

        self.schedule = schedule

        if self.verbosity > 0:
            print()
            print(f"Final preschedule: {self.schedule}")
            print("+ " * 40)

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | None = None,
        performance: pd.DataFrame | np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
        """
        Return the predicted schedule.

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

    def get_preschedule_config(self) -> dict[str, float]:
        """Get the optimized preschedule configuration (only non-zero times)"""
        if (
            self.algorithms
            and self.runtimes_preschedule is not None
            and self.ialgos_preschedule is not None
        ):
            return {
                self.algorithms[alg_idx]: time
                for alg_idx, time in zip(
                    self.ialgos_preschedule, self.runtimes_preschedule
                )
                if time > 0
            }
        return {}

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, Any] | None = None,
        budget: float | None = None,
        maximize: bool = False,
        presolver_name: str | None = None,
        **kwargs: Any,
    ) -> ASAPv2:
        """
        Create an ASAPv2 presolver from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            The configuration.
        pre_prefix : str, default=""
            Prefix for the configuration keys.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ASAPv2
            The initialized presolver.
        """
        if presolver_name:
            prefix = f"{presolver_name}:"
        else:
            prefix = ""

        # AbstractPresolver adds presolver_budget to the config
        # We use it to set max_runtime_preschedule
        presolver_budget = configuration.get(
            f"{prefix}presolver_budget", configuration.get("presolver_budget")
        )

        init_params = kwargs.copy()
        if presolver_budget is not None:
            init_params["max_runtime_preschedule"] = presolver_budget

        return cls(**init_params)
