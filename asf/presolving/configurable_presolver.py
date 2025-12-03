"""
Configurable Presolver - A presolver with a configurable schedule via ConfigSpace.

This presolver allows users to define the presolving schedule through a
configuration space, specifying which algorithms to use and for how long.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from asf.presolving.presolver import AbstractPresolver

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Configuration,
        Categorical,
        Float,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class ConfigurablePresolver(AbstractPresolver):
    """
    A presolver that uses a configurable schedule defined via ConfigSpace.

    The configuration space allows specifying:
    - Whether to use each algorithm (True/False)
    - Time budget allocation for each algorithm

    This enables hyperparameter optimization of the presolving schedule.

    Attributes:
        schedule (list[tuple[str, float]]): The configured schedule of (algorithm, time) pairs.
        algorithm_config (dict[str, tuple[bool, float]]): Per-algorithm configuration
            mapping algorithm name to (use_algorithm, time_budget).
    """

    PREFIX = "configurable_presolver"

    def __init__(
        self,
        budget: float = 30.0,
        maximize: bool = False,
        algorithm_config: dict[str, tuple[bool, float]] | None = None,
        **kwargs,
    ):
        """
        Initialize the ConfigurablePresolver.

        Args:
            budget: Total time budget for pre-solving (used as upper bound for validation).
            maximize: If True, maximize performance values instead of minimize.
            algorithm_config: Dictionary mapping algorithm names to (use_algorithm, time_budget).
                If None, the presolver will not have any algorithms in its schedule.
        """
        super().__init__(budget=budget, maximize=maximize)
        self.algorithm_config = algorithm_config or {}
        self.schedule: list[tuple[str, float]] = []
        self.algorithms: list[str] = []

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the presolver - builds the schedule from the algorithm_config.

        The fit method validates that configured algorithms exist in the performance
        data and builds the final schedule.

        Args:
            features: DataFrame of instance features (not used, but required by interface).
            performance: DataFrame of runtimes/performance (n_instances x n_algorithms).
        """
        self.algorithms = list(performance.columns)
        self.schedule = []

        # Build schedule from algorithm_config
        for algo_name, (use_algo, time_budget) in self.algorithm_config.items():
            if use_algo and algo_name in self.algorithms and time_budget > 0:
                self.schedule.append((algo_name, time_budget))

        # Sort by time budget (shorter times first - run quick solvers first)
        self.schedule.sort(key=lambda x: x[1])

    def predict(self) -> list[tuple[str, float]]:
        """
        Return the configured pre-solve schedule.

        Returns:
            List of (algorithm_name, cutoff_time) tuples representing the schedule.
        """
        return self.schedule

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        cs_transform: dict[str, dict[str, type]] | None = None,
        algorithms: list[str] | None = None,
        max_time_per_algo: float = 30.0,
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
        **kwargs,
    ) -> tuple[ConfigurationSpace, dict[str, dict[str, type]]]:
        """
        Get the configuration space for the ConfigurablePresolver.

        The configuration space includes:
        - For each algorithm: a boolean to enable/disable it
        - For each algorithm: a float for the time budget (conditional on being enabled)

        Args:
            cs: The configuration space to use. If None, a new one will be created.
            cs_transform: A dictionary for transforming configuration space values.
            algorithms: List of algorithm names to include in the configuration space.
                If None, raises ValueError.
            max_time_per_algo: Maximum time budget that can be allocated per algorithm.
            pre_prefix: Prefix for parameter names (for hierarchical configuration spaces).
            parent_param: Parent parameter for conditional configuration.
            parent_value: Value of parent parameter that activates these parameters.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tuple of (ConfigurationSpace, transformation dictionary).

        Raises:
            RuntimeError: If ConfigSpace is not installed.
            ValueError: If algorithms list is None or empty.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if algorithms is None or len(algorithms) == 0:
            raise ValueError(
                "algorithms must be provided to create ConfigurablePresolver configuration space"
            )

        if cs is None:
            cs = ConfigurationSpace()

        if cs_transform is None:
            cs_transform = {}

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{ConfigurablePresolver.PREFIX}"
        else:
            prefix = ConfigurablePresolver.PREFIX

        all_params = []
        all_conditions = []

        for algo in algorithms:
            # Sanitize algorithm name for use as parameter name
            safe_algo_name = algo.replace(":", "_").replace(" ", "_")

            # Boolean parameter: whether to use this algorithm
            use_algo_param = Categorical(
                name=f"{prefix}:use_{safe_algo_name}",
                items=[True, False],
                default=False,
            )
            all_params.append(use_algo_param)

            # Float parameter: time budget for this algorithm (conditional on use_algo=True)
            time_param = Float(
                name=f"{prefix}:time_{safe_algo_name}",
                bounds=(0.1, max_time_per_algo),
                default=min(5.0, max_time_per_algo),
                log=True,  # Log scale for time values
            )
            all_params.append(time_param)

            # Time is only relevant if the algorithm is enabled
            time_condition = EqualsCondition(
                child=time_param,
                parent=use_algo_param,
                value=True,
            )
            all_conditions.append(time_condition)

            # If there's a parent parameter, add conditions for the use_algo param
            if parent_param is not None:
                parent_condition = EqualsCondition(
                    child=use_algo_param,
                    parent=parent_param,
                    value=parent_value,
                )
                all_conditions.append(parent_condition)

        # Store algorithm list in transform for reconstruction
        cs_transform[f"{prefix}:algorithms"] = algorithms

        cs.add(all_params + all_conditions)

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, dict[str, type]],
        pre_prefix: str = "",
        **kwargs,
    ) -> "ConfigurablePresolver":
        """
        Create a ConfigurablePresolver instance from a configuration.

        Args:
            configuration: The configuration object or dictionary.
            cs_transform: The transformation dictionary for the configuration space.
            pre_prefix: Prefix for parameter names.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A ConfigurablePresolver instance configured according to the configuration.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{ConfigurablePresolver.PREFIX}"
        else:
            prefix = ConfigurablePresolver.PREFIX

        # Get algorithms list from transform
        algorithms = cs_transform.get(f"{prefix}:algorithms", [])

        # Build algorithm_config from configuration
        algorithm_config = {}
        total_budget = 0.0

        for algo in algorithms:
            safe_algo_name = algo.replace(":", "_").replace(" ", "_")
            use_key = f"{prefix}:use_{safe_algo_name}"
            time_key = f"{prefix}:time_{safe_algo_name}"

            use_algo = configuration.get(use_key, False)
            if use_algo:
                time_budget = configuration.get(time_key, 5.0)
                algorithm_config[algo] = (True, time_budget)
                total_budget += time_budget
            else:
                algorithm_config[algo] = (False, 0.0)

        return ConfigurablePresolver(
            budget=total_budget if total_budget > 0 else kwargs.get("budget", 30.0),
            algorithm_config=algorithm_config,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a string representation of the presolver."""
        enabled = [
            f"{algo}={time:.2f}s"
            for algo, (use, time) in self.algorithm_config.items()
            if use
        ]
        return f"ConfigurablePresolver(schedule=[{', '.join(enabled)}])"
