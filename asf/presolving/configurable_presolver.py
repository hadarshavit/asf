"""
Configurable Presolver - A presolver with a configurable schedule via ConfigSpace.

This presolver allows users to define the presolving schedule through a
configuration space, specifying which algorithms to use and for how long.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from asf.presolving.presolver import AbstractPresolver

try:
    from ConfigSpace import (
        Categorical,
        Configuration,
        ConfigurationSpace,
        EqualsCondition,
        Float,
    )

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

    Parameters
    ----------
    budget : float, default=30.0
        Total time budget for pre-solving.
    maximize : bool, default=False
        If True, maximize performance values instead of minimize.
    algorithm_config : dict[str, tuple[bool, float]] or None, default=None
        Dictionary mapping algorithm names to (use_algorithm, time_budget).
    **kwargs : Any
        Additional keyword arguments.
    """

    PREFIX: str = "configurable_presolver"

    def __init__(
        self,
        budget: float = 30.0,
        maximize: bool = False,
        algorithm_config: dict[str, tuple[bool, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(budget=budget, maximize=maximize)
        self.algorithm_config = algorithm_config or {}
        self.schedule: list[tuple[str, float]] = []
        self.algorithms: list[str] = []

    def fit(
        self,
        features: pd.DataFrame | np.ndarray | None,
        performance: pd.DataFrame | np.ndarray | None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the presolver - builds the schedule from the algorithm_config.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The instance features.
        performance : pd.DataFrame or np.ndarray
            The algorithm performances.
        """
        if performance is None:
            raise ValueError(
                "ConfigurablePresolver requires performance data for fitting."
            )

        if isinstance(performance, pd.DataFrame):
            self.algorithms = list(performance.columns)
        else:
            self.algorithms = [f"a{i}" for i in range(cast(Any, performance).shape[1])]

        self.schedule = []

        # Build schedule from algorithm_config
        for algo_name, (use_algo, time_budget) in self.algorithm_config.items():
            if use_algo and algo_name in self.algorithms and time_budget > 0:
                self.schedule.append((algo_name, time_budget))

        # Sort by time budget (shorter times first - run quick solvers first)
        self.schedule.sort(key=lambda x: x[1])

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | None = None,
        performance: pd.DataFrame | np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
        """
        Return the configured pre-solve schedule.

        Parameters
        ----------
        features : pd.DataFrame or None, default=None
            The features for the instances. If provided, the schedule will be
            returned as a dictionary mapping instance IDs to the schedule.
        performance : pd.DataFrame or None, default=None
            The algorithm performances. Not used by ConfigurablePresolver.

        Returns
        -------
        list or dict
            The presolving schedule. If `features` is None, returns a list of
            (algorithm_name, time_budget) pairs. If `features` is provided,
            returns a dictionary mapping instance IDs to their respective schedules.
        """
        if features is not None:
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features)
            return {str(inst): self.schedule for inst in features.index}
        return self.schedule

    @classmethod
    def get_configuration_space(
        cls,
        cs: ConfigurationSpace | None = None,
        cs_transform: dict[str, Any] | None = None,
        parent_param: Any | None = None,
        parent_value: str | None = None,
        total_budget: float | None = None,
        algorithms: list[str] | None = None,
        max_time_per_algo: float = 30.0,
        pre_prefix: str = "",
        **kwargs: Any,
    ) -> tuple[ConfigurationSpace, dict[str, Any]]:
        """
        Get the configuration space.

        The configuration space includes:
        - For each algorithm: a boolean to enable/disable it
        - For each algorithm: a float for the time budget (conditional)

        Parameters
        ----------
        cs : ConfigurationSpace or None, default=None
            The configuration space to use.
        cs_transform : dict or None, default=None
            A dictionary for transforming configuration space values.
        algorithms : list[str] or None, default=None
            List of algorithm names to include in the configuration space.
        max_time_per_algo : float, default=30.0
            Maximum time budget that can be allocated per algorithm.
        pre_prefix : str, default=""
            Prefix for parameter names.
        parent_param : Any or None, default=None
            Parent parameter for conditional configuration.
        parent_value : str or None, default=None
            Value of parent parameter that activates these parameters.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            The configuration space and transformation dictionary.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        ValueError
            If algorithms list is None or empty.
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
            prefix = f"{pre_prefix}:{cls.PREFIX}"
        else:
            prefix = cls.PREFIX

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
                log=True,
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

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, Any] | None = None,
        budget: float | None = None,
        maximize: bool = False,
        presolver_name: str | None = None,
        pre_prefix: str = "",
        **kwargs: Any,
    ) -> ConfigurablePresolver:
        """
        Create a ConfigurablePresolver instance from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            The configuration object or dictionary.
        cs_transform : dict
            The transformation dictionary.
        pre_prefix : str, default=""
            Prefix for parameter names.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ConfigurablePresolver
            A ConfigurablePresolver instance.

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
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
        algorithms = (cs_transform or {}).get(f"{prefix}:algorithms", [])

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
                algorithm_config[algo] = (True, float(time_budget))
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
