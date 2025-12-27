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
    import ConfigSpace  # noqa: F401

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
    presolver_budget : float, default=30.0
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
        init_params: dict[str, Any] | None = None,
        presolver_budget: float = 30.0,
        maximize: bool = False,
        algorithm_config: dict[str, tuple[bool, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        params = init_params if isinstance(init_params, dict) else {}
        params.update(kwargs)

        if "presolver_budget" in params:
            presolver_budget = params.pop("presolver_budget")
            params.pop("budget", None)
        else:
            presolver_budget = params.pop("budget", presolver_budget)
        maximize = params.pop("maximize", maximize)
        algorithm_config = params.pop("algorithm_config", algorithm_config)

        # Special handling for reconstruction from configuration
        if algorithm_config is None:
            # Try to build it from leftovers in params if they follow the naming convention
            # from _define_hyperparameters
            algorithm_config = {}
            # We need the algorithm list to know which keys to look for
            # This information should be passed in init_params or kwargs
            algorithms = params.pop("algorithms", [])
            for algo in algorithms:
                safe_algo_name = algo.replace(":", "_").replace(" ", "_")
                use_key = f"use_{safe_algo_name}"
                time_key = f"time_{safe_algo_name}"
                if use_key in params:
                    use_algo = params.pop(use_key)
                    time_val = params.pop(time_key, 5.0)
                    algorithm_config[algo] = (bool(use_algo), float(time_val))

        super().__init__(presolver_budget=presolver_budget, maximize=maximize, **params)
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

    @staticmethod
    def _define_hyperparameters(
        total_budget: float | None = None,
        algorithms: list[str] | None = None,
        max_time_per_algo: float = 30.0,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for ConfigurablePresolver.
        """
        from ConfigSpace import (
            Categorical,
            EqualsCondition,
            Float,
        )

        hps, conds, forbs = AbstractPresolver._define_hyperparameters(
            total_budget=total_budget, **kwargs
        )

        if not algorithms:
            return hps, conds, forbs

        for algo in algorithms:
            # Sanitize algorithm name for use as parameter name
            safe_algo_name = algo.replace(":", "_").replace(" ", "_")

            # Boolean parameter: whether to use this algorithm
            use_algo_param = Categorical(
                name=f"use_{safe_algo_name}",
                items=[True, False],
                default=False,
            )
            hps.append(use_algo_param)

            # Float parameter: time budget for this algorithm (conditional on use_algo=True)
            time_param = Float(
                name=f"time_{safe_algo_name}",
                bounds=(0.1, max_time_per_algo),
                default=min(5.0, max_time_per_algo),
                log=True,
            )
            hps.append(time_param)

            # Time is only relevant if the algorithm is enabled
            time_condition = EqualsCondition(
                child=time_param,
                parent=use_algo_param,
                value=True,
            )
            conds.append(time_condition)

        return hps, conds, forbs

    def __repr__(self) -> str:
        """Return a string representation of the presolver."""
        enabled = [
            f"{algo}={time:.2f}s"
            for algo, (use, time) in self.algorithm_config.items()
            if use
        ]
        return f"ConfigurablePresolver(schedule=[{', '.join(enabled)}])"
