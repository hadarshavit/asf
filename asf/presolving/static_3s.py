"""
Static 3S presolver - Resource-constrained set covering problem.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

try:
    from ConfigSpace import Configuration  # noqa: F401

    _HAS_CONFIGSPACE = True
except ImportError:
    _HAS_CONFIGSPACE = False

from scipy.optimize import Bounds, LinearConstraint, milp

from asf.presolving.presolver import AbstractPresolver, resolve_presolver_budget


class Static3S(AbstractPresolver):
    """
    Compute a static presolve schedule by solving a resource-constrained set
    covering problem (RCSCP).

    The formulation is solved with SciPy's HiGHS-backed mixed-integer solver.

    Parameters
    ----------
    runcount_limit : float, default=0.0
        Dummy parameter for compatibility.
    presolver_budget : float, default=200.0
        Overall time budget for the preschedule.
    max_candidates_per_solver : int, default=20
        Max distinct candidate times per solver to consider.
    **kwargs : Any
        Additional keyword arguments.
    """

    PREFIX: str = "static_3s"

    def __init__(
        self,
        init_params: dict[str, Any] | None = None,
        runcount_limit: float = 0.0,
        presolver_budget: float | None = None,
        max_candidates_per_solver: int = 20,
        **kwargs: Any,
    ) -> None:
        params = init_params if isinstance(init_params, dict) else {}
        params.update(kwargs)

        presolver_budget = resolve_presolver_budget(presolver_budget, params, 200.0)
        runcount_limit = params.pop("runcount_limit", runcount_limit)
        max_candidates_per_solver = params.pop(
            "max_candidates_per_solver", max_candidates_per_solver
        )

        super().__init__(presolver_budget=presolver_budget, **params)
        self.runcount_limit = float(runcount_limit)
        self.max_candidates_per_solver = int(max_candidates_per_solver)
        self.schedule: list[tuple[str, float]] | None = None
        self.algorithms: list[str] = []

    @staticmethod
    def _define_hyperparameters(
        total_budget: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for Static3S.
        """
        from ConfigSpace import Integer

        hps, conds, forbs = AbstractPresolver._define_hyperparameters(
            total_budget=total_budget, **kwargs
        )

        hps.append(
            Integer(
                "max_candidates_per_solver",
                bounds=(5, 100),
                default=20,
            )
        )

        return hps, conds, forbs

    def fit(
        self,
        features: pd.DataFrame | np.ndarray | None,
        performance: pd.DataFrame | np.ndarray | None,
        **kwargs: Any,
    ) -> None:
        """
        Build a static schedule from performance data.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The instance features. Not used.
        performance : pd.DataFrame or np.ndarray
            The algorithm performances (n_instances x n_algorithms).
        """
        if performance is None:
            raise ValueError("Static3S requires performance data for fitting.")
        self.schedule = None

        if isinstance(performance, pd.DataFrame):
            perf = performance.copy()
            instances = list(perf.index)
            self.algorithms = list(perf.columns)
        else:
            perf = pd.DataFrame(performance)
            instances = list(range(len(perf)))
            self.algorithms = [f"a{i}" for i in range(cast(Any, performance).shape[1])]

        # Build candidate (solver, time) pairs.
        candidates = {}
        for s in self.algorithms:
            vals = perf[s].replace([np.inf, -np.inf], np.nan).dropna()
            vals = vals[vals <= self.presolver_budget]
            if vals.empty:
                candidates[s] = []
                continue
            uniq = np.unique(vals.values)
            uniq = np.sort(uniq)
            if len(uniq) > self.max_candidates_per_solver:
                idx = np.linspace(
                    0, len(uniq) - 1, self.max_candidates_per_solver
                ).astype(int)
                uniq = uniq[idx]
            candidates[s] = list(map(float, uniq))

        # No available candidates fallback
        any_cands = any(len(v) > 0 for v in candidates.values())
        if not any_cands:
            self.schedule = []
            return

        actions = [(s, t) for s, times in candidates.items() for t in times]
        n_actions = len(actions)
        n_instances = len(instances)
        n_vars = n_actions + n_instances
        objective = np.zeros(n_vars)
        objective[:n_actions] = [t for _, t in actions]
        objective[n_actions:] = self.presolver_budget + 1.0

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        for row_i, i in enumerate(instances):
            row = np.zeros(n_vars)
            row[n_actions + row_i] = 1.0
            for action_i, (s, t) in enumerate(actions):
                try:
                    rt_f = float(perf.at[i, s])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(rt_f) and rt_f <= t:
                    row[action_i] = 1.0
            rows.append(row)
            lower.append(1.0)
            upper.append(np.inf)

        row = np.zeros(n_vars)
        row[:n_actions] = objective[:n_actions]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(self.presolver_budget)

        for solver in self.algorithms:
            row = np.zeros(n_vars)
            for action_i, (s, _) in enumerate(actions):
                if s == solver:
                    row[action_i] = 1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(1.0)

        result = milp(
            objective,
            integrality=np.ones(n_vars),
            bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
            constraints=LinearConstraint(np.asarray(rows), lower, upper),
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                "Static3S could not find a feasible schedule: " + result.message
            )

        chosen = [
            (s, float(t))
            for action_i, (s, t) in enumerate(actions)
            if result.x[action_i] > 0.5
        ]
        chosen.sort(key=lambda x: x[1])

        total_time = sum(t for _, t in chosen)
        if total_time < float(self.presolver_budget) and chosen:
            remaining = float(self.presolver_budget) - total_time
            alg, last_t = chosen[-1]
            chosen[-1] = (alg, float(last_t + remaining))

        self.schedule = chosen

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
        if self.schedule is None:
            raise ValueError("Static3S has not been fitted yet.")
        if features is not None:
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features)
            return {str(inst): self.schedule for inst in features.index}
        return self.schedule

    def get_preschedule_config(self) -> dict[str, float]:
        """Get the optimized preschedule configuration."""
        if self.schedule is None:
            return {}
        return {alg: time for alg, time in self.schedule}

    def get_configuration(self) -> dict[str, Any]:
        """Get the configuration of the fitted presolver."""
        return {
            "algorithms": self.algorithms,
            "presolver_budget": self.presolver_budget,
            "preschedule_config": self.get_preschedule_config(),
        }
