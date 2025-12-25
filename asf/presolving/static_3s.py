"""
Static 3S presolver - Resource-constrained set covering problem.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

try:
    from ConfigSpace import Configuration

    _HAS_CONFIGSPACE = True
except ImportError:
    _HAS_CONFIGSPACE = False

try:
    import pulp

    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False

from asf.presolving.presolver import AbstractPresolver


class Static3S(AbstractPresolver):
    """
    Compute a static presolve schedule by solving a resource-constrained set
    covering problem (RCSCP).

    If an IP solver (pulp) is available the exact formulation is solved,
    otherwise it raises an ImportError as the greedy heuristic is not implemented.

    Parameters
    ----------
    runcount_limit : float, default=0.0
        Dummy parameter for compatibility.
    budget : float, default=200.0
        Overall time budget for the preschedule.
    max_candidates_per_solver : int, default=20
        Max distinct candidate times per solver to consider.
    **kwargs : Any
        Additional keyword arguments.
    """

    PREFIX: str = "static_3s"

    def __init__(
        self,
        runcount_limit: float = 0.0,
        budget: float = 200.0,
        max_candidates_per_solver: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(budget=budget)
        self.runcount_limit = float(runcount_limit)
        self.budget = float(budget)
        self.max_candidates_per_solver = int(max_candidates_per_solver)
        self.schedule: list[tuple[str, float]] | None = None
        self.algorithms: list[str] = []

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
            vals = vals[vals <= self.budget]
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

        if not _HAS_PULP:
            raise ImportError(
                "pulp is required to use Static3S presolver. Please install pulp."
            )

        prob = pulp.LpProblem("static_schedule_rcscp", pulp.LpMinimize)
        x_vars = {}
        for s, times in candidates.items():
            for t in times:
                var = pulp.LpVariable(
                    f"x_{s}_{t:.4f}".replace(".", "_"), cat=pulp.LpBinary
                )
                x_vars[(s, t)] = var

        y_vars = {}
        for i in instances:
            y_vars[i] = pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary)

        # Objective: (C+1)*sum y_i + sum t * x_{s,t}
        bigC = self.budget + 1.0
        prob += bigC * pulp.lpSum([y_vars[i] for i in instances]) + pulp.lpSum(
            [t * var for (s, t), var in x_vars.items()]
        )

        # Covering constraints
        for i in instances:
            terms = [y_vars[i]]
            for (s, t), var in x_vars.items():
                # if solver s solves instance i within time t
                rt = perf.at[i, s]
                if pd.isna(rt):
                    continue
                try:
                    rt_f = float(rt)
                except Exception:
                    continue
                if rt_f <= t:
                    terms.append(var)
            prob += pulp.lpSum(terms) >= 1

        # Resource constraint
        prob += pulp.lpSum([t * var for (s, t), var in x_vars.items()]) <= self.budget

        # Solver selection constraints
        for s in self.algorithms:
            solver_x_vars = [
                var for (solver_name, time), var in x_vars.items() if solver_name == s
            ]
            prob += pulp.lpSum(solver_x_vars) <= 1, f"One_selection_{s}"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen = []
        for (s, t), var in x_vars.items():
            try:
                val = var.value()
            except Exception:
                val = None
            if val is not None and float(val) > 0.5:
                chosen.append((s, float(t)))
        chosen.sort(key=lambda x: x[1])

        total_time = sum(t for _, t in chosen)
        if total_time < float(self.budget) and chosen:
            remaining = float(self.budget) - total_time
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
            "budget": self.budget,
            "preschedule_config": self.get_preschedule_config(),
        }

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, Any] | None = None,
        budget: float | None = None,
        maximize: bool = False,
        presolver_name: str | None = None,
        **kwargs: Any,
    ) -> Static3S:
        """
        Create a Static3S presolver instance from a configuration.

        Parameters
        ----------
        configuration : dict
            The configuration.
        cs_transform : dict
            The transformation dictionary.
        budget : float or None, default=None
            Budget for the presolver.
        maximize : bool, default=False
            Whether to maximize the metric (not used).
        presolver_name : str or None, default=None
            Name of the presolver.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Static3S
            The Static3S presolver instance.

        Raises
        ------
        ValueError
            If budget is not provided.
        """
        if budget is None and presolver_name is not None:
            budget_key = f"{presolver_name}:presolver_budget"
            if budget_key in configuration:
                budget = configuration[budget_key]

        if budget is None:
            raise ValueError("Budget must be provided for Static3S presolver")

        return cls(budget=budget, **kwargs)
