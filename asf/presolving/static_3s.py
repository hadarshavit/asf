from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import pulp

    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False

from asf.presolving.presolver import AbstractPresolver


class Static3S(AbstractPresolver):
    """
    Compute a static presolve schedule by solving a resource-constrained set
    covering problem (RCSCP). If an IP solver (pulp) is available the exact
    formulation is solved, otherwise a greedy heuristic is used.

    The produced schedule is a list of (solver_name, time) tuples.

    Attributes:
        budget (float): overall time budget for the preschedule.
        max_candidates_per_solver (int): max distinct candidate times per solver.
    """

    PREFIX = "static_3s"

    def __init__(
        self,
        runcount_limit: float = 0.0,
        budget: float = 200.0,
        max_candidates_per_solver: int = 20,
        **kwargs,
    ):
        super().__init__(runcount_limit=runcount_limit, budget=budget)
        self.runcount_limit = float(runcount_limit)
        self.budget = float(budget)
        self.max_candidates_per_solver = int(max_candidates_per_solver)
        self.schedule: List[Tuple[str, float]] = None
        self.algorithms: List[str] = []

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Build a static schedule from training performance matrix (instances x solvers).
        """
        perf = performance.copy()
        instances = list(perf.index)
        self.algorithms = list(perf.columns)

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
        else:
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
            prob += (
                pulp.lpSum([t * var for (s, t), var in x_vars.items()]) <= self.budget
            )

            # Solver selection constraints
            for s in self.algorithms:
                solver_x_vars = [
                    var
                    for (solver_name, time), var in x_vars.items()
                    if solver_name == s
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
            return

    def predict(
        self, features: pd.DataFrame | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        if self.schedule is None:
            raise ValueError("Must call fit() before predict()")
        if features is None:
            return {"default": self.schedule}
        out: dict[str, list[tuple[str, float]]] = {}
        for instance_id in features.index:
            out[instance_id] = list(self.schedule)
        return out

    def get_preschedule_config(self) -> dict[str, float]:
        return {alg: time for alg, time in self.schedule}

    def get_configuration(self) -> dict:
        return {
            "algorithms": self.algorithms,
            "budget": self.budget,
            "preschedule_config": self.get_preschedule_config(),
        }
