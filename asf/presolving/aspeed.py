"""
Answer Set Programming (ASP) based presolver.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from asf.presolving.presolver import AbstractPresolver

try:
    import clingo
    import clingo.script

    clingo.script.enable_python()
    CLINGO_AVAIL = True
except ImportError:
    CLINGO_AVAIL = False


class Aspeed(AbstractPresolver):
    """
    A presolver class that uses Answer Set Programming (ASP) to compute a schedule.

    Parameters
    ----------
    budget : float, default=30.0
        The total time budget for the presolver.
    aspeed_cutoff : int, default=60
        Time limit for the ASP solver in seconds.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    cores : int, default=1
        Number of CPU cores to use for the ASP solver.
    data_threshold : int, default=300
        Minimum number of instances to use for subsampling.
    data_fraction : float, default=0.3
        Fraction of instances to use for subsampling if above data_threshold.
    """

    def __init__(
        self,
        budget: float = 30.0,
        aspeed_cutoff: int = 60,
        maximize: bool = False,
        cores: int = 1,
        data_threshold: int = 300,
        data_fraction: float = 0.3,
    ) -> None:
        if not CLINGO_AVAIL:
            raise ImportError(
                "clingo is not installed. Please install it to use the Aspeed presolver."
            )
        super().__init__(budget=budget, maximize=maximize)
        self.cores = cores
        self.data_threshold = data_threshold
        self.data_fraction = data_fraction
        self.aspeed_cutoff = aspeed_cutoff
        self.schedule: list[tuple[str, float]] = []
        self.algorithms: list[str] = []

    def fit(
        self,
        features: pd.DataFrame | np.ndarray | None,
        performance: pd.DataFrame | np.ndarray | None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the presolver to the data.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The instance features.
        performance : pd.DataFrame or np.ndarray
            The algorithm performances.
        """
        if performance is None:
            raise ValueError("Aspeed requires performance data for fitting.")
        if isinstance(performance, pd.DataFrame):
            perf_frame = performance
            self.algorithms = list(performance.columns)
        else:
            perf_frame = pd.DataFrame(performance)
            self.algorithms = [f"a{i}" for i in range(performance.shape[1])]

        # ASP program with dynamic number of cores
        asp_program = """
#script(python)

from clingo import Number, Tuple_, Function
from clingo.symbol import parse_term

ts = {}
def insert(i,s,t):
  key = str(s)
  if not ts.get(key):
    ts[key] = []
  ts[key].append([i,t])
  return parse_term("1")

def order(s):
  key = str(s)
  if not ts.get(key):
    ts[key] = []
  ts[key].sort(key=lambda x: int(x[1].number))
  p = None
  r = []
  for i, v in ts[key]:
    if p:
      r.append(Tuple_([p,i]))
    p = i
  return Tuple_(r)

#end.

#const cores=1.

solver(S)  :- time(_,S,_).
time(S,T)  :- time(_,S,T).
unit(1..cores).

insert(@insert(I,S,T)) :- time(I,S,T).
order(I,K,S) :- insert(_), solver(S), (I,K) = @order(S).

{ slice(U,S,T) : time(S,T), T <= K, unit(U) } 1 :-
  solver(S), kappa(K).
slice(S,T) :- slice(_,S,T).

 :- not #sum { T,S : slice(U,S,T) } K, kappa(K), unit(U).

solved(I,S) :- slice(S,T), time(I,S,T).
solved(I,S) :- solved(J,S), order(I,J,S).
solved(I)   :- solved(I,_).

#maximize { 1@2,I: solved(I) }.
#minimize { T*T@1,S : slice(S,T)}.

#show slice/3.
    """

        # Create a Clingo Control object
        ctl = clingo.Control(arguments=[f"-t{self.cores}"])
        ctl.add(asp_program)

        # Subsample if needed
        if perf_frame.shape[0] > self.data_threshold:
            random_indx = np.random.choice(
                range(perf_frame.shape[0]),
                size=min(
                    perf_frame.shape[0],
                    max(
                        int(perf_frame.shape[0] * self.data_fraction),
                        self.data_threshold,
                    ),
                ),
                replace=True,
            )
            perf_frame = perf_frame.iloc[random_indx, :]

        times = [
            "time(i%d, %d, %d)." % (i, j, max(1, math.ceil(perf_frame.iloc[i, j])))
            for i in range(perf_frame.shape[0])
            for j in range(perf_frame.shape[1])
        ]

        kappa = "kappa(%d)." % (self.budget)
        data_in = "\n".join(times) + "\n" + kappa
        ctl.add(data_in)

        try:
            ctl.ground([("base", [])])
        except Exception:
            ctl.ground()

        def clingo_callback(model: clingo.Model) -> None:
            """Callback function to process the Clingo model."""
            schedule_dict = {}
            for symbol in model.symbols(shown=True):
                try:
                    algo = self.algorithms[symbol.arguments[1].number]
                except Exception:
                    algo = str(symbol.arguments[1])
                runcount_limit = symbol.arguments[2].number
                schedule_dict[algo] = runcount_limit

            self.schedule = sorted(schedule_dict.items(), key=lambda x: x[1])

        try:
            with ctl.solve(on_model=clingo_callback, async_=True) as handle:
                if handle.wait(self.aspeed_cutoff):
                    handle.get()
                else:
                    handle.cancel()
        except Exception as e:
            print(f"Clingo solving failed: {e}")
            self.schedule = []

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
