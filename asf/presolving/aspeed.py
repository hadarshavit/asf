import clingo.script
import pandas as pd
import numpy as np
import math
import clingo
from asf.presolving.presolver import AbstractPresolver


clingo.script.enable_python()


class Aspeed(AbstractPresolver):
    """
    A presolver class that uses Answer Set Programming (ASP) to compute a schedule for solving instances.

    Attributes:
        cores (int): Number of CPU cores to use.
        cutoff (int): Time limit for solving.
        data_threshold (int): Minimum number of instances to use.
        data_fraction (float): Fraction of instances to use.
        schedule (list): Computed schedule of algorithms and their runcount_limits.
    """

    def __init__(
        self,
        runcount_limit: float = 100.0,
        budget: float = 30.0,
        aspeed_cutoff: int = 60,
        maximize: bool = False,
        cores: int = 1,
        data_threshold: int = 300,
        data_fraction: float = 0.3,
    ) -> None:
        """
        Initializes the Aspeed presolver.

        Args:
            metadata (dict): Metadata for the presolver.
            cores (int): Number of CPU cores to use.
            cutoff (int): Time limit for solving.
        """
        super().__init__(
            budget=budget, runcount_limit=runcount_limit, maximize=maximize
        )
        self.cores = cores
        self.data_threshold = data_threshold  # minimal number of instances to use
        self.data_fraction = data_fraction  # fraction of instances to use
        self.aspeed_cutoff = aspeed_cutoff  # time limit for solving
        self.schedule: list[tuple[str, float]] = []

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the presolver to the given features and performance data.

        Args:
            features (pd.DataFrame): A DataFrame containing feature data.
            performance (pd.DataFrame): A DataFrame containing performance data.
        """

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

        # Create a Clingo Control object with the specified number of threads
        ctl = clingo.Control(
            arguments=[
                f"--parallel-mode={self.cores}",
            ]  # f"--time-limit={self.aspeed_cutoff}"]
        )

        # # Register external Python functions
        # ctl.register_external("insert", insert)
        # ctl.register_external("order", order)

        # Load the ASP program
        ctl.add(asp_program)

        # if the instance set is too large, we subsample it
        if performance.shape[0] > self.data_threshold:
            random_indx = np.random.choice(
                range(performance.shape[0]),
                size=min(
                    performance.shape[0],
                    max(
                        int(performance.shape[0] * self.data_fraction),
                        self.data_threshold,
                    ),
                ),
                replace=True,
            )
            performance = performance[random_indx, :]

        times = [
            "time(i%d, %d, %d)." % (i, j, max(1, math.ceil(performance.iloc[i, j])))
            for i in range(performance.shape[0])
            for j in range(performance.shape[1])
        ]

        kappa = "kappa(%d)." % (self.budget)

        data_in = " ".join(times) + " " + kappa
        ctl.add(data_in)
        # Ground the logic program
        ctl.ground()

        def clingo_callback(model: clingo.Model) -> bool:
            """
            Callback function to process the Clingo model.

            Args:
                model (clingo.Model): The Clingo model.

            Returns:
                bool: Always returns False to stop after the first model.
            """
            schedule_dict = {}
            for slice in model.symbols(shown=True):
                algo = self.algorithms[slice.arguments[1].number]
                runcount_limit = slice.arguments[2].number
                schedule_dict[algo] = runcount_limit
                self.schedule = sorted(schedule_dict.items(), key=lambda x: x[1])
            return False

        # Solve the logic program
        result = ctl.solve(yield_=False, on_model=clingo_callback)
        if result.satisfiable:
            assert self.schedule is not None
        else:
            self.schedule = []

    def predict(self) -> dict[str, list[tuple[str, float]]]:
        """
        Predicts the schedule based on the fitted model.

        Returns:
            dict[str, list[tuple[str, float]]]: A dictionary containing the schedule.
        """
        return self.schedule
