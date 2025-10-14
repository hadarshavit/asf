import pandas as pd
import numpy as np
import math
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
        if not CLINGO_AVAIL:
            raise ImportError(
                "clingo is not installed. Please install it to use the Aspeed presolver."
            )
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

        # remember algorithm names (expects DataFrame columns)
        try:
            self.algorithms = list(performance.columns)
        except Exception:
            # if performance is provided as numpy array, create default names
            self.algorithms = [f"a{i}" for i in range(performance.shape[1])]

        # Create a Clingo Control object with the specified number of threads
        # Use -t (threads) argument which is broadly supported
        ctl = clingo.Control(arguments=[f"-t{self.cores}"])

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
            # keep it as a pandas DataFrame view using iloc
            performance = performance.iloc[random_indx, :]

        times = [
            "time(i%d, %d, %d)." % (i, j, max(1, math.ceil(performance.iloc[i, j])))
            for i in range(performance.shape[0])
            for j in range(performance.shape[1])
        ]

        kappa = "kappa(%d)." % (self.budget)

        # join facts with newlines (more readable for clingo) and add kappa
        data_in = "\n".join(times) + "\n" + kappa
        ctl.add(data_in)

        # Ground the logic program (ground the default 'base' part)
        try:
            ctl.ground([("base", [])])
        except Exception:
            # fallback to grounding everything
            ctl.ground()

        def clingo_callback(model: clingo.Model):
            """Callback function to process the Clingo model."""
            schedule_dict = {}
            for slice in model.symbols(shown=True):
                try:
                    algo = self.algorithms[slice.arguments[1].number]
                except Exception:
                    algo = str(slice.arguments[1])
                runcount_limit = slice.arguments[2].number
                schedule_dict[algo] = runcount_limit

            # sort by allocated time
            self.schedule = sorted(schedule_dict.items(), key=lambda x: x[1])

        # Use async solve and a timeout similar to AutoFolio
        try:
            with ctl.solve(on_model=clingo_callback, async_=True) as handle:
                if handle.wait(self.aspeed_cutoff):
                    handle.get()
                else:
                    # timeout: cancel and keep whatever was found (if any)
                    handle.cancel()
        except Exception as e:
            # If solving failed, leave schedule empty and report
            print(f"Clingo solving failed: {e}")
            self.schedule = []

    def predict(self) -> dict[str, list[tuple[str, float]]]:
        """
        Predicts the schedule based on the fitted model.

        Returns:
            dict[str, list[tuple[str, float]]]: A dictionary containing the schedule.
        """
        return self.schedule
