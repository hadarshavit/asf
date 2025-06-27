import pandas as pd
from abc import abstractmethod


class AbstractPresolver:
    def __init__(
        self,
        budget: float,
        presolver_cutoff: float,
        maximize: bool = False,
    ):
        self.budget = budget
        self.presolver_cutoff = presolver_cutoff
        self.maximize = maximize

    @abstractmethod
    def fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self) -> dict[str, list[tuple[str, float]]]:
        pass
