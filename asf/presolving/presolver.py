"""
Abstract base class for algorithm presolvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from asf.utils.configurable import ConfigurableMixin

try:
    import ConfigSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractPresolver(ABC, ConfigurableMixin):
    """
    Abstract base class for algorithm presolvers.

    A presolver selects a sequence of algorithms to run for a fixed budget
    before a selector is used.

    Parameters
    ----------
    presolver_budget : float
        The total time budget for the presolver.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    """

    def __init__(
        self,
        presolver_budget: float,
        maximize: bool = False,
        **kwargs: Any,
    ) -> None:
        self.presolver_budget = float(presolver_budget)
        self.maximize = bool(maximize)

    @abstractmethod
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
        features : pd.DataFrame, np.ndarray, or None
            The instance features.
        performance : pd.DataFrame, np.ndarray, or None
            The algorithm performances.
        **kwargs : Any
            Additional keyword arguments.
        """
        pass

    @abstractmethod
    def predict(
        self,
        features: pd.DataFrame | np.ndarray | None = None,
        performance: pd.DataFrame | np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
        """
        Predict the presolving schedule.

        Parameters
        ----------
        features : pd.DataFrame, np.ndarray, or None, default=None
            The features for the instances.
        performance : pd.DataFrame, np.ndarray, or None, default=None
            The algorithm performances.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list of tuple or dict
            A list of (algorithm_name, time_budget) pairs, OR a dict mapping instance names to such lists.
        """
        pass

    @staticmethod
    def _define_hyperparameters(
        total_budget: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define common hyperparameters for presolvers.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        from ConfigSpace import Float

        # Normalize budget parameter name
        if total_budget is None:
            total_budget = kwargs.get("budget", kwargs.get("total_budget"))

        hyperparameters = []
        if total_budget is not None:
            total_budget = float(total_budget)
            upper_budget = max(1.1, 0.1 * total_budget)
            hyperparameters.append(
                Float(
                    name="presolver_budget",
                    bounds=(1.0, upper_budget),
                    default=min(10.0, upper_budget),
                    log=True,
                )
            )

        return hyperparameters, [], []
