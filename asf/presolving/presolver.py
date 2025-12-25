"""
Abstract base class for algorithm presolvers.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

try:
    from ConfigSpace import (
        Configuration,
        ConfigurationSpace,
        EqualsCondition,
        UniformFloatHyperparameter,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractPresolver:
    """
    Abstract base class for algorithm presolvers.

    A presolver selects a sequence of algorithms to run for a fixed budget
    before a selector is used.

    Parameters
    ----------
    budget : float
        The total time budget for the presolver.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    """

    def __init__(
        self,
        budget: float,
        maximize: bool = False,
    ) -> None:
        self.budget = budget
        self.maximize = maximize

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

    @classmethod
    def get_configuration_space(
        cls,
        cs: ConfigurationSpace | None = None,
        cs_transform: dict[str, Any] | None = None,
        parent_param: Any | None = None,
        parent_value: str | None = None,
        total_budget: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Get the configuration space.

        Parameters
        ----------
        cs : ConfigurationSpace or None, default=None
            The configuration space to use.
        cs_transform : dict or None, default=None
            A dictionary for transforming configuration space values.
        parent_param : Any or None, default=None
            Parent parameter for conditional configuration.
        parent_value : str or None, default=None
            Value of parent parameter that activates these parameters.
        total_budget : float or None, default=None
            Total budget available.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The configuration space and transformation dictionary (or just CS).

        Raises
        ------
        RuntimeError
            If ConfigSpace is not installed.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if cs is None:
            cs = ConfigurationSpace()
        if cs_transform is None:
            cs_transform = {}

        if parent_param is not None and parent_value is not None:
            if total_budget is not None:
                upper_budget = max(1.1, 0.1 * total_budget)
                presolver_budget_param = UniformFloatHyperparameter(
                    name=f"{parent_value}:presolver_budget",
                    lower=1,
                    upper=upper_budget,
                    default_value=min(10, upper_budget),
                    log=True,
                )
                cs.add(presolver_budget_param)

                condition = EqualsCondition(
                    presolver_budget_param, parent_param, parent_value
                )
                cs.add(condition)

        return cs, cs_transform

    @classmethod
    def get_from_configuration(
        cls,
        configuration: Configuration | dict[str, Any],
        cs_transform: dict[str, Any] | None = None,
        budget: float | None = None,
        maximize: bool = False,
        presolver_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a presolver instance from a configuration.

        Parameters
        ----------
        configuration : Configuration or dict
            The configuration object or dictionary.
        cs_transform : dict or None
            The transformation dictionary.
        budget : float or None, default=None
            Budget for the presolver.
        maximize : bool, default=False
            Whether to maximize the metric.
        presolver_name : str or None, default=None
            Name of the presolver.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The presolver instance.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )

        if budget is None and presolver_name is not None and cs_transform is None:
            # Fallback logic if cs_transform is not provided or needed
            pass

        if budget is None and presolver_name is not None:
            budget_key = f"{presolver_name}:presolver_budget"
            if budget_key in configuration:
                budget = configuration[budget_key]

        raise NotImplementedError(
            "get_from_configuration() is not implemented for this presolver"
        )
