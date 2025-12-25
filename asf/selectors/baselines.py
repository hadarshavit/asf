from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin


class SingleBestSolver(ConfigurableMixin, AbstractSelector):
    """
    Single Best Solver (SBS) selector.

    Always selects the algorithm with the best average performance across all
    training instances. This represents the baseline performance achievable
    without any instance-specific selection.

    Attributes
    ----------
    best_algorithm : str or None
        The name of the algorithm with the best aggregate performance.
    """

    PREFIX = "sbs"

    def __init__(
        self,
        budget: int | None = None,
        maximize: bool = False,
        feature_groups: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SingleBestSolver.

        Parameters
        ----------
        budget : int or None, default=None
            The budget for the selector.
        maximize : bool, default=False
            Indicates whether to maximize the performance metric.
        feature_groups : list[str] or None, default=None
            Groups of features to be considered.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(
            budget=budget,
            maximize=maximize,
            feature_groups=feature_groups,
            **kwargs,
        )
        self.best_algorithm: str | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Find the single best algorithm based on aggregate performance.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        # Apply PAR10 penalty for comparison
        if self.budget is not None:
            perf_penalized = np.where(
                performance <= self.budget, performance, self.budget * 10
            )
        else:
            perf_penalized = performance.values

        # Aggregate performance across all instances
        perf_sum = np.sum(perf_penalized, axis=0)

        if self.maximize:
            best_idx = np.argmax(perf_sum)
        else:
            best_idx = np.argmin(perf_sum)

        self.best_algorithm = performance.columns[best_idx]

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
                Predict the single best algorithm for all instances.

                Parameters
                ----------
                features : pd.DataFrame or None
                    The input features.
                performance : pd.DataFrame or None, default=None
                    The performance data.

                Returns
        -------
                dict
                    Dictionary mapping instance IDs to the single best algorithm.
        """
        indices = features.index if features is not None else [0]
        return {
            str(instance): [(str(self.best_algorithm), float(self.budget or 0))]
            for instance in indices
        }

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for SingleBestSolver.

        Returns
        -------
        tuple
            Empty hyperparameters, conditions, and forbiddens.
        """
        return [], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[SingleBestSolver]:
        """
                Create a SingleBestSolver from a clean configuration.

                Parameters
                ----------
                clean_config : dict
                    The clean configuration.
                **kwargs : Any
                    Additional keyword arguments.

                Returns
        -------
                partial
                    Partial function for SingleBestSolver.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SingleBestSolver, **config)


class VirtualBestSolver(ConfigurableMixin, AbstractSelector):
    """
    Virtual Best Solver (VBS) / Oracle selector.

    Always selects the best algorithm for each specific instance.
    This represents the upper bound of performance achievable by any
    algorithm selector (requires oracle knowledge of true performance).

    Note: This selector "cheats" by using the test performance data.
    """

    PREFIX = "vbs"

    def __init__(
        self,
        budget: int | None = None,
        maximize: bool = False,
        feature_groups: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the VirtualBestSolver.

        Parameters
        ----------
        budget : int or None, default=None
            The budget for the selector.
        maximize : bool, default=False
            Indicates whether to maximize the performance metric.
        feature_groups : list[str] or None, default=None
            Groups of features to be considered.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(
            budget=budget,
            maximize=maximize,
            feature_groups=feature_groups,
            **kwargs,
        )
        self._performance: pd.DataFrame | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Store the performance data for oracle predictions.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        self._performance = performance

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each instance (oracle).

        If performance data is provided at prediction time, use it.
        Otherwise, fall back to training performance.

        Parameters
        ----------
        features : pd.DataFrame or None
            The input features.
        performance : pd.DataFrame or None, default=None
            The performance data.

        Returns
        -------
        dict
            Dictionary mapping instance IDs to the best algorithm.
        """
        # Use provided performance or fall back to stored
        perf = performance if performance is not None else self._performance

        if perf is None:
            raise ValueError(
                "VirtualBestSolver requires performance data. "
                "Either provide it at fit time or pass it to predict."
            )

        indices = features.index if features is not None else perf.index

        result: dict[str, list[tuple[str, float]]] = {}
        for instance in indices:
            if instance not in perf.index:
                # Fall back to first algorithm if instance not found
                result[str(instance)] = [
                    (str(self.algorithms[0]), float(self.budget or 0))
                ]
                continue

            instance_perf = perf.loc[instance]

            # Apply PAR10 penalty for comparison
            if self.budget is not None:
                instance_perf_penalized = np.where(
                    instance_perf <= self.budget, instance_perf, self.budget * 10
                )
            else:
                instance_perf_penalized = instance_perf.values

            if self.maximize:
                best_idx = int(np.argmax(instance_perf_penalized))
            else:
                best_idx = int(np.argmin(instance_perf_penalized))

            best_algorithm = str(perf.columns[best_idx])
            result[str(instance)] = [(best_algorithm, float(self.budget or 0))]

        return result

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for VirtualBestSolver.

        Returns
        -------
        tuple
            Empty hyperparameters, conditions, and forbiddens.
        """
        return [], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[VirtualBestSolver]:
        """
        Create a VirtualBestSolver from a clean configuration.

        Parameters
        ----------
        clean_config : dict
            The clean configuration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for VirtualBestSolver.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(VirtualBestSolver, **config)
