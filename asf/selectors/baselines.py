"""
Baseline selectors: Single Best Solver (SBS) and Virtual Best Solver (VBS).

These provide upper and lower bounds for algorithm selection performance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin
from functools import partial
from typing import Any


class SingleBestSolver(ConfigurableMixin, AbstractSelector):
    """
    Single Best Solver (SBS) selector.

    Always selects the algorithm with the best average performance across all
    training instances. This represents the baseline performance achievable
    without any instance-specific selection.
    """

    PREFIX = "sbs"

    PREFIX = "sbs"

    def __init__(
        self,
        budget: int | None = None,
        maximize: bool = False,
        feature_groups: list[str] | None = None,
        **kwargs,
    ):
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
        **kwargs,
    ) -> None:
        """
        Find the single best algorithm based on aggregate performance.
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
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the single best algorithm for all instances.
        """
        return {
            instance: [(self.best_algorithm, self.budget)]
            for instance in features.index
        }

    @staticmethod
    def _define_hyperparameters(**kwargs):
        return [], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SingleBestSolver, **config)


class VirtualBestSolver(ConfigurableMixin, AbstractSelector):
    """
    Virtual Best Solver (VBS) / Oracle selector.

    Always selects the best algorithm for each specific instance.
    This represents the upper bound of performance achievable by any
    algorithm selector (requires oracle knowledge of true performance).

    Note: This selector "cheats" by using the test performance data,
    so it should only be used as an upper bound reference, not as a
    practical selector.
    """

    PREFIX = "vbs"

    PREFIX = "vbs"

    PREFIX = "vbs"

    def __init__(
        self,
        budget: int | None = None,
        maximize: bool = False,
        feature_groups: list[str] | None = None,
        **kwargs,
    ):
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
        **kwargs,
    ) -> None:
        """
        Store the performance data for oracle predictions.
        """
        self._performance = performance

    def _predict(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each instance (oracle).

        If performance data is provided at prediction time, use it.
        Otherwise, fall back to training performance (for instances in training set).
        """
        # Use provided performance or fall back to stored
        perf = performance if performance is not None else self._performance

        if perf is None:
            raise ValueError(
                "VirtualBestSolver requires performance data. "
                "Either provide it at fit time or pass it to predict."
            )

        result = {}
        for instance in features.index:
            if instance not in perf.index:
                # Fall back to first algorithm if instance not found
                result[instance] = [(self.algorithms[0], self.budget)]
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
                best_idx = np.argmax(instance_perf_penalized)
            else:
                best_idx = np.argmin(instance_perf_penalized)

            best_algorithm = perf.columns[best_idx]
            result[instance] = [(best_algorithm, self.budget)]

        return result

    @staticmethod
    def _define_hyperparameters(**kwargs):
        return [], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs,
    ) -> partial:
        config = clean_config.copy()
        config.update(kwargs)
        return partial(VirtualBestSolver, **config)
