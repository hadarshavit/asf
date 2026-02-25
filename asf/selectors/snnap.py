from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asf.predictors.random_forest import RandomForestRegressorWrapper
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SNNAP(ConfigurableMixin, AbstractSelector):
    """
    SNNAP (Solver-based Nearest Neighbor for Algorithm Portfolio) selector.

    Uses per-algorithm performance prediction models and Jaccard distance on predicted
    top algorithms to find similar instances, then selects from neighbors' best algorithms.

    Attributes
    ----------
    k : int
        Number of similar instances (neighbors) to use.
    top_n : int
        Number of top algorithms to consider for Jaccard distance calculation.
    algorithm_models : dict[str, AbstractPredictor] or None
        Per-algorithm regression models for predicting scaled runtime.
    scaled_performance_df : pd.DataFrame or None
        Z-score normalized performance matrix (per instance).
    original_performance_df : pd.DataFrame or None
        Original unnormalized performance data.
    training_top_n_sets : list[set[str]] or None
        Pre-computed top-n algorithm sets for each training instance.
    features_df : pd.DataFrame or None
        Training features.
    """

    PREFIX = "snnap"
    RETURN_TYPE = "single"

    def __init__(
        self,
        k: int = 5,
        top_n: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SNNAP selector.

        Parameters
        ----------
        k : int, default=5
            Number of nearest neighbors (similar instances) to consider.
        top_n : int, default=3
            Number of top algorithms to use for Jaccard distance calculation.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.k = int(k)
        self.top_n = int(top_n)

        self.features_df: pd.DataFrame | None = None
        self.original_performance_df: pd.DataFrame | None = None
        self.scaled_performance_df: pd.DataFrame | None = None
        self.algorithm_models: dict[str, Any] | None = None
        self.training_top_n_sets: list[set[str]] | None = None

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit per-algorithm regression models on z-score normalized performance.

        Parameters
        ----------
        features : pd.DataFrame
            The training features (instances x features).
        performance : pd.DataFrame
            The training performance data (instances x algorithms).
        """
        self.features_df = features.copy()
        self.original_performance_df = performance[self.algorithms].copy()

        # Step 1: Z-score normalize performance per instance (row-wise)
        self.scaled_performance_df = self.original_performance_df.copy()
        for idx in self.scaled_performance_df.index:
            row = self.scaled_performance_df.loc[idx]
            mean_val = row.mean()
            std_val = row.std()
            if std_val > 0:
                self.scaled_performance_df.loc[idx] = (row - mean_val) / std_val
            else:
                # If all values are the same, set to 0
                self.scaled_performance_df.loc[idx] = 0

        # Step 2: Train one regressor per algorithm
        self.algorithm_models = {}
        for algo in self.algorithms:
            y = self.scaled_performance_df[algo]
            model = RandomForestRegressorWrapper()
            model.fit(features.values, y.values)
            self.algorithm_models[str(algo)] = model

        # Step 3: Pre-compute top-n sets for each training instance
        self.training_top_n_sets = []
        for train_idx in range(len(self.scaled_performance_df)):
            train_scaled_perf = self.scaled_performance_df.iloc[train_idx]
            # For minimize: lowest scaled runtime is best. For maximize: highest is best.
            ascending = not self.maximize
            train_sorted = train_scaled_perf.sort_values(ascending=ascending)
            train_top_n = set(str(algo) for algo in train_sorted.index[: self.top_n])
            self.training_top_n_sets.append(train_top_n)

    @staticmethod
    def _jaccard_distance(set1: set[str], set2: set[str]) -> float:
        """
        Compute Jaccard distance between two sets.

        Parameters
        ----------
        set1 : set[str]
            First set of algorithm names.
        set2 : set[str]
            Second set of algorithm names.

        Returns
        -------
        float
            Jaccard distance: 1 - |intersection| / |union|
        """
        if not set1 or not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 1.0
        return 1.0 - (intersection / union)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm using Jaccard distance on top-n algorithms.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame, optional
            Unused, kept for interface compatibility.

        Returns
        -------
        dict
            Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("SNNAP requires features for prediction.")
        if (
            self.algorithm_models is None
            or self.features_df is None
            or self.original_performance_df is None
            or self.scaled_performance_df is None
            or self.training_top_n_sets is None
        ):
            raise RuntimeError("SNNAP must be fitted before prediction.")

        predictions: dict[str, list[tuple[str, float]]] = {}

        for instance_name in features.index:
            # Step 1: Use per-algorithm models to predict scaled runtimes for query instance
            query_features = features.loc[[instance_name]]
            predicted_scaled_runtimes = {}
            for algo in self.algorithms:
                model = self.algorithm_models[str(algo)]
                pred = model.predict(query_features.values)[0]
                predicted_scaled_runtimes[algo] = pred

            # Step 2: Identify query's predicted top-n algorithms (respecting maximize flag)
            sorted_algos = sorted(
                predicted_scaled_runtimes.items(),
                key=lambda x: x[1],
                reverse=self.maximize,  # If maximize, sort descending; if minimize, ascending
            )
            query_top_n = set(str(algo) for algo, _ in sorted_algos[: self.top_n])

            # Step 3: Compute Jaccard distance for each training instance
            jaccard_distances = []
            for train_idx in range(len(self.features_df)):
                # Use pre-computed top-n set
                train_top_n = self.training_top_n_sets[train_idx]

                # Compute Jaccard distance
                jdist = self._jaccard_distance(query_top_n, train_top_n)
                jaccard_distances.append((train_idx, jdist))

            # Step 4: Select k neighbors with lowest Jaccard distance
            jaccard_distances.sort(key=lambda x: x[1])
            k_actual = min(self.k, len(self.features_df))
            neighbor_idxs = [idx for idx, _ in jaccard_distances[:k_actual]]

            # Step 5: Compute mean runtime for ALL algorithms across neighbors, then select best
            # This differs from selecting best-per-neighbor and averaging only winners
            runtimes_by_algo: dict[str, list[float]] = {}
            for ni in neighbor_idxs:
                neighbor_perf = self.original_performance_df.iloc[ni]
                # For each algorithm, collect its actual runtime in this neighbor
                for algo in self.algorithms:
                    if algo in neighbor_perf.index and pd.notna(neighbor_perf[algo]):
                        if algo not in runtimes_by_algo:
                            runtimes_by_algo[algo] = []
                        runtimes_by_algo[algo].append(float(neighbor_perf[algo]))

            if not runtimes_by_algo:
                predictions[str(instance_name)] = []
                continue

            # Compute mean runtime for all algorithms
            mean_runtimes = {
                algo: float(np.mean(times)) for algo, times in runtimes_by_algo.items()
            }
            # Select algorithm with best mean runtime
            if self.maximize:
                chosen = max(mean_runtimes.items(), key=lambda x: x[1])[0]
            else:
                chosen = min(mean_runtimes.items(), key=lambda x: x[1])[0]

            predictions[str(instance_name)] = [(str(chosen), float(self.budget or 0))]

        return predictions

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for SNNAP.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        k_param = Integer(
            name="k",
            bounds=(1, 50),
            default=5,
        )

        top_n_param = Integer(
            name="top_n",
            bounds=(1, 20),
            default=3,
        )

        return [k_param, top_n_param], [], []
