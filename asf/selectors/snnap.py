from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SNNAP(ConfigurableMixin, AbstractSelector):
    """
    SNNAP (Simple Nearest Neighbor Algorithm Portfolio) selector.

    Attributes
    ----------
    k : int
        Number of neighbors to use.
    metric : str
        Distance metric for NearestNeighbors.
    random_state : int or None
        Random seed for reproducibility.
    nn_model : NearestNeighbors or None
        Trained NearestNeighbors model.
    features_df : pd.DataFrame or None
        Training features.
    performance_df : pd.DataFrame or None
        Training performance.
    """

    PREFIX = "snnap"
    RETURN_TYPE = "single"

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SNNAP selector.

        Parameters
        ----------
        k : int, default=5
            Number of neighbors to use.
        metric : str, default='euclidean'
            Distance metric for NearestNeighbors.
        random_state : int or None, default=None
            Random seed for reproducibility.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.k = int(k)
        self.metric = str(metric)
        self.random_state = random_state

        self.features_df: pd.DataFrame | None = None
        self.performance_df: pd.DataFrame | None = None
        self.nn_model: NearestNeighbors | None = None

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the NearestNeighbors model.

        Parameters
        ----------
        features : pd.DataFrame
            The training features.
        performance : pd.DataFrame
            The training performance data.
        """
        self.features_df = features.copy()
        self.performance_df = performance.copy()

        n_neighbors = min(self.k, len(self.features_df))
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self.nn_model.fit(self.features_df.values)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
                Predict the best algorithm for each instance.

                Parameters
                ----------
                features : pd.DataFrame
                    The input features.

                Returns
        -------
                dict
                    Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("SNNAP requires features for prediction.")
        if (
            self.nn_model is None
            or self.features_df is None
            or self.performance_df is None
        ):
            raise RuntimeError("SNNAP must be fitted before prediction.")

        predictions: dict[str, list[tuple[str, float]]] = {}
        for instance_name in features.index:
            x = features.loc[[instance_name]].values
            n_neighbors = min(self.k, len(self.features_df))
            _, neighbor_idxs = self.nn_model.kneighbors(x, n_neighbors=n_neighbors)
            neighbor_idxs = neighbor_idxs.flatten()

            votes: dict[str, int] = {}
            runtimes_for_candidates: dict[str, list[float]] = {}

            for ni in neighbor_idxs:
                neighbor_perf = self.performance_df.iloc[ni]
                valid = neighbor_perf.dropna()
                if valid.empty:
                    continue
                # Best algorithm for this neighbor
                best_algo = str(valid.idxmax() if self.maximize else valid.idxmin())
                votes[best_algo] = votes.get(best_algo, 0) + 1
                runtimes_for_candidates.setdefault(best_algo, []).append(
                    float(valid.loc[best_algo])
                )

            if not votes:
                predictions[str(instance_name)] = []
                continue

            # Identify candidate(s) with max votes
            max_votes = max(votes.values())
            candidates = [a for a, c in votes.items() if c == max_votes]

            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                # Tie-break: Smallest mean runtime (or largest mean performance)
                mean_perfs = {
                    algo: float(np.mean(runtimes_for_candidates[algo]))
                    for algo in candidates
                }
                if self.maximize:
                    chosen = max(mean_perfs.items(), key=lambda x: x[1])[0]
                else:
                    chosen = min(mean_perfs.items(), key=lambda x: x[1])[0]

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

        metric_param = Categorical(
            name="metric",
            items=["euclidean", "manhattan", "minkowski", "cosine"],
            default="euclidean",
        )

        return [k_param, metric_param], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[SNNAP]:
        """
        Create a partial function from a clean configuration.

        Parameters
        ----------
        clean_config : dict
            The clean configuration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for SNNAP.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SNNAP, **config)
