import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from asf.selectors.abstract_selector import AbstractSelector

try:
    from ConfigSpace import (
        Categorical,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


from asf.utils.configurable import ConfigurableMixin
from functools import partial
from typing import Any


class SNNAP(ConfigurableMixin, AbstractSelector):
    """
    SNNAP (Simple Nearest Neighbor Algorithm Portfolio) selector.

    Args:
      k (int): number of neighbors to use (default 5).
      metric (str): distance metric for NearestNeighbors (default 'euclidean').
    random_state (int | None): Random seed for reproducibility.
    """

    PREFIX = "snnap"
    RETURN_TYPE = "single"

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric
        self.random_state = random_state

        self.features: pd.DataFrame | None = None
        self.performance: pd.DataFrame | None = None
        self.nn_model: NearestNeighbors | None = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Store training data and fit the NearestNeighbors model.

        Args:
            features: DataFrame (instances x features)
            performance: DataFrame (instances x algorithms)
        """
        self.features = features.copy()
        self.performance = performance.copy()

        n_neighbors = min(self.k, len(self.features))
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self.nn_model.fit(self.features.values)

    def _predict(
        self,
        features: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str | None, float]]]:
        """
        Predict the single best algorithm for each instance using majority vote among k neighbors.

        Returns:
            dict: instance_id -> [(algorithm_name or None, budget)]
        """
        if features is None:
            raise ValueError("Features must be provided for prediction.")
        if self.nn_model is None or self.features is None or self.performance is None:
            raise RuntimeError("SNNAPSelector must be fitted before prediction.")

        predictions: dict[str, list[tuple[str | None, float]]] = {}
        for idx, instance in enumerate(features.index):
            x = features.loc[instance].values.reshape(1, -1)
            n_neighbors = min(self.k, len(self.features))
            dists, neighbor_idxs = self.nn_model.kneighbors(x, n_neighbors=n_neighbors)
            neighbor_idxs = neighbor_idxs.flatten()

            votes: dict[str, int] = {}
            runtimes_for_candidates: dict[str, list[float]] = {}

            for ni in neighbor_idxs:
                neighbor_perf = self.performance.iloc[ni]
                valid = neighbor_perf.dropna()
                if valid.empty:
                    continue
                best_algo = valid.idxmin()
                votes[best_algo] = votes.get(best_algo, 0) + 1
                runtimes_for_candidates.setdefault(best_algo, []).append(
                    valid.loc[best_algo]
                )

            if not votes:
                predictions[instance] = [(None, self.budget)]
                continue

            max_votes = max(votes.values())
            candidates = [a for a, c in votes.items() if c == max_votes]

            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                # tie-break: choose candidate with smallest mean runtime across recorded neighbor runtimes
                mean_runtimes = {
                    a: np.mean(runtimes_for_candidates[a])
                    for a in candidates
                    if a in runtimes_for_candidates
                    and len(runtimes_for_candidates[a]) > 0
                }
                if not mean_runtimes:
                    # No candidates have recorded runtimes; fallback to None
                    chosen = None
                else:
                    chosen = min(mean_runtimes.items(), key=lambda x: x[1])[0]
            if chosen is None:
                predictions[instance] = [(None, self.budget)]
            else:
                predictions[instance] = [(chosen, self.budget)]

        return predictions

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for SNNAP."""
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
        **kwargs,
    ) -> partial:
        """
        Create a partial function from a clean (unprefixed) configuration.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(SNNAP, **config)
