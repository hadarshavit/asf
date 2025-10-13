import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.g_means import GMeans


class ISAC(AbstractSelector):
    """
    ISAC (Instance-Specific Algorithm Configuration) selector.

    Clusters instances in feature space using a user-provided clusterer (default: GMeans) and assigns to each cluster the best algorithm
    (by mean or median performance). For a new instance, predicts the cluster and
    recommends the cluster's best algorithm.

    Args:
        clusterer (object): An object with fit(X) and predict(X) methods (e.g., GMeans, KMeans).
            If None, uses GMeans by default.
        clusterer_kwargs (dict): Optional keyword arguments to instantiate the clusterer if not provided.
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional arguments for the parent class.

    Note:
        It is recommended to scale features before using ISACSelector.
    """

    def __init__(
        self,
        clusterer: Optional[Any] = GMeans,
        clusterer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clusterer = clusterer
        self.clusterer_kwargs = clusterer_kwargs or {}
        self.clusterer_instance = None
        self.cluster_to_best_algo = {}

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the ISAC selector.

        Args:
            features (pd.DataFrame): Feature matrix (instances x features).
            performance (pd.DataFrame): Performance matrix (instances x algorithms).
        """
        self.clusterer = self.clusterer(**self.clusterer_kwargs)

        if callable(self.clusterer):
            self.clusterer_instance = self.clusterer(
                random_state=self.random_state, **self.clusterer_kwargs
            )
        elif hasattr(self.clusterer, "fit") and hasattr(self.clusterer, "predict"):
            self.clusterer_instance = self.clusterer
        else:
            raise ValueError(
                "clusterer must be a class or an instance with fit/predict"
            )

        self.clusterer_instance.fit(features.values)
        cluster_labels = self.clusterer_instance.predict(features.values)

        # For each cluster, find the best algorithm (lowest mean performance)
        n_clusters = len(np.unique(cluster_labels))
        for cluster_id in range(n_clusters):
            idxs = np.where(cluster_labels == cluster_id)[0]
            if len(idxs) == 0:
                continue
            cluster_perf = performance.iloc[idxs]
            algo_means = cluster_perf.mean(axis=0)
            best_algo = algo_means.idxmin()
            self.cluster_to_best_algo[cluster_id] = best_algo

    def _predict(
        self,
        features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predict the best algorithm for each instance based on its cluster.

        Args:
            features (pd.DataFrame): Feature matrix for test instances.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping from instance name to [(algorithm, budget)].
        """
        if features is None:
            raise ValueError("Features must be provided for prediction.")
        if self.clusterer_instance is None:
            raise RuntimeError("ISACSelector must be fitted before prediction.")

        cluster_labels = self.clusterer_instance.predict(features.values)
        predictions = {}
        for idx, instance in enumerate(features.index):
            cluster_id = cluster_labels[idx]
            best_algo = self.cluster_to_best_algo.get(cluster_id, None)
            predictions[instance] = (
                [(best_algo, self.budget)] if best_algo else [(None, self.budget)]
            )
        return predictions
