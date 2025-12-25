from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.clustering.wrappers import (
    AgglomerativeClusteringWrapper,
    DBSCANWrapper,
    GMeansWrapper,
    KMeansWrapper,
)
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Float,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class ISAC(ConfigurableMixin, AbstractSelector):
    """
    ISAC (Instance-Specific Algorithm Configuration) selector.

    Clusters instances in feature space and assigns to each cluster the best
    algorithm (by mean performance).

    Attributes
    ----------
    clusterer : type or Callable or Any
        The clusterer class, partial, or instance.
    clusterer_kwargs : dict[str, Any]
        Arguments for clusterer instantiation.
    clusterer_instance : Any or None
        The trained clusterer instance.
    cluster_to_best_algo : dict[int, str]
        Mapping from cluster ID to best algorithm name.
    """

    PREFIX = "isac"
    RETURN_TYPE = "single"

    def __init__(
        self,
        clusterer: type | Callable[..., Any] | Any = GMeansWrapper,
        clusterer_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ISAC selector.

        Parameters
        ----------
        clusterer : type or Callable or Any, default=GMeansWrapper
            The clusterer class, partial, or instance.
        clusterer_kwargs : dict[str, Any] or None, default=None
            Arguments for clusterer instantiation.
        random_state : int or None, default=None
            Random state for the clusterer.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.clusterer = clusterer
        self.clusterer_kwargs = clusterer_kwargs or {}
        self.random_state = random_state
        self.clusterer_instance: Any | None = None
        self.cluster_to_best_algo: dict[int, str] = {}

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the ISAC selector.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (instances x features).
        performance : pd.DataFrame
            Performance matrix (instances x algorithms).
        """
        if isinstance(self.clusterer, type) or isinstance(self.clusterer, partial):
            self.clusterer_instance = self.clusterer(
                random_state=self.random_state, **self.clusterer_kwargs
            )
        elif hasattr(self.clusterer, "fit") and hasattr(self.clusterer, "predict"):
            self.clusterer_instance = self.clusterer
        else:
            raise ValueError(
                "clusterer must be a class, partial, or an instance with fit/predict"
            )

        self.clusterer_instance.fit(features.values)  # type: ignore[attr-defined]
        cluster_labels = self.clusterer_instance.predict(features.values)  # type: ignore[attr-defined]

        n_clusters = len(np.unique(cluster_labels))
        for cluster_id in range(n_clusters):
            idxs = np.where(cluster_labels == cluster_id)[0]
            if len(idxs) == 0:
                continue
            cluster_perf = performance.iloc[idxs]
            algo_means = cluster_perf.mean(axis=0)
            best_algo = algo_means.idxmin()
            self.cluster_to_best_algo[int(cluster_id)] = str(best_algo)

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
            Feature matrix for test instances.

        Returns
        -------
        dict
            Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("ISAC require features for prediction.")
        cluster_labels = self.clusterer_instance.predict(features.values)  # type: ignore[attr-defined]
        predictions: dict[str, list[tuple[str, float]]] = {}
        for idx, instance in enumerate(features.index):
            cluster_id = int(cluster_labels[idx])
            best_algo = self.cluster_to_best_algo.get(cluster_id)
            if best_algo:
                predictions[str(instance)] = [(str(best_algo), float(self.budget or 0))]
            else:
                predictions[str(instance)] = []
        return predictions

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for ISAC.

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

        clusterer_param = ClassChoice(
            name="clusterer",
            choices=[
                GMeansWrapper,
                KMeansWrapper,
                AgglomerativeClusteringWrapper,
                DBSCANWrapper,
            ],
            default=GMeansWrapper,
        )

        return [clusterer_param], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[ISAC]:
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
            Partial function for ISAC.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(ISAC, **config)
