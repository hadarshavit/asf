import numpy as np
import pandas as pd
from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.g_means import GMeans
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Integer,
        Float,
        Configuration,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class ISAC(AbstractSelector):
    """
    ISAC (Instance-Specific Algorithm Configuration) selector.

    Clusters instances in feature space using a user-provided clusterer (default: GMeans) and assigns to each cluster the best algorithm
    (by mean or median performance). For a new instance, predicts the cluster and recommends the cluster's best algorithm.

    Args:
        clusterer (object): An object with fit(X) and predict(X) methods (e.g., GMeans, KMeans).
            If None, uses GMeans by default.
        clusterer_kwargs (dict): Optional keyword arguments to instantiate the clusterer if not provided.
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional arguments for the parent class.

    Note:
        It is recommended to scale features before using ISACSelector.
    """

    PREFIX = "isac"
    RETURN_TYPE = "single"

    def __init__(
        self,
        clusterer: object | None = GMeans,
        clusterer_kwargs: dict | None = None,
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
        features: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
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

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            cs_transform: dict[str, dict] | None = None,
            pre_prefix: str = "",
            parent_param: Hyperparameter | None = None,
            parent_value: str | None = None,
            **kwargs,
        ) -> tuple[ConfigurationSpace, dict[str, dict]]:
            """
            Get the configuration space for ISAC.

            Args:
                cs: The configuration space to use. If None, a new one will be created.
                cs_transform: A dictionary for transforming configuration space parameters.
                pre_prefix: Prefix for parameter names.
                parent_param: Parent parameter for conditional configuration.
                parent_value: Value of parent parameter that activates these hyperparameters.
                **kwargs: Additional keyword arguments.

            Returns:
                Tuple[ConfigurationSpace, Dict[str, dict]]: The configuration space and its transformation dictionary.
            """
            if cs is None:
                cs = ConfigurationSpace()

            if cs_transform is None:
                cs_transform = dict()

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{ISAC.PREFIX}"
            else:
                prefix = ISAC.PREFIX

            clusterer_param = Categorical(
                name=f"{prefix}:clusterer",
                items=["GMeans", "KMeans", "AgglomerativeClustering", "DBSCAN"],
                default="GMeans",
            )

            cs_transform[f"{prefix}:clusterer"] = {
                "GMeans": GMeans,
                "KMeans": KMeans,
                "AgglomerativeClustering": AgglomerativeClustering,
                "DBSCAN": DBSCAN,
            }

            # GMeans hyperparameters
            gmeans_min_samples = Float(
                name=f"{prefix}:gmeans:min_samples",
                bounds=(0.0001, 0.1),
                default=0.001,
                log=True,
            )
            gmeans_significance = Categorical(
                name=f"{prefix}:gmeans:significance",
                items=[0.15, 0.1, 0.05, 0.025, 0.001],
                default=0.05,
            )
            gmeans_n_init = Integer(
                name=f"{prefix}:gmeans:n_init",
                bounds=(1, 10),
                default=5,
            )

            # KMeans hyperparameters
            kmeans_n_clusters = Integer(
                name=f"{prefix}:kmeans:n_clusters",
                bounds=(2, 20),
                default=5,
            )

            # AgglomerativeClustering hyperparameters
            agg_n_clusters = Integer(
                name=f"{prefix}:agg:n_clusters",
                bounds=(2, 20),
                default=5,
            )
            agg_linkage = Categorical(
                name=f"{prefix}:agg:linkage",
                items=["ward", "complete", "average", "single"],
                default="ward",
            )

            # DBSCAN hyperparameters
            dbscan_eps = Float(
                name=f"{prefix}:dbscan:eps",
                bounds=(0.1, 2.0),
                default=0.5,
            )
            dbscan_min_samples = Integer(
                name=f"{prefix}:dbscan:min_samples",
                bounds=(2, 10),
                default=5,
            )

            params = [
                clusterer_param,
                gmeans_min_samples,
                gmeans_significance,
                gmeans_n_init,
                kmeans_n_clusters,
                agg_n_clusters,
                agg_linkage,
                dbscan_eps,
                dbscan_min_samples,
            ]

            conditions = []
            conditions.extend(
                [
                    EqualsCondition(
                        child=gmeans_min_samples,
                        parent=clusterer_param,
                        value="GMeans",
                    ),
                    EqualsCondition(
                        child=gmeans_significance,
                        parent=clusterer_param,
                        value="GMeans",
                    ),
                    EqualsCondition(
                        child=gmeans_n_init,
                        parent=clusterer_param,
                        value="GMeans",
                    ),
                ]
            )

            conditions.append(
                EqualsCondition(
                    child=kmeans_n_clusters,
                    parent=clusterer_param,
                    value="KMeans",
                )
            )

            conditions.extend(
                [
                    EqualsCondition(
                        child=agg_n_clusters,
                        parent=clusterer_param,
                        value="AgglomerativeClustering",
                    ),
                    EqualsCondition(
                        child=agg_linkage,
                        parent=clusterer_param,
                        value="AgglomerativeClustering",
                    ),
                ]
            )

            conditions.extend(
                [
                    EqualsCondition(
                        child=dbscan_eps,
                        parent=clusterer_param,
                        value="DBSCAN",
                    ),
                    EqualsCondition(
                        child=dbscan_min_samples,
                        parent=clusterer_param,
                        value="DBSCAN",
                    ),
                ]
            )

            if parent_param is not None:
                for param in params:
                    conditions.append(
                        EqualsCondition(
                            child=param,
                            parent=parent_param,
                            value=parent_value,
                        )
                    )

            cs.add(params + conditions)

            return cs, cs_transform

        @staticmethod
        def get_from_configuration(
            configuration: Configuration,
            cs_transform: dict[str, dict],
            pre_prefix: str = "",
            **kwargs,
        ) -> "ISAC":
            """
            Get the ISAC selector from a given configuration.

            Args:
                configuration: The configuration object.
                cs_transform: The transformation dictionary for the configuration space.
                pre_prefix: Prefix for parameter names.
                **kwargs: Additional keyword arguments for ISAC initialization.

            Returns:
                ISAC: An instance of ISAC configured according to the given configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{ISAC.PREFIX}"
            else:
                prefix = ISAC.PREFIX

            clusterer_class = cs_transform[f"{prefix}:clusterer"][
                configuration[f"{prefix}:clusterer"]
            ]

            clusterer_kwargs = {}
            if configuration[f"{prefix}:clusterer"] == "GMeans":
                clusterer_kwargs["min_samples"] = configuration[
                    f"{prefix}:gmeans:min_samples"
                ]
                clusterer_kwargs["significance"] = configuration[
                    f"{prefix}:gmeans:significance"
                ]
                clusterer_kwargs["n_init"] = configuration[f"{prefix}:gmeans:n_init"]
            elif configuration[f"{prefix}:clusterer"] == "KMeans":
                clusterer_kwargs["n_clusters"] = configuration[
                    f"{prefix}:kmeans:n_clusters"
                ]
            elif configuration[f"{prefix}:clusterer"] == "AgglomerativeClustering":
                clusterer_kwargs["n_clusters"] = configuration[
                    f"{prefix}:agg:n_clusters"
                ]
                clusterer_kwargs["linkage"] = configuration[f"{prefix}:agg:linkage"]
            elif configuration[f"{prefix}:clusterer"] == "DBSCAN":
                clusterer_kwargs["eps"] = configuration[f"{prefix}:dbscan:eps"]
                clusterer_kwargs["min_samples"] = configuration[
                    f"{prefix}:dbscan:min_samples"
                ]

            return ISAC(
                clusterer=clusterer_class,
                clusterer_kwargs=clusterer_kwargs,
                **kwargs,
            )
