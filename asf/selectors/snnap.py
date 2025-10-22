import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from asf.selectors.abstract_selector import AbstractSelector

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Integer,
        Configuration,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SNNAP(AbstractSelector):
    """
    SNNAP (Simple Nearest Neighbor Algorithm Portfolio) selector.

    Args:
      k (int): number of neighbors to use (default 5).
      metric (str): distance metric for NearestNeighbors (default 'euclidean').
    random_state (int | None): Random seed for reproducibility.
    """

    PREFIX = "snnap"

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
            Get the configuration space for SNNAP.

            Args:
                cs (Optional[ConfigurationSpace]): The configuration space to use. If None, a new one will be created.
                cs_transform (Optional[Dict[str, dict]]): A dictionary for transforming configuration space parameters.
                pre_prefix (str): Prefix for parameter names.
                parent_param (Optional[Hyperparameter]): Parent parameter for conditional configuration.
                parent_value (Optional[str]): Value of parent parameter that activates these hyperparameters.
                **kwargs: Additional keyword arguments.

            Returns:
                Tuple[ConfigurationSpace, Dict[str, dict]]: The configuration space and its transformation dictionary.
            """
            if cs is None:
                cs = ConfigurationSpace()

            if cs_transform is None:
                cs_transform = dict()

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SNNAP.PREFIX}"
            else:
                prefix = SNNAP.PREFIX

            k_param = Integer(
                name=f"{prefix}:k",
                bounds=(1, 50),
                default=5,
            )

            metric_param = Categorical(
                name=f"{prefix}:metric",
                items=["euclidean", "manhattan", "minkowski", "cosine"],
                default="euclidean",
            )

            params = [k_param, metric_param]

            # e.g. parent_param could be "selector_class" with parent_value "SNNAP"
            if parent_param is not None:
                conditions = [
                    EqualsCondition(
                        child=param,
                        parent=parent_param,
                        value=parent_value,
                    )
                    for param in params
                ]
            else:
                conditions = []

            cs.add(params + conditions)

            return cs, cs_transform

        @staticmethod
        def get_from_configuration(
            configuration: Configuration,
            cs_transform: dict[str, dict],
            pre_prefix: str = "",
            **kwargs,
        ) -> "SNNAP":
            """
            Get the SNNAP selector from a given configuration.

            Args:
                configuration (Configuration): The configuration object.
                cs_transform (Dict[str, dict]): The transformation dictionary for the configuration space.
                pre_prefix (str): Prefix for parameter names.
                **kwargs: Additional keyword arguments for SNNAP initialization.

            Returns:
                SNNAP: An instance of SNNAP configured according to the given configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SNNAP.PREFIX}"
            else:
                prefix = SNNAP.PREFIX

            k = configuration[f"{prefix}:k"]
            metric = configuration[f"{prefix}:metric"]

            return SNNAP(
                k=k,
                metric=metric,
                **kwargs,
            )
