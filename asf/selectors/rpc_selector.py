from typing import Any, List, Dict, Tuple, Type
import inspect
import numpy as np
import pandas as pd
from asf.predictors import RandomForestClassifierWrapper
from asf.selectors.abstract_selector import AbstractSelector


class RPCSelector(AbstractSelector):
    """
    Ranking by Pairwise Comparison (RPC) for Algorithm Selection.

    RPC decomposes the k-label ranking problem into m = k(k-1)/2 binary
    classification tasks (one for each pair of algorithms).
    """

    PREFIX = "rpc"
    RETURN_TYPE = "single"

    def __init__(
        self,
        classifier_class: Type = RandomForestClassifierWrapper,
        n_estimators: int = 100,
        classifier_kwargs: Dict | None = None,
        random_state: int = 42,
        top_n: int = 1,
        **kwargs,
    ):
        """
        Initialize RPC selector.

        Args:
            classifier_class: The classifier class to use for pairwise tasks.
            n_estimators: Number of estimators for ensemble classifiers.
            classifier_kwargs: Additional kwargs to pass to each classifier.
            random_state: Random seed for reproducibility.
            top_n: Number of top algorithms to return. If > 1, selector returns
                    parallel portfolios (list of algorithm names). If 1, returns
                    a single (algorithm, budget) tuple.
            **kwargs: Additional arguments for parent class.
        """
        super().__init__(**kwargs)
        self.classifier_class = classifier_class
        self.n_estimators = int(n_estimators)
        self.classifier_kwargs = dict(classifier_kwargs or {})

        # Only set defaults if the classifier accepts these parameters
        sig_params = set(inspect.signature(classifier_class).parameters.keys())
        if (
            "n_estimators" in sig_params
            and "n_estimators" not in self.classifier_kwargs
        ):
            self.classifier_kwargs["n_estimators"] = int(n_estimators)

        if (
            "random_state" in sig_params
            and "random_state" not in self.classifier_kwargs
        ):
            self.classifier_kwargs["random_state"] = int(random_state)

        self.random_state = int(random_state)
        self.top_n = max(1, int(top_n))

        # Set return type dynamically based on top_n
        self.RETURN_TYPE = "parallel" if self.top_n > 1 else "single"

        self.classifiers: Dict[Tuple[str, str], Any] = {}
        self.algorithms: List[str] = []
        self.pairs: List[Tuple[str, str]] = []

    def _create_pairs(self, algorithms: List[str]) -> List[Tuple[str, str]]:
        """Create all unique pairs of algorithms (i, j) where i < j."""
        pairs = []
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                pairs.append((algorithms[i], algorithms[j]))
        return pairs

    def _create_binary_labels(
        self, performance: pd.DataFrame, algo_i: str, algo_j: str
    ) -> np.ndarray:
        """
        Create binary labels for pairwise comparison.

        Args:
            performance: Performance matrix (n_instances, n_algorithms)
            algo_i: First algorithm
            algo_j: Second algorithm

        Returns:
            Binary labels: 1 if algo_i < algo_j (better), 0 otherwise
        """
        perf_i = performance[algo_i].values
        perf_j = performance[algo_j].values
        # 1 if algo_i is better (lower runtime), 0 otherwise
        return (perf_i < perf_j).astype(int)

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs) -> None:
        """
        Train binary classifiers for all algorithm pairs.

        Args:
            features: Instance features (n_instances, n_features)
            performance: Algorithm performance (n_instances, n_algorithms)
        """
        self.algorithms = list(performance.columns)
        self.pairs = self._create_pairs(self.algorithms)

        X = features.values

        for algo_i, algo_j in self.pairs:
            y_ij = self._create_binary_labels(performance, algo_i, algo_j)

            clf = self.classifier_class(**self.classifier_kwargs)
            clf.fit(X, y_ij)

            self.classifiers[(algo_i, algo_j)] = clf

    def _compute_copeland_scores(self, features: pd.DataFrame) -> np.ndarray:
        """
        Compute Copeland scores for each instance.

        Args:
            features: Instance features (n_instances, n_features)

        Returns:
            Array of shape (n_instances, n_algorithms) with Copeland scores
        """
        n_instances = len(features)
        n_algorithms = len(self.algorithms)

        scores = np.zeros((n_instances, n_algorithms))
        X = features.values

        for algo_i, algo_j in self.pairs:
            i_idx = self.algorithms.index(algo_i)
            j_idx = self.algorithms.index(algo_j)

            clf = self.classifiers[(algo_i, algo_j)]
            predictions = clf.predict(X)  # 1 if algo_i better, 0 if algo_j better

            scores[predictions == 1, i_idx] += 1
            scores[predictions == 0, j_idx] += 1

        return scores

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict algorithm(s) for each instance using Copeland scores.

        Args:
            features: Instance features

        Returns:
            If top_n == 1: dict[str, list[tuple[str, float]]] with single (algo, budget).
            If top_n > 1: dict[str, list[tuple[str, float]]] with equal time slices.
        """
        if not self.classifiers:
            raise RuntimeError("The selector has not been fitted yet.")

        budget = getattr(self, "budget", None)
        if budget is None:
            budget = float("inf")

        assert isinstance(features, pd.DataFrame)
        scores = self._compute_copeland_scores(features)
        predictions = {}

        for inst_idx, inst_name in enumerate(features.index):
            inst_scores = scores[inst_idx, :]
            order = np.argsort(-inst_scores)
            top_indices = order[: self.top_n]
            top_algos = [self.algorithms[i] for i in top_indices]

            if self.top_n == 1:
                predictions[inst_name] = [(top_algos[0], float(budget))]
            else:
                time_slice = float(budget) / len(top_algos)
                predictions[inst_name] = [(algo, time_slice) for algo in top_algos]

        return predictions
