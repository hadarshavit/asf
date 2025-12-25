import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

from asf.selectors.abstract_selector import AbstractSelector
from asf.presolving.aspeed import Aspeed, CLINGO_AVAIL
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
        Categorical,
        Integer,
        EqualsCondition,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial
from typing import Any


class ISA(ConfigurableMixin, AbstractSelector):
    """
    ISA (Instance-Specific Aspeed) selector.
    """

    PREFIX = "isa"
    RETURN_TYPE = "schedule"

    def __init__(
        self,
        k: int = 10,
        use_k_tuning: bool = True,
        n_folds: int = 5,
        k_candidates: list[int] | None = None,
        aspeed_cutoff: int = 30,
        cores: int = 1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the ISA selector.

        Args:
            k (int): Number of neighbors for k-NN.
            use_k_tuning (bool): Whether to tune k using cross-validation.
            n_folds (int): Number of folds for cross-validation when tuning k.
            k_candidates (list[int]): Candidate k values to consider when tuning.
            aspeed_cutoff (int): Time limit for the internal aspeed solver.
            cores (int): Number of cores for the internal aspeed solver.
            random_state (int): Random seed for reproducibility.
            **kwargs: Additional arguments for the parent class.
        """
        if not CLINGO_AVAIL:
            raise ImportError("clingo is not installed. Please install it to use ISA.")
        super().__init__(**kwargs)
        self.k = k
        self.use_k_tuning = use_k_tuning
        self.n_folds = n_folds
        self.k_candidates = [3, 5, 10, 15, 20] if k_candidates is None else k_candidates
        self.aspeed_cutoff = aspeed_cutoff
        self.cores = cores
        self.random_state = random_state

        self.reduced_features: pd.DataFrame | None = None
        self.reduced_performance: pd.DataFrame | None = None
        self.knn: NearestNeighbors | None = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the ISA selector.

        This involves reducing the instance set and tuning k.

        Args:
            features (pd.DataFrame): Training features (instances x features).
            performance (pd.DataFrame): Training performance (instances x algorithms).
        """
        # Instance Set Reduction
        is_solved = performance < self.budget
        solved_by_all = is_solved.all(axis=1)
        solved_by_none = ~is_solved.any(axis=1)
        trivial_mask = solved_by_all | solved_by_none

        self.reduced_features = features[~trivial_mask].copy()
        self.reduced_performance = performance[~trivial_mask].copy()

        if self.reduced_features.empty:
            return

        if self.use_k_tuning:
            self.k = self._tune_k()

        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.reduced_features)), metric="euclidean"
        )
        self.knn.fit(self.reduced_features.values)

    def _tune_k(self) -> int:
        """
        Find the best k using cross-validation on the reduced instance set.
        """
        if len(self.reduced_features) < self.n_folds:
            return self.k

        best_k = self.k
        best_score = float("inf")
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        instance_indices = np.arange(len(self.reduced_features))

        for candidate_k in self.k_candidates:
            fold_scores = []
            for train_idx, val_idx in kf.split(instance_indices):
                train_features = self.reduced_features.iloc[train_idx]
                train_perf = self.reduced_performance.iloc[train_idx]
                val_features = self.reduced_features.iloc[val_idx]
                val_perf = self.reduced_performance.iloc[val_idx]

                if len(train_features) < candidate_k:
                    continue

                knn = NearestNeighbors(n_neighbors=candidate_k, metric="euclidean")
                knn.fit(train_features.values)

                total_runtime = 0.0
                for _, instance_row in val_features.iterrows():
                    x = instance_row.values.reshape(1, -1)
                    _, neighbor_idxs = knn.kneighbors(x)
                    neighbor_perf = train_perf.iloc[neighbor_idxs.flatten()]

                    schedule = self._get_aspeed_schedule(neighbor_perf)

                    instance_actual_perf = val_perf.loc[instance_row.name]
                    solved = False
                    for algo, _ in schedule:
                        runtime = instance_actual_perf.get(algo)
                        if runtime is not None and runtime < self.budget:
                            total_runtime += runtime
                            solved = True
                            break
                    if not solved:
                        total_runtime += self.budget

                avg_runtime = total_runtime / len(val_features)
                fold_scores.append(avg_runtime)

            mean_score = np.mean(fold_scores) if fold_scores else float("inf")
            if mean_score < best_score:
                best_score = mean_score
                best_k = candidate_k

        return best_k

    def _get_aspeed_schedule(
        self, performance_subset: pd.DataFrame
    ) -> list[tuple[str, float]]:
        """
        Run aspeed on performance data to get a schedule.
        """
        aspeed_presolver = Aspeed(
            budget=self.budget,
            aspeed_cutoff=self.aspeed_cutoff,
            cores=self.cores,
        )

        aspeed_presolver.fit(features=None, performance=performance_subset)
        schedule = aspeed_presolver.predict()
        schedule.sort(key=lambda x: x[1])

        total_time = sum(time for _, time in schedule)
        remaining_time = self.budget - total_time

        if remaining_time > 0:
            max_idx = max(range(len(schedule)), key=lambda i: schedule[i][1])
            algo, time = schedule[max_idx]
            schedule[max_idx] = (algo, time + remaining_time)

        return schedule

    def _predict(self, features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        """
        For each test instance, find neighbors and compute a schedule using aspeed.

        Args:
            features (pd.DataFrame): Feature matrix for the test instances.

        Returns:
            Dict[str, List[Tuple[str, float]]]: A schedule for each instance.
        """
        if self.knn is None:
            # This happens if the training data was all trivial
            return {instance: [] for instance in features.index}

        predictions = {}
        for instance_name, instance_row in features.iterrows():
            x = instance_row.values.reshape(1, -1)
            _, neighbor_idxs = self.knn.kneighbors(x)
            neighbor_perf = self.reduced_performance.iloc[neighbor_idxs.flatten()]

            schedule = self._get_aspeed_schedule(neighbor_perf)
            predictions[instance_name] = schedule

        return predictions

    @staticmethod
    def _define_hyperparameters(**kwargs):
        """Define hyperparameters for ISA."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        k_param = Integer(
            name="k",
            bounds=(1, 50),
            default=10,
        )

        use_k_tuning_param = Categorical(
            name="use_k_tuning",
            items=[True, False],
            default=True,
        )

        n_folds_param = Integer(
            name="n_folds",
            bounds=(2, 10),
            default=5,
        )

        k_candidates_param = Categorical(
            name="k_candidates",
            items=["small", "medium", "broad"],
            default="medium",
        )

        aspeed_cutoff_param = Integer(
            name="aspeed_cutoff",
            bounds=(1, 300),
            default=30,
        )

        cores_param = Integer(
            name="cores",
            bounds=(1, 8),
            default=1,
        )

        params = [
            k_param,
            use_k_tuning_param,
            n_folds_param,
            k_candidates_param,
            aspeed_cutoff_param,
            cores_param,
        ]

        conditions = [
            EqualsCondition(n_folds_param, use_k_tuning_param, True),
            EqualsCondition(k_candidates_param, use_k_tuning_param, True),
        ]

        return params, conditions, []

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

        # Helper to map k_candidates categorical to list
        k_candidates_map = {
            "small": [3, 5, 10],
            "medium": [3, 5, 10, 15, 20],
            "broad": [3, 5, 10, 15, 20, 30, 50],
        }

        use_k = config.get("use_k_tuning", True)

        if use_k:
            k_candidates_str = config.get("k_candidates", "medium")
            config["k_candidates"] = k_candidates_map[k_candidates_str]
        else:
            # Default fallback if not tuning k
            config["k_candidates"] = [3, 5, 10, 15, 20]  # Default from __init__
            if "n_folds" not in config:
                config["n_folds"] = 5

        config.update(kwargs)
        return partial(ISA, **config)
