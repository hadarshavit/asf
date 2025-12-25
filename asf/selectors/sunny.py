from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class SUNNY(ConfigurableMixin, AbstractSelector):
    """
    SUNNY/SUNNY-AS2 algorithm selector.

    This selector uses k-nearest neighbors (k-NN) in feature space to construct
    a schedule. When SUNNY-AS2 is enabled, k is optimized.

    Attributes
    ----------
    k : int
        Number of neighbors for k-NN.
    use_v2 : bool
        Whether to tune k using cross-validation.
    n_folds : int
        Number of folds for cross-validation when tuning.
    k_candidates : list[int]
        Candidate k values for tuning.
    random_state : int
        Random seed for reproducibility.
    use_tsunny : bool
        Whether to tune the maximum number of algorithms.
    algorithm_limit : int or None
        Manual cap on the number of algorithms in each schedule.
    tuned_algorithm_limit : int or None
        Tuned cap on the number of algorithms.
    features_df : pd.DataFrame or None
        Training features.
    performance_df : pd.DataFrame or None
        Training performance.
    knn : NearestNeighbors or None
        Trained k-NN model.
    """

    PREFIX = "sunny"
    RETURN_TYPE = "schedule"

    def __init__(
        self,
        k: int = 10,
        use_v2: bool = False,
        n_folds: int = 5,
        k_candidates: list[int] | None = None,
        random_state: int = 42,
        use_tsunny: bool = False,
        algorithm_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SUNNY selector.

        Parameters
        ----------
        k : int, default=10
            Number of neighbors for k-NN.
        use_v2 : bool, default=False
            Whether to tune k using cross-validation (SUNNY-AS2).
        n_folds : int, default=5
            Number of folds for cross-validation when tuning.
        k_candidates : list[int] or None, default=None
            Candidate k values to consider when tuning.
        random_state : int, default=42
            Random seed for reproducibility.
        use_tsunny : bool, default=False
            Whether to tune the max number of algorithms via cross-validation.
        algorithm_limit : int or None, default=None
            If set, cap the number of algorithms in each schedule.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.k = int(k)
        self.use_v2 = bool(use_v2)
        self.n_folds = int(n_folds)
        self.k_candidates = k_candidates or [3, 5, 7, 10, 20, 50]
        self.random_state = int(random_state)
        self.use_tsunny = bool(use_tsunny)
        self.algorithm_limit = algorithm_limit
        self.tuned_algorithm_limit: int | None = None

        self.features_df: pd.DataFrame | None = None
        self.performance_df: pd.DataFrame | None = None
        self.knn: NearestNeighbors | None = None

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the SUNNY selector.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        """
        self.features_df = features.copy()
        perf = performance.copy()
        budget = float(self.budget or 1e10)
        perf[perf > budget] = np.nan
        self.performance_df = perf

        if self.use_v2:
            self.k = self._tune_k()

        if self.use_tsunny and self.algorithm_limit is None:
            self.tuned_algorithm_limit = self._tune_algorithm_limit()

        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.features_df)), metric="euclidean"
        )
        self.knn.fit(self.features_df.values)

    def _tune_k(self) -> int:
        """
        Tune the neighborhood size k via cross-validation.

        Returns
        -------
        int
            The best k value found.
        """
        if self.features_df is None or self.performance_df is None:
            return self.k

        best_k = self.k
        best_score = float("inf")
        n_splits = min(self.n_folds, max(2, len(self.features_df)))

        if n_splits < 2:
            return best_k

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        indices = np.arange(len(self.features_df))

        for candidate_k in self.k_candidates:
            fold_scores = []
            for train_idx, val_idx in kf.split(indices):
                train_feat = self.features_df.iloc[train_idx]
                train_perf = self.performance_df.iloc[train_idx]
                val_feat = self.features_df.iloc[val_idx]
                val_perf = self.performance_df.iloc[val_idx]

                if len(train_feat) == 0:
                    continue

                knn = NearestNeighbors(
                    n_neighbors=min(candidate_k, len(train_feat)),
                    metric="euclidean",
                )
                knn.fit(train_feat.values)

                total_cost = 0.0
                for instance in val_feat.index:
                    x = val_feat.loc[[instance]].values
                    _, n_idx = knn.kneighbors(
                        x, n_neighbors=min(candidate_k, len(train_feat))
                    )
                    n_perf = train_perf.iloc[n_idx.flatten()]
                    schedule = self._construct_sunny_schedule(n_perf)

                    inst_perf = val_perf.loc[instance]
                    solved = False
                    for algo, _ in schedule:
                        runtime = inst_perf[algo]
                        if not pd.isna(runtime) and runtime <= float(
                            self.budget or 1e10
                        ):
                            total_cost += float(runtime)
                            solved = True
                            break
                    if not solved:
                        total_cost += float(self.budget or 1e10)

                fold_scores.append(total_cost / len(val_feat))

            mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
            if mean_score < best_score:
                best_score = mean_score
                best_k = int(candidate_k)

        return best_k

    def _tune_algorithm_limit(self) -> int:
        """
        Tune the maximum number of algorithms via cross-validation.

        Returns
        -------
        int
            Best limit found.
        """
        if self.features_df is None or self.performance_df is None:
            return len(self.algorithms)

        n_solvers = len(self.performance_df.columns)
        if n_solvers <= 1:
            return n_solvers

        n_splits = min(self.n_folds, max(2, len(self.features_df)))
        if n_splits < 2:
            return n_solvers

        best_lam = n_solvers
        best_score = float("inf")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        indices = np.arange(len(self.features_df))

        for lam in range(1, n_solvers + 1):
            fold_scores = []
            for train_idx, val_idx in kf.split(indices):
                train_feat = self.features_df.iloc[train_idx]
                train_perf = self.performance_df.iloc[train_idx]
                val_feat = self.features_df.iloc[val_idx]
                val_perf = self.performance_df.iloc[val_idx]

                knn = NearestNeighbors(
                    n_neighbors=min(self.k, len(train_feat)), metric="euclidean"
                )
                knn.fit(train_feat.values)

                total_cost = 0.0
                for instance in val_feat.index:
                    x = val_feat.loc[[instance]].values
                    _, n_idx = knn.kneighbors(
                        x, n_neighbors=min(self.k, len(train_feat))
                    )
                    n_perf = train_perf.iloc[n_idx.flatten()]
                    schedule = self._construct_sunny_schedule(n_perf, lam_limit=lam)

                    inst_perf = val_perf.loc[instance]
                    solved = False
                    for algo, _ in schedule:
                        runtime = inst_perf[algo]
                        if not pd.isna(runtime) and runtime <= float(
                            self.budget or 1e10
                        ):
                            total_cost += float(runtime)
                            solved = True
                            break
                    if not solved:
                        total_cost += float(self.budget or 1e10)
                fold_scores.append(total_cost / len(val_feat))

            mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
            if mean_score < best_score:
                best_score = mean_score
                best_lam = lam

        return best_lam

    def _mine_solvers(
        self,
        neighbor_perf: pd.DataFrame,
        cutoff: int,
        already_selected: list[str] | None = None,
        already_covered: set[str] | None = None,
    ) -> list[str]:
        """
        Recursive greedy set cover to identify a portfolio.
        """
        if already_selected is None:
            already_selected = []
        if already_covered is None:
            already_covered = set()

        remaining_instances = set(neighbor_perf.index) - already_covered
        if len(already_selected) >= cutoff or not remaining_instances:
            return already_selected

        best_solver: str | None = None
        best_cover: set[str] = set()
        best_runtime = float("inf")

        for algo in self.algorithms:
            if algo in already_selected:
                continue
            covers = (
                set(neighbor_perf.index[neighbor_perf[algo].notna()])
                & remaining_instances
            )

            if not best_solver or len(covers) > len(best_cover):
                best_solver = algo
                best_cover = covers
                best_runtime = (
                    float(neighbor_perf.loc[list(covers), algo].sum())
                    if covers
                    else float("inf")
                )
            elif len(covers) == len(best_cover) and len(covers) > 0:
                runtime = float(neighbor_perf.loc[list(covers), algo].sum())
                if runtime < best_runtime:
                    best_solver = algo
                    best_cover = covers
                    best_runtime = runtime

        if not best_solver or not best_cover:
            return already_selected

        already_selected.append(str(best_solver))
        already_covered |= best_cover
        return self._mine_solvers(
            neighbor_perf, cutoff, already_selected, already_covered
        )

    def _construct_sunny_schedule(
        self, neighbor_perf: pd.DataFrame, lam_limit: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Construct a SUNNY schedule.
        """
        lam = (
            lam_limit
            or self.algorithm_limit
            or self.tuned_algorithm_limit
            or len(self.algorithms)
        )
        lam = max(1, min(int(lam), len(self.algorithms)))

        cutoff = min(self.k, lam, len(self.algorithms))
        best_pfolio = self._mine_solvers(neighbor_perf, cutoff)

        valid_perf = neighbor_perf.notna()
        slots = {algo: int(valid_perf[algo].sum()) for algo in best_pfolio}
        n_unsolved = len(
            set(neighbor_perf.index)
            - set().union(
                *(set(neighbor_perf.index[valid_perf[a]]) for a in best_pfolio)
            )
        )

        total_slots = sum(slots.values()) + n_unsolved
        if total_slots == 0:
            slots = {algo: 1 for algo in best_pfolio}
            total_slots = len(best_pfolio)

        budget = float(self.budget or 1e10)
        schedule: list[tuple[str, float]] = []
        for algo in best_pfolio:
            allocated = budget * (slots[algo] / total_slots)
            schedule.append((str(algo), float(allocated)))

        # Sort by mean running time in neighborhood
        avg_times = neighbor_perf[[a for a, _ in schedule]].mean(axis=0).to_dict()
        schedule.sort(key=lambda x: avg_times.get(x[0], float("inf")))

        # Allocate remaining budget
        used = sum(t for _, t in schedule)
        remaining = max(0.0, budget - used)
        if n_unsolved > 0 and remaining > 0:
            # Add to the globally best solver in the neighborhood
            best_global = str(valid_perf.sum(axis=0).idxmax())
            for i, (a, t) in enumerate(schedule):
                if a == best_global:
                    schedule[i] = (a, t + remaining)
                    break
            else:
                if len(schedule) < lam:
                    schedule.append((best_global, remaining))
                    # Resort
                    avg_t = (
                        float(neighbor_perf[best_global].mean())
                        if best_global in neighbor_perf.columns
                        else float("inf")
                    )
                    avg_times[best_global] = avg_t
                    schedule.sort(key=lambda x: avg_times.get(x[0], float("inf")))
                else:
                    a_last, t_last = schedule[-1]
                    schedule[-1] = (a_last, t_last + remaining)

        return schedule

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict schedules for each instance.
        """
        if self.knn is None or self.performance_df is None:
            raise RuntimeError("SUNNY must be fitted.")

        if features is None:
            raise ValueError("Sunny require features for prediction.")
        predictions: dict[str, list[tuple[str, float]]] = {}
        for instance in features.index:
            x = features.loc[[instance]].values
            _, n_idx = self.knn.kneighbors(x, n_neighbors=self.k)
            n_perf = self.performance_df.iloc[n_idx.flatten()]
            predictions[str(instance)] = self._construct_sunny_schedule(n_perf)
        return predictions

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for SUNNY.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        use_v2_param = Categorical(name="use_v2", items=[True, False], default=False)
        k_param = Integer(name="k", bounds=(1, 50), default=10)
        n_folds_param = Integer(name="n_folds", bounds=(3, 10), default=5)
        k_candidates_param = Categorical(
            name="k_candidates", items=["small", "medium", "broad"], default="medium"
        )

        params = [use_v2_param, k_param, n_folds_param, k_candidates_param]
        conditions = [
            EqualsCondition(n_folds_param, use_v2_param, True),
            EqualsCondition(k_candidates_param, use_v2_param, True),
        ]
        return params, conditions, []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[SUNNY]:
        """
        Create a partial function from a clean configuration.
        """
        config = clean_config.copy()
        k_map = {
            "small": [3, 5, 7],
            "medium": [3, 5, 7, 10, 20],
            "broad": [3, 5, 7, 10, 20, 50],
        }
        if config.get("use_v2"):
            config["k_candidates"] = k_map[config.get("k_candidates", "medium")]
        else:
            config["n_folds"] = 5
            config["k_candidates"] = [3, 5, 7, 10, 20, 50]

        config.update(kwargs)
        return partial(SUNNY, **config)
