import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from asf.selectors.abstract_model_based_selector import AbstractSelector
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold


class SunnySelector(AbstractSelector):
    """
    SUNNY/SUNNY-AS2 algorithm selector.

    This selector uses k-nearest neighbors (k-NN) in feature space to construct a schedule. When SUNNY-A2 is enabled, k is optimized.
    """

    def __init__(
        self,
        k: int = 10,
        use_v2: bool = False,
        random_state: int = 42,
        n_folds: int = 5,
        k_candidates: list[int] = [3, 5, 7, 10, 20, 50],
        **kwargs,
    ):
        """
        Initialize the SUNNY selector.

        Args:
            k (int): Number of neighbors for k-NN.
            use_v2 (bool): Whether to tune k using cross-validation.
            budget (float): Total time budget for the schedule.
            random_state (int): Random seed.
            **kwargs: Additional arguments for the parent class.
        """
        super().__init__(**kwargs)
        self.k = k
        self.use_v2 = use_v2
        self.random_state = random_state
        self.features = None
        self.performance = None
        self.knn = None
        self.n_folds = n_folds
        self.k_candidates = k_candidates

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fit the SUNNY selector on the training data.

        Caps all performance values above the budget as unsolved (NaN).
        If use_v2 is True, tunes k using internal cross-validation.

        Args:
            features (pd.DataFrame): Training features (instances x features).
            performance (pd.DataFrame): Training performance matrix (instances x algorithms).
        """
        self.features = features.copy()
        perf = performance.copy()
        perf[perf > self.budget] = np.nan
        self.performance = perf

        # SUNNY-AS2: tune k using cross-validation if requested
        if self.use_v2:
            best_k = self.k
            best_score = float("inf")
            kf = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            instance_indices = np.arange(len(self.features))

            for candidate_k in self.k_candidates:
                fold_scores = []
                for train_idx, val_idx in kf.split(instance_indices):
                    train_features = self.features.iloc[train_idx]
                    train_perf = self.performance.iloc[train_idx]
                    val_features = self.features.iloc[val_idx]
                    val_perf = self.performance.iloc[val_idx]

                    knn = NearestNeighbors(
                        n_neighbors=min(candidate_k, len(train_features)),
                        metric="euclidean",
                    )
                    knn.fit(train_features.values)

                    # For each validation instance, get schedule and compute achieved runtime
                    total_runtime = 0.0
                    n_instances = 0
                    for idx, instance in enumerate(val_features.index):
                        x = val_features.loc[instance].values.reshape(1, -1)
                        dists, neighbor_idxs = knn.kneighbors(
                            x, n_neighbors=min(candidate_k, len(train_features))
                        )
                        neighbor_idxs = neighbor_idxs.flatten()
                        neighbor_perf = train_perf.iloc[neighbor_idxs]
                        schedule = self._construct_sunny_schedule(neighbor_perf)

                        # Evaluate: take the first algorithm in the schedule that solves the instance, or assign budget if none solve it
                        instance_perf = val_perf.loc[instance]
                        solved = False
                        for algo, _ in schedule:
                            runtime = instance_perf[algo]
                            if not np.isnan(runtime) and runtime <= self.budget:
                                total_runtime += runtime
                                solved = True
                                break
                        if not solved:
                            total_runtime += self.budget  # Penalize unsolved
                        n_instances += 1

                    avg_runtime = (
                        total_runtime / n_instances if n_instances > 0 else float("inf")
                    )
                    fold_scores.append(avg_runtime)

                mean_score = np.mean(fold_scores)
                if mean_score < best_score:
                    best_score = mean_score
                    best_k = candidate_k

            self.k = best_k

        # Fit final model with optimal k
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.features)), metric="euclidean"
        )
        self.knn.fit(self.features.values)

    def _mine_solvers(
        self,
        neighbor_perf: pd.DataFrame,
        cutoff: int,
        already_selected: Optional[List[str]] = None,
        already_covered: Optional[set] = None,
    ) -> List[str]:
        """
        Recursive greedy set cover to identify a portfolio of solvers.
        Tie-break by minimum total runtime on solved instances.

        Args:
            neighbor_perf (pd.DataFrame): Performance matrix for the k nearest neighbors.
            cutoff (int): Maximum number of solvers to select.
            already_selected (Optional[List[str]]): Solvers already selected (for recursion).
            already_covered (Optional[set]): Instances already covered (for recursion).

        Returns:
            List[str]: List of selected solver names.
        """
        if already_selected is None:
            already_selected = []
        if already_covered is None:
            already_covered = set()

        remaining_instances = set(neighbor_perf.index) - already_covered
        if len(already_selected) >= cutoff or not remaining_instances:
            return already_selected

        # For each solver, count how many new instances it can solve
        best_solver = None
        best_cover = set()
        best_runtime = None
        for algo in self.algorithms:
            if algo in already_selected:
                continue
            # Instances this solver solves and are not yet covered
            covers = (
                set(neighbor_perf.index[neighbor_perf[algo].notna()])
                & remaining_instances
            )
            if not best_solver or len(covers) > len(best_cover):
                best_solver = algo
                best_cover = covers
                # For tie-breaking, sum runtime on these instances
                best_runtime = (
                    neighbor_perf.loc[list(covers), algo].sum() if covers else np.inf
                )
            elif len(covers) == len(best_cover):
                runtime = (
                    neighbor_perf.loc[list(covers), algo].sum() if covers else np.inf
                )
                if runtime < best_runtime:
                    best_solver = algo
                    best_cover = covers
                    best_runtime = runtime

        if not best_cover:
            return already_selected

        already_selected.append(best_solver)
        already_covered |= best_cover
        return self._mine_solvers(
            neighbor_perf, cutoff, already_selected, already_covered
        )

    def _construct_sunny_schedule(
        self, neighbor_perf: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        """
        Construct a SUNNY schedule for a given neighborhood.

        Uses recursive greedy set cover to select a portfolio, allocates time slices
        proportionally to solved counts, and (if needed) adds a backup solver.

        Args:
            neighbor_perf (pd.DataFrame): Performance matrix for the k nearest neighbors.

        Returns:
            List[Tuple[str, float]]: List of (algorithm, allocated_time) tuples, sorted by average runtime.
        """
        # 1. H_sel: Select portfolio using recursive greedy set cover
        cutoff = min(self.k, len(self.algorithms))
        best_pfolio = self._mine_solvers(neighbor_perf, cutoff)

        # Count solved/unsolved instances for each selected solver
        solved_mask = neighbor_perf.notna()
        slots = {algo: solved_mask[algo].sum() for algo in best_pfolio}

        covered = set()
        for algo in best_pfolio:
            covered |= set(neighbor_perf.index[solved_mask[algo]])
        n_unsolved = len(set(neighbor_perf.index) - covered)

        # Total time slots = sum of solved counts + unsolved
        total_slots = sum(slots.values()) + n_unsolved
        if total_slots == 0:
            # fallback: equal allocation
            slots = {algo: 1 for algo in best_pfolio}
            total_slots = len(best_pfolio)

        # 2. H_all: Allocate time slices proportionally
        schedule = []
        for algo in best_pfolio:
            t = self.budget * (slots[algo] / total_slots)
            schedule.append((algo, t))

        # If there are unsolved instances, allocate remaining time to backup solver
        time_used = sum(t for _, t in schedule)
        if n_unsolved > 0:
            backup_time = self.budget - time_used
            if backup_time > 0:
                backup_algo = solved_mask.sum(axis=0).idxmax()
                schedule.append((backup_algo, backup_time))

        # 3. H_sch: Sort by average runtime (ascending) among neighbors
        avg_times = neighbor_perf[[algo for algo, _ in schedule]].mean(axis=0).to_dict()
        schedule.sort(key=lambda x: avg_times.get(x[0], float("inf")))

        return schedule

    def _predict(
        self,
        features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predict a SUNNY schedule for each instance in the provided features.

        Args:
            features (pd.DataFrame): Feature matrix for the test instances.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping from instance name to schedule (list of (algorithm, time) tuples).
        """
        if features is None:
            raise ValueError("Features must be provided for prediction.")

        predictions = {}
        for idx, instance in enumerate(features.index):
            x = features.loc[instance].values.reshape(1, -1)
            dists, neighbor_idxs = self.knn.kneighbors(x, n_neighbors=self.k)
            neighbor_idxs = neighbor_idxs.flatten()
            neighbor_perf = self.performance.iloc[neighbor_idxs]

            schedule = self._construct_sunny_schedule(neighbor_perf)
            predictions[instance] = schedule

        return predictions
