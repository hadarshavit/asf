import pandas as pd
import numpy as np
from asf.selectors.abstract_model_based_selector import AbstractSelector
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold


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


class SUNNY(AbstractSelector):
    """
    SUNNY/SUNNY-AS2 algorithm selector.

    This selector uses k-nearest neighbors (k-NN) in feature space to construct a schedule. When SUNNY-A2 is enabled, k is optimized.
    """

    PREFIX = "sunny"

    def __init__(
        self,
        k: int = 10,
        use_v2: bool = False,
        n_folds: int = 5,
        k_candidates: list[int] = [3, 5, 7, 10, 20, 50],
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the SUNNY selector.

        Args:
            k (int): Number of neighbors for k-NN.
            use_v2 (bool): Whether to tune k using cross-validation.
            n_folds (int): Number of folds for cross-validation when tuning k.
            k_candidates (list[int]): Candidate k values to consider when tuning.
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
                            total_runtime += self.budget
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
        already_selected: list[str] | None = None,
        already_covered: set | None = None,
    ) -> list[str]:
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
    ) -> list[tuple[str, float]]:
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
        features: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
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
            Get the configuration space for SUNNY selector.

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
                prefix = f"{pre_prefix}:{SUNNY.PREFIX}"
            else:
                prefix = SUNNY.PREFIX

            use_v2_param = Categorical(
                name=f"{prefix}:use_v2",
                items=[True, False],
                default=False,
            )

            k_param = Integer(
                name=f"{prefix}:k",
                bounds=(1, 50),
                default=10,
            )

            n_folds_param = Integer(
                name=f"{prefix}:n_folds",
                bounds=(3, 10),
                default=5,
            )

            k_candidates_param = Categorical(
                name=f"{prefix}:k_candidates",
                items=["small", "medium", "broad"],
                default="medium",
            )

            cs_transform[f"{prefix}:k_candidates"] = {
                "small": [3, 5, 7],
                "medium": [3, 5, 7, 10, 20],
                "broad": [3, 5, 7, 10, 20, 50],
            }

            params = [use_v2_param, k_param, n_folds_param, k_candidates_param]

            conditions = [
                EqualsCondition(
                    child=n_folds_param,
                    parent=use_v2_param,
                    value=True,
                ),
                EqualsCondition(
                    child=k_candidates_param,
                    parent=use_v2_param,
                    value=True,
                ),
            ]

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
        ) -> "SUNNY":
            """
            Get the SUNNY selector from a given configuration.

            Args:
                configuration: The configuration object.
                cs_transform: The transformation dictionary for the configuration space.
                pre_prefix: Prefix for parameter names.
                **kwargs: Additional keyword arguments for SUNNY initialization.

            Returns:
                Sunny: An instance of SUNNY configured according to the given configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{SUNNY.PREFIX}"
            else:
                prefix = SUNNY.PREFIX

            use_v2 = configuration[f"{prefix}:use_v2"]
            k = configuration[f"{prefix}:k"]

            if use_v2:
                n_folds = configuration[f"{prefix}:n_folds"]
                k_candidates = cs_transform[f"{prefix}:k_candidates"][
                    configuration[f"{prefix}:k_candidates"]
                ]
            else:
                n_folds = 5
                k_candidates = [3, 5, 7, 10, 20, 50]

            return SUNNY(
                k=k,
                use_v2=use_v2,
                n_folds=n_folds,
                k_candidates=k_candidates,
                **kwargs,
            )
