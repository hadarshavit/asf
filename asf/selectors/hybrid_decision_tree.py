from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from asf.selectors.abstract_selector import AbstractSelector
from asf.utils.configurable import ConfigurableMixin

try:
    from ConfigSpace import Categorical, Float, Integer

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class HybridDecisionTree:
    """
    A decision tree that uses a hybrid loss combining regression and ranking.
    """

    RETURN_TYPE = "single"

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        lambda_param: float = 0.5,
        max_thresholds: int = 32,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the Hybrid Decision Tree.

        Args:
            max_depth: Maximum depth of the tree.
            min_samples_split: Minimum number of samples required to split an internal node.
            lambda_param: Weighting parameter for combining regression and ranking losses (0 <= lambda_param <= 1).
            max_thresholds: Maximum number of thresholds to consider per feature when searching for splits.
            random_state: Seed for the random number generator.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_param = lambda_param
        self.max_thresholds = max_thresholds
        self.rng = np.random.RandomState(random_state)

        # Tree structure
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["HybridDecisionTree"] = None
        self.right: Optional["HybridDecisionTree"] = None

        # Node labels
        self.regression_label: Optional[np.ndarray] = None
        self.is_leaf: bool = False

    def _compute_regression_label(self, y: np.ndarray) -> np.ndarray:
        """Compute regression label as mean performance."""
        return np.mean(y, axis=0)

    def _vectorized_rank_correlation(
        self, y_ranks: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute average Spearman correlation using vectorized operations.

        Args:
            y_ranks: Pre-computed ranks of performance values (n_instances, n_algorithms)
            y_pred: Predicted performance values (n_algorithms,)

        Returns:
            Average correlation across instances
        """
        n_instances = y_ranks.shape[0]
        if n_instances == 0:
            return 0.0

        # Rank y_pred once
        y_pred_ranks = np.argsort(np.argsort(y_pred)).astype(float)

        # Compute Pearson correlation on ranks (Spearman)
        y_ranks_centered = y_ranks - np.mean(y_ranks, axis=1, keepdims=True)
        y_pred_centered = y_pred_ranks - np.mean(y_pred_ranks)

        # Check if y_pred has zero variance (constant values)
        y_pred_var = np.sum(y_pred_centered**2)
        if y_pred_var < 1e-10:
            return 0.0

        numerator = np.sum(y_ranks_centered * y_pred_centered, axis=1)
        denominator = np.sqrt(np.sum(y_ranks_centered**2, axis=1) * y_pred_var)

        # Avoid division by zero
        correlations = np.where(denominator > 1e-10, numerator / denominator, 0.0)

        return np.mean(correlations)

    def _regression_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error."""
        return np.mean((y - y_pred) ** 2)

    def _ranking_loss(self, y_ranks: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Vectorized Spearman correlation loss using pre-computed ranks.

        Args:
            y_ranks: Pre-computed ranks of performance values (n_instances, n_algorithms)
            y_pred: Predicted performance values (regression label)

        Returns:
            Loss value in [0, 1]: (1 - avg_correlation) / 2
        """
        if y_ranks.shape[0] == 0:
            return 1.0

        avg_corr = self._vectorized_rank_correlation(y_ranks, y_pred)
        # Convert to loss: (1 - correlation) / 2 to scale to [0, 1]
        return (1.0 - avg_corr) / 2.0

    def _hybrid_loss(
        self, y: np.ndarray, y_ranks: np.ndarray, regression_label: np.ndarray
    ) -> float:
        """
        Combined regression and ranking loss.
        Args:
            y: True performance values (n_instances, n_algorithms)
            y_ranks: Pre-computed ranks (n_instances, n_algorithms)
            regression_label: Predicted performances (mean of y for this node)

        Returns:
            Weighted combination of regression and ranking losses
        """
        reg_loss = self._regression_loss(y, regression_label)
        rank_loss = self._ranking_loss(y_ranks, regression_label)

        return self.lambda_param * reg_loss + (1 - self.lambda_param) * rank_loss

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split the data to minimize the hybrid loss.

        Args:
            X (np.ndarray): Feature matrix of shape (n_instances, n_features).
            y (np.ndarray): Target values of shape (n_instances, n_algorithms).

        Returns:
            Tuple[Optional[int], Optional[float]]: The index of the best feature to split on and the threshold value.
                Returns (None, None) if no valid split is found.
        """
        n_instances, n_features = X.shape

        if n_instances < self.min_samples_split:
            return None, None

        # Pre-compute ranks ONCE for all instances (only done once per node)
        y_ranks = np.argsort(np.argsort(y, axis=1), axis=1).astype(float)

        best_loss = float("inf")
        best_feature = None
        best_threshold = None

        # Try each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            if thresholds.size > self.max_thresholds:
                # Subsample candidate thresholds via quantiles to speed up search
                quantiles = np.linspace(0, 100, self.max_thresholds)
                thresholds = np.unique(np.percentile(feature_values, quantiles))

            # Try each threshold
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Slice pre-computed ranks (no recalculation)
                y_ranks_left = y_ranks[left_mask]
                y_ranks_right = y_ranks[right_mask]

                left_reg_label = self._compute_regression_label(y_left)
                right_reg_label = self._compute_regression_label(y_right)

                # Compute weighted loss
                n_left = len(y_left)
                n_right = len(y_right)

                left_loss = self._hybrid_loss(y_left, y_ranks_left, left_reg_label)
                right_loss = self._hybrid_loss(y_right, y_ranks_right, right_reg_label)

                weighted_loss = (n_left / n_instances) * left_loss + (
                    n_right / n_instances
                ) * right_loss

                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> None:
        """
        Fit the hybrid decision tree.

        Args:
            X: Feature matrix (n_instances, n_features)
            y: Performance matrix (n_instances, n_algorithms)
            depth: Current depth in the tree
        """
        n_instances = X.shape[0]

        self.regression_label = self._compute_regression_label(y)

        if depth >= self.max_depth or n_instances < self.min_samples_split:
            self.is_leaf = True
            return

        feature_idx, threshold = self._find_best_split(X, y)
        if feature_idx is None:
            self.is_leaf = True
            return
        self.feature_idx = feature_idx
        self.threshold = threshold

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        self.left = HybridDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            lambda_param=self.lambda_param,
            max_thresholds=self.max_thresholds,
            random_state=self.rng.randint(0, 10000),
        )
        self.left.fit(X_left, y_left, depth + 1)

        self.right = HybridDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            lambda_param=self.lambda_param,
            max_thresholds=self.max_thresholds,
            random_state=self.rng.randint(0, 10000),
        )
        self.right.fit(X_right, y_right, depth + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict performance for instances using regression labels.

        Args:
            X: Feature matrix (n_instances, n_features)

        Returns:
            Predicted performances (n_instances, n_algorithms)
        """
        if self.regression_label is None:
            raise ValueError("Tree has not been fitted. Call fit() before predict().")

        predictions = np.zeros((X.shape[0], len(self.regression_label)))

        for i in range(X.shape[0]):
            predictions[i, :] = self._predict_single(X[i, :])

        return predictions

    def _predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Predict performances for a single instance.

        Args:
            x: Feature vector for a single instance (n_features,)

        Returns:
            Predicted performances for the single instance (n_algorithms,)
        """
        if self.is_leaf:
            assert self.regression_label is not None
            return self.regression_label

        if x[self.feature_idx] <= self.threshold:
            assert self.left is not None
            return self.left._predict_single(x)
        else:
            assert self.right is not None
            return self.right._predict_single(x)


class HARRIS(ConfigurableMixin, AbstractSelector):
    """
    Hybrid Ranking and Regression Forests for Algorithm Selection.

    HARRIS builds an ensemble of decision trees trained with a hybrid loss
    that combines regression (MSE) and ranking (Spearman correlation) objectives.
    Each tree makes splits based on both objectives, but returns predictions
    based on regression labels at leaf nodes.

    References
    ----------
    Fehring, et al. (2022).
    "HARRIS-Hybrid Algorithm Selection using Regression and Ranking."
    https://arxiv.org/pdf/2210.17341
    """

    PREFIX = "harris"
    RETURN_TYPE = "single"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        lambda_param: float = 0.5,
        max_features: Optional[str] = "sqrt",
        max_thresholds: int = 32,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize HARRIS selector.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples required to split a node.
            lambda_param: Balance between regression (λ=1) and ranking (λ=0).
                         λ ∈ [0, 1], controls hybrid loss.
            max_features: Number of features to consider for splits:
                         - "sqrt": sqrt(n_features)
                         - "log2": log2(n_features)
                         - int: exact number
                         - None: all features
            max_thresholds: Maximum candidate thresholds per feature (quantile sampled)
            random_state: Random seed.
            **kwargs: Additional arguments for parent class.
        """
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.lambda_param = float(lambda_param)
        self.max_features = max_features
        self.max_thresholds = int(max_thresholds)
        self.random_state = int(random_state)

        self.trees: List[HybridDecisionTree] = []
        self.algorithms: List[str] = []
        self.feature_indices_per_tree: List[np.ndarray] = []

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for HARRIS.

        The defaults mirror the constructor defaults, while the ranges keep the
        search space compact enough for selector-level HPO.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("n_estimators", bounds=(10, 200), default=100, log=True),
            Integer("max_depth", bounds=(2, 20), default=10),
            Integer("min_samples_split", bounds=(2, 20), default=2),
            Float("lambda_param", bounds=(0.0, 1.0), default=0.5),
            Categorical(
                "max_features",
                items=["sqrt", "log2", "all"],
                default="sqrt",
            ),
            Integer("max_thresholds", bounds=(4, 128), default=32, log=True),
        ]
        return hyperparameters, [], []

    def _scale_performance(self, performance: pd.DataFrame) -> np.ndarray:
        """Scale performance to [0, 1] using min-max normalization."""
        y = performance.values

        # Min-max scaling per algorithm
        y_min = np.nanmin(y, axis=0, keepdims=True)
        y_max = np.nanmax(y, axis=0, keepdims=True)

        # Avoid division by zero
        y_range = y_max - y_min
        y_range[y_range == 0] = 1.0

        return (y - y_min) / y_range

    def _get_n_features_to_sample(self, n_total_features: int) -> int:
        """Determine number of features to sample for each tree."""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_total_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_total_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        else:
            return n_total_features

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs) -> None:
        """
        Train the HARRIS forest.

        Args:
            features: DataFrame of instance features.
            performance: DataFrame of algorithm performance (runtimes).
        """
        self.algorithms = list(performance.columns)
        self.trees = []
        self.feature_indices_per_tree = []

        X = features.values
        y = self._scale_performance(performance)

        n_instances, n_features = X.shape
        n_features_to_sample = self._get_n_features_to_sample(n_features)

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_indices = rng.choice(n_instances, size=n_instances, replace=True)
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]

            # Random feature subset
            feature_indices = rng.choice(
                n_features, size=n_features_to_sample, replace=False
            )
            X_boot_subset = X_boot[:, feature_indices]

            # Train tree
            tree = HybridDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                lambda_param=self.lambda_param,
                max_thresholds=self.max_thresholds,
                random_state=rng.randint(0, 1000000),
            )
            tree.fit(X_boot_subset, y_boot)

            self.trees.append(tree)
            self.feature_indices_per_tree.append(feature_indices)

    def _predict(
        self, features: pd.DataFrame | None, performance: pd.DataFrame | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each instance.

        Args:
            features (pd.DataFrame): Feature matrix with instances as rows and features as columns.

        Returns:
            Dict[str, List[Tuple[str, float]]]: A dictionary mapping each instance name to a list containing
                a tuple of the best algorithm (str) and the budget (float).
        """
        budget = getattr(self, "budget", None)
        if budget is None:
            budget = float("inf")

        assert isinstance(features, pd.DataFrame)
        X = features.values
        predictions = {}

        for inst_idx, inst_name in enumerate(features.index):
            x = X[inst_idx : inst_idx + 1, :]

            # Aggregate predictions from all trees
            tree_predictions = []
            for tree, feature_indices in zip(self.trees, self.feature_indices_per_tree):
                x_subset = x[:, feature_indices]
                pred = tree.predict(x_subset)[0, :]
                tree_predictions.append(pred)

            # Average predictions across trees
            avg_prediction = np.mean(tree_predictions, axis=0)

            # Select best algorithm based on maximize parameter
            if self.maximize:
                best_algo_idx = np.argmax(avg_prediction)
            else:
                best_algo_idx = np.argmin(avg_prediction)
            best_algo = self.algorithms[best_algo_idx]

            predictions[str(inst_name)] = [(str(best_algo), float(budget))]

        return predictions


class tuned_harris(HARRIS):
    """Named HARRIS variant used for tuned selector experiments."""

    PREFIX = "tuned_harris"

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define a bounded HARRIS search space for selector tuning.

        Deep trees with tiny leaves and all features are prohibitively slow on
        larger ASlib folds, especially because SMAC evaluates each config inside
        cross-validation. Keep the tuned variant in the useful, tractable part
        of the search space.
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        hyperparameters = [
            Integer("n_estimators", bounds=(10, 60), default=30, log=True),
            Integer("max_depth", bounds=(2, 12), default=8),
            Integer("min_samples_split", bounds=(5, 30), default=10),
            Float("lambda_param", bounds=(0.0, 1.0), default=0.5),
            Categorical(
                "max_features",
                items=["sqrt", "log2"],
                default="sqrt",
            ),
            Integer("max_thresholds", bounds=(4, 32), default=16, log=True),
        ]
        return hyperparameters, [], []
