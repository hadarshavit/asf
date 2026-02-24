from typing import List, Any, cast
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors import AbstractPredictor, RandomForestRegressorWrapper


class APPS(AbstractSelector):
    """
    Automatic Parallel Portfolio Selector based on the approach by Kashgarani and Kotthoff.

    This selector predicts performance distributions for each algorithm and selects
    a parallel portfolio based on the overlap of each algorithm's predicted runtime
    distribution with the best algorithm's distribution.

    The size of the portfolio is controlled by the p_intersection threshold, which
    determines the minimum overlap required for an algorithm to be included.
    """

    PREFIX = "parallel_portfolio"
    RETURN_TYPE = "portfolio"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestRegressorWrapper,
        p_intersection: float = 0.82,
        n_estimators_for_std: int = 10,
        random_state: int = 42,
        use_jackknife: bool = True,
        n_jackknife_folds: int | None = None,
        **kwargs,
    ):
        """
        Initialize the Parallel Portfolio Selector.

        Args:
            model_class: The predictor class to use for performance modeling.
            p_intersection: Threshold for distribution overlap.
                          Higher values -> smaller portfolios.
            n_estimators_for_std: Number of Jackknife iterations (n_instances) or bootstrap samples.
            random_state: Random seed for reproducibility.
            use_jackknife: If True, use Jackknife method; if False, use bootstrap.
            n_jackknife_folds: Number of folds for Jackknife. If None, uses leave-one-out.
                              E.g., 10 means 10-fold cross-validation (10 models per algorithm).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.model_class = model_class
        self.p_intersection = float(p_intersection)
        self.n_estimators_for_std = int(n_estimators_for_std)
        self.random_state = int(random_state)
        self.use_jackknife = bool(use_jackknife)
        self.n_jackknife_folds = n_jackknife_folds

        self.predictors: (
            List[List[tuple[AbstractPredictor, Any]]] | List[List[AbstractPredictor]]
        ) = []
        self.algorithms: List[str] = []
        self.features_train: pd.DataFrame | None = None
        self.performance_train: pd.DataFrame | None = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs) -> None:
        """
        Train ensemble of performance models for each algorithm.
        Uses either Jackknife or bootstrap method based on use_jackknife flag.

        Args:
            features: DataFrame of instance features.
            performance: DataFrame of algorithm performance (runtimes).
        """
        self.algorithms = list(performance.columns)
        self.features_train = features.copy()
        self.performance_train = performance.copy()

        rng = np.random.RandomState(self.random_state)
        n_instances = len(features)

        if self.use_jackknife:
            # Jackknife method: K-fold or leave-one-out
            if self.n_jackknife_folds is None:
                n_splits = n_instances
            else:
                n_splits = self.n_jackknife_folds

            kfold = KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )

            for algo_idx, _ in enumerate(self.algorithms):
                algo_models = []
                algo_performance = performance.iloc[:, algo_idx]

                for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(features)):
                    X_train = features.iloc[train_idx]
                    y_train = algo_performance.iloc[train_idx]

                    model = self.model_class()
                    model.fit(X_train, y_train)
                    # Store (model, test_indices) for this fold
                    algo_models.append((model, test_idx))

                self.predictors.append(algo_models)
        else:
            # Bootstrap method: random sampling with replacement
            for algo_idx, _ in enumerate(self.algorithms):
                algo_models = []
                algo_performance = performance.iloc[:, algo_idx]

                for _ in range(self.n_estimators_for_std):
                    sample_indices = rng.choice(
                        n_instances, size=n_instances, replace=True
                    )
                    X_boot = features.iloc[sample_indices]
                    y_boot = algo_performance.iloc[sample_indices]

                    model = self.model_class()
                    model.fit(X_boot, y_boot)
                    algo_models.append(model)

                self.predictors.append(algo_models)

    def _predict_with_uncertainty(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for each algorithm on each instance.
        For Jackknife: uses leave-one-out predictions to estimate std dev.
        For bootstrap: uses bootstrap ensemble predictions to estimate std dev.

        Args:
            features: DataFrame of instance features.

        Returns:
            means: Array of shape (n_instances, n_algorithms) with mean predictions
            stds: Array of shape (n_instances, n_algorithms) with std dev predictions
        """
        n_instances = len(features)
        n_algorithms = len(self.algorithms)

        if self.use_jackknife:
            # Jackknife: compute mean predictions and std dev across K-fold models
            means = np.zeros((n_instances, n_algorithms))
            stds = np.zeros((n_instances, n_algorithms))

            for algo_idx in range(n_algorithms):
                # For each test instance, collect predictions from all models
                all_predictions = []

                # Type narrowing: ensure we have the right structure
                algo_models = self.predictors[algo_idx]
                if not isinstance(algo_models, list):
                    raise RuntimeError("Expected list of models")

                for model_tuple in algo_models:
                    if isinstance(model_tuple, tuple):
                        model, test_idx = model_tuple
                    else:
                        model = model_tuple
                    pred = model.predict(features)  # type: ignore[union-attr]
                    all_predictions.append(pred)

                all_predictions = np.array(all_predictions)  # (n_folds, n_test)
                means[:, algo_idx] = np.mean(all_predictions, axis=0)
                stds[:, algo_idx] = np.std(all_predictions, axis=0)

            stds = np.maximum(stds, 1e-6)
            return means, stds
        else:
            # Bootstrap: compute mean and std dev across bootstrap samples
            all_predictions = np.zeros(
                (n_instances, n_algorithms, self.n_estimators_for_std)
            )

            for algo_idx in range(n_algorithms):
                for model_idx, model in enumerate(self.predictors[algo_idx]):
                    all_predictions[:, algo_idx, model_idx] = model.predict(features)

            means = np.mean(all_predictions, axis=2)
            stds = np.std(all_predictions, axis=2)
            stds = np.maximum(stds, 1e-6)

            return means, stds

    def _solve_intersection_vectorized(
        self,
        mu_best: float,
        sigma_best: float,
        mu_candidates: np.ndarray,
        sigma_candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized computation of PDF intersection points for multiple candidates.

        Args:
            mu_best: Scalar or array of best algorithm means
            sigma_best: Scalar or array of best algorithm stds
            mu_candidates: Array of candidate means (n_candidates,)
            sigma_candidates: Array of candidate stds (n_candidates,)

        Returns:
            Array of intersection points c for each candidate
        """
        # Handle equal variance case
        equal_var = np.abs(sigma_best - sigma_candidates) < 1e-9
        c = np.where(equal_var, (mu_best + mu_candidates) / 2.0, 0.0)

        # Compute for unequal variance cases
        unequal_mask = ~equal_var
        if np.any(unequal_mask):
            sigma_best_arr = np.asarray(sigma_best)
            mu_best_arr = np.asarray(mu_best)

            if sigma_best_arr.ndim == 0:
                sigma_best_ue = cast(float, sigma_best_arr.item())
            else:
                sigma_best_ue = sigma_best_arr[unequal_mask]

            if mu_best_arr.ndim == 0:
                mu_best_ue = cast(float, mu_best_arr.item())
            else:
                mu_best_ue = mu_best_arr[unequal_mask]

            var1 = np.asarray(sigma_best_ue) ** 2
            var2 = np.asarray(sigma_candidates[unequal_mask]) ** 2

            a = 0.5 / var1 - 0.5 / var2
            mu_best_arr = np.asarray(mu_best_ue)
            b = mu_candidates[unequal_mask] / var2 - mu_best_arr / var1
            c_coeff = (
                (mu_best_arr**2) / (2 * var1)
                - (mu_candidates[unequal_mask] ** 2) / (2 * var2)
                - np.log(
                    np.asarray(sigma_candidates[unequal_mask])
                    / np.asarray(sigma_best_ue)
                )
            )

            delta = b**2 - 4 * a * c_coeff
            delta = np.maximum(delta, 0)  # Clamp negative deltas

            sqrt_delta = np.sqrt(delta)
            x1 = (-b - sqrt_delta) / (2 * a)
            x2 = (-b + sqrt_delta) / (2 * a)

            midpoint = (mu_best_ue + mu_candidates[unequal_mask]) / 2.0

            # Choose x closest to midpoint
            c_unequal = np.where(np.abs(x1 - midpoint) < np.abs(x2 - midpoint), x1, x2)

            c[unequal_mask] = c_unequal

        return c

    def _compute_overlap_vectorized(
        self,
        mu_best: float,
        sigma_best: float,
        mu_candidates: np.ndarray,
        sigma_candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized computation of distribution overlaps.

        Args:
            mu_best: Best algorithm mean
            sigma_best: Best algorithm std
            mu_candidates: Array of candidate means
            sigma_candidates: Array of candidate stds

        Returns:
            Array of overlap values for each candidate
        """
        c = self._solve_intersection_vectorized(
            float(mu_best), float(sigma_best), mu_candidates, sigma_candidates
        )

        # Vectorized CDF computation
        p_cand_left = stats.norm.cdf(c, loc=mu_candidates, scale=sigma_candidates)
        p_best_right = 1.0 - stats.norm.cdf(c, loc=mu_best, scale=sigma_best)

        return p_cand_left + p_best_right

    def _solve_intersection(
        self, mu1: float, sigma1: float, mu2: float, sigma2: float
    ) -> float:
        """
        Solve for the PDF intersection point between two Gaussians.
        Assumes mu1 <= mu2; returns a point between the means when possible.
        """

        if abs(sigma1 - sigma2) < 1e-9:
            return (mu1 + mu2) / 2.0

        var1 = sigma1**2
        var2 = sigma2**2

        a = 0.5 / var1 - 0.5 / var2
        b = mu2 / var2 - mu1 / var1
        c = (mu1**2) / (2 * var1) - (mu2**2) / (2 * var2) - np.log(sigma2 / sigma1)

        delta = b**2 - 4 * a * c
        if delta < 0:
            return (mu1 + mu2) / 2.0

        sqrt_delta = np.sqrt(delta)
        x1 = (-b - sqrt_delta) / (2 * a)
        x2 = (-b + sqrt_delta) / (2 * a)

        midpoint = (mu1 + mu2) / 2.0
        in_range_1 = mu1 <= x1 <= mu2
        in_range_2 = mu1 <= x2 <= mu2

        if in_range_1 and not in_range_2:
            return x1
        if in_range_2 and not in_range_1:
            return x2

        return x1 if abs(x1 - midpoint) < abs(x2 - midpoint) else x2

    def _compute_overlap(
        self,
        mu_best: float,
        sigma_best: float,
        mu_candidate: float,
        sigma_candidate: float,
    ) -> float:
        """
        Compute distribution overlap as in Kashgarani & Kotthoff.
        Overlap = P(candidate <= c) + P(best >= c), where c is the PDF intersection.
        """

        c = self._solve_intersection(mu_best, sigma_best, mu_candidate, sigma_candidate)

        p_cand_left = stats.norm.cdf(c, loc=mu_candidate, scale=sigma_candidate)
        p_best_right = 1.0 - stats.norm.cdf(c, loc=mu_best, scale=sigma_best)

        return p_cand_left + p_best_right

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict parallel portfolio for each instance.

        Returns:
            Dictionary mapping instance names to lists of (algorithm, budget) tuples.
        """
        if features is None:
            raise ValueError("features cannot be None for APPS prediction")

        budget = getattr(self, "budget", None)
        if budget is None:
            budget = float("inf")

        means, stds = self._predict_with_uncertainty(features)
        predictions: dict[str, list[tuple[str, float]]] = {}

        # Vectorized ranking across all instances
        ranks = np.argsort(means, axis=1)  # (n_instances, n_algorithms)
        best_algo_indices = ranks[:, 0]

        mu_best_all = means[np.arange(len(means)), best_algo_indices]
        sigma_best_all = stds[np.arange(len(stds)), best_algo_indices]

        for inst_idx, inst_name in enumerate(features.index):
            mu_best = mu_best_all[inst_idx]
            sigma_best = sigma_best_all[inst_idx]
            best_algo_idx = best_algo_indices[inst_idx]

            portfolio = [self.algorithms[best_algo_idx]]

            # Vectorized overlap computation for all candidates
            mu_candidates = means[inst_idx, :]
            sigma_candidates = stds[inst_idx, :]
            overlaps = self._compute_overlap_vectorized(
                mu_best, sigma_best, mu_candidates, sigma_candidates
            )

            # Find algorithms that meet threshold (excluding the best one)
            for rank_idx in range(1, len(self.algorithms)):
                algo_idx = ranks[inst_idx, rank_idx]
                if overlaps[algo_idx] >= self.p_intersection:
                    portfolio.append(self.algorithms[algo_idx])

            predictions[inst_name] = [(algo, float(budget)) for algo in portfolio]

        return predictions
