from typing import List
import numpy as np
import pandas as pd
from scipy import stats

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
        p_intersection: float = 0.1,
        n_estimators_for_std: int = 10,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the Parallel Portfolio Selector.

        Args:
            model_class: The predictor class to use for performance modeling.
            p_intersection: Threshold for distribution overlap.
                          Higher values -> smaller portfolios.
            n_estimators_for_std: Number of bootstrap models to estimate std deviation.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.model_class = model_class
        self.p_intersection = float(p_intersection)
        self.n_estimators_for_std = int(n_estimators_for_std)
        self.random_state = int(random_state)

        self.predictors: List[List[AbstractPredictor]] = []
        self.algorithms: List[str] = []

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs) -> None:
        """
        Train ensemble of performance models for each algorithm.

        Args:
            features: DataFrame of instance features.
            performance: DataFrame of algorithm performance (runtimes).
        """
        self.algorithms = list(performance.columns)

        rng = np.random.RandomState(self.random_state)
        n_instances = len(features)

        # Train bootstrap ensemble for each algorithm
        for algo_idx, _ in enumerate(self.algorithms):
            algo_models = []
            algo_performance = performance.iloc[:, algo_idx]

            for _ in range(self.n_estimators_for_std):
                sample_indices = rng.choice(n_instances, size=n_instances, replace=True)
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

        Args:
            features: DataFrame of instance features.

        Returns:
            means: Array of shape (n_instances, n_algorithms) with mean predictions
            stds: Array of shape (n_instances, n_algorithms) with std dev predictions
        """
        n_instances = len(features)
        n_algorithms = len(self.algorithms)

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

        for inst_idx, inst_name in enumerate(features.index):
            inst_means = means[inst_idx, :]
            inst_stds = stds[inst_idx, :]

            ranks = np.argsort(inst_means)
            best_algo_idx = ranks[0]

            mu_best = inst_means[best_algo_idx]
            sigma_best = inst_stds[best_algo_idx]

            portfolio = [self.algorithms[best_algo_idx]]

            for rank_idx in range(1, len(self.algorithms)):
                algo_idx = ranks[rank_idx]
                mu_candidate = inst_means[algo_idx]
                sigma_candidate = inst_stds[algo_idx]

                overlap = self._compute_overlap(
                    mu_best, sigma_best, mu_candidate, sigma_candidate
                )

                if overlap >= self.p_intersection:
                    portfolio.append(self.algorithms[algo_idx])

            predictions[inst_name] = [(algo, float(budget)) for algo in portfolio]

        return predictions
