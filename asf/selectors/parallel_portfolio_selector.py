from typing import List, Dict
import numpy as np
import pandas as pd
from scipy import stats

from asf.selectors.abstract_selector import AbstractSelector
from asf.predictors import AbstractPredictor, RandomForestRegressorWrapper


class APPS(AbstractSelector):
    """
    Automatic Parallel Portfolio Selector based on the approach by Kashgarani and Kotthoff.

    This selector predicts performance distributions for each algorithm and selects
    a parallel portfolio based on the winning probability of each algorithm compared
    to the best-predicted algorithm.

    The size of the portfolio is controlled by the p_intersection threshold, which
    determines the minimum winning probability required for an algorithm to be included.
    """

    PREFIX = "parallel_portfolio"
    RETURN_TYPE = "schedule"

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
            p_intersection: Threshold for winning probability.
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
        """
        self.algorithms = list(performance.columns)

        rng = np.random.RandomState(self.random_state)
        n_instances = len(features)

        # Train bootstrap ensemble for each algorithm
        for algo_idx, algorithm in enumerate(self.algorithms):
            algo_models = []
            algo_performance = performance.iloc[:, algo_idx]

            for boot_idx in range(self.n_estimators_for_std):
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

    def _compute_winning_probability(
        self,
        mu_best: float,
        sigma_best: float,
        mu_candidate: float,
        sigma_candidate: float,
    ) -> float:
        """
        Compute P(A_candidate < A_best) using the difference distribution.
        Args:
            mu_best: Mean performance of the best algorithm.
            sigma_best: Std dev of the best algorithm.
            mu_candidate: Mean performance of the candidate algorithm.
            sigma_candidate: Std dev of the candidate algorithm.
        """
        mu_diff = mu_best - mu_candidate
        sigma_diff = np.sqrt(sigma_best**2 + sigma_candidate**2)
        sigma_diff = np.maximum(sigma_diff, 1e-6)

        # P(A_candidate < A_best) is P(D > 0) = 1 - CDF(0)
        winning_prob = 1.0 - stats.norm.cdf(0, loc=mu_diff, scale=sigma_diff)

        return winning_prob

    def _predict(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Predict parallel portfolio for each instance.

        Returns:
            Dictionary mapping instance names to lists of algorithm names.
        """
        means, stds = self._predict_with_uncertainty(features)
        predictions = {}

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

                winning_prob = self._compute_winning_probability(
                    mu_best, sigma_best, mu_candidate, sigma_candidate
                )

                if winning_prob >= self.p_intersection:
                    portfolio.append(self.algorithms[algo_idx])

            predictions[inst_name] = portfolio

        return predictions
