import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class _DummyModel:
    """Dummy model class to satisfy AbstractModelBasedSelector requirements."""

    pass


class CollaborativeFilteringSelector(AbstractModelBasedSelector):
    """
    Collaborative filtering selector using SGD matrix factorization (ALORS-style).
    """

    def __init__(
        self,
        n_components: int = 10,
        n_iter: int = 100,
        lr: float = 0.001,
        reg: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initializes the CollaborativeFilteringSelector.

        Args:
            n_components (int): Number of latent factors.
            n_iter (int): Number of iterations for SGD.
            lr (float): Learning rate for SGD.
            reg (float): Regularization strength.
            random_state (int): Random seed for initialization.
            **kwargs: Additional arguments for the parent classes.
        """
        super().__init__(model_class=_DummyModel, **kwargs)
        self.n_components = n_components
        self.n_iter = n_iter
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.U = None  # Instance latent factors
        self.V = None  # Algorithm latent factors
        self.performance_matrix = None
        self.algorithms = []
        self.feature_mapper = None
        self.feature_names = []
        self.scaler = None  # Add scaler attribute

        # Bias terms
        self.mu = None  # Global mean
        self.b_U = None  # Instance biases
        self.b_V = None  # Algorithm biases

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the collaborative filtering model to the given data.

        Args:
            features (pd.DataFrame): DataFrame containing problem instance features.
            performance (pd.DataFrame): DataFrame where columns are algorithms and rows are instances.
        """
        self.algorithms = list(performance.columns)
        self.performance_matrix = performance.copy()
        np.random.seed(self.random_state)

        n_instances, n_algorithms = performance.shape
        # Initialize latent factors
        self.U = np.random.normal(scale=0.1, size=(n_instances, self.n_components))
        self.V = np.random.normal(scale=0.1, size=(n_algorithms, self.n_components))

        # Get observed entries
        observed = ~performance.isna()
        rows, cols = np.where(observed.values)

        # --- Bias initialization ---
        # Global mean from observed entries
        self.mu = np.nanmean(performance.values)
        # Instance and algorithm biases
        self.b_U = np.zeros(n_instances)
        self.b_V = np.zeros(n_algorithms)

        # SGD optimization with bias terms
        for it in range(self.n_iter):
            for i, j in zip(rows, cols):
                r_ij = performance.values[i, j]
                pred = (
                    self.mu + self.b_U[i] + self.b_V[j] + np.dot(self.U[i], self.V[j])
                )
                if np.isnan(r_ij) or np.isnan(pred):
                    continue
                err = r_ij - pred
                err = np.clip(err, -10, 10)
                # Update latent factors
                self.U[i] += self.lr * (err * self.V[j] - self.reg * self.U[i])
                self.V[j] += self.lr * (err * self.U[i] - self.reg * self.V[j])
                # Update biases with L2 regularization
                self.b_U[i] += self.lr * (err - self.reg * self.b_U[i])
                self.b_V[j] += self.lr * (err - self.reg * self.b_V[j])

        # Cold-start feature mapper training with standardized features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features.values)
        self.feature_mapper = Ridge(alpha=1.0)
        self.feature_mapper.fit(X_scaled, self.U)
        self.feature_names = list(features.columns)

    def _predict_cold_start(
        self, instance_features: pd.Series, instance_name: str
    ) -> Tuple[str, float]:
        """
        Predict the best algorithm for a single instance using only its features (cold-start).
        """
        # Align and scale features
        X = instance_features[self.feature_names].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        U_new = self.feature_mapper.predict(X_scaled)
        # Compute scores with global and algorithm bias
        scores = self.mu + self.b_V + np.dot(U_new, self.V.T).flatten()
        scores = np.asarray(scores).flatten()
        best_idx = np.argmin(scores)
        best_algo = self.algorithms[best_idx]
        best_score = scores[best_idx]
        return best_algo, best_score

    def _predict(
        self,
        features: Optional[pd.DataFrame] = None,
        performance: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predicts the best algorithm for instances according to the scenario described.
        """
        if self.U is None or self.V is None or self.performance_matrix is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        predictions = {}

        # Case 1: Return best algorithm for training instances
        if features is None and performance is None:
            pred_matrix = (
                self.mu + self.b_U[:, None] + self.b_V[None, :] + (self.U @ self.V.T)
            )
            for idx, instance in enumerate(self.performance_matrix.index):
                scores = np.asarray(pred_matrix[idx]).flatten()
                best_idx = np.argmin(scores)
                best_algo = self.algorithms[best_idx]
                best_score = scores[best_idx]
                predictions[instance] = [(best_algo, best_score)]
            return predictions

        # Case 2: Performance is not None (ALORS-style prediction for new instances)
        if performance is not None:
            rng = np.random.RandomState(self.random_state)
            for i, instance in enumerate(performance.index):
                perf_row = performance.loc[instance]
                if not perf_row.isnull().all():
                    # Infer latent factors for this instance using observed entries
                    u = rng.normal(scale=0.1, size=(self.n_components,))
                    for _ in range(20):  # few SGD steps
                        for j, algo in enumerate(self.algorithms):
                            if not pd.isna(perf_row[algo]):
                                r_ij = perf_row[algo]
                                pred = self.mu + self.b_V[j] + np.dot(u, self.V[j])
                                err = r_ij - pred
                                u += self.lr * (err * self.V[j] - self.reg * u)
                    scores = self.mu + self.b_V[None, :] + np.dot(u, self.V.T)
                    scores = np.asarray(scores).flatten()
                    best_idx = np.argmin(scores)
                    best_algo = self.algorithms[best_idx]
                    best_score = scores[best_idx]
                    predictions[instance] = [(best_algo, best_score)]
                else:
                    # True cold-start within the warm-start batch: use features if available
                    if features is None:
                        # Fallback to average if no features are available
                        avg_scores = self.performance_matrix.mean()
                        scores = np.asarray(avg_scores.values).flatten()
                        best_idx = np.argmin(scores)
                        best_algo = self.algorithms[best_idx]
                        best_score = scores[best_idx]
                        predictions[instance] = [(best_algo, best_score)]
                    else:
                        instance_features = features.loc[instance]
                        best_algo, best_score = self._predict_cold_start(
                            instance_features, instance
                        )
                        predictions[instance] = [(best_algo, best_score)]
                    continue
            return predictions

        # Case 3: Features is not None, Performance is None (cold start)
        if features is not None and performance is None:
            for instance in features.index:
                instance_features = features.loc[instance]
                best_algo, best_score = self._predict_cold_start(
                    instance_features, instance
                )
                predictions[instance] = [(best_algo, best_score)]
            return predictions

        return predictions
