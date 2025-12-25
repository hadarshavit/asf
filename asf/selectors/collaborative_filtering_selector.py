from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.predictors.linear_model import RidgeRegressorWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        Configuration,
        ConfigurationSpace,
        Float,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class CollaborativeFilteringSelector(ConfigurableMixin, AbstractModelBasedSelector):
    """
    Collaborative filtering selector using SGD matrix factorization (ALORS-style).

    Attributes
    ----------
    n_components : int
        Number of latent factors.
    n_iter : int
        Number of iterations for SGD.
    lr : float
        Learning rate for SGD.
    reg : float
        Regularization strength.
    random_state : int
        Random seed for initialization.
    U : np.ndarray or None
        Instance latent factors.
    V : np.ndarray or None
        Algorithm latent factors.
    performance_matrix : pd.DataFrame or None
        The performance data used for training.
    model : Any or None
        The regressor model to predict latent factors from features.
    mu : float or None
        Global mean of observed performance entries.
    b_U : np.ndarray or None
        Instance biases.
    b_V : np.ndarray or None
        Algorithm biases.
    """

    PREFIX = "collaborative_filtering"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type | Callable[..., Any] = RidgeRegressorWrapper,
        n_components: int = 10,
        n_iter: int = 100,
        lr: float = 0.001,
        reg: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CollaborativeFilteringSelector.

        Parameters
        ----------
        model_class : type or Callable, default=RidgeRegressorWrapper
            The regressor wrapper to predict latent factors from features.
        n_components : int, default=10
            Number of latent factors.
        n_iter : int, default=100
            Number of iterations for SGD.
        lr : float, default=0.001
            Learning rate for SGD.
        reg : float, default=0.1
            Regularization strength.
        random_state : int, default=42
            Random seed for initialization.
        **kwargs : Any
            Additional keyword arguments for the parent classes.
        """
        super().__init__(model_class=model_class, **kwargs)
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.reg = float(reg)
        self.random_state = int(random_state)
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.performance_matrix: pd.DataFrame | None = None
        self.model: Any | None = None

        self.mu: float | None = None
        self.b_U: np.ndarray | None = None
        self.b_V: np.ndarray | None = None

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Fit the collaborative filtering model.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.algorithms = [str(a) for a in performance.columns]
        self.performance_matrix = performance.copy()
        rng = np.random.RandomState(self.random_state)

        n_instances, n_algorithms = performance.shape
        self.U = rng.normal(scale=0.1, size=(n_instances, self.n_components)).astype(
            float
        )
        self.V = rng.normal(scale=0.1, size=(n_algorithms, self.n_components)).astype(
            float
        )

        observed = ~performance.isna()
        rows, cols = np.where(observed.values)
        vals = performance.values

        self.mu = float(np.nanmean(vals))
        self.b_U = np.zeros(n_instances, dtype=float)
        self.b_V = np.zeros(n_algorithms, dtype=float)

        for _ in range(self.n_iter):
            for i, j in zip(rows, cols):
                r_ij = float(vals[i, j])
                pred = (
                    float(self.mu or 0)
                    + self.b_U[i]
                    + self.b_V[j]
                    + float(np.dot(self.U[i], self.V[j]))
                )

                err = r_ij - pred
                err = np.clip(err, -10.0, 10.0)

                self.U[i] += self.lr * (err * self.V[j] - self.reg * self.U[i])
                self.V[j] += self.lr * (err * self.U[i] - self.reg * self.V[j])
                self.b_U[i] += self.lr * (err - self.reg * self.b_U[i])
                self.b_V[j] += self.lr * (err - self.reg * self.b_V[j])

        self.model = self.model_class()
        if self.model is None:
            raise RuntimeError("Model could not be initialized.")

        self.model.fit(features.values, self.U)

    def _predict_cold_start(
        self, instance_features: pd.Series, instance_name: str
    ) -> tuple[str, float]:
        """
        Predict for a single instance using features.

        Parameters
        ----------
        instance_features : pd.Series
            Features of the instance.
        instance_name : str
            Name of the instance.

        Returns
        -------
        tuple
            Tuple of (best_algorithm, score).
        """
        if self.model is None or self.V is None or self.b_V is None:
            raise RuntimeError("Model has not been fitted.")

        X = instance_features[self.features].values.reshape(1, -1)
        U_new = self.model.predict(X)

        scores = float(self.mu or 0) + self.b_V + np.dot(U_new, self.V.T).flatten()
        scores = np.asarray(scores, dtype=float).flatten()

        idx = int(np.argmin(scores))
        return str(self.algorithms[idx]), float(scores[idx])

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict best algorithm for instances.

        Parameters
        ----------
        features : pd.DataFrame or None, default=None
            The input features.
        performance : pd.DataFrame or None, default=None
            The performance data.

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if (
            self.U is None
            or self.V is None
            or self.performance_matrix is None
            or self.b_U is None
            or self.b_V is None
        ):
            raise ValueError("Model has not been fitted.")

        predictions: dict[str, list[tuple[str, float]]] = {}

        # Case 1: Return best algorithm for training instances
        if features is None and performance is None:
            pred_matrix = (
                float(self.mu or 0)
                + self.b_U[:, None]
                + self.b_V[None, :]
                + (self.U @ self.V.T)
            )
            for idx, instance in enumerate(self.performance_matrix.index):
                scores = np.asarray(pred_matrix[idx], dtype=float).flatten()
                best_idx = int(np.argmin(scores))
                predictions[str(instance)] = [
                    (str(self.algorithms[best_idx]), float(self.budget or 0))
                ]
            return predictions

        # Case 2: Warm-start prediction with some observed performance
        if performance is not None:
            rng = np.random.RandomState(self.random_state)
            for instance in performance.index:
                perf_row = performance.loc[instance]
                if not perf_row.isnull().all():
                    u = rng.normal(scale=0.1, size=(self.n_components,)).astype(float)
                    # SGD refinement for instance factors
                    for _ in range(20):
                        for j, _ in enumerate(self.algorithms):
                            r_ij = perf_row.iloc[j]
                            if not pd.isna(r_ij):
                                pred = (
                                    float(self.mu or 0)
                                    + self.b_V[j]
                                    + float(np.dot(u, self.V[j]))
                                )
                                err = float(r_ij) - pred
                                u += self.lr * (err * self.V[j] - self.reg * u)

                    scores = (
                        float(self.mu or 0) + self.b_V + np.dot(u, self.V.T).flatten()
                    )
                    scores = np.asarray(scores, dtype=float).flatten()
                    best_idx = int(np.argmin(scores))
                    predictions[str(instance)] = [
                        (str(self.algorithms[best_idx]), float(self.budget or 0))
                    ]
                else:
                    if features is None:
                        # Fallback to global average if nothing else available
                        avg_scores = self.performance_matrix.mean(axis=0)
                        best_idx = int(np.argmin(avg_scores.values))
                        predictions[str(instance)] = [
                            (str(self.algorithms[best_idx]), float(self.budget or 0))
                        ]
                    else:
                        best_algo, _ = self._predict_cold_start(
                            features.loc[instance], str(instance)
                        )
                        predictions[str(instance)] = [
                            (best_algo, float(self.budget or 0))
                        ]
            return predictions

        # Case 3: Cold-start prediction using only features
        if features is not None and performance is None:
            for instance in features.index:
                best_algo, _ = self._predict_cold_start(
                    features.loc[instance], str(instance)
                )
                predictions[str(instance)] = [(best_algo, float(self.budget or 0))]
            return predictions

        return predictions

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
                Define hyperparameters for CollaborativeFilteringSelector.

                Parameters
                ----------
                model_class : list[type] or None, default=None
                    List of model classes.
                **kwargs : Any
                    Additional keyword arguments.

                Returns
        -------
                tuple
                    Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if model_class is None:
            model_class = [RidgeRegressorWrapper]

        model_class_param = ClassChoice(
            name="model_class",
            choices=model_class,
            default=model_class[0],
        )

        n_components_param = Integer(
            name="n_components",
            bounds=(5, 50),
            default=10,
        )

        n_iter_param = Integer(
            name="n_iter",
            bounds=(50, 500),
            default=100,
        )

        lr_param = Float(
            name="lr",
            bounds=(1e-5, 1e-1),
            log=True,
            default=0.001,
        )

        reg_param = Float(
            name="reg",
            bounds=(1e-4, 1.0),
            log=True,
            default=0.1,
        )

        params = [
            model_class_param,
            n_components_param,
            n_iter_param,
            lr_param,
            reg_param,
        ]

        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[CollaborativeFilteringSelector]:
        """
        Create a partial function from a clean configuration.

        Parameters
        ----------
        clean_config : dict
            The clean configuration.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        partial
            Partial function for CollaborativeFilteringSelector.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(CollaborativeFilteringSelector, **config)
