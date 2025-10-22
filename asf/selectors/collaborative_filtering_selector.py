import pandas as pd
import numpy as np
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.predictors.linear_model import RidgeRegressorWrapper

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Integer,
        Float,
        Configuration,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class CollaborativeFilteringSelector(AbstractModelBasedSelector):
    """
    Collaborative filtering selector using SGD matrix factorization (ALORS-style).
    """

    PREFIX = "collaborative_filtering"

    def __init__(
        self,
        model_class=RidgeRegressorWrapper,
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
            model_class: The regressor wrapper to predict latent factors from features.
            n_components (int): Number of latent factors.
            n_iter (int): Number of iterations for SGD.
            lr (float): Learning rate for SGD.
            reg (float): Regularization strength.
            random_state (int): Random seed for initialization.
            **kwargs: Additional arguments for the parent classes.
        """
        super().__init__(model_class=model_class, **kwargs)
        self.n_components = n_components
        self.n_iter = n_iter
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.U = None  # Instance latent factors
        self.V = None  # Algorithm latent factors
        self.performance_matrix = None
        self.model = None

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

        self.model = self.model_class()
        self.model.fit(features.values, self.U)

    def _predict_cold_start(
        self, instance_features: pd.Series, instance_name: str
    ) -> tuple[str, float]:
        """
        Predict the best algorithm for a single instance using only its features (cold-start).
        """
        # Align and scale features
        X = instance_features[self.features].values.reshape(1, -1)
        U_new = self.model.predict(X)
        # Compute scores with global and algorithm bias
        scores = self.mu + self.b_V + np.dot(U_new, self.V.T).flatten()
        scores = np.asarray(scores).flatten()
        best_idx = np.argmin(scores)
        best_algo = self.algorithms[best_idx]
        best_score = scores[best_idx]
        return best_algo, best_score

    def _predict(
        self,
        features: pd.DataFrame | None = None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
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
                predictions[instance] = [(best_algo, self.budget)]
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
                    predictions[instance] = [(best_algo, self.budget)]
                else:
                    # True cold-start within the warm-start batch: use features if available
                    if features is None:
                        # Fallback to average if no features are available
                        avg_scores = self.performance_matrix.mean()
                        scores = np.asarray(avg_scores.values).flatten()
                        best_idx = np.argmin(scores)
                        best_algo = self.algorithms[best_idx]
                        predictions[instance] = [(best_algo, self.budget)]
                    else:
                        instance_features = features.loc[instance]
                        best_algo, _ = self._predict_cold_start(
                            instance_features, instance
                        )
                        predictions[instance] = [(best_algo, self.budget)]
                    continue
            return predictions

        # Case 3: Features is not None, Performance is None (cold start)
        if features is not None and performance is None:
            for instance in features.index:
                instance_features = features.loc[instance]
                best_algo, _ = self._predict_cold_start(instance_features, instance)
                predictions[instance] = [(best_algo, self.budget)]
            return predictions

        return predictions

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            cs_transform: dict[str, dict] | None = None,
            model_class: list[type] | None = None,
            pre_prefix: str = "",
            parent_param: Hyperparameter | None = None,
            parent_value: str | None = None,
            **kwargs,
        ) -> tuple[ConfigurationSpace, dict[str, dict]]:
            """
            Get the configuration space for CollaborativeFilteringSelector.

            Args:
                cs: The configuration space to use. If None, a new one will be created.
                cs_transform: A dictionary for transforming configuration space parameters.
                model_class: List of regressor wrapper classes to choose from.
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

            if model_class is None:
                model_class = [RidgeRegressorWrapper]

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{CollaborativeFilteringSelector.PREFIX}"
            else:
                prefix = CollaborativeFilteringSelector.PREFIX

            model_class_param = Categorical(
                name=f"{prefix}:model_class",
                items=[str(c.__name__) for c in model_class],
            )

            cs_transform[f"{prefix}:model_class"] = {
                str(c.__name__): c for c in model_class
            }

            n_components_param = Integer(
                name=f"{prefix}:n_components",
                bounds=(5, 50),
                default=10,
            )

            n_iter_param = Integer(
                name=f"{prefix}:n_iter",
                bounds=(50, 500),
                default=100,
            )

            lr_param = Float(
                name=f"{prefix}:lr",
                bounds=(1e-5, 1e-1),
                log=True,
                default=0.001,
            )

            reg_param = Float(
                name=f"{prefix}:reg",
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

            if parent_param is not None:
                conditions = [
                    EqualsCondition(
                        child=param,
                        parent=parent_param,
                        value=parent_value,
                    )
                    for param in params
                ]
            else:
                conditions = []

            cs.add(params + conditions)

            for mc in model_class:
                mc.get_configuration_space(
                    cs=cs,
                    pre_prefix=f"{prefix}:model_class",
                    parent_param=model_class_param,
                    parent_value=str(mc.__name__),
                    **kwargs,
                )

            return cs, cs_transform

        @staticmethod
        def get_from_configuration(
            configuration: Configuration,
            cs_transform: dict[str, dict],
            pre_prefix: str = "",
            **kwargs,
        ) -> "CollaborativeFilteringSelector":
            """
            Get the CollaborativeFilteringSelector from a given configuration.

            Args:
                configuration: The configuration object.
                cs_transform: The transformation dictionary for the configuration space.
                pre_prefix: Prefix for parameter names.
                **kwargs: Additional keyword arguments for CollaborativeFilteringSelector initialization.

            Returns:
                CollaborativeFilteringSelector: An instance configured according to the given configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{CollaborativeFilteringSelector.PREFIX}"
            else:
                prefix = CollaborativeFilteringSelector.PREFIX

            model_cls = cs_transform[f"{prefix}:model_class"][
                configuration[f"{prefix}:model_class"]
            ]
            model_ctor = model_cls.get_from_configuration(
                configuration, pre_prefix=f"{prefix}:model_class"
            )

            n_components = configuration[f"{prefix}:n_components"]
            n_iter = configuration[f"{prefix}:n_iter"]
            lr = configuration[f"{prefix}:lr"]
            reg = configuration[f"{prefix}:reg"]

            return CollaborativeFilteringSelector(
                model_class=model_ctor,
                n_components=n_components,
                n_iter=n_iter,
                lr=lr,
                reg=reg,
                **kwargs,
            )
