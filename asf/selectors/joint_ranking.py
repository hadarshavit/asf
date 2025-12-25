from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

from asf.predictors.ranking_mlp import RankingMLP
from asf.selectors.abstract_selector import AbstractSelector
from asf.selectors.feature_generator import AbstractFeatureGenerator
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import ConfigurationSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class JointRanking(ConfigurableMixin, AbstractSelector, AbstractFeatureGenerator):
    """
    Ranking-based algorithm selector.

    Combines feature generation and model-based selection to predict algorithm
    performance.

    Reference:
        Ortuzk et al. (2022)

    Attributes
    ----------
    model : RankingMLP or Callable or None
        The model used for ranking.
    """

    PREFIX = "joint_ranking"
    RETURN_TYPE = "single"

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "AbstractSelector":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __init__(
        self,
        model: RankingMLP | Callable[..., RankingMLP] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the JointRanking selector.

        Parameters
        ----------
        model : RankingMLP or Callable or None, default=None
            The model to be used for ranking algorithms.
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractSelector.__init__(self, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.model = model

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the ranking model.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The performance data.
        """
        if self.algorithm_features is None:
            encoder = OneHotEncoder(sparse_output=False)
            self.algorithm_features = pd.DataFrame(
                encoder.fit_transform(np.array(self.algorithms).reshape(-1, 1)),
                index=list(self.algorithms),  # type: ignore[arg-type]
                columns=[f"algo_{i}" for i in range(len(self.algorithms))],  # type: ignore[arg-type]
            )

        if self.model is None:
            self.model = RankingMLP(
                input_size=len(self.features) + len(self.algorithms)
            )
        elif callable(self.model) and not isinstance(self.model, RankingMLP):
            self.model = self.model(
                input_size=len(self.features) + len(self.algorithms)
            )

        if self.model is None:
            raise RuntimeError("Model could not be initialized.")

        self.model.fit(
            X=features[self.features],
            Y=performance,
            algorithm_features=self.algorithm_features,
        )

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the best algorithm for each instance.

        Parameters
        ----------
        features : pd.DataFrame
            The query instance features.

        Returns
        -------
        dict
            Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("JointRanking require features for prediction.")
        predictions = self.generate_features(features)

        results: dict[str, list[tuple[str, float]]] = {}
        for i, instance_name in enumerate(features.index):
            idx = (
                int(np.argmax(predictions.iloc[i]))
                if self.maximize
                else int(np.argmin(predictions.iloc[i]))
            )
            results[str(instance_name)] = [
                (str(self.algorithms[idx]), float(self.budget or 0))
            ]
        return results

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for each algorithm.

        Parameters
        ----------
        features : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the predictions for each algorithm.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        predictions = np.zeros((base_features.shape[0], len(self.algorithms)))
        selected_features = base_features[self.features]

        for i, algorithm in enumerate(self.algorithms):
            if self.algorithm_features is None:
                raise RuntimeError("Algorithm features are missing.")

            data = selected_features.assign(**self.algorithm_features.loc[algorithm])
            # Ensure column order matches training
            data = data[self.algorithm_features.columns.to_list() + self.features]
            prediction = self.model.predict(data)  # type: ignore[attr-defined]
            predictions[:, i] = prediction.flatten()

        return pd.DataFrame(predictions, columns=list(self.algorithms))  # type: ignore[arg-type]

    @staticmethod
    def _define_hyperparameters(
        model: list[type] | None = None, **kwargs: Any
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
                Define hyperparameters for JointRanking.

                Parameters
                ----------
                model : list[type] or None, default=None
                    List of model classes to choose from.
                **kwargs : Any
                    Additional keyword arguments.

                Returns
        -------
                tuple
                    Tuple of (hyperparameters, conditions, forbiddens).
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if model is None:
            model = [RankingMLP]

        model_param = ClassChoice(
            name="model",
            choices=model,
            default=model[0],
        )

        return [model_param], [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[JointRanking]:
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
            Partial function for JointRanking.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(JointRanking, **config)
