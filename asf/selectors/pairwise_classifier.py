from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from asf.predictors import (
    AbstractPredictor,
    RandomForestClassifierWrapper,
    XGBoostClassifierWrapper,
)
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import AbstractFeatureGenerator
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        Categorical,
        ConfigurationSpace,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class PairwiseClassifier(
    ConfigurableMixin, AbstractModelBasedSelector, AbstractFeatureGenerator
):
    """
    Selector using pairwise comparison of algorithms.

    Attributes
    ----------
    classifiers : list[AbstractPredictor]
        Trained classifiers for pairwise comparisons.
    use_weights : bool
        Whether to use weights based on performance differences.
    """

    PREFIX = "pairwise_classifier"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestClassifierWrapper,
        use_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PairwiseClassifier.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=RandomForestClassifierWrapper
            The classifier model class used for pairwise comparisons.
        use_weights : bool, default=True
            Whether to use weights based on performance differences.
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.classifiers: list[AbstractPredictor] = []
        self.use_weights = bool(use_weights)

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the pairwise classifiers.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        """
        if self.algorithm_features is not None:
            raise ValueError("PairwiseClassifier does not use algorithm features.")

        self.classifiers = []
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1 :]:
                val1 = performance[algorithm].to_numpy(dtype=float)
                val2 = performance[other_algorithm].to_numpy(dtype=float)

                if self.maximize:
                    diffs = (val1 > val2).astype(int)
                else:
                    diffs = (val1 < val2).astype(int)

                cur_model = self.model_class()
                if cur_model is None:
                    raise RuntimeError("Classifier could not be initialized.")

                cur_model.fit(
                    features,
                    diffs,
                    sample_weight=None if not self.use_weights else np.abs(val1 - val2),
                )
                self.classifiers.append(cur_model)

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
            raise ValueError("PairwiseClassifier require features for prediction.")
        votes = self.generate_features(features)
        result: dict[str, list[tuple[str, float]]] = {}
        for instance in features.index:
            best_algo = votes.loc[instance].idxmax()
            result[str(instance)] = [(str(best_algo), float(self.budget or 0))]
        return result

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """
                Generate vote counts for each algorithm.

                Parameters
                ----------
                base_features : pd.DataFrame
                    The input features.

                Returns
        -------
                pd.DataFrame
                    DataFrame of vote counts for each algorithm.
        """
        # Ensure input is a DataFrame
        if not isinstance(base_features, pd.DataFrame):
            cols = (
                self.features
                if self.features
                else [f"f_{i}" for i in range(base_features.shape[1])]
            )
            features_df = pd.DataFrame(base_features, columns=list(cols))  # type: ignore[arg-type]
        else:
            features_df = base_features

        votes = pd.DataFrame(0, index=features_df.index, columns=list(self.algorithms))  # type: ignore[arg-type]
        cnt = 0
        for i, algo1 in enumerate(self.algorithms):
            for _j, algo2 in enumerate(self.algorithms[i + 1 :]):
                pred = self.classifiers[cnt].predict(features_df)
                # 1 means algo1 is better, 0 means algo2 is better
                votes.loc[features_df.index[pred == 1], algo1] += 1
                votes.loc[features_df.index[pred == 0], algo2] += 1
                cnt += 1
        return votes

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for PairwiseClassifier.

        Parameters
        ----------
        model_class : list[type[AbstractPredictor]] or None, default=None
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
            model_class = [RandomForestClassifierWrapper, XGBoostClassifierWrapper]

        hyperparameters = [
            ClassChoice("model_class", choices=model_class),
            Categorical("use_weights", items=[True, False], default=True),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[PairwiseClassifier]:
        """
                Create a partial function from a clean configuration.

                Parameters
        -------
                clean_config : dict
                    The clean configuration.
                **kwargs : Any
                    Additional keyword arguments.

                Returns
                -------
                partial
                    Partial function for PairwiseClassifier.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(PairwiseClassifier, **config)
