from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from asf.predictors import (
    AbstractPredictor,
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper,
)
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import AbstractFeatureGenerator
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import (  # noqa: F401
        ConfigurationSpace,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class PairwiseRegressor(
    ConfigurableMixin, AbstractModelBasedSelector, AbstractFeatureGenerator
):
    """
    Selector using pairwise regression of algorithms.

    Attributes
    ----------
    regressors : list[AbstractPredictor]
        Trained regressors for pairwise comparisons.
    """

    PREFIX = "pairwise_regressor"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestRegressorWrapper,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PairwiseRegressor.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=RandomForestRegressorWrapper
            The regression model class used for pairwise comparisons.
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.regressors: list[AbstractPredictor] = []

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the pairwise regressors.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        """
        if self.algorithm_features is not None:
            raise ValueError("PairwiseRegressor does not use algorithm features.")

        self.regressors = []
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1 :]:
                val1 = performance[algorithm].to_numpy(dtype=float)
                val2 = performance[other_algorithm].to_numpy(dtype=float)

                diffs = val1 - val2
                cur_model = self.model_class()
                if cur_model is None:
                    raise RuntimeError("Regressor could not be initialized.")

                cur_model.fit(features, diffs)
                self.regressors.append(cur_model)

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
            raise ValueError("PairwiseRegressor require features for prediction.")
        scores = self.generate_features(features)
        result: dict[str, list[tuple[str, float]]] = {}
        for instance in features.index:
            # If maximizing, we want the highest combined score (algo1 - algo2 > 0)
            # If minimizing, we want the lowest combined score
            if self.maximize:
                best_algo = scores.loc[instance].idxmax()
            else:
                best_algo = scores.loc[instance].idxmin()
            result[str(instance)] = [(str(best_algo), float(self.budget or 0))]
        return result

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """
                Generate pairwise comparisons for each algorithm.

                Parameters
                ----------
                features : pd.DataFrame
                    The input features.

                Returns
        -------
                pd.DataFrame
                    DataFrame of aggregated regression values for each algorithm.
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

        scores = pd.DataFrame(
            0.0,
            index=features_df.index,
            columns=pd.Index(list(self.algorithms)),
        )
        cnt = 0
        for i, algo1 in enumerate(self.algorithms):
            for _j, algo2 in enumerate(self.algorithms[i + 1 :]):
                pred = self.regressors[cnt].predict(features_df)
                # pred is algo1_perf - algo2_perf
                scores.loc[features_df.index, algo1] += pred
                scores.loc[features_df.index, algo2] -= pred
                cnt += 1
        return scores

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for PairwiseRegressor.

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
            model_class = [RandomForestRegressorWrapper, XGBoostRegressorWrapper]

        hyperparameters = [
            ClassChoice("model_class", choices=model_class),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[PairwiseRegressor]:
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
                    Partial function for PairwiseRegressor.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(PairwiseRegressor, **config)
