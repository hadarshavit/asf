from __future__ import annotations

import pandas as pd
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ConfigurableMixin, ClassChoice

try:
    import ConfigSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
from asf.predictors import (
    AbstractPredictor,
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper,
)


class PairwiseRegressor(
    ConfigurableMixin, AbstractModelBasedSelector, AbstractFeatureGenerator
):
    """
    PairwiseRegressor is a selector that uses pairwise regression of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        model_class (type): The regression model class to be used for pairwise comparisons.
    regressors (list[AbstractPredictor]): List of trained regressors for pairwise comparisons.
    """

    PREFIX = "pairwise_regressor"
    RETURN_TYPE = "single"

    def __init__(self, model_class: type = RandomForestRegressorWrapper, **kwargs):
        """
        Initializes the PairwiseRegressor with a given model class and hierarchical feature generator.

        Args:
            model_class (type): The regression model class to be used for pairwise comparisons.
            kwargs: Additional keyword arguments for the parent classes.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.regressors: list[AbstractPredictor] = []

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the pairwise regressors using the provided features and performance data.

        Args:
            features (pd.DataFrame): The feature data for the instances.
            performance (pd.DataFrame): The performance data for the algorithms.
        """
        assert self.algorithm_features is None, (
            "PairwiseRegressor does not use algorithm features."
        )
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1 :]:
                algo1_times = performance[algorithm]
                algo2_times = performance[other_algorithm]

                diffs = algo1_times - algo2_times
                cur_model = self.model_class()
                cur_model.fit(
                    features,
                    diffs,
                    sample_weight=None,
                )
                self.regressors.append(cur_model)

    def _predict(self, features: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
        """
        Predicts the best algorithm for each instance using the trained pairwise regressors.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            dict[str, list[tuple[str, float]]]: A dictionary mapping instance names to the predicted best algorithm
            and the associated budget.
            Example: {instance_name: [(algorithm_name, budget)]}
        """
        predictions_sum = self.generate_features(features)
        return {
            instance_name: [
                (
                    predictions_sum.loc[instance_name].idxmax()
                    if self.maximize
                    else predictions_sum.loc[instance_name].idxmin(),
                    self.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }

    def generate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features for the pairwise regressors.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            pd.DataFrame: A DataFrame of predictions for each instance and algorithm pair.
        """
        cnt = 0
        predictions_sum = pd.DataFrame(0, index=features.index, columns=self.algorithms)
        for i, algorithm in enumerate(self.algorithms):
            for j, other_algorithm in enumerate(self.algorithms[i + 1 :]):
                prediction = self.regressors[cnt].predict(features)
                predictions_sum[algorithm] += prediction
                predictions_sum[other_algorithm] -= prediction
                cnt += 1

        return predictions_sum

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] = None,
        **kwargs,  # Accept additional kwargs from mixin
    ):
        """
        Define hyperparameters for PairwiseRegressor.

        Parameters
        ----------
        model_class : list[type[AbstractPredictor]], optional
            List of model classes to include in the configuration space.
            Defaults to [RandomForestRegressorWrapper, XGBoostRegressorWrapper].
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if model_class is None:
            model_class = [RandomForestRegressorWrapper, XGBoostRegressorWrapper]

        hyperparameters = [
            ClassChoice("model_class", choices=model_class),
        ]
        return hyperparameters, [], []
