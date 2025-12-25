import numpy as np
import pandas as pd

try:
    from ConfigSpace import (
        Categorical,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.predictors import (
    AbstractPredictor,
    RandomForestClassifierWrapper,
    XGBoostClassifierWrapper,
)
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
from asf.utils.configurable import ConfigurableMixin, ClassChoice


class PairwiseClassifier(
    ConfigurableMixin, AbstractModelBasedSelector, AbstractFeatureGenerator
):
    """
    PairwiseClassifier is a selector that uses pairwise comparison of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        PREFIX (str): Prefix used for configuration space parameters.
        classifiers (List[AbstractPredictor]): List of trained classifiers for pairwise comparisons.
        use_weights (bool): Whether to use weights based on performance differences.
    """

    PREFIX = "pairwise_classifier"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestClassifierWrapper,
        use_weights: bool = True,
        **kwargs,
    ):
        """
        Initializes the PairwiseClassifier with a given model class and hierarchical feature generator.

        Args:
            model_class (type[AbstractPredictor]): The classifier model to be used for pairwise comparisons.
            use_weights (bool): Whether to use weights based on performance differences. Defaults to True.
            **kwargs: Additional keyword arguments for the parent class.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.classifiers: list[AbstractPredictor] = []
        self.use_weights: bool = use_weights

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the pairwise classifiers using the provided features and performance data.

        Args:
            features (pd.DataFrame): The feature data for the instances.
            performance (pd.DataFrame): The performance data for the algorithms.
        """
        assert self.algorithm_features is None, (
            "PairwiseClassifier does not use algorithm features."
        )
        for i, algorithm in enumerate(self.algorithms):
            for other_algorithm in self.algorithms[i + 1 :]:
                algo1_times = performance[algorithm]
                algo2_times = performance[other_algorithm]

                if self.maximize:
                    diffs = algo1_times > algo2_times
                else:
                    diffs = algo1_times < algo2_times

                # Ensure diffs are integers (0/1), not boolean
                diffs = diffs.astype(int)

                cur_model = self.model_class()
                cur_model.fit(
                    features,
                    diffs,
                    sample_weight=None
                    if not self.use_weights
                    else np.abs(algo1_times - algo2_times),
                )
                self.classifiers.append(cur_model)

    def _predict(
        self, features: pd.DataFrame
    ) -> dict[str, list[tuple[str, int | float]]]:
        """
        Predicts the best algorithm for each instance using the trained pairwise classifiers.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            dict[str, list[tuple[str, int | float]]]: A dictionary mapping instance names to the predicted best algorithm and budget.
            Example: {instance_name: [(algorithm_name, budget)]}
        """
        predictions_sum = self.generate_features(features)
        result = {
            instance_name: [
                (
                    predictions_sum.loc[instance_name].idxmax(),
                    self.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }
        return result

    def generate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features for the pairwise classifiers.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            pd.DataFrame: A DataFrame of predictions for each instance and algorithm pair.
        """
        # Ensure we are working with a pandas DataFrame
        if not isinstance(features, pd.DataFrame):
            if hasattr(self, "features") and isinstance(self.features, list):
                cols = self.features
            else:
                cols = [f"f_{i}" for i in range(features.shape[1])]
            features = pd.DataFrame(features, index=range(len(features)), columns=cols)

        cnt = 0
        predictions_sum = pd.DataFrame(0, index=features.index, columns=self.algorithms)
        for i, algorithm in enumerate(self.algorithms):
            for j, other_algorithm in enumerate(self.algorithms[i + 1 :]):
                prediction = self.classifiers[cnt].predict(features)
                # prediction is an array of 0s and 1s
                # 1 means algorithm is better, 0 means other_algorithm is better
                predictions_sum.loc[features.index[prediction == 1], algorithm] += 1
                predictions_sum.loc[
                    features.index[prediction == 0], other_algorithm
                ] += 1
                cnt += 1

        return predictions_sum

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] = None,
        **kwargs,  # Accept additional kwargs from mixin
    ):
        """
        Define hyperparameters for PairwiseClassifier.

        Parameters
        ----------
        model_class : list[type[AbstractPredictor]], optional
            List of model classes to include in the configuration space.
            Defaults to [RandomForestClassifierWrapper, XGBoostClassifierWrapper].
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
