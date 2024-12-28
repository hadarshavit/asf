import numpy as np
import pandas as pd

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
    DummyFeatureGenerator,
)


class PerformanceModel(AbstractModelBasedSelector, AbstractFeatureGenerator):
    """
    PerformancePredictor is a class that predicts the performance of algorithms
    based on given features. It can handle both single-target and multi-target
    regression models.

    Attributes:
        model_class: The class of the regression model to be used.
        metadata: Metadata containing information about the algorithms.
        use_multi_target: Boolean indicating whether to use multi-target regression.
        normalize: Method to normalize the performance data.
        regressors: List of trained regression models.
    """

    def __init__(
        self,
        model_class,
        metadata,
        use_multi_target=False,
        normalize="log",
        hierarchical_generator=DummyFeatureGenerator(),
    ):
        """
        Initializes the PerformancePredictor with the given parameters.

        Args:
            model_class: The class of the regression model to be used.
            metadata: Metadata containing information about the algorithms.
            use_multi_target: Boolean indicating whether to use multi-target regression.
            normalize: Method to normalize the performance data.
            hierarchical_generator: Feature generator to be used.
        """
        AbstractModelBasedSelector.__init__(
            self, model_class, metadata, hierarchical_generator
        )
        AbstractFeatureGenerator.__init__(self)
        self.regressors = []
        self.use_multi_target = use_multi_target
        self.normalize = normalize

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the regression models to the given features and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        if self.normalize == "log":
            performance = np.log(performance + 1e-8)

        if self.use_multi_target:
            self.regressors = self.model_class()
            self.regressors.fit(features, performance)
        else:
            for i, algorithm in enumerate(self.metadata.algorithms):
                algo_times = performance.iloc[:, i]

                cur_model = self.model_class()
                cur_model.fit(features, algo_times)
                self.regressors.append(cur_model)

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the performance of algorithms for the given features.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions = self.generate_features(features)

        return {
            instance_name: [
                (
                    self.metadata.algorithms[np.argmin(predictions[i])],
                    self.metadata.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }

    def generate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the given features using the trained models.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            DataFrame containing the predictions for each algorithm.
        """
        if self.use_multi_target:
            predictions = self.regressors.predict(features)
        else:
            predictions = np.zeros((features.shape[0], len(self.metadata.algorithms)))
            for i, algorithm in enumerate(self.metadata.algorithms):
                prediction = self.regressors[i].predict(features)
                predictions[:, i] = prediction

        return predictions
