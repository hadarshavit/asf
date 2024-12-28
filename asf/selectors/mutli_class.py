import pandas as pd
import numpy as np
from asf.selectors.feature_generator import DummyFeatureGenerator
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class MultiClassClassifier(AbstractModelBasedSelector):
    """
    MultiClassClassifier is a class that predicts the best algorithm for a given instance
    using a multi-class classification model.

    Attributes:
        model_class: The class of the classification model to be used.
        metadata: Metadata containing information about the algorithms.
        classifier: The trained classification model.
    """

    def __init__(
        self, model_class, metadata, hierarchical_generator=DummyFeatureGenerator()
    ):
        """
        Initializes the MultiClassClassifier with the given parameters.

        Args:
            model_class: The class of the classification model to be used.
            metadata: Metadata containing information about the algorithms.
            hierarchical_generator: Feature generator to be used.
        """
        AbstractModelBasedSelector.__init__(
            self, model_class, metadata, hierarchical_generator
        )
        self.classifier = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the classification model to the given feature and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """
        self.classifier = self.model_class()
        self.classifier.fit(features, np.argmin(performance.values, axis=1))

    def _predict(self, features: pd.DataFrame):
        """
        Predicts the best algorithm for each instance in the given feature data.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            A dictionary mapping instance names to the predicted best algorithm.
        """
        predictions = self.classifier.predict(features)

        return {
            instance_name: self.metadata.algorithms[predictions[i]]
            for i, instance_name in enumerate(features.index)
        }
