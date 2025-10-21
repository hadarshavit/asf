import pandas as pd
import numpy as np
from asf.predictors import AbstractPredictor
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class MultiClassClassifier(AbstractModelBasedSelector):
    """
    A selector that uses a multi-class classification model to predict the best algorithm
    for a given set of features and performance data.
    """

    def __init__(self, model_class: type[AbstractPredictor], **kwargs):
        """
        Initializes the MultiClassClassifier.

        Args:
            model_class: The class of the model to be used for classification.
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        self.classifier: object = None

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the classification model to the given feature and performance data.

        Args:
            features (pd.DataFrame): DataFrame containing the feature data.
                Each row corresponds to an instance, and each column corresponds to a feature.
            performance (pd.DataFrame): DataFrame containing the performance data.
                Each row corresponds to an instance, and each column corresponds to an algorithm.
        """
        assert self.algorithm_features is None, (
            "MultiClassClassifier does not use algorithm features."
        )
        self.classifier = self.model_class()
        # Use the index of the algorithm with the best performance (lowest value) as the target
        self.classifier.fit(features, np.argmin(performance.values, axis=1))

    def _predict(self, features: pd.DataFrame) -> dict:
        """
        Predicts the best algorithm for each instance in the given feature data using simple multi-class classification.

        Args:
            features (pd.DataFrame): DataFrame containing the feature data.
                Each row corresponds to an instance, and each column corresponds to a feature.

        Returns:
            dict: A dictionary mapping instance names (index of the features DataFrame)
                  to a list containing a tuple of the predicted best algorithm and the budget.
                  Example: {instance_name: [(algorithm_name, budget)]}
        """
        predictions = self.classifier.predict(features)

        return {
            instance_name: [(self.algorithms[predictions[i]], self.budget)]
            for i, instance_name in enumerate(features.index)
        }
