from __future__ import annotations


import numpy as np
import pandas as pd

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ConfigurableMixin, ClassChoice

try:
    import ConfigSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.predictors import (
    AbstractPredictor,
    RandomForestClassifierWrapper,
    XGBoostClassifierWrapper,
)


class MultiClassClassifier(ConfigurableMixin, AbstractModelBasedSelector):
    """
    A selector that uses a multi-class classification model to predict the best algorithm
    for a given set of features and performance data.
    """

    PREFIX = "multi_class_classifier"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestClassifierWrapper,
        **kwargs,
    ):
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

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] = None,
        **kwargs,  # Accept additional kwargs from mixin
    ):
        """
        Define hyperparameters for MultiClassClassifier.

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
        ]
        return hyperparameters, [], []
