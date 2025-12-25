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
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import ConfigurationSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class MultiClassClassifier(ConfigurableMixin, AbstractModelBasedSelector):
    """
    Multi-class classification algorithm selector.

    Attributes
    ----------
    classifier : AbstractPredictor or None
        The trained classification model.
    """

    PREFIX = "multi_class_classifier"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestClassifierWrapper,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MultiClassClassifier.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=RandomForestClassifierWrapper
            The class of the model to be used for classification.
        **kwargs : Any
            Additional keyword arguments.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        self.classifier: AbstractPredictor | None = None

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the classification model.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        """
        if self.algorithm_features is not None:
            raise ValueError("MultiClassClassifier does not use algorithm features.")

        self.classifier = self.model_class()
        if self.classifier is None:
            raise RuntimeError("Classifier could not be initialized.")

        # Best algorithm (lowest value) per instance
        target = np.argmin(performance.values, axis=1)
        self.classifier.fit(features, target)

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
            The input features.

        Returns
        -------
        dict
            Mapping from instance names to algorithm schedules.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier has not been fitted.")

        if features is None:
            raise ValueError("MultiClassClassifier require features for prediction.")
        predictions = self.classifier.predict(features)

        results: dict[str, list[tuple[str, float]]] = {}
        for i, instance_name in enumerate(features.index):
            idx = int(predictions[i])
            results[str(instance_name)] = [
                (str(self.algorithms[idx]), float(self.budget or 0))
            ]
        return results

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for MultiClassClassifier.

        Parameters
        ----------
        model_class : list[type[AbstractPredictor]] or None, default=None
            List of model classes to include in the configuration space.
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
            ClassChoice("model_class", choices=model_class, default=model_class[0]),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[MultiClassClassifier]:
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
            Partial function for MultiClassClassifier.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(MultiClassClassifier, **config)
