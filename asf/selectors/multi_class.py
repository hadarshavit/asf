from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

try:
    from ConfigSpace import (
        Categorical,
        Configuration,
        ConfigurationSpace,
        EqualsCondition,
    )
    from ConfigSpace.hyperparameters import Hyperparameter

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.predictors import (
    AbstractPredictor,
    RandomForestClassifierWrapper,
    XGBoostClassifierWrapper,
)


class MultiClassClassifier(AbstractModelBasedSelector):
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
    def get_configuration_space(
        cs: ConfigurationSpace | None = None,
        cs_transform: dict[str, dict[str, type]] | None = None,
        model_class: list[type[AbstractPredictor]] = [
            RandomForestClassifierWrapper,
            XGBoostClassifierWrapper,
        ],
        pre_prefix: str = "",
        parent_param: Hyperparameter | None = None,
        parent_value: str | None = None,
        **kwargs,
    ) -> tuple[ConfigurationSpace, dict[str, dict[str, type]]]:
        """
        Get the configuration space for the predictor.

        Args:
            cs (Optional[ConfigurationSpace]): The configuration space to use. If None, a new one will be created.
            cs_transform (Optional[Dict[str, Dict[str, type]]]): A dictionary for transforming configuration space values.
            model_class (List[type]): The list of model classes to use. Defaults to [RandomForestRegressorWrapper, XGBoostRegressorWrapper].
            hierarchical_generator (Optional[List[AbstractFeatureGenerator]]): List of hierarchical feature generators.
            kwargs: Additional keyword arguments to pass to the model class.

        Returns:
            Tuple[ConfigurationSpace, Dict[str, Dict[str, type]]]: The configuration space and its transformation dictionary.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if cs is None:
            cs = ConfigurationSpace()

        if cs_transform is None:
            cs_transform = {}

        if pre_prefix != "":
            prefix = f"{pre_prefix}:{MultiClassClassifier.PREFIX}"
        else:
            prefix = MultiClassClassifier.PREFIX

        model_class_param = Categorical(
            name=f"{prefix}:model_class",
            items=[str(c.__name__) for c in model_class],
        )

        cs_transform[f"{prefix}:model_class"] = {
            str(c.__name__): c for c in model_class
        }

        params = [model_class_param]

        if parent_param is not None:
            conditions = [
                EqualsCondition(
                    child=param,
                    parent=parent_param,
                    value=parent_value,
                )
                for param in params
            ]
        else:
            conditions = []

        cs.add(params + conditions)

        for model in model_class:
            model.get_configuration_space(
                cs=cs,
                pre_prefix=f"{prefix}:model_class",
                parent_param=model_class_param,
                parent_value=str(model.__name__),
                **kwargs,
            )

        return cs, cs_transform

    @staticmethod
    def get_from_configuration(
        configuration: Configuration,
        cs_transform: dict[str, dict[str, type]],
        pre_prefix: str = "",
        **kwargs,
    ) -> partial:
        """
        Get the configuration space for the predictor.

        Args:
            configuration (Configuration): The configuration object.
            cs_transform (Dict[str, Dict[str, type]]): The transformation dictionary for the configuration space.

        Returns:
            partial: A partial function to initialize the MultiClassClassifier with the given configuration.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError(
                "ConfigSpace is not installed. Install optional extra with: pip install 'asf[configspace]'"
            )
        if pre_prefix != "":
            prefix = f"{pre_prefix}:{MultiClassClassifier.PREFIX}"
        else:
            prefix = MultiClassClassifier.PREFIX

        model_class = cs_transform[f"{prefix}:model_class"][
            configuration[f"{prefix}:model_class"]
        ]

        model = model_class.get_from_configuration(
            configuration, pre_prefix=f"{prefix}:model_class"
        )

        return MultiClassClassifier(
            model_class=model,
            hierarchical_generator=None,
            **kwargs,
        )
