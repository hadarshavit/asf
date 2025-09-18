import numpy as np
import pandas as pd

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Configuration,
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
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
from functools import partial
from typing import Optional, List, Dict, Tuple, Union


class PairwiseClassifier(AbstractModelBasedSelector, AbstractFeatureGenerator):
    PREFIX = "pairwise_classifier"
    """
    PairwiseClassifier is a selector that uses pairwise comparison of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        PREFIX (str): Prefix used for configuration space parameters.
        classifiers (List[AbstractPredictor]): List of trained classifiers for pairwise comparisons.
        use_weights (bool): Whether to use weights based on performance differences.
    """

    def __init__(
        self, model_class: type[AbstractPredictor], use_weights: bool = True, **kwargs
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
        self.classifiers: List[AbstractPredictor] = []
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
    ) -> Dict[str, List[Tuple[str, Union[int, float]]]]:
        """
        Predicts the best algorithm for each instance using the trained pairwise classifiers.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            Dict[str, List[Tuple[str, Union[int, float]]]]: A dictionary mapping instance names to the predicted best algorithm and budget.
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
        cnt = 0
        predictions_sum = pd.DataFrame(0, index=features.index, columns=self.algorithms)
        for i, algorithm in enumerate(self.algorithms):
            for j, other_algorithm in enumerate(self.algorithms[i + 1 :]):
                prediction = self.classifiers[cnt].predict(features)
                # Ensure prediction is a boolean Series with the correct index
                if not isinstance(prediction, pd.Series):
                    prediction = pd.Series(prediction, index=features.index)
                if prediction.dtype != bool:
                    prediction = prediction.astype(bool)
                predictions_sum.loc[prediction, algorithm] += 1
                predictions_sum.loc[~prediction, other_algorithm] += 1
                cnt += 1

        return predictions_sum

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            cs_transform: Optional[Dict[str, dict]] = None,
            model_class: List[type[AbstractPredictor]] = [
                RandomForestClassifierWrapper,
                XGBoostClassifierWrapper,
            ],
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
            **kwargs,
        ) -> Tuple[ConfigurationSpace, Dict[str, dict]]:
            """
            Get the configuration space for the predictor.

            Args:
                cs (Optional[ConfigurationSpace]): The configuration space to use. If None, a new one will be created.
                cs_transform (Optional[Dict[str, dict]]): A dictionary for transforming configuration space parameters.
                model_class (List[type[AbstractPredictor]]): The list of model classes to use. Defaults to [RandomForestClassifierWrapper, XGBoostClassifierWrapper].
                hierarchical_generator (Optional[List[AbstractFeatureGenerator]]): List of hierarchical feature generators.
                **kwargs: Additional keyword arguments to pass to the model class.

            Returns:
                Tuple[ConfigurationSpace, Dict[str, dict]]: The configuration space and its transformation dictionary.
            """
            if cs is None:
                cs = ConfigurationSpace()

            if cs_transform is None:
                cs_transform = dict()

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{PairwiseClassifier.PREFIX}"
            else:
                prefix = PairwiseClassifier.PREFIX

            model_class_param = Categorical(
                name=f"{prefix}:model_class",
                items=[str(c.__name__) for c in model_class],
            )

            cs_transform[f"{prefix}:model_class"] = {
                str(c.__name__): c for c in model_class
            }

            use_weights_param = Categorical(
                name=f"{prefix}:use_weights",
                items=[True, False],
            )

            params = [model_class_param, use_weights_param]

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
            cs_transform: Dict[str, dict],
            pre_prefix: str = "",
            **kwargs,
        ) -> partial:
            """
            Get the predictor from a given configuration.

            Args:
                configuration (Configuration): The configuration object.
                cs_transform (Dict[str, dict]): The transformation dictionary for the configuration space.

            Returns:
                partial: A partial function to initialize the PairwiseClassifier with the given configuration.
            """

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{PairwiseClassifier.PREFIX}"
            else:
                prefix = PairwiseClassifier.PREFIX

            model_class = cs_transform[f"{prefix}:model_class"][
                configuration[f"{prefix}:model_class"]
            ]
            use_weights = configuration[f"{prefix}:use_weights"]

            model = model_class.get_from_configuration(
                configuration, pre_prefix=f"{prefix}:model_class"
            )

            return PairwiseClassifier(
                model_class=model,
                use_weights=use_weights,
                hierarchical_generator=None,
                **kwargs,
            )
