import pandas as pd
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector

try:
    from ConfigSpace.hyperparameters import Hyperparameter

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

try:
    from ConfigSpace import (
        ConfigurationSpace,
        Categorical,
        Configuration,
        EqualsCondition,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
from functools import partial
from typing import Optional, List, Dict, Tuple


class PairwiseRegressor(AbstractModelBasedSelector, AbstractFeatureGenerator):
    PREFIX = "pairwise_regressor"
    """
    PairwiseRegressor is a selector that uses pairwise regression of algorithms
    to predict the best algorithm for a given instance.

    Attributes:
        model_class (type): The regression model class to be used for pairwise comparisons.
        regressors (List[AbstractPredictor]): List of trained regressors for pairwise comparisons.
    """

    def __init__(self, model_class: type, **kwargs):
        """
        Initializes the PairwiseRegressor with a given model class and hierarchical feature generator.

        Args:
            model_class (type): The regression model class to be used for pairwise comparisons.
            kwargs: Additional keyword arguments for the parent classes.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.regressors: List[AbstractPredictor] = []

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

    def _predict(self, features: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predicts the best algorithm for each instance using the trained pairwise regressors.

        Args:
            features (pd.DataFrame): The feature data for the instances.

        Returns:
            Dict[str, List[Tuple[str, float]]]: A dictionary mapping instance names to the predicted best algorithm
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

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: Optional[ConfigurationSpace] = None,
            cs_transform: Optional[Dict[str, Dict[str, type]]] = None,
            model_class: List[type[AbstractPredictor]] = [
                RandomForestRegressorWrapper,
                XGBoostRegressorWrapper,
            ],
            pre_prefix: str = "",
            parent_param: Optional[Hyperparameter] = None,
            parent_value: Optional[str] = None,
            **kwargs,
        ) -> Tuple[ConfigurationSpace, Dict[str, Dict[str, type]]]:
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
            if cs is None:
                cs = ConfigurationSpace()

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{PairwiseRegressor.PREFIX}"
            else:
                prefix = PairwiseRegressor.PREFIX

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
            cs_transform: Dict[str, Dict[str, type]],
            pre_prefix: str = "",
            **kwargs,
        ) -> partial:
            """
            Get the configuration space for the predictor.

            Args:
                configuration (Configuration): The configuration object.
                cs_transform (Dict[str, Dict[str, type]]): The transformation dictionary for the configuration space.

            Returns:
                partial: A partial function to initialize the PairwiseRegressor with the given configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{PairwiseRegressor.PREFIX}"
            else:
                prefix = PairwiseRegressor.PREFIX

            model_class = cs_transform[f"{prefix}:model_class"][
                configuration[f"{prefix}:model_class"]
            ]

            model = model_class.get_from_configuration(
                configuration, pre_prefix=f"{prefix}:model_class"
            )

            return PairwiseRegressor(
                model_class=model,
                hierarchical_generator=None,
                **kwargs,
            )
