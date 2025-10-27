import inspect

import numpy as np
import pandas as pd

from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
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

from functools import partial

from asf.predictors import (
    AbstractPredictor,
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper,
)
from asf.selectors.feature_generator import AbstractFeatureGenerator


class PerformanceModel(AbstractModelBasedSelector, AbstractFeatureGenerator):
    PREFIX = "performance_model "
    """
    PerformanceModel is a class that predicts the performance of algorithms
    based on given features. It can handle both single-target and multi-target
    regression models.

    Attributes:
        model_class (type): The class of the regression model to be used.
        use_multi_target (bool): Indicates whether to use multi-target regression.
        normalize (str): Method to normalize the performance data. Default is "log".
        regressors (list | object): List of trained regression models or a single model for multi-target regression.
        algorithm_features (pd.DataFrame | None): Features specific to each algorithm, if applicable.
        algorithms (list[str]): List of algorithm names.
        maximize (bool): Whether to maximize or minimize the performance metric.
        budget (float): Budget associated with the predictions.
    """

    def __init__(
        self,
        model_class: type,
        use_multi_target: bool = False,
        normalize: AbstractNormalization = LogNormalization(),
        **kwargs,
    ):
        """
        Initializes the PerformanceModel with the given parameters.

        Args:
            model_class (type): The class of the regression model to be used.
            use_multi_target (bool): Indicates whether to use multi-target regression.
            normalize (AbstractNormalization): Method to normalize the performance data. Default is LogNormalization.
            **kwargs: Additional arguments for the parent classes.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.regressors: list | object = []
        self.use_multi_target: bool = use_multi_target
        self.normalize: AbstractNormalization = normalize

    def _fit(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """
        Fits the regression models to the given features and performance data.

        Args:
            features (pd.DataFrame): DataFrame containing the feature data.
            performance (pd.DataFrame): DataFrame containing the performance data.
        """

        if self.normalize is not None:
            performance = self.normalize.fit_transform(performance)

        regressor_init_args = {}
        if "input_size" in inspect.signature(self.model_class).parameters.keys():
            regressor_init_args["input_size"] = features.shape[1]

        if self.use_multi_target:
            assert self.algorithm_features is None, (
                "PerformanceModel does not use algorithm features for multi-target regression."
            )
            self.regressors = self.model_class(**regressor_init_args)
            self.regressors.fit(features, performance)
        else:
            if self.algorithm_features is None:
                for i, algorithm in enumerate(self.algorithms):
                    algo_times = performance.iloc[:, i]

                    cur_model = self.model_class(**regressor_init_args)
                    cur_model.fit(features, algo_times)
                    self.regressors.append(cur_model)
            else:
                train_data = []
                for i, algorithm in enumerate(self.algorithms):
                    data = pd.merge(
                        features,
                        self.algorithm_features.loc[algorithm],
                        left_index=True,
                        right_index=True,
                    )
                    data = pd.merge(
                        data, performance.iloc[:, i], left_index=True, right_index=True
                    )
                    train_data.append(data)
                train_data = pd.concat(train_data)
                self.regressors = self.model_class(**regressor_init_args)
                self.regressors.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

    def _predict(self, features: pd.DataFrame) -> dict[str, list[tuple]]:
        """
        Predicts the performance of algorithms for the given features.

        Args:
            features (pd.DataFrame): DataFrame containing the feature data.

        Returns:
            dict[str, list[tuple]]: A dictionary mapping instance names to the predicted best algorithm
            and the associated budget.
        """
        predictions = self.generate_features(features)

        return {
            instance_name: [
                (
                    self.algorithms[
                        np.argmax(predictions[i])
                        if self.maximize
                        else np.argmin(predictions[i])
                    ],
                    self.budget,
                )
            ]
            for i, instance_name in enumerate(features.index)
        }

    def generate_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given features using the trained models.

        Args:
            features (pd.DataFrame): DataFrame containing the feature data.

        Returns:
            np.ndarray: Array containing the predictions for each algorithm.
        """
        if self.use_multi_target:
            predictions = self.regressors.predict(features)
        else:
            if self.algorithm_features is None:
                predictions = np.zeros((features.shape[0], len(self.algorithms)))
                for i, algorithm in enumerate(self.algorithms):
                    prediction = self.regressors[i].predict(features)
                    predictions[:, i] = prediction
            else:
                predictions = np.zeros((features.shape[0], len(self.algorithms)))
                for i, algorithm in enumerate(self.algorithms):
                    data = pd.merge(
                        features,
                        self.algorithm_features.loc[algorithm],
                        left_index=True,
                        right_index=True,
                    )
                    prediction = self.regressors.predict(data)
                    predictions[:, i] = prediction

        return predictions

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None,
            cs_transform: dict[str, dict[str, type]] | None = None,
            model_class: list[type[AbstractPredictor]] = [
                RandomForestRegressorWrapper,
                XGBoostRegressorWrapper,
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
            if cs is None:
                cs = ConfigurationSpace()

            if cs_transform is None:
                cs_transform = {}

            if pre_prefix != "":
                prefix = f"{pre_prefix}:{PerformanceModel.PREFIX}"
            else:
                prefix = PerformanceModel.PREFIX

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
                partial: A partial function to initialize the PerformanceModel with the given configuration.
            """
            if pre_prefix != "":
                prefix = f"{pre_prefix}:{PerformanceModel.PREFIX}"
            else:
                prefix = PerformanceModel.PREFIX

            model_class = cs_transform[f"{prefix}:model_class"][
                configuration[f"{prefix}:model_class"]
            ]

            model = model_class.get_from_configuration(
                configuration, pre_prefix=f"{prefix}:model_class"
            )

            return PerformanceModel(
                model_class=model,
                hierarchical_generator=None,
                **kwargs,
            )
