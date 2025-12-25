import inspect

import numpy as np
import pandas as pd

from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.configurable import ConfigurableMixin, ClassChoice

try:
    import ConfigSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False

from asf.predictors import (
    AbstractPredictor,
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper,
)
from asf.selectors.feature_generator import AbstractFeatureGenerator


class PerformanceModel(
    ConfigurableMixin, AbstractModelBasedSelector, AbstractFeatureGenerator
):
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

    PREFIX = "performance_model"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type = RandomForestRegressorWrapper,
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

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] = None,
        **kwargs,  # Accept additional kwargs from mixin
    ):
        """
        Define hyperparameters for PerformanceModel.

        Parameters
        ----------
        model_class : list[type[AbstractPredictor]], optional
            List of model classes to include in the configuration space.
            Defaults to [RandomForestRegressorWrapper, XGBoostRegressorWrapper].
        """
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        if model_class is None:
            model_class = [RandomForestRegressorWrapper, XGBoostRegressorWrapper]

        hyperparameters = [
            ClassChoice("model_class", choices=model_class),
        ]
        return hyperparameters, [], []
