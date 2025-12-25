from __future__ import annotations

import inspect
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from asf.predictors import (
    AbstractPredictor,
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper,
)
from asf.preprocessing.performance_scaling import (
    AbstractNormalization,
    LogNormalization,
)
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.selectors.feature_generator import AbstractFeatureGenerator
from asf.utils.configurable import ClassChoice, ConfigurableMixin

try:
    from ConfigSpace import ConfigurationSpace  # noqa: F401

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class PerformanceModel(
    ConfigurableMixin, AbstractModelBasedSelector, AbstractFeatureGenerator
):
    """
    PerformanceModel predicts algorithm performance based on instance features.

    It can handle both single-target (one model per algorithm) and multi-target
    regression models.

    Attributes
    ----------
    model_class : type
        The class of the regression model to be used.
    use_multi_target : bool
        Whether to use multi-target regression.
    normalize : AbstractNormalization
        Method to normalize the performance data.
    regressors : list or object
        Trained regression models.
    """

    PREFIX = "performance_model"
    RETURN_TYPE = "single"

    def __init__(
        self,
        model_class: type[AbstractPredictor] = RandomForestRegressorWrapper,
        use_multi_target: bool = False,
        normalize: AbstractNormalization | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PerformanceModel.

        Parameters
        ----------
        model_class : type[AbstractPredictor], default=RandomForestRegressorWrapper
            The class of the regression model to be used.
        use_multi_target : bool, default=False
            Indicates whether to use multi-target regression.
        normalize : AbstractNormalization or None, default=None
            Method to normalize performance data. If None, defaults to LogNormalization().
        **kwargs : Any
            Additional arguments for the parent classes.
        """
        AbstractModelBasedSelector.__init__(self, model_class, **kwargs)
        AbstractFeatureGenerator.__init__(self)
        self.regressors: list[AbstractPredictor] | AbstractPredictor | None = None
        self.use_multi_target = bool(use_multi_target)
        self.normalize = normalize if normalize is not None else LogNormalization()

    def _fit(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """
        Fit the regression models.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The performance data.
        """
        if self.normalize is not None:
            performance = self.normalize.fit_transform(performance)

        regressor_init_args: dict[str, Any] = {}
        # Safely check for input_size if it's a type (standard wrapper classes usually have it)
        try:
            sig = inspect.signature(self.model_class)
            if "input_size" in sig.parameters:
                regressor_init_args["input_size"] = features.shape[1]
        except (ValueError, TypeError):
            pass

        if self.use_multi_target:
            if self.algorithm_features is not None:
                raise ValueError(
                    "PerformanceModel does not use algorithm features for multi-target regression."
                )
            self.regressors = self.model_class(**regressor_init_args)
            self.regressors.fit(features, performance)
        else:
            if self.algorithm_features is None:
                self.regressors = []
                for i, _ in enumerate(self.algorithms):
                    algo_times = performance.iloc[:, i]
                    cur_model = self.model_class(**regressor_init_args)
                    cur_model.fit(features, algo_times)
                    self.regressors.append(cur_model)
            else:
                train_data_list = []
                for i, algorithm in enumerate(self.algorithms):
                    # Align algorithm features with instance features
                    data = pd.merge(
                        features,
                        self.algorithm_features.loc[[algorithm]]
                        .reindex([algorithm] * len(features))
                        .set_index(features.index),
                        left_index=True,
                        right_index=True,
                    )
                    data = pd.merge(
                        data,
                        performance.iloc[:, [i]],
                        left_index=True,
                        right_index=True,
                    )
                    train_data_list.append(data)
                train_data = pd.concat(train_data_list)
                self.regressors = self.model_class(**regressor_init_args)
                self.regressors.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

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
            Mapping from instance name to algorithm schedules.
        """
        if features is None:
            raise ValueError("PerformanceModel require features for prediction.")
        predictions = self.generate_features(features)

        results: dict[str, list[tuple[str, float]]] = {}
        for i, instance_name in enumerate(features.index):
            idx = (
                int(np.argmax(predictions.iloc[i]))
                if self.maximize
                else int(np.argmin(predictions.iloc[i]))
            )
            results[str(instance_name)] = [
                (str(self.algorithms[idx]), float(self.budget or 0))
            ]
        return results

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for each algorithm.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.

        Returns
        -------
        np.ndarray
            Predicted performance for each algorithm (n_instances x n_algorithms).
        """
        if self.regressors is None:
            raise RuntimeError("Model has not been fitted.")

        predictions = np.zeros((base_features.shape[0], len(self.algorithms)))

        if self.use_multi_target:
            if not isinstance(self.regressors, AbstractPredictor):
                raise RuntimeError("Multi-target regressor missing.")
            predictions = self.regressors.predict(base_features)
            if isinstance(predictions, pd.DataFrame):
                predictions = predictions.values
        else:
            if self.algorithm_features is None:
                if not isinstance(self.regressors, list):
                    raise RuntimeError("Individual regressors missing.")
                for i, _ in enumerate(self.algorithms):
                    predictions[:, i] = np.asarray(
                        self.regressors[i].predict(base_features)
                    ).flatten()
            else:
                if not isinstance(self.regressors, AbstractPredictor):
                    raise RuntimeError("Joint regressor missing.")
                for i, algorithm in enumerate(self.algorithms):
                    data = pd.merge(
                        base_features,
                        self.algorithm_features.loc[[algorithm]]
                        .reindex([algorithm] * len(base_features))
                        .set_index(base_features.index),
                        left_index=True,
                        right_index=True,
                    )
                    predictions[:, i] = self.regressors.predict(data)

        return pd.DataFrame(
            predictions,
            index=base_features.index,
            columns=pd.Index(list(self.algorithms)),
        )

    @staticmethod
    def _define_hyperparameters(
        model_class: list[type[AbstractPredictor]] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Define hyperparameters for PerformanceModel.

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
            model_class = [RandomForestRegressorWrapper, XGBoostRegressorWrapper]

        hyperparameters = [
            ClassChoice("model_class", choices=model_class, default=model_class[0]),
        ]
        return hyperparameters, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls,
        clean_config: dict[str, Any],
        **kwargs: Any,
    ) -> partial[PerformanceModel]:
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
            Partial function for PerformanceModel.
        """
        config = clean_config.copy()
        config.update(kwargs)
        return partial(PerformanceModel, **config)
