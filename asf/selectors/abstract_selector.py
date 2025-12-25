from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from asf.selectors.feature_generator import AbstractFeatureGenerator

try:
    from ConfigSpace import Configuration, ConfigurationSpace
    from ConfigSpace.hyperparameters import CategoricalHyperparameter as Categorical

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractSelector(ABC):
    """
    Abstract base class for algorithm selectors.

    Provides a framework for fitting, predicting, and managing hierarchical feature
    generators and configuration spaces.

    Attributes
    ----------
    maximize : bool
        Indicates whether the objective is to maximize or minimize the performance metric.
    budget : float or None
        The budget for the selector, if applicable.
    feature_groups : list[str] or None
        Groups of features to be considered during selection.
    hierarchical_generator : AbstractFeatureGenerator or None
        A generator for hierarchical features, if applicable.
    algorithm_features : pd.DataFrame or None
        Additional features related to algorithms, if provided.
    prediction_mode : str
        Mode for predictions ('aslib', 'pandas', 'numpy').
    algorithms : list[str]
        List of algorithm names seen during fitting.
    features : list[str]
        List of feature names seen during fitting.
    """

    def __init__(
        self,
        budget: float | None = None,
        maximize: bool = False,
        feature_groups: list[str] | None = None,
        hierarchical_generator: AbstractFeatureGenerator | None = None,
        prediction_mode: str = "aslib",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the AbstractSelector.

        Parameters
        ----------
        budget : float or None, default=None
            The budget for the selector, if applicable.
        maximize : bool, default=False
            Indicates whether to maximize the performance metric.
        feature_groups : list[str] or None, default=None
            Groups of features to be considered during selection.
        hierarchical_generator : AbstractFeatureGenerator or None, default=None
            A generator for hierarchical features, if applicable.
        prediction_mode : str, default="aslib"
            Mode for predictions ('aslib', 'pandas', 'numpy').
        **kwargs : Any
            Additional keyword arguments.
        """
        self.maximize = bool(maximize)
        self.budget = float(budget) if budget is not None else None
        self.feature_groups = feature_groups
        self.hierarchical_generator = hierarchical_generator
        self.algorithm_features: pd.DataFrame | None = None
        self.prediction_mode = str(prediction_mode)
        self.algorithms: list[str] = []
        self.features: list[str] = []

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        performance: pd.DataFrame | np.ndarray,
        algorithm_features: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the selector.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            The input features.
        performance : pd.DataFrame or np.ndarray
            The algorithm performance data.
        algorithm_features : pd.DataFrame or None, optional
            Additional features related to algorithms.
        **kwargs : Any
            Additional keyword arguments for fitting.
        """
        if (
            isinstance(features, pd.DataFrame) and isinstance(performance, np.ndarray)
        ) or (
            isinstance(features, np.ndarray) and isinstance(performance, pd.DataFrame)
        ):
            raise ValueError(
                "Mixed input types (DataFrame and numpy array) are not allowed."
            )
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(
                features,
                columns=pd.Index([f"f_{i}" for i in range(features.shape[1])]),
            )
        if isinstance(performance, np.ndarray):
            performance = pd.DataFrame(
                performance,
                columns=pd.Index([f"algo_{i}" for i in range(performance.shape[1])]),
            )

        if not isinstance(features, pd.DataFrame) or not isinstance(
            performance, pd.DataFrame
        ):
            raise ValueError(
                "Features and performance must be pandas DataFrames or numpy arrays."
            )

        if self.hierarchical_generator is not None:
            self.hierarchical_generator.fit(features, performance, algorithm_features)
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )

        self.algorithms = [str(a) for a in performance.columns]
        self.features = [str(f) for f in features.columns]
        self.algorithm_features = algorithm_features

        self._fit(features, performance, **kwargs)

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]] | pd.Series | np.ndarray:
        """
        Predict algorithm selections/rankings.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray or None
            The input features for prediction.
        performance : pd.DataFrame or None, default=None
            Partial performance data if available (e.g., for oracle selectors).

        Returns
        -------
        dict or pd.Series or np.ndarray
            Predicted selections in the specified prediction_mode.
        """
        if features is None:
            df_features: pd.DataFrame | None = None
        elif isinstance(features, np.ndarray):
            cols = (
                self.features
                if self.features
                else [f"f_{i}" for i in range(features.shape[1])]
            )
            df_features = pd.DataFrame(features, columns=pd.Index(cols))
        elif isinstance(features, pd.DataFrame):
            df_features = features
        else:
            raise ValueError(
                "Features must be a numpy array, pandas DataFrame, or None."
            )

        if self.hierarchical_generator is not None and df_features is not None:
            df_features = pd.concat(
                [
                    df_features,
                    self.hierarchical_generator.generate_features(df_features),
                ],
                axis=1,
            )

        # Call the internal _predict
        scheds = self._predict(df_features, performance=performance)

        if self.prediction_mode == "aslib":
            if self.feature_groups is None:
                return scheds

            fg_steps = list(self.feature_groups)
            return {
                str(instance): fg_steps + list(scheds.get(str(instance), []))
                for instance in (
                    df_features.index if df_features is not None else scheds.keys()
                )
            }
        elif self.prediction_mode == "pandas":
            if df_features is None:
                raise ValueError("Pandas mode requires features.")
            return pd.Series(
                {
                    instance: scheds.get(str(instance), [(None, 0.0)])[0][0]
                    for instance in df_features.index
                }
            )
        elif self.prediction_mode == "numpy":
            if df_features is None:
                raise ValueError("Numpy mode requires features.")
            labels = [
                scheds.get(str(instance), [(None, 0.0)])[0][0]
                for instance in df_features.index
            ]
            encoder = OneHotEncoder(sparse_output=False, categories=[self.algorithms])
            return encoder.fit_transform(np.array(labels).reshape(-1, 1))
        else:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")

    def save(self, path: str) -> None:
        """
        Save the selector instance.

        Parameters
        ----------
        path : str
            File path to save to.
        """
        pass

    @classmethod
    def load(cls, path: str) -> "AbstractSelector":
        """
        Load a selector instance.

        Parameters
        ----------
        path : str
            File path to load from.

        Returns
        -------
        AbstractSelector
            The loaded selector instance.
        """
        raise NotImplementedError(f"{cls.__name__} does not support loading from file.")

    @staticmethod
    def get_configuration_space(
        cs: ConfigurationSpace | None = None, **kwargs: Any
    ) -> ConfigurationSpace:
        """
        Get the configuration space.

        Parameters
        ----------
        cs : ConfigurationSpace or None, optional
            Base configuration space.
        **kwargs : Any
            Additional options.

        Returns
        -------
        ConfigurationSpace
            The configuration space.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError("ConfigSpace is not available.")
        raise NotImplementedError("Subclasses must implement get_configuration_space.")

    @staticmethod
    def get_from_configuration(configuration: Configuration) -> AbstractSelector:
        """
        Create an instance from a configuration.

        Parameters
        ----------
        configuration : Configuration
            The configuration object.

        Returns
        -------
        AbstractSelector
            The initialized selector.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError("ConfigSpace is not available.")
        raise NotImplementedError("Subclasses must implement get_from_configuration.")

    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """
        Internal fit implementation.
        """
        raise NotImplementedError("Subclasses must implement _fit.")

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Internal predict implementation.
        """
        raise NotImplementedError("Subclasses must implement _predict.")

    @staticmethod
    def _add_hierarchical_generator_space(
        cs: ConfigurationSpace,
        hierarchical_generator: list[AbstractFeatureGenerator] | None = None,
        **kwargs: Any,
    ) -> ConfigurationSpace:
        """
        Add hierarchical generator options to the configuration space.
        """
        if not CONFIGSPACE_AVAILABLE:
            raise RuntimeError("ConfigSpace is not available.")
        if hierarchical_generator:
            if "hierarchical_generator" not in cs:
                cs.add(
                    Categorical(
                        name="hierarchical_generator", choices=hierarchical_generator
                    )
                )
            for g in hierarchical_generator:
                g.get_configuration_space(cs=cs, **kwargs)
        return cs
