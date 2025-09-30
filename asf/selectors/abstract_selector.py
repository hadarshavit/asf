import pandas as pd
from asf.selectors.feature_generator import (
    AbstractFeatureGenerator,
)
import numpy as np

from typing import Optional

try:
    from ConfigSpace import ConfigurationSpace, Categorical, Configuration

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class AbstractSelector:
    """
    AbstractSelector is a base class for implementing feature selection algorithms.
    It provides a framework for fitting, predicting, and managing hierarchical feature
    generators and configuration spaces.

    Attributes
    ----------
    maximize : bool
        Indicates whether the objective is to maximize or minimize the performance metric.
    budget : int or None
        The budget for the selector, if applicable.
    feature_groups : list[str] or None
        Groups of features to be considered during selection.
    hierarchical_generator : AbstractFeatureGenerator or None
        A generator for hierarchical features, if applicable.
    algorithm_features : pd.DataFrame or None
        Additional features related to algorithms, if provided.
    """

    def __init__(
        self,
        budget: int | None = None,
        maximize: bool = False,
        feature_groups: list[str] | None = None,
        hierarchical_generator: AbstractFeatureGenerator | None = None,
    ):
        """
        Initialize the AbstractSelector.

        Parameters
        ----------
        budget : int or None, optional
            The budget for the selector, if applicable. Defaults to None.
        maximize : bool, optional
            Indicates whether to maximize the performance metric. Defaults to False.
        feature_groups : list[str] or None, optional
            Groups of features to be considered during selection. Defaults to None.
        hierarchical_generator : AbstractFeatureGenerator or None, optional
            A generator for hierarchical features, if applicable. Defaults to None.
        """
        self.maximize = maximize
        self.budget = budget
        self.feature_groups = feature_groups
        self.hierarchical_generator = hierarchical_generator
        self.algorithm_features: pd.DataFrame | None = None

    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """
        Fit the selector to the given features and performance data.

        Parameters
        ----------
        features : pd.DataFrame
            The input features for the selector.
        performance : pd.DataFrame
            The performance data corresponding to the features.
        algorithm_features : pd.DataFrame or None, optional
            Additional features related to algorithms, if provided. Defaults to None.
        **kwargs : dict
            Additional keyword arguments for fitting.
        """
        if isinstance(features, np.ndarray) and isinstance(performance, np.ndarray):
            features = pd.DataFrame(
                features,
                index=range(len(features)),
                columns=[f"f_{i}" for i in range(features.shape[1])],
            )
            performance = pd.DataFrame(
                performance,
                index=range(len(performance)),
                columns=[f"algo_{i}" for i in range(performance.shape[1])],
            )
        elif isinstance(features, pd.DataFrame) and isinstance(
            performance, pd.DataFrame
        ):
            pass
        else:
            raise ValueError(
                "features and performance must be either numpy arrays or pandas DataFrames"
            )

        if self.hierarchical_generator is not None:
            self.hierarchical_generator.fit(features, performance, algorithm_features)
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )
        self.algorithms: list[str] = performance.columns.to_list()
        self.features: list[str] = features.columns.to_list()
        self.algorithm_features = algorithm_features
        self._fit(features, performance, **kwargs)

    def predict(
        self, features: pd.DataFrame, performance: Optional[pd.DataFrame] = None
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Predict the ranking or selection of features for the given input features.

        Parameters
        ----------
        features : pd.DataFrame
            The input features for prediction.
        performance : pd.DataFrame or None, optional
            The (partial) performance data corresponding to the features, if applicable. Defaults to None.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            A dictionary where keys are algorithm names and values are lists of tuples
            containing feature names and their corresponding scores.
        """
        if self.hierarchical_generator is not None:
            features = pd.concat(
                [features, self.hierarchical_generator.generate_features(features)],
                axis=1,
            )
        if performance is None:
            return self._predict(features)
        else:
            return self._predict(features, performance)

    def save(self, path: str) -> None:
        """
        Save the selector's state to the specified path.

        Parameters
        ----------
        path : str
            The file path where the selector's state will be saved.
        """
        pass

    def load(self, path: str) -> None:
        """
        Load the selector's state from the specified path.

        Parameters
        ----------
        path : str
            The file path from which the selector's state will be loaded.
        """
        pass

    if CONFIGSPACE_AVAILABLE:

        @staticmethod
        def get_configuration_space(
            cs: ConfigurationSpace | None = None, **kwargs
        ) -> ConfigurationSpace:
            """
            Get the configuration space for the selector.

            Parameters
            ----------
            cs : ConfigurationSpace or None, optional
                The configuration space to use. If None, a new one will be created.
            **kwargs : dict
                Additional keyword arguments for configuration space creation.

            Returns
            -------
            ConfigurationSpace
                The configuration space for the selector.

            Raises
            ------
            NotImplementedError
                If the method is not implemented in a subclass.
            """
            raise NotImplementedError(
                "get_configuration_space() is not implemented for this selector"
            )

        @staticmethod
        def get_from_configuration(configuration: Configuration) -> "AbstractSelector":
            """
            Create a selector instance from a configuration.

            Parameters
            ----------
            configuration : Configuration
                The configuration object.

            Returns
            -------
            AbstractSelector
                The selector instance.

            Raises
            ------
            NotImplementedError
                If the method is not implemented in a subclass.
            """
            raise NotImplementedError(
                "get_from_configuration() is not implemented for this selector"
            )

        @staticmethod
        def _add_hierarchical_generator_space(
            cs: ConfigurationSpace,
            hierarchical_generator: list[AbstractFeatureGenerator] | None = None,
            **kwargs,
        ) -> ConfigurationSpace:
            """
            Add the hierarchical generator space to the configuration space.

            Parameters
            ----------
            cs : ConfigurationSpace
                The configuration space to use.
            hierarchical_generator : list[AbstractFeatureGenerator] or None, optional
                The list of hierarchical generators to add. Defaults to None.
            **kwargs : dict
                Additional keyword arguments to pass to the model class.

            Returns
            -------
            ConfigurationSpace
                The updated configuration space.
            """
            if hierarchical_generator is not None:
                if "hierarchical_generator" in cs:
                    return

                cs.add(
                    Categorical(
                        name="hierarchical_generator",
                        items=hierarchical_generator,
                    )
                )

                for generator in hierarchical_generator:
                    generator.get_configuration_space(cs=cs, **kwargs)

            return cs
