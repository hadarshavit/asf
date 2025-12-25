from __future__ import annotations

from typing import Any

import pandas as pd


class AbstractFeatureGenerator:
    """
    Abstract base class for generating additional features.

    Subclasses should implement the methods to define specific feature
    generation logic based on a set of base features.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the AbstractFeatureGenerator.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.
        """
        pass

    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the generator to the data.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        performance : pd.DataFrame
            The algorithm performance data.
        algorithm_features : pd.DataFrame or None, optional
            Additional features related to algorithms.
        **kwargs : Any
            Additional keyword arguments.
        """
        pass

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """
                Generate additional features based on the provided base features.

                Parameters
                ----------
                base_features : pd.DataFrame
                    The input DataFrame containing the base features.

                Returns
        -------
                pd.DataFrame
                    A DataFrame containing the generated features.
        """
        raise NotImplementedError("Subclasses must implement generate_features.")

    @staticmethod
    def get_configuration_space(**kwargs: Any) -> Any:
        """
        Get the configuration space.
        """
        return None


class DummyFeatureGenerator(AbstractFeatureGenerator):
    """
    Feature generator that does nothing.
    """

    def generate_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=base_features.index)
