from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from asf.utils.configurable import ConfigurableMixin
from asf.utils.g_means import GMeans

try:
    from ConfigSpace import (
        Categorical,
        Float,
        Integer,
    )

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False


class GMeansWrapper(ConfigurableMixin):
    """
    Wrapper for GMeans clustering.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments passed to GMeans.
    """

    PREFIX: str = "gmeans"

    def __init__(self, **kwargs: Any) -> None:
        self.model = GMeans(**kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray) -> GMeansWrapper:
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns
        -------
        GMeansWrapper
            The fitted wrapper.
        """
        self.model.fit(X.values if isinstance(X, pd.DataFrame) else X)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        return self.model.predict(X.values if isinstance(X, pd.DataFrame) else X)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for GMeans."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Float("min_samples", (0.0001, 0.1), default=0.001, log=True),
            Categorical("significance", [0.15, 0.1, 0.05, 0.025, 0.001], default=0.05),
            Integer("n_init", (1, 10), default=5),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls, clean_config: dict[str, Any], **kwargs: Any
    ) -> partial:
        """Create a partial class wrapper."""
        config = clean_config.copy()
        config.update(kwargs)
        return partial(GMeansWrapper, **config)


class KMeansWrapper(ConfigurableMixin):
    """
    Wrapper for KMeans clustering.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments passed to KMeans.
    """

    PREFIX: str = "kmeans"

    def __init__(self, **kwargs: Any) -> None:
        self.model = KMeans(**kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray) -> KMeansWrapper:
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns
        -------
        KMeansWrapper
            The fitted wrapper.
        """
        self.model.fit(X.values if isinstance(X, pd.DataFrame) else X)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        return self.model.predict(X.values if isinstance(X, pd.DataFrame) else X)

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for KMeans."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Integer("n_clusters", (2, 20), default=5),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls, clean_config: dict[str, Any], **kwargs: Any
    ) -> partial:
        """Create a partial class wrapper."""
        config = clean_config.copy()
        config.update(kwargs)
        return partial(KMeansWrapper, **config)


class AgglomerativeClusteringWrapper(ConfigurableMixin):
    """
    Wrapper for AgglomerativeClustering.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments passed to AgglomerativeClustering.
    """

    PREFIX: str = "agglomerative_clustering"

    def __init__(self, **kwargs: Any) -> None:
        self.model = AgglomerativeClustering(**kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray) -> AgglomerativeClusteringWrapper:
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns
        -------
        AgglomerativeClusteringWrapper
            The fitted wrapper.
        """
        self.model.fit(X.values if isinstance(X, pd.DataFrame) else X)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
                Predict labels (not supported by default).

                Parameters
                ----------
                X : pd.DataFrame or np.ndarray
                    The input data.

                Returns
        -------
                np.ndarray
                    The predicted labels.

                Raises
                ------
                NotImplementedError
                    If predict is not supported.
        """
        if hasattr(self.model, "predict"):
            return getattr(self.model, "predict")(X)
        raise NotImplementedError("AgglomerativeClustering does not support predict()")

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for AgglomerativeClustering."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Integer("n_clusters", (2, 20), default=5),
            Categorical(
                "linkage", ["ward", "complete", "average", "single"], default="ward"
            ),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls, clean_config: dict[str, Any], **kwargs: Any
    ) -> partial:
        """Create a partial class wrapper."""
        config = clean_config.copy()
        config.update(kwargs)
        return partial(AgglomerativeClusteringWrapper, **config)


class DBSCANWrapper(ConfigurableMixin):
    """
    Wrapper for DBSCAN clustering.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments passed to DBSCAN.
    """

    PREFIX: str = "dbscan"

    def __init__(self, **kwargs: Any) -> None:
        self.model = DBSCAN(**kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray) -> DBSCANWrapper:
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns
        -------
        DBSCANWrapper
            The fitted wrapper.
        """
        self.model.fit(X.values if isinstance(X, pd.DataFrame) else X)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
                Predict labels (not supported by default).

                Parameters
                ----------
                X : pd.DataFrame or np.ndarray
                    The input data.

                Returns
        -------
                np.ndarray
                    The predicted labels.

                Raises
                ------
                NotImplementedError
                    If predict is not supported.
        """
        if hasattr(self.model, "predict"):
            return getattr(self.model, "predict")(X)
        raise NotImplementedError("DBSCAN does not support predict()")

    @staticmethod
    def _define_hyperparameters(
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Define hyperparameters for DBSCAN."""
        if not CONFIGSPACE_AVAILABLE:
            return [], [], []

        params = [
            Float("eps", (0.1, 2.0), default=0.5),
            Integer("min_samples", (2, 10), default=5),
        ]
        return params, [], []

    @classmethod
    def _get_from_clean_configuration(
        cls, clean_config: dict[str, Any], **kwargs: Any
    ) -> partial:
        """Create a partial class wrapper."""
        config = clean_config.copy()
        config.update(kwargs)
        return partial(DBSCANWrapper, **config)
