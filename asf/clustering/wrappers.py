from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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


class AbstractClustering(ABC, ConfigurableMixin):
    """
    Abstract base class for all clustering wrappers.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray) -> AbstractClustering:
        """Fit the clustering model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        pass

    def _ensure_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Ensure input is a numpy array."""
        return X.values if isinstance(X, pd.DataFrame) else X


class GMeansWrapper(AbstractClustering):
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
        self.model.fit(self._ensure_array(X))
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
        return partial(GMeansWrapper, **clean_config)


class KMeansWrapper(AbstractClustering):
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
        self.model.fit(self._ensure_array(X))
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
        return self.model.predict(self._ensure_array(X))

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
        return partial(KMeansWrapper, **clean_config)


class AgglomerativeClusteringWrapper(AbstractClustering):
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
        self._X_fit: np.ndarray | None = None

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
        self._X_fit = self._ensure_array(X)
        self.model.fit(self._X_fit)
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

        if self._X_fit is None:
            raise ValueError("Model must be fitted before calling predict.")

        X_idx, _ = pairwise_distances_argmin_min(self._ensure_array(X), self._X_fit)
        return self.model.labels_[X_idx]

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
        return partial(AgglomerativeClusteringWrapper, **clean_config)


class DBSCANWrapper(AbstractClustering):
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
        self._X_fit: np.ndarray | None = None

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
        self._X_fit = self._ensure_array(X)
        self.model.fit(self._X_fit)
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

        if self._X_fit is None:
            raise ValueError("Model must be fitted before calling predict.")

        X_idx, _ = pairwise_distances_argmin_min(self._ensure_array(X), self._X_fit)
        return self.model.labels_[X_idx]

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
        return partial(DBSCANWrapper, **clean_config)
