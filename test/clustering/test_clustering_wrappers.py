"""Tests for clustering wrappers."""

import numpy as np
import pandas as pd
import pytest

from asf.clustering.wrappers import (
    GMeansWrapper,
    KMeansWrapper,
    AgglomerativeClusteringWrapper,
    DBSCANWrapper,
)


class TestGMeansWrapper:
    """Tests for GMeansWrapper."""

    def test_fit_with_numpy(self):
        """Test fit with numpy array."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.5, 1.8],
                [5.0, 8.0],
                [8.0, 8.0],
                [1.0, 0.6],
                [9.0, 11.0],
            ]
        )

        wrapper = GMeansWrapper()
        result = wrapper.fit(X)

        assert hasattr(wrapper, "model")
        assert result is wrapper  # Should return self

    def test_fit_with_dataframe(self):
        """Test fit with DataFrame."""
        X = pd.DataFrame(
            {
                "f1": [1.0, 1.5, 5.0, 8.0, 1.0, 9.0],
                "f2": [2.0, 1.8, 8.0, 8.0, 0.6, 11.0],
            }
        )

        wrapper = GMeansWrapper()
        wrapper.fit(X)

        assert hasattr(wrapper, "model")

    def test_predict_with_numpy(self):
        """Test predict method."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.5, 1.8],
                [5.0, 8.0],
                [8.0, 8.0],
            ]
        )

        wrapper = GMeansWrapper()
        wrapper.fit(X)
        labels = wrapper.predict(X)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 4

    def test_predict_with_dataframe(self):
        """Test predict with DataFrame."""
        X = pd.DataFrame({"f1": [1.0, 1.5, 5.0, 8.0], "f2": [2.0, 1.8, 8.0, 8.0]})

        wrapper = GMeansWrapper()
        wrapper.fit(X)
        labels = wrapper.predict(X)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 4


class TestKMeansWrapper:
    """Tests for KMeansWrapper."""

    def test_fit_with_numpy(self):
        """Test fit with numpy array."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.5, 1.8],
                [5.0, 8.0],
                [8.0, 8.0],
            ]
        )

        wrapper = KMeansWrapper(n_clusters=2)
        result = wrapper.fit(X)

        assert hasattr(wrapper, "model")
        assert result is wrapper

    def test_fit_with_dataframe(self):
        """Test fit with DataFrame."""
        X = pd.DataFrame(
            {
                "f1": [1.0, 1.5, 5.0, 8.0],
                "f2": [2.0, 1.8, 8.0, 8.0],
            }
        )

        wrapper = KMeansWrapper(n_clusters=2)
        wrapper.fit(X)

        assert hasattr(wrapper, "model")

    def test_predict(self):
        """Test predict method."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.5, 1.8],
                [5.0, 8.0],
                [8.0, 8.0],
            ]
        )

        wrapper = KMeansWrapper(n_clusters=2)
        wrapper.fit(X)
        labels = wrapper.predict(X)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 4
        assert len(set(labels)) <= 2

    def test_predict_with_dataframe(self):
        """Test predict with DataFrame input."""
        X = pd.DataFrame({"f1": [1.0, 1.5, 5.0, 8.0], "f2": [2.0, 1.8, 8.0, 8.0]})

        wrapper = KMeansWrapper(n_clusters=2)
        wrapper.fit(X)
        labels = wrapper.predict(X)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 4


class TestAgglomerativeClusteringWrapper:
    """Tests for AgglomerativeClusteringWrapper."""

    def test_fit_with_numpy(self):
        """Test fit with numpy array."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.5, 1.8],
                [5.0, 8.0],
                [8.0, 8.0],
            ]
        )

        wrapper = AgglomerativeClusteringWrapper(n_clusters=2)
        result = wrapper.fit(X)

        assert hasattr(wrapper, "model")
        assert result is wrapper

    def test_fit_with_dataframe(self):
        """Test fit with DataFrame."""
        X = pd.DataFrame(
            {
                "f1": [1.0, 1.5, 5.0, 8.0],
                "f2": [2.0, 1.8, 8.0, 8.0],
            }
        )

        wrapper = AgglomerativeClusteringWrapper(n_clusters=2)
        wrapper.fit(X)

        assert hasattr(wrapper, "model")

    def test_predict_raises_not_implemented(self):
        """Test that predict raises NotImplementedError."""
        X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])

        wrapper = AgglomerativeClusteringWrapper(n_clusters=2)
        wrapper.fit(X)

        with pytest.raises(NotImplementedError):
            wrapper.predict(X)

    def test_model_has_labels(self):
        """Test that underlying model has labels_ after fit."""
        X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])

        wrapper = AgglomerativeClusteringWrapper(n_clusters=2)
        wrapper.fit(X)

        # The underlying sklearn model has labels_
        assert hasattr(wrapper.model, "labels_")
        assert len(wrapper.model.labels_) == 4


class TestDBSCANWrapper:
    """Tests for DBSCANWrapper."""

    def test_fit_with_numpy(self):
        """Test fit with numpy array."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [5.0, 8.0],
                [5.1, 8.1],
            ]
        )

        wrapper = DBSCANWrapper(eps=0.5, min_samples=2)
        result = wrapper.fit(X)

        assert hasattr(wrapper, "model")
        assert result is wrapper

    def test_fit_with_dataframe(self):
        """Test fit with DataFrame."""
        X = pd.DataFrame(
            {
                "f1": [1.0, 1.1, 5.0, 5.1],
                "f2": [2.0, 2.1, 8.0, 8.1],
            }
        )

        wrapper = DBSCANWrapper(eps=0.5, min_samples=2)
        wrapper.fit(X)

        assert hasattr(wrapper, "model")

    def test_model_has_labels(self):
        """Test that underlying model has labels_ after fit."""
        X = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 8.0], [5.1, 8.1]])

        wrapper = DBSCANWrapper(eps=0.5, min_samples=2)
        wrapper.fit(X)

        # The underlying sklearn model has labels_
        assert hasattr(wrapper.model, "labels_")
        assert len(wrapper.model.labels_) == 4

    def test_predict_raises_not_implemented(self):
        """Test that predict raises NotImplementedError (DBSCAN doesn't support it)."""
        X = np.array(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [5.0, 8.0],
                [5.1, 8.1],
            ]
        )

        wrapper = DBSCANWrapper(eps=0.5, min_samples=2)
        wrapper.fit(X)

        with pytest.raises(NotImplementedError):
            wrapper.predict(X)
