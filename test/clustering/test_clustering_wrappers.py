"""Tests for clustering wrappers."""

import numpy as np
import pandas as pd
import pytest

from asf.utils.g_means import GMeans
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

    def test_keeps_split_model_after_non_gaussian_test(self, monkeypatch):
        """Accepted GMeans splits must leave usable centers for prediction."""

        class FakeAndersonResult:
            statistic = 1.0
            critical_values = [0.0, 0.0, 0.0, 0.0, 0.0]

        monkeypatch.setattr(
            "asf.utils.g_means.anderson", lambda _: FakeAndersonResult()
        )
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [10.0, 10.0],
                [10.1, 10.1],
            ]
        )

        model = GMeans(random_state=0).fit(X)
        labels = model.predict(X)

        assert model.clusters
        assert len(labels) == len(X)


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

    def test_ignores_selector_metadata_kwargs(self):
        """Selector-level configuration kwargs should not leak into sklearn."""
        wrapper = KMeansWrapper(
            n_clusters=2,
            budget=100.0,
            maximize=False,
            n_algorithms=3,
        )

        assert wrapper.model.n_clusters == 2


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

    def test_ignores_unsupported_isac_kwargs(self):
        """ISAC may pass random_state; selector config may pass budget."""
        wrapper = AgglomerativeClusteringWrapper(
            n_clusters=2,
            random_state=1,
            budget=100.0,
        )

        assert wrapper.model.n_clusters == 2


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
