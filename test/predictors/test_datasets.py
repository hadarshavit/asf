"""Tests for predictor utility datasets."""

import numpy as np
import pandas as pd
import pytest
from asf.predictors.utils.datasets import RegressionDataset, RankingDataset


torch = pytest.importorskip("torch")


class TestRegressionDataset:
    """Tests for RegressionDataset."""

    def test_init_with_dataframes(self):
        """Test initialization with DataFrames."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        performance = pd.DataFrame({"algo1": [10.0, 20.0, 30.0]})

        dataset = RegressionDataset(features, performance)

        assert len(dataset) == 3
        assert dataset.features.shape == (3, 2)
        assert dataset.performance.shape == (3, 1)

    def test_init_with_numpy(self):
        """Test initialization with numpy arrays."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        performance = np.array([[10.0], [20.0], [30.0]])

        dataset = RegressionDataset(features, performance)

        assert len(dataset) == 3
        assert dataset.features.shape == (3, 2)

    def test_getitem(self):
        """Test __getitem__ returns correct data."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        performance = pd.DataFrame({"algo1": [10.0, 20.0, 30.0]})

        dataset = RegressionDataset(features, performance)
        feat, perf = dataset[0]

        assert torch.is_tensor(feat)
        assert torch.is_tensor(perf)
        assert feat[0].item() == 1.0
        assert perf[0].item() == 10.0

    def test_len(self):
        """Test __len__ returns correct length."""
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        performance = pd.DataFrame({"algo1": [10.0, 20.0, 30.0, 40.0, 50.0]})

        dataset = RegressionDataset(features, performance)

        assert len(dataset) == 5

    def test_custom_dtype(self):
        """Test with custom dtype."""
        features = pd.DataFrame({"f1": [1.0, 2.0]})
        performance = pd.DataFrame({"algo1": [10.0, 20.0]})

        dataset = RegressionDataset(features, performance, dtype=torch.float64)

        assert dataset.features.dtype == torch.float64
        assert dataset.performance.dtype == torch.float64

    def test_sorts_by_index(self):
        """Test that data is sorted by index."""
        features = pd.DataFrame({"f1": [3.0, 1.0, 2.0]}, index=[2, 0, 1])
        performance = pd.DataFrame({"algo1": [30.0, 10.0, 20.0]}, index=[2, 0, 1])

        dataset = RegressionDataset(features, performance)

        # After sorting by index, first item should be from original index 0
        feat, perf = dataset[0]
        assert feat[0].item() == 1.0
        assert perf[0].item() == 10.0


class TestRankingDataset:
    """Tests for RankingDataset."""

    def test_init(self):
        """Test initialization."""
        features = pd.DataFrame(
            {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}, index=["i1", "i2", "i3"]
        )
        performance = pd.DataFrame(
            {"algo1": [10.0, 20.0, 30.0], "algo2": [15.0, 25.0, 35.0]},
            index=["i1", "i2", "i3"],
        )
        algorithm_features = pd.DataFrame({"af1": [0.1, 0.2]}, index=["algo1", "algo2"])

        dataset = RankingDataset(features, performance, algorithm_features)

        assert len(dataset) == 3  # 3 unique instances
        assert dataset.features_cols == ["f1", "f2"]
        assert dataset.algorithm_features_cols == ["af1"]

    def test_len(self):
        """Test __len__ returns number of unique instances."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [10.0, 20.0], "algo2": [15.0, 25.0]}, index=["i1", "i2"]
        )
        algorithm_features = pd.DataFrame({"af1": [0.1, 0.2]}, index=["algo1", "algo2"])

        dataset = RankingDataset(features, performance, algorithm_features)

        assert len(dataset) == 2

    def test_getitem_returns_triplets(self):
        """Test __getitem__ returns triplets for ranking."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [10.0, 20.0], "algo2": [15.0, 25.0], "algo3": [5.0, 30.0]},
            index=["i1", "i2"],
        )
        algorithm_features = pd.DataFrame(
            {"af1": [0.1, 0.2, 0.3]}, index=["algo1", "algo2", "algo3"]
        )

        dataset = RankingDataset(features, performance, algorithm_features)
        (
            (main_feats, smaller_feats, larger_feats),
            (main_perf, smaller_perf, larger_perf),
        ) = dataset[0]

        assert torch.is_tensor(main_feats)
        assert torch.is_tensor(smaller_feats)
        assert torch.is_tensor(larger_feats)

    def test_custom_dtype(self):
        """Test with custom dtype."""
        features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["i1", "i2"])
        performance = pd.DataFrame(
            {"algo1": [10.0, 20.0], "algo2": [15.0, 25.0]}, index=["i1", "i2"]
        )
        algorithm_features = pd.DataFrame({"af1": [0.1, 0.2]}, index=["algo1", "algo2"])

        dataset = RankingDataset(
            features, performance, algorithm_features, dtype=torch.float64
        )

        assert dataset._dtype == torch.float64
