"""Tests for asf.analysis module.

Tests for feature and performance analysis functions, including
statistics, correlation, clustering, CDF, baselines, and contribution values.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")


class TestFeatureStatistics:
    """Tests for feature statistics functions."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
                "feature3": np.random.randn(100) + 5,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_get_feature_statistics(self, sample_features):
        """Test basic feature statistics computation."""
        from asf.analysis import get_feature_statistics

        stats = get_feature_statistics(sample_features)
        assert isinstance(stats, pd.DataFrame)
        assert set(stats.index) == {"feature1", "feature2", "feature3"}
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert "min" in stats.columns
        assert "max" in stats.columns


class TestFeatureCorrelation:
    """Tests for feature correlation functions."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
                "feature3": np.random.randn(100) + 5,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_feature_correlation(self, sample_features):
        """Test feature correlation matrix computation."""
        from asf.analysis import compute_feature_correlation

        corr_matrix, feature_order = compute_feature_correlation(sample_features)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert len(feature_order) == 3
        assert corr_matrix.shape == (3, 3)

    def test_plot_feature_correlation(self, sample_features):
        """Test feature correlation plot generation."""
        from asf.analysis import plot_feature_correlation
        import plotly.graph_objects as go

        fig = plot_feature_correlation(sample_features)
        assert isinstance(fig, go.Figure)

    def test_plot_feature_correlation_return_data(self, sample_features):
        """Test feature correlation plot with return_data=True."""
        from asf.analysis import plot_feature_correlation

        result = plot_feature_correlation(sample_features, return_data=True)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestFeatureBoxPlots:
    """Tests for feature box plot functions."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_box_plot_data(self, sample_features):
        """Test box plot data computation."""
        from asf.analysis import compute_box_plot_data

        data = compute_box_plot_data(sample_features, "feature1")
        assert isinstance(data, dict)
        assert "median" in data
        assert "q1" in data or "q25" in data

    def test_plot_feature_box_single(self, sample_features):
        """Test single feature box plot."""
        from asf.analysis import plot_feature_box
        import plotly.graph_objects as go

        fig = plot_feature_box(sample_features, feature_name="feature1")
        assert isinstance(fig, go.Figure)

    def test_plot_feature_box_all(self, sample_features):
        """Test all features box plot."""
        from asf.analysis import plot_feature_box

        result = plot_feature_box(sample_features)
        # Returns list of (name, figure) tuples when no feature_name specified
        assert isinstance(result, list)


class TestFeatureImportance:
    """Tests for feature importance functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample feature and performance data for testing."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50) * 2,
                "feature3": np.random.randn(50) + 5,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        performance = pd.DataFrame(
            {
                "algo1": np.random.rand(50) * 100,
                "algo2": np.random.rand(50) * 100,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        return features, performance

    def test_compute_feature_importance(self, sample_data):
        """Test feature importance computation."""
        from asf.analysis import compute_feature_importance

        features, performance = sample_data
        importance = compute_feature_importance(
            features, performance, n_estimators=10, top_n=3
        )
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) <= 3

    def test_plot_feature_importance(self, sample_data):
        """Test feature importance plot generation."""
        from asf.analysis import plot_feature_importance
        import plotly.graph_objects as go

        features, performance = sample_data
        fig = plot_feature_importance(features, performance, n_estimators=10, top_n=3)
        assert isinstance(fig, go.Figure)


class TestFeatureClustering:
    """Tests for feature clustering functions."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
                "feature3": np.random.randn(100) + 5,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_feature_clusters(self, sample_features):
        """Test feature clustering computation."""
        from asf.analysis import compute_feature_clusters

        cluster_data = compute_feature_clusters(
            sample_features, n_clusters_range=(2, 5), random_state=42
        )
        assert isinstance(cluster_data, dict)
        assert "labels" in cluster_data
        assert "n_clusters" in cluster_data

    def test_plot_feature_clusters(self, sample_features):
        """Test feature clustering plot generation."""
        from asf.analysis import plot_feature_clusters
        import plotly.graph_objects as go

        fig = plot_feature_clusters(
            sample_features, n_clusters_range=(2, 5), random_state=42
        )
        assert isinstance(fig, go.Figure)


class TestFeaturePCA:
    """Tests for feature PCA functions."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
                "feature3": np.random.randn(100) + 5,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_pca_features(self, sample_features):
        """Test PCA feature computation."""
        from asf.analysis import compute_pca_features

        features_2d, pca, scaler = compute_pca_features(sample_features, n_components=2)
        assert features_2d.shape == (100, 2)

    def test_plot_feature_pca(self, sample_features):
        """Test PCA feature plot generation."""
        from asf.analysis import plot_feature_pca
        import plotly.graph_objects as go

        fig = plot_feature_pca(sample_features)
        assert isinstance(fig, go.Figure)


class TestSummarizeFeatures:
    """Tests for feature summary function."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_summarize_features(self, sample_features):
        """Test feature summary computation."""
        from asf.analysis import summarize_features

        summary = summarize_features(sample_features)
        assert isinstance(summary, dict)
        assert "n_instances" in summary
        assert "n_features" in summary


class TestPerformanceBaselines:
    """Tests for performance baseline functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(100) * 100,
                "algo2": np.random.rand(100) * 100,
                "algo3": np.random.rand(100) * 100,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_baselines(self, sample_performance):
        """Test baseline computation."""
        from asf.analysis import compute_baselines

        baselines = compute_baselines(sample_performance, maximize=False)
        assert isinstance(baselines, dict)
        assert "vbs_score" in baselines
        assert "bsa_score" in baselines

    def test_compute_greedy_portfolio(self, sample_performance):
        """Test greedy portfolio computation."""
        from asf.analysis import compute_greedy_portfolio

        portfolio = compute_greedy_portfolio(
            sample_performance, max_algos=3, maximize=False
        )
        assert isinstance(portfolio, list)
        assert len(portfolio) <= 3


class TestAlgorithmCorrelation:
    """Tests for algorithm correlation functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(100) * 100,
                "algo2": np.random.rand(100) * 100,
                "algo3": np.random.rand(100) * 100,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_algorithm_correlation(self, sample_performance):
        """Test algorithm correlation computation."""
        from asf.analysis import compute_algorithm_correlation

        corr_matrix, algo_order = compute_algorithm_correlation(sample_performance)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert len(algo_order) == 3
        assert corr_matrix.shape == (3, 3)

    def test_plot_algorithm_correlation(self, sample_performance):
        """Test algorithm correlation plot generation."""
        from asf.analysis import plot_algorithm_correlation
        import plotly.graph_objects as go

        fig = plot_algorithm_correlation(sample_performance)
        assert isinstance(fig, go.Figure)


class TestPerformanceCDF:
    """Tests for performance CDF functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(100) * 100,
                "algo2": np.random.rand(100) * 100,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_performance_cdf(self, sample_performance):
        """Test performance CDF computation."""
        from asf.analysis import compute_performance_cdf

        cdf_data = compute_performance_cdf(sample_performance)
        assert isinstance(cdf_data, dict)
        assert "algo1" in cdf_data
        assert "algo2" in cdf_data

    def test_plot_performance_cdf(self, sample_performance):
        """Test performance CDF plot generation."""
        from asf.analysis import plot_performance_cdf
        import plotly.graph_objects as go

        fig = plot_performance_cdf(sample_performance)
        assert isinstance(fig, go.Figure)


class TestPerformanceBoxPlots:
    """Tests for performance box plot functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(100) * 100,
                "algo2": np.random.rand(100) * 100,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_performance_box_plot_data(self, sample_performance):
        """Test performance box plot data computation."""
        from asf.analysis import compute_performance_box_plot_data

        data = compute_performance_box_plot_data(sample_performance)
        assert isinstance(data, dict)
        assert "algo1" in data
        assert "algo2" in data

    def test_plot_performance_box(self, sample_performance):
        """Test performance box plot generation."""
        from asf.analysis import plot_performance_box
        import plotly.graph_objects as go

        fig = plot_performance_box(sample_performance)
        assert isinstance(fig, go.Figure)


class TestScatterPlots:
    """Tests for scatter plot functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(100) * 100,
                "algo2": np.random.rand(100) * 100,
                "algo3": np.random.rand(100) * 100,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_compute_scatter_data(self, sample_performance):
        """Test scatter data computation."""
        from asf.analysis import compute_scatter_data

        data = compute_scatter_data(sample_performance, "algo1", "algo2")
        assert isinstance(data, dict)
        assert "x" in data
        assert "y" in data

    def test_plot_scatter(self, sample_performance):
        """Test scatter plot generation."""
        from asf.analysis import plot_scatter
        import plotly.graph_objects as go

        fig = plot_scatter(sample_performance, "algo1", "algo2")
        assert isinstance(fig, go.Figure)

    def test_plot_all_scatter(self, sample_performance):
        """Test all scatter plots generation."""
        from asf.analysis import plot_all_scatter

        result = plot_all_scatter(sample_performance)
        assert isinstance(result, list)
        # Should have 3 pairs for 3 algorithms
        assert len(result) == 3


class TestContributionValues:
    """Tests for contribution value functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing (small for Shapley computation)."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(20) * 100,
                "algo2": np.random.rand(20) * 100,
                "algo3": np.random.rand(20) * 100,
            },
            index=[f"inst_{i}" for i in range(20)],
        )

    def test_compute_contribution_values(self, sample_performance):
        """Test contribution values computation."""
        from asf.analysis import compute_contribution_values

        values = compute_contribution_values(sample_performance, maximize=False)
        assert isinstance(values, dict)
        assert "averages" in values
        assert "marginals" in values
        assert "shapleys" in values

    def test_plot_contribution_pie(self, sample_performance):
        """Test contribution pie chart generation."""
        from asf.analysis import plot_contribution_pie
        import plotly.graph_objects as go

        fig = plot_contribution_pie(sample_performance, contribution_type="marginals")
        assert isinstance(fig, go.Figure)


class TestInstanceHardness:
    """Tests for instance hardness functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample performance and feature data for testing."""
        np.random.seed(42)
        performance = pd.DataFrame(
            {
                "algo1": np.random.rand(50) * 100,
                "algo2": np.random.rand(50) * 100,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50) * 2,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        return performance, features

    def test_compute_instance_hardness(self, sample_data):
        """Test instance hardness computation."""
        from asf.analysis import compute_instance_hardness

        performance, features = sample_data
        hardness = compute_instance_hardness(
            performance, features, maximize=False, eps=0.1
        )
        assert isinstance(hardness, dict)

    def test_plot_instance_hardness(self, sample_data):
        """Test instance hardness plot generation."""
        from asf.analysis import plot_instance_hardness
        import plotly.graph_objects as go

        performance, features = sample_data
        fig = plot_instance_hardness(performance, features, maximize=False, eps=0.1)
        assert isinstance(fig, go.Figure)


class TestAlgorithmFootprint:
    """Tests for algorithm footprint functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample performance and feature data for testing."""
        np.random.seed(42)
        performance = pd.DataFrame(
            {
                "algo1": np.random.rand(50) * 100,
                "algo2": np.random.rand(50) * 100,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50) * 2,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        return performance, features

    def test_compute_algorithm_footprint(self, sample_data):
        """Test algorithm footprint computation."""
        from asf.analysis import compute_algorithm_footprint

        performance, features = sample_data
        footprint = compute_algorithm_footprint(
            performance, features, algorithm="algo1", maximize=False, eps=0.1
        )
        assert isinstance(footprint, dict)

    def test_plot_algorithm_footprint(self, sample_data):
        """Test algorithm footprint plot generation."""
        from asf.analysis import plot_algorithm_footprint
        import plotly.graph_objects as go

        performance, features = sample_data
        fig = plot_algorithm_footprint(
            performance, features, algorithm="algo1", maximize=False, eps=0.1
        )
        assert isinstance(fig, go.Figure)


class TestSummarizePerformance:
    """Tests for performance summary function."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(100) * 100,
                "algo2": np.random.rand(100) * 100,
            },
            index=[f"inst_{i}" for i in range(100)],
        )

    def test_summarize_performance(self, sample_performance):
        """Test performance summary computation."""
        from asf.analysis import summarize_performance

        summary = summarize_performance(sample_performance, maximize=False)
        assert isinstance(summary, dict)
        assert "n_instances" in summary
        assert "n_algorithms" in summary


class TestCriticalDistance:
    """Tests for critical distance functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(30) * 100,
                "algo2": np.random.rand(30) * 100,
                "algo3": np.random.rand(30) * 100,
            },
            index=[f"inst_{i}" for i in range(30)],
        )

    def test_compute_critical_distance(self, sample_performance):
        """Test critical distance computation."""
        from asf.analysis import compute_critical_distance

        cd_result = compute_critical_distance(
            sample_performance, alpha=0.05, maximize=False
        )
        assert isinstance(cd_result, dict)
        assert "critical_distance" in cd_result
        assert "average_ranks" in cd_result

    def test_plot_critical_distance(self, sample_performance):
        """Test critical distance plot generation."""
        from asf.analysis import plot_critical_distance
        import plotly.graph_objects as go

        fig = plot_critical_distance(sample_performance, alpha=0.05, maximize=False)
        assert isinstance(fig, go.Figure)


class TestAlgorithmSimilarity:
    """Tests for algorithm similarity functions."""

    @pytest.fixture
    def sample_performance(self):
        """Sample performance data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "algo1": np.random.rand(50) * 100,
                "algo2": np.random.rand(50) * 100,
                "algo3": np.random.rand(50) * 100,
            },
            index=[f"inst_{i}" for i in range(50)],
        )

    def test_compute_algorithm_similarity(self, sample_performance):
        """Test algorithm similarity computation."""
        from asf.analysis import compute_algorithm_similarity

        similarity = compute_algorithm_similarity(sample_performance)
        assert isinstance(similarity, dict)

    def test_plot_algorithm_similarity_heatmap(self, sample_performance):
        """Test algorithm similarity heatmap plot generation."""
        from asf.analysis import plot_algorithm_similarity_heatmap
        import plotly.graph_objects as go

        fig = plot_algorithm_similarity_heatmap(sample_performance)
        assert isinstance(fig, go.Figure)

    def test_plot_algorithm_win_matrix(self, sample_performance):
        """Test algorithm win matrix plot generation."""
        from asf.analysis import plot_algorithm_win_matrix
        import plotly.graph_objects as go

        fig = plot_algorithm_win_matrix(sample_performance)
        assert isinstance(fig, go.Figure)

    def test_plot_algorithm_dendrogram(self, sample_performance):
        """Test algorithm dendrogram plot generation."""
        from asf.analysis import plot_algorithm_dendrogram
        import plotly.graph_objects as go

        fig = plot_algorithm_dendrogram(sample_performance)
        assert isinstance(fig, go.Figure)


class TestFeaturePerformanceCorrelation:
    """Tests for feature-performance correlation functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample feature and performance data for testing."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50) * 2,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        performance = pd.DataFrame(
            {
                "algo1": np.random.rand(50) * 100,
                "algo2": np.random.rand(50) * 100,
            },
            index=[f"inst_{i}" for i in range(50)],
        )
        return features, performance

    def test_compute_feature_performance_correlation(self, sample_data):
        """Test feature-performance correlation computation."""
        from asf.analysis import compute_feature_performance_correlation

        features, performance = sample_data
        corr = compute_feature_performance_correlation(features, performance)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (2, 2)  # 2 features x 2 algos

    def test_compute_feature_performance_difference_correlation(self, sample_data):
        """Test feature-performance difference correlation computation."""
        from asf.analysis import compute_feature_performance_difference_correlation

        features, performance = sample_data
        # Function requires algo1 and algo2 arguments
        corr = compute_feature_performance_difference_correlation(
            features, performance, algo1="algo1", algo2="algo2"
        )
        assert isinstance(corr, pd.DataFrame)

    def test_plot_feature_performance_correlation(self, sample_data):
        """Test feature-performance correlation plot generation."""
        from asf.analysis import plot_feature_performance_correlation
        import plotly.graph_objects as go

        features, performance = sample_data
        fig = plot_feature_performance_correlation(features, performance)
        assert isinstance(fig, go.Figure)
