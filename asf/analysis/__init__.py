"""Analysis module for ASF.

Provides functions for analyzing features and performance in algorithm selection scenarios.
"""

from __future__ import annotations

from asf.analysis.features_analysis import (
    get_feature_statistics,
    compute_feature_correlation,
    plot_feature_correlation,
    compute_box_plot_data,
    plot_feature_box,
    compute_feature_importance,
    plot_feature_importance,
    compute_feature_clusters,
    plot_feature_clusters,
    compute_feature_runstatus_distribution,
    plot_feature_runstatus_bar,
    compute_feature_cost_cdf,
    plot_feature_cost_cdf,
    compute_pca_features,
    plot_feature_pca,
    summarize_features,
)

from asf.analysis.performance_analysis import (
    compute_baselines,
    compute_greedy_portfolio,
    compute_algorithm_correlation,
    plot_algorithm_correlation,
    compute_performance_cdf,
    plot_performance_cdf,
    compute_box_plot_data as compute_performance_box_plot_data,
    plot_performance_box,
    compute_runstatus_distribution,
    plot_runstatus_bar,
    compute_scatter_data,
    plot_scatter,
    plot_all_scatter,
    compute_contribution_values,
    plot_contribution_pie,
    compute_instance_hardness,
    plot_instance_hardness,
    compute_algorithm_footprint,
    plot_algorithm_footprint,
    summarize_performance,
    # Critical distance analysis
    compute_critical_distance,
    plot_critical_distance,
    # Algorithm similarity analysis
    compute_algorithm_similarity,
    plot_algorithm_similarity_heatmap,
    plot_algorithm_win_matrix,
    plot_algorithm_dendrogram,
    # Feature-performance correlation
    compute_feature_performance_correlation,
    compute_feature_performance_difference_correlation,
    plot_feature_performance_correlation,
    compute_feature_algorithm_selection_correlation,
    plot_feature_selection_predictors,
)

__all__ = [
    # Feature analysis
    # Feature statistics
    "get_feature_statistics",
    # Correlation analysis
    "compute_feature_correlation",
    "plot_feature_correlation",
    # Box plots
    "compute_box_plot_data",
    "plot_feature_box",
    # Feature importance
    "compute_feature_importance",
    "plot_feature_importance",
    # Clustering
    "compute_feature_clusters",
    "plot_feature_clusters",
    # Run status
    "compute_feature_runstatus_distribution",
    "plot_feature_runstatus_bar",
    # Feature cost CDF
    "compute_feature_cost_cdf",
    "plot_feature_cost_cdf",
    # PCA
    "compute_pca_features",
    "plot_feature_pca",
    # Summary
    "summarize_features",
    # Performance analysis
    # Baselines
    "compute_baselines",
    "compute_greedy_portfolio",
    # Correlation
    "compute_algorithm_correlation",
    "plot_algorithm_correlation",
    # CDF
    "compute_performance_cdf",
    "plot_performance_cdf",
    # Box plots
    "compute_performance_box_plot_data",
    "plot_performance_box",
    # Run status
    "compute_runstatus_distribution",
    "plot_runstatus_bar",
    # Scatter plots
    "compute_scatter_data",
    "plot_scatter",
    "plot_all_scatter",
    # Contribution values
    "compute_contribution_values",
    "plot_contribution_pie",
    # Instance hardness
    "compute_instance_hardness",
    "plot_instance_hardness",
    # Footprints
    "compute_algorithm_footprint",
    "plot_algorithm_footprint",
    # Summary
    "summarize_performance",
    # Critical distance analysis
    "compute_critical_distance",
    "plot_critical_distance",
    # Algorithm similarity analysis
    "compute_algorithm_similarity",
    "plot_algorithm_similarity_heatmap",
    "plot_algorithm_win_matrix",
    "plot_algorithm_dendrogram",
    # Feature-performance correlation
    "compute_feature_performance_correlation",
    "compute_feature_performance_difference_correlation",
    "plot_feature_performance_correlation",
    "compute_feature_algorithm_selection_correlation",
    "plot_feature_selection_predictors",
]
