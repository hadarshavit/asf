"""
Feature analysis functions for ASF.

This module provides functions for analyzing features in algorithm selection scenarios,
including correlation analysis, clustering, box/violin plots, and feature importance.
All plotting functions use Plotly and provide an option to return data without plotting.
"""

import numpy as np
import pandas as pd
from numpy import corrcoef
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

import plotly.express as px
import plotly.graph_objects as go


def get_feature_statistics(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute basic statistics for each feature.

    Args:
        features: DataFrame with feature values, rows are instances, columns are features.

    Returns:
        DataFrame with statistics (mean, std, min, max, median, missing count) for each feature.
    """
    stats = pd.DataFrame(
        {
            "mean": features.mean(),
            "std": features.std(),
            "min": features.min(),
            "max": features.max(),
            "median": features.median(),
            "missing_count": features.isna().sum(),
            "missing_pct": features.isna().sum() / len(features) * 100,
        }
    )
    return stats


def compute_feature_correlation(
    features: pd.DataFrame,
    method: str = "pearson",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute correlation matrix between features using hierarchical clustering for ordering.

    Args:
        features: DataFrame with feature values, rows are instances, columns are features.
        method: Correlation method ('pearson' or 'spearman').

    Returns:
        Tuple of:
            - Correlation matrix as DataFrame with features ordered by hierarchical clustering.
            - List of feature names in clustered order.
    """
    feature_data = features.fillna(features.mean())
    feature_names = list(features.columns)
    n_features = len(feature_names)

    # Compute correlation matrix
    if method == "pearson":
        data = np.zeros((n_features, n_features)) + 1
        feature_values = feature_data.values
        for i in range(n_features):
            for j in range(i + 1, n_features):
                rho = corrcoef([feature_values[:, i], feature_values[:, j]])[0, 1]
                if np.isnan(rho):  # is nan if one feature vec is constant
                    rho = 0
                data[i, j] = rho
                data[j, i] = rho
    else:  # spearman
        data = feature_data.corr(method="spearman").values
        data = np.nan_to_num(data, nan=0)

    # Hierarchical clustering for ordering
    link = linkage(data * -1, "ward")  # input is distance -> * -1

    sorted_features = [[a] for a in feature_names]
    for link_item in link:
        new_cluster = sorted_features[int(link_item[0])][:]
        new_cluster.extend(sorted_features[int(link_item[1])][:])
        sorted_features.append(new_cluster)

    sorted_features = sorted_features[-1]

    # Resort data according to clustering
    indx_list = []
    for f in feature_names:
        indx_list.append(sorted_features.index(f))
    indx_list = np.argsort(indx_list)
    data = data[indx_list, :]
    data = data[:, indx_list]

    correlation_df = pd.DataFrame(data, index=sorted_features, columns=sorted_features)

    return correlation_df, sorted_features


def plot_feature_correlation(
    features: pd.DataFrame,
    method: str = "pearson",
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame, list[str]]:
    """
    Plot correlation heatmap between features with hierarchical clustering.

    Args:
        features: DataFrame with feature values, rows are instances, columns are features.
        method: Correlation method ('pearson' or 'spearman').
        return_data: If True, also return the correlation data.

    Returns:
        Plotly Figure, or tuple of (Figure, correlation_df, sorted_features) if return_data is True.
    """
    correlation_df, sorted_features = compute_feature_correlation(features, method)

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=sorted_features,
            y=sorted_features,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=f"Feature Correlation ({method.capitalize()})",
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        width=800,
        height=800,
    )

    if return_data:
        return fig, correlation_df, sorted_features
    return fig


def compute_box_plot_data(
    features: pd.DataFrame,
    feature_name: str,
) -> dict:
    """
    Compute box plot statistics for a single feature.

    Args:
        features: DataFrame with feature values.
        feature_name: Name of the feature to analyze.

    Returns:
        Dictionary with statistics for the feature.
    """
    vec = features[feature_name].dropna().values
    return {
        "feature_name": feature_name,
        "values": vec,
        "mean": np.mean(vec),
        "median": np.median(vec),
        "std": np.std(vec),
        "min": np.min(vec),
        "max": np.max(vec),
        "q25": np.percentile(vec, 25),
        "q75": np.percentile(vec, 75),
        "count": len(vec),
        "missing": features[feature_name].isna().sum(),
    }


def plot_feature_box(
    features: pd.DataFrame,
    feature_name: str | None = None,
    return_data: bool = False,
) -> (
    go.Figure
    | list[tuple[str, go.Figure]]
    | tuple[go.Figure | list[tuple[str, go.Figure]], list[dict]]
):
    """
    Create box plots for features.

    Args:
        features: DataFrame with feature values.
        feature_name: Name of specific feature to plot. If None, plots all features.
        return_data: If True, also return the statistics data.

    Returns:
        If feature_name is provided: Plotly Figure (or tuple with data if return_data).
        If feature_name is None: List of (feature_name, Figure) tuples (or tuple with list of data dicts).
    """
    if feature_name is not None:
        data = compute_box_plot_data(features, feature_name)
        vec = data["values"]

        fig = go.Figure()

        fig.add_trace(
            go.Box(x=vec, name=feature_name, orientation="h", showlegend=False)
        )

        fig.update_layout(
            title=f"Distribution of {feature_name}",
            height=300,
            width=800,
        )

        if return_data:
            return fig, data
        return fig

    # Plot all features
    figures = []
    all_data = []
    for feat in sorted(features.columns):
        data = compute_box_plot_data(features, feat)
        all_data.append(data)
        vec = data["values"]

        fig = go.Figure()

        fig.add_trace(go.Box(x=vec, name=feat, orientation="h", showlegend=False))

        fig.update_layout(
            title=f"Distribution of {feat}",
            height=300,
            width=800,
        )

        figures.append((feat, fig))

    if return_data:
        return figures, all_data
    return figures


def compute_feature_importance(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    n_estimators: int = 100,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Compute feature importance using pairwise random forest classification.

    Trains random forests for each pair of algorithms to predict which performs better,
    then averages the feature importances across all forests.

    Args:
        features: DataFrame with feature values, rows are instances, columns are features.
        performance: DataFrame with performance values, rows are instances, columns are algorithms.
        n_estimators: Number of trees in each random forest.
        top_n: Number of top features to return.

    Returns:
        DataFrame with feature importance statistics (median, q25, q75) for top features.
    """
    # Fill missing values
    features_filled = features.fillna(features.mean())

    algorithms = performance.columns.tolist()
    importances = []

    # Train pairwise classifiers
    for algo1, algo2 in combinations(algorithms, 2):
        # Create binary labels: 1 if algo1 is better (lower), 0 otherwise
        y = (performance[algo1] < performance[algo2]).astype(int)

        # Skip if all labels are the same
        if y.nunique() < 2:
            continue

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        try:
            rf.fit(features_filled, y)
            if not np.isnan(rf.feature_importances_).any():
                importances.append(rf.feature_importances_)
        except Exception:
            continue

    if not importances:
        return pd.DataFrame()

    importances = np.array(importances)
    median_importance = np.median(importances, axis=0)
    q25 = np.percentile(importances, 25, axis=0)
    q75 = np.percentile(importances, 75, axis=0)

    feature_names = np.array(features.columns)

    # Sort by median importance
    n_feat = min(len(feature_names), top_n)
    indices = np.argsort(median_importance)[::-1]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names[indices[:n_feat]],
            "median_importance": median_importance[indices[:n_feat]],
            "q25": q25[indices[:n_feat]],
            "q75": q75[indices[:n_feat]],
        }
    )

    return importance_df


def plot_feature_importance(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    n_estimators: int = 100,
    top_n: int = 15,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame]:
    """
    Plot feature importance based on pairwise random forest classification.

    Args:
        features: DataFrame with feature values.
        performance: DataFrame with performance values.
        n_estimators: Number of trees in each random forest.
        top_n: Number of top features to show.
        return_data: If True, also return the importance data.

    Returns:
        Plotly Figure, or tuple of (Figure, importance_df) if return_data is True.
    """
    importance_df = compute_feature_importance(
        features, performance, n_estimators, top_n
    )

    if importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Could not compute feature importance", showarrow=False)
        if return_data:
            return fig, importance_df
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=importance_df["feature"],
            y=importance_df["median_importance"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=importance_df["q75"] - importance_df["median_importance"],
                arrayminus=importance_df["median_importance"] - importance_df["q25"],
            ),
            marker_color="indianred",
        )
    )

    fig.update_layout(
        title="Feature Importance (Pairwise Random Forest)",
        xaxis_title="Feature",
        yaxis_title="Importance",
        xaxis=dict(tickangle=45),
        width=800,
        height=500,
    )

    if return_data:
        return fig, importance_df
    return fig


def compute_feature_clusters(
    features: pd.DataFrame,
    n_clusters_range: tuple[int, int] = (2, 12),
    random_state: int = 42,
) -> dict:
    """
    Cluster instances in feature space using PCA and k-means.

    Uses silhouette score to determine optimal number of clusters.

    Args:
        features: DataFrame with feature values.
        n_clusters_range: Range of cluster numbers to try (min, max).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with clustering results including:
            - features_2d: 2D PCA-transformed features
            - labels: Cluster labels for each instance
            - n_clusters: Optimal number of clusters
            - silhouette_scores: Silhouette scores for each k
            - instances: Instance names
    """
    # Fill missing values
    features_filled = features.fillna(features.mean())
    feature_values = features_filled.values

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_values)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=random_state)
    features_2d = pca.fit_transform(features_scaled)

    # Find optimal number of clusters
    scores = []
    for n_clusters in range(n_clusters_range[0], n_clusters_range[1]):
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        y_pred = km.fit_predict(features_2d)
        score = silhouette_score(features_2d, y_pred)
        scores.append(score)

    # Use maximum silhouette score (not minimum as in original)
    best_score = max(scores)
    best_k = scores.index(best_score) + n_clusters_range[0]

    # Final clustering with optimal k
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km.fit_predict(features_2d)

    return {
        "features_2d": features_2d,
        "labels": labels,
        "n_clusters": best_k,
        "silhouette_scores": scores,
        "instances": features.index.tolist(),
        "pca_explained_variance": pca.explained_variance_ratio_,
    }


def plot_feature_clusters(
    features: pd.DataFrame,
    n_clusters_range: tuple[int, int] = (2, 12),
    random_state: int = 42,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict]:
    """
    Plot instances clustered in 2D PCA feature space.

    Args:
        features: DataFrame with feature values.
        n_clusters_range: Range of cluster numbers to try.
        random_state: Random seed for reproducibility.
        return_data: If True, also return the clustering data.

    Returns:
        Plotly Figure, or tuple of (Figure, cluster_data) if return_data is True.
    """
    cluster_data = compute_feature_clusters(features, n_clusters_range, random_state)

    features_2d = cluster_data["features_2d"]
    labels = cluster_data["labels"]
    instances = cluster_data["instances"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            mode="markers",
            marker=dict(
                color=labels,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cluster"),
            ),
            text=instances,
            hovertemplate="Instance: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )

    explained_var = cluster_data["pca_explained_variance"]
    fig.update_layout(
        title=f"Instance Clustering (k={cluster_data['n_clusters']})",
        xaxis_title=f"PC1 ({explained_var[0] * 100:.1f}% var)",
        yaxis_title=f"PC2 ({explained_var[1] * 100:.1f}% var)",
        width=800,
        height=600,
    )

    if return_data:
        return fig, cluster_data
    return fig


def compute_feature_runstatus_distribution(
    feature_runstatus: pd.DataFrame,
    stati: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the distribution of run statuses for each feature group.

    Args:
        feature_runstatus: DataFrame with runstatus for each feature group.
        stati: List of status values to include. Defaults to common statuses.

    Returns:
        DataFrame with frequency of each status per feature group.
    """
    if stati is None:
        stati = ["ok", "timeout", "memout", "presolved", "crash", "other", "unknown"]

    n_instances = len(feature_runstatus)

    distribution = {}
    for status in stati:
        distribution[status] = (feature_runstatus == status).sum() / n_instances

    return pd.DataFrame(distribution)


def plot_feature_runstatus_bar(
    feature_runstatus: pd.DataFrame,
    stati: list[str] | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame]:
    """
    Plot stacked bar chart of feature run status distribution.

    Args:
        feature_runstatus: DataFrame with runstatus for each feature group.
        stati: List of status values to include.
        return_data: If True, also return the distribution data.

    Returns:
        Plotly Figure, or tuple of (Figure, distribution_df) if return_data is True.
    """
    if stati is None:
        stati = ["ok", "timeout", "memout", "presolved", "crash", "other", "unknown"]

    distribution_df = compute_feature_runstatus_distribution(feature_runstatus, stati)

    fig = go.Figure()

    colors = px.colors.qualitative.Set2[: len(stati)]

    for i, status in enumerate(stati):
        fig.add_trace(
            go.Bar(
                x=distribution_df.index,
                y=distribution_df[status],
                name=status,
                marker_color=colors[i] if i < len(colors) else None,
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Feature Group Run Status Distribution",
        xaxis_title="Feature Group",
        yaxis_title="Frequency",
        xaxis=dict(tickangle=45),
        legend_title="Status",
        width=800,
        height=500,
    )

    if return_data:
        return fig, distribution_df
    return fig


def compute_feature_cost_cdf(
    feature_costs: pd.DataFrame,
    n_points: int = 1000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute CDF data for feature computation costs.

    Args:
        feature_costs: DataFrame with feature costs for each feature group.
        n_points: Number of points to sample for CDF.

    Returns:
        Dictionary mapping feature group names to (x, y) CDF data.
    """
    min_val = max(0.0005, feature_costs.min().min())

    cdfs = {}
    for col in feature_costs.columns:
        values = feature_costs[col].dropna().sort_values()
        if len(values) == 0:
            continue

        # Create CDF
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)

        # Add start point
        x = np.concatenate([[min_val], x])
        y = np.concatenate([[0], y])

        cdfs[col] = (x, y)

    return cdfs


def plot_feature_cost_cdf(
    feature_costs: pd.DataFrame,
    log_scale: bool = True,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict]:
    """
    Plot CDF of feature computation costs.

    Args:
        feature_costs: DataFrame with feature costs for each feature group.
        log_scale: Whether to use log scale for x-axis.
        return_data: If True, also return the CDF data.

    Returns:
        Plotly Figure, or tuple of (Figure, cdf_data) if return_data is True.
    """
    cdf_data = compute_feature_cost_cdf(feature_costs)

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    for i, (name, (x, y)) in enumerate(cdf_data.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title="Feature Cost CDF",
        xaxis_title="Cost",
        yaxis_title="P(x < X)",
        xaxis_type="log" if log_scale else "linear",
        legend_title="Feature Group",
        width=800,
        height=500,
    )

    if return_data:
        return fig, cdf_data
    return fig


def compute_pca_features(
    features: pd.DataFrame,
    n_components: int = 2,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """
    Apply PCA to reduce feature dimensionality.

    Args:
        features: DataFrame with feature values.
        n_components: Number of PCA components.

    Returns:
        Tuple of (transformed features, fitted PCA, fitted scaler).
    """
    features_filled = features.fillna(features.mean())

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_filled.values)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    return features_pca, pca, scaler


def plot_feature_pca(
    features: pd.DataFrame,
    color_by: pd.Series | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, np.ndarray, PCA]:
    """
    Plot instances in 2D PCA feature space.

    Args:
        features: DataFrame with feature values.
        color_by: Optional series to color points by (e.g., algorithm performance).
        return_data: If True, also return the PCA data.

    Returns:
        Plotly Figure, or tuple of (Figure, features_2d, pca) if return_data is True.
    """
    features_2d, pca, _ = compute_pca_features(features, n_components=2)

    instances = features.index.tolist()

    if color_by is not None:
        fig = go.Figure(
            go.Scatter(
                x=features_2d[:, 0],
                y=features_2d[:, 1],
                mode="markers",
                marker=dict(
                    color=color_by,
                    colorscale="Viridis",
                    showscale=True,
                ),
                text=instances,
                hovertemplate="Instance: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )
    else:
        fig = go.Figure(
            go.Scatter(
                x=features_2d[:, 0],
                y=features_2d[:, 1],
                mode="markers",
                text=instances,
                hovertemplate="Instance: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

    explained_var = pca.explained_variance_ratio_
    fig.update_layout(
        title="Instance Feature Space (PCA)",
        xaxis_title=f"PC1 ({explained_var[0] * 100:.1f}% var)",
        yaxis_title=f"PC2 ({explained_var[1] * 100:.1f}% var)",
        width=800,
        height=600,
    )

    if return_data:
        return fig, features_2d, pca
    return fig


def summarize_features(
    features: pd.DataFrame,
    feature_costs: pd.DataFrame | None = None,
    feature_runstatus: pd.DataFrame | None = None,
) -> dict:
    """
    Generate a comprehensive summary of feature data.

    Args:
        features: DataFrame with feature values.
        feature_costs: Optional DataFrame with feature computation costs.
        feature_runstatus: Optional DataFrame with feature runstatus.

    Returns:
        Dictionary with summary statistics and information.
    """
    summary = {
        "n_instances": len(features),
        "n_features": len(features.columns),
        "feature_names": features.columns.tolist(),
        "missing_values": features.isna().sum().sum(),
        "missing_pct": features.isna().sum().sum() / features.size * 100,
        "feature_statistics": get_feature_statistics(features),
    }

    if feature_costs is not None:
        summary["feature_cost_stats"] = {
            "total_mean": feature_costs.sum(axis=1).mean(),
            "total_std": feature_costs.sum(axis=1).std(),
            "per_group_mean": feature_costs.mean(),
            "per_group_std": feature_costs.std(),
        }

    if feature_runstatus is not None:
        summary["runstatus_distribution"] = compute_feature_runstatus_distribution(
            feature_runstatus
        )

    return summary
