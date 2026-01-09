"""
Performance analysis functions for ASF.

This module provides functions for analyzing algorithm performance in algorithm selection scenarios,
including baseline computation, correlation analysis, CDF plots, box plots, and contribution analysis.
All plotting functions use Plotly and provide an option to return data without plotting.
"""

from __future__ import annotations

from typing import Any, Callable

import io
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.special import comb
from scipy.stats import friedmanchisquare, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_baselines(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float | None = None,
    runstatus: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compute baseline performance metrics: VBS and BSA.

    VBS (Virtual Best Solver) and BSA (Best Single Algorithm).

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values, rows are instances, columns are algorithms.
    maximize : bool, default=False
        Whether to maximize (True) or minimize (False) performance.
    budget : float or None, default=None
        Algorithm cutoff time/budget for runtime scenarios.
    runstatus : pd.DataFrame or None, default=None
        Optional DataFrame with run status for each algorithm/instance.

    Returns
    -------
    dict
        Dictionary with baseline metrics including VBS score, BSA score, and best algorithm name.
    """
    if maximize:
        vbs_score = float(performance.max(axis=1).mean())
        algo_perfs = performance.mean(axis=0)
        best_algo = str(algo_perfs.idxmax())
        bsa_score = float(algo_perfs[best_algo])
    else:
        vbs_score = float(performance.min(axis=1).mean())
        algo_perfs = performance.mean(axis=0)
        best_algo = str(algo_perfs.idxmin())
        bsa_score = float(algo_perfs[best_algo])

    result: dict[str, Any] = {
        "vbs_score": vbs_score,
        "bsa_score": bsa_score,
        "best_algorithm": best_algo,
        "algorithm_means": algo_perfs.to_dict(),
    }

    # Count unsolvable instances if runstatus is provided
    if runstatus is not None:
        unsolvable = int(np.sum(np.sum(runstatus.values == "ok", axis=1) == 0))
        result["unsolvable_instances"] = unsolvable

        # Compute clean scores (excluding unsolvable instances)
        if budget is not None and unsolvable > 0:
            n_inst = len(performance)
            vbs_clean = (vbs_score * n_inst - 10 * budget * unsolvable) / (
                n_inst - unsolvable
            )
            bsa_clean = (bsa_score * n_inst - 10 * budget * unsolvable) / (
                n_inst - unsolvable
            )
            result["vbs_score_clean"] = vbs_clean
            result["bsa_score_clean"] = bsa_clean

    return result


def compute_greedy_portfolio(
    performance: pd.DataFrame,
    max_algos: int | None = None,
    maximize: bool = False,
) -> list[tuple[str, float]]:
    """
    Build a greedy portfolio by iteratively adding algorithms.

    Algorithms are added based on which most improves VBS.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    max_algos : int or None, default=None
        Maximum number of algorithms to select. If None, selects all.
    maximize : bool, default=False
        Whether to maximize (True) or minimize (False) performance.

    Returns
    -------
    list of tuple of (str, float)
        List of tuples (algorithm_name, vbs_score_after_adding).
    """
    if max_algos is None:
        max_algos = len(performance.columns)

    perf_data = performance.copy()
    if maximize:
        perf_data = perf_data * -1

    # Start with best single algorithm
    bsa = str(perf_data.mean(axis=0).idxmin())
    bsa_score = float(perf_data.mean(axis=0).min())

    selected = [(bsa, -bsa_score if maximize else bsa_score)]
    remaining = set(performance.columns) - {bsa}

    def get_vbs(data: pd.DataFrame) -> float:
        return float(data.min(axis=1).mean())

    for _ in range(1, max_algos):
        if not remaining:
            break

        current_algos = [a[0] for a in selected]
        best_addition: tuple[str | None, float] = (
            None,
            get_vbs(perf_data[current_algos]),
        )

        for algo in remaining:
            test_algos = current_algos + [algo]
            vbs = get_vbs(perf_data[test_algos])
            if vbs < best_addition[1]:
                best_addition = (algo, vbs)

        if best_addition[0] is None:
            break

        score = -best_addition[1] if maximize else best_addition[1]
        selected.append((str(best_addition[0]), score))
        remaining.remove(best_addition[0])

    return selected


def compute_algorithm_correlation(
    performance: pd.DataFrame,
    method: str = "spearman",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute correlation matrix between algorithms.

    Uses hierarchical clustering for ordering the matrix.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    method : str, default="spearman"
        Correlation method ('spearman' or 'pearson').

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        Correlation matrix as DataFrame and algorithm names in clustered order.
    """
    algos = list(performance.columns)
    n_algos = len(algos)
    perf_data = performance.values

    # Compute correlation matrix
    data = np.zeros((n_algos, n_algos)) + 1
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            y_i = np.array(perf_data[:, i], dtype=np.float64)
            y_j = np.array(perf_data[:, j], dtype=np.float64)

            # Add small noise if all zeros to avoid correlation issues
            if np.sum(perf_data[:, i]) == 0:
                y_i += np.random.rand(y_i.shape[0]) * 0.00001
            if np.sum(perf_data[:, j]) == 0:
                y_j += np.random.rand(y_j.shape[0]) * 0.00001

            if method == "spearman":
                rho, _ = spearmanr(y_i, y_j)
            else:
                rho = np.corrcoef(y_i, y_j)[0, 1]

            if np.isnan(rho):
                rho = 0.0
            data[i, j] = rho
            data[j, i] = rho

    # Hierarchical clustering for ordering
    link = linkage(data * -1, "ward")

    sorted_algos_list: list[list[str]] = [[a] for a in algos]
    for link_item in link:
        new_cluster = sorted_algos_list[int(link_item[0])][:]
        new_cluster.extend(sorted_algos_list[int(link_item[1])][:])
        sorted_algos_list.append(new_cluster)

    sorted_algos = sorted_algos_list[-1]

    # Resort data according to clustering
    indx_list = [sorted_algos.index(a) for a in algos]
    indx_list_arr = np.argsort(indx_list)
    data = data[indx_list_arr, :]
    data = data[:, indx_list_arr]

    correlation_df = pd.DataFrame(
        data, index=pd.Index(sorted_algos), columns=pd.Index(sorted_algos)
    )

    return correlation_df, sorted_algos


def plot_algorithm_correlation(
    performance: pd.DataFrame,
    method: str = "spearman",
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame, list[str]]:
    """
    Plot correlation heatmap between algorithms.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    method : str, default="spearman"
        Correlation method ('spearman' or 'pearson').
    return_data : bool, default=False
        If True, also return the correlation data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, correlation_df, sorted_algos).
    """
    correlation_df, sorted_algos = compute_algorithm_correlation(performance, method)

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=sorted_algos,
            y=sorted_algos,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=f"Algorithm Correlation ({method.capitalize()})",
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        width=800,
        height=800,
    )

    if return_data:
        return fig, correlation_df, sorted_algos
    return fig


def compute_performance_cdf(
    performance: pd.DataFrame,
    budget: float | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute CDF data for algorithm performance.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    budget : float or None, default=None
        Optional cutoff value for runtime scenarios.

    Returns
    -------
    dict
        Dictionary mapping algorithm names to (x, y) CDF data.
    """
    if budget is not None:
        max_val = budget
    else:
        max_val = float(performance.max().max())

    min_val = max(0.0005, float(performance.min().min()))

    cdfs = {}
    for algo in performance.columns:
        values = performance[algo].dropna().sort_values()
        if len(values) == 0:
            continue

        # Clip values to max_val
        values_clipped = values.clip(upper=max_val)

        # Create CDF
        x = np.sort(values_clipped.values)
        y = np.arange(1, len(x) + 1) / len(x)

        # Add start point
        x = np.concatenate([[min_val], x])
        y = np.concatenate([[0], y])

        cdfs[str(algo)] = (x, y)

    return cdfs


def plot_performance_cdf(
    performance: pd.DataFrame,
    budget: float | None = None,
    log_scale: bool = True,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Plot CDF of algorithm performance.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    budget : float or None, default=None
        Optional cutoff value for runtime scenarios.
    log_scale : bool, default=True
        Whether to use log scale for x-axis.
    return_data : bool, default=False
        If True, also return the CDF data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, cdf_data).
    """
    cdf_data = compute_performance_cdf(performance, budget)

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    for i, (algo, (x, y)) in enumerate(cdf_data.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=algo,
                line=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title="Performance CDF",
        xaxis_title="Performance",
        yaxis_title="P(x < X)",
        xaxis_type="log" if log_scale else "linear",
        legend_title="Algorithm",
        width=900,
        height=600,
    )

    if return_data:
        return fig, cdf_data
    return fig


def compute_box_plot_data(
    performance: pd.DataFrame,
    budget: float | None = None,
    log_scale: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Compute box plot statistics for each algorithm.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    budget : float or None, default=None
        Optional cutoff value to clip performance.
    log_scale : bool, default=False
        Whether to log-transform the data.

    Returns
    -------
    dict
        Dictionary mapping algorithm names to their statistics.
    """
    data_df = performance.copy()
    if budget is not None:
        data_df = data_df.clip(upper=budget)

    if log_scale:
        # Replace 0 with NaN before log transform to avoid -inf
        data_df = np.log10(data_df.replace(0, np.nan))

    stats: dict[str, dict[str, Any]] = {}
    for algo in data_df.columns:
        values = data_df[algo].dropna().values
        if len(values) == 0:
            continue
        stats[str(algo)] = {
            "values": values,
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
        }

    return stats


def plot_performance_box(
    performance: pd.DataFrame,
    budget: float | None = None,
    log_scale: bool = False,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, dict[str, Any]]]:
    """
    Plot box plots for algorithm performance.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    budget : float or None, default=None
        Optional cutoff value to clip performance.
    log_scale : bool, default=False
        Whether to use log scale for y-axis.
    return_data : bool, default=False
        If True, also return the box plot data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, stats).
    """
    data_df = performance.copy()
    if budget is not None:
        data_df = data_df.clip(upper=budget)

    stats = compute_box_plot_data(performance, budget, log_scale=False)

    fig = go.Figure()

    for algo in data_df.columns:
        fig.add_trace(
            go.Box(
                y=data_df[algo],
                name=str(algo),
                boxpoints=False,
            )
        )

    fig.update_layout(
        title="Algorithm Performance Distribution",
        yaxis_title="Performance",
        yaxis_type="log" if log_scale else "linear",
        xaxis=dict(tickangle=45),
        width=900,
        height=600,
    )

    if return_data:
        return fig, stats
    return fig


def compute_runstatus_distribution(
    runstatus: pd.DataFrame,
    stati: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the distribution of run statuses for each algorithm.

    Parameters
    ----------
    runstatus : pd.DataFrame
        DataFrame with runstatus for each algorithm/instance.
    stati : list of str or None, default=None
        List of status values to include. Defaults to common statuses.

    Returns
    -------
    pd.DataFrame
        DataFrame with frequency of each status per algorithm.
    """
    if stati is None:
        stati = ["ok", "timeout", "memout", "not_applicable", "crash", "other"]

    n_instances = len(runstatus)

    distribution = {}
    for status in stati:
        distribution[status] = (runstatus == status).sum() / n_instances

    return pd.DataFrame(distribution)


def plot_runstatus_bar(
    runstatus: pd.DataFrame,
    stati: list[str] | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame]:
    """
    Plot stacked bar chart of algorithm run status distribution.

    Parameters
    ----------
    runstatus : pd.DataFrame
        DataFrame with runstatus for each algorithm/instance.
    stati : list of str or None, default=None
        List of status values to include.
    return_data : bool, default=False
        If True, also return the distribution data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, distribution_df).
    """
    if stati is None:
        stati = ["ok", "timeout", "memout", "not_applicable", "crash", "other"]

    distribution_df = compute_runstatus_distribution(runstatus, stati)

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
        title="Algorithm Run Status Distribution",
        xaxis_title="Algorithm",
        yaxis_title="Frequency",
        xaxis=dict(tickangle=45),
        legend_title="Status",
        width=900,
        height=600,
    )

    if return_data:
        return fig, distribution_df
    return fig


def compute_scatter_data(
    performance: pd.DataFrame,
    algo1: str,
    algo2: str,
) -> dict[str, Any]:
    """
    Compute scatter plot data for two algorithms.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    algo1 : str
        First algorithm name.
    algo2 : str
        Second algorithm name.

    Returns
    -------
    dict
        Dictionary with x, y values and statistics.
    """
    x = performance[algo1].values
    y = performance[algo2].values

    # Compute wins/ties/losses
    wins_algo1 = int(np.sum(x < y))
    wins_algo2 = int(np.sum(y < x))
    ties = int(np.sum(x == y))

    return {
        "x": x,
        "y": y,
        "algo1": algo1,
        "algo2": algo2,
        "wins_algo1": wins_algo1,
        "wins_algo2": wins_algo2,
        "ties": ties,
        "instances": performance.index.tolist(),
    }


def plot_scatter(
    performance: pd.DataFrame,
    algo1: str,
    algo2: str,
    budget: float | None = None,
    log_scale: bool = True,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot scatter plot comparing two algorithms.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    algo1 : str
        First algorithm name.
    algo2 : str
        Second algorithm name.
    budget : float or None, default=None
        Optional max value for axes.
    log_scale : bool, default=True
        Whether to use log scale.
    return_data : bool, default=False
        If True, also return the scatter data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, scatter_data).
    """
    scatter_data = compute_scatter_data(performance, algo1, algo2)

    x = scatter_data["x"]
    y = scatter_data["y"]

    if budget is None:
        budget = float(max(x.max(), y.max()))

    fig = go.Figure()

    # Diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0.001 if log_scale else 0, budget],
            y=[0.001 if log_scale else 0, budget],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
    )

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="blue", opacity=0.5),
            text=scatter_data["instances"],
            hovertemplate=f"Instance: %{{text}}<br>{algo1}: %{{x:.3f}}<br>{algo2}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{algo1} vs {algo2}",
        xaxis_title=algo1,
        yaxis_title=algo2,
        xaxis_type="log" if log_scale else "linear",
        yaxis_type="log" if log_scale else "linear",
        xaxis=dict(
            range=[np.log10(0.001), np.log10(budget)] if log_scale else [0, budget]
        ),
        yaxis=dict(
            range=[np.log10(0.001), np.log10(budget)] if log_scale else [0, budget]
        ),
        width=700,
        height=700,
    )

    if return_data:
        return fig, scatter_data
    return fig


def plot_all_scatter(
    performance: pd.DataFrame,
    budget: float | None = None,
    log_scale: bool = True,
    return_data: bool = False,
) -> (
    list[tuple[str, str, go.Figure]]
    | tuple[list[tuple[str, str, go.Figure]], list[dict[str, Any]]]
):
    """
    Generate scatter plots for all pairs of algorithms.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    budget : float or None, default=None
        Optional max value for axes.
    log_scale : bool, default=True
        Whether to use log scale.
    return_data : bool, default=False
        If True, also return the scatter data.

    Returns
    -------
    list or tuple
        List of (algo1, algo2, figure), or tuple with list of data dicts.
    """
    algos = list(performance.columns)
    plots = []
    all_data = []

    for i, algo1 in enumerate(algos):
        for algo2 in algos[i + 1 :]:
            if return_data:
                fig, data = plot_scatter(
                    performance, algo1, algo2, budget, log_scale, return_data=True
                )
                all_data.append(data)
            else:
                fig = plot_scatter(performance, algo1, algo2, budget, log_scale)
            plots.append((algo1, algo2, fig))

    if return_data:
        return plots, all_data
    return plots


def compute_contribution_values(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute contribution values.

    Computes average performance, marginal contribution, and Shapley values.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether to maximize (True) or minimize (False) performance.
    budget : float or None, default=None
        Optional cutoff time for runtime scenarios.

    Returns
    -------
    dict
        Dictionary with 'averages', 'marginals', and 'shapleys' sub-dictionaries.
    """
    algorithms = list(performance.columns)
    instances = list(performance.index)

    perf_data = performance.copy()
    if maximize:
        perf_data = perf_data * -1

    max_perf = float(perf_data.max().max())

    # Metric function: higher is better
    def metric(algo: str, inst: Any) -> float:
        if budget is not None:
            perf = budget - min(budget, perf_data.loc[inst, algo])
            return float(perf)
        else:
            return float(max_perf - perf_data.loc[inst, algo])

    # Compute Shapley values
    shapleys = _compute_vbs_shapley(instances, algorithms, metric)

    # Compute marginal contributions
    def get_vbs(data: pd.DataFrame) -> float:
        return float(data.min(axis=1).mean())

    all_vbs = get_vbs(perf_data)
    marginals = {}
    for algo in algorithms:
        remaining = [a for a in algorithms if a != algo]
        rem_vbs = get_vbs(perf_data[remaining])
        marginals[str(algo)] = float(rem_vbs - all_vbs)

    # Compute average performance
    averages = {}
    for algo in algorithms:
        averages[str(algo)] = float(performance[algo].mean())

    return {
        "averages": averages,
        "marginals": marginals,
        "shapleys": shapleys,
    }


def _compute_vbs_shapley(
    instances: list[Any],
    algorithms: list[str],
    metric: Callable[[str, Any], float],
) -> dict[str, float]:
    """
    Compute Shapley values for algorithms in the VBS game.

    Based on the algorithm by Alexandre Frechette et al.

    Parameters
    ----------
    instances : list
        List of instance names.
    algorithms : list of str
        List of algorithm names.
    metric : Callable
        Function (algo, instance) -> float, higher is better.

    Returns
    -------
    dict
        Dictionary mapping algorithm names to their Shapley values.
    """
    n = len(algorithms)
    m = len(instances)

    shapleys: dict[str, float] = {}

    for instance in instances:
        # Sort algorithms from worst to best for this instance
        instance_algorithms = sorted(algorithms, key=lambda a: metric(a, instance))

        for i, ialgorithm in enumerate(instance_algorithms):
            pos = 1
            neg = n - i - 1

            metricvalue = metric(ialgorithm, instance)
            value = metricvalue / float(m)

            # Shapley value for positive literals
            pos_shap = value / float(pos * comb(pos + neg, neg, exact=True))

            # Shapley value for negative literals
            if neg > 0:
                neg_shap = -value / float(neg * comb(pos + neg, pos, exact=True))
            else:
                neg_shap = 0.0

            # Update Shapley values
            for j in range(i, len(instance_algorithms)):
                jalgorithm = instance_algorithms[j]

                if jalgorithm not in shapleys:
                    shapleys[jalgorithm] = 0.0

                if j == i:
                    shapleys[jalgorithm] += pos_shap
                else:
                    shapleys[jalgorithm] += neg_shap

    return shapleys


def plot_contribution_pie(
    performance: pd.DataFrame,
    contribution_type: str = "shapleys",
    maximize: bool = False,
    budget: float | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, dict[str, float]]]:
    """
    Plot pie chart of algorithm contributions.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    contribution_type : str, default="shapleys"
        Type of contribution ('averages', 'marginals', or 'shapleys').
    maximize : bool, default=False
        Whether to maximize performance.
    budget : float or None, default=None
        Optional cutoff time.
    return_data : bool, default=False
        If True, also return the contribution data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, contributions).
    """
    contributions = compute_contribution_values(performance, maximize, budget)
    data = contributions[contribution_type]

    # Normalize for pie chart
    labels = list(data.keys())
    values = list(data.values())
    min_val = min(values) if values else 0.0
    if min_val < 0:
        # Shift to positive for pie chart
        values = [v - min_val + 0.001 for v in values]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.4f}<extra></extra>",
            )
        ]
    )

    titles = {
        "averages": "Average Performance",
        "marginals": "Marginal Contribution",
        "shapleys": "Shapley Values",
    }

    fig.update_layout(
        title=f"Algorithm Contribution ({titles.get(contribution_type, contribution_type)})",
        width=800,
        height=600,
    )

    if return_data:
        return fig, contributions
    return fig


def compute_instance_hardness(
    performance: pd.DataFrame,
    features: pd.DataFrame,
    maximize: bool = False,
    eps: float = 0.05,
    runstatus: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compute instance hardness based on VBS overlap.

    Hardness is defined by the number of algorithms within eps% of VBS performance.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    features : pd.DataFrame
        DataFrame with feature values for PCA projection.
    maximize : bool, default=False
        Whether to maximize performance.
    eps : float, default=0.05
        Threshold percentage from VBS performance.
    runstatus : pd.DataFrame or None, default=None
        Optional DataFrame with run status.

    Returns
    -------
    dict
        Dictionary with hardness data and 2D PCA coordinates.
    """
    # Fill missing features
    features_filled = features.fillna(features.mean())

    # Scale and PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_filled.values)

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    # Compute VBS performance
    if maximize:
        vbs_perf = performance.max(axis=1)
    else:
        vbs_perf = performance.min(axis=1)

    algorithms = list(performance.columns)
    instances = list(performance.index)

    # Count how many algorithms solve each instance within eps% of VBS
    hardness = pd.Series(0, index=instances)

    for algo in algorithms:
        algo_perf = performance[algo]

        if maximize:
            threshold = vbs_perf * (1 - eps)
            within_threshold = algo_perf >= threshold
        else:
            threshold = vbs_perf * (1 + eps)
            within_threshold = algo_perf <= threshold

        # Also check runstatus if provided
        if runstatus is not None and algo in runstatus.columns:
            within_threshold = within_threshold & (runstatus[algo] == "ok")

        hardness.loc[within_threshold[within_threshold].index] += 1

    return {
        "hardness": hardness.values,
        "features_2d": features_2d,
        "instances": instances,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "n_algorithms": len(algorithms),
    }


def plot_instance_hardness(
    performance: pd.DataFrame,
    features: pd.DataFrame,
    maximize: bool = False,
    eps: float = 0.05,
    runstatus: pd.DataFrame | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot instances in 2D PCA space colored by hardness.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    features : pd.DataFrame
        DataFrame with feature values.
    maximize : bool, default=False
        Whether to maximize performance.
    eps : float, default=0.05
        Threshold percentage from VBS.
    runstatus : pd.DataFrame or None, default=None
        Optional DataFrame with run status.
    return_data : bool, default=False
        If True, also return the hardness data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, hardness_data).
    """
    hardness_data = compute_instance_hardness(
        performance, features, maximize, eps, runstatus
    )

    features_2d = hardness_data["features_2d"]
    hardness = hardness_data["hardness"]
    instances = hardness_data["instances"]
    n_algos = hardness_data["n_algorithms"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            mode="markers",
            marker=dict(
                color=hardness,
                colorscale="Jet",
                cmin=0,
                cmax=n_algos,
                showscale=True,
                colorbar=dict(title="# Algorithms<br>within ε%"),
            ),
            text=instances,
            hovertemplate="Instance: %{text}<br>Hardness: %{marker.color}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )

    explained_var = hardness_data["pca_explained_variance"]
    fig.update_layout(
        title=f"Instance Hardness (ε={eps * 100:.0f}%)",
        xaxis_title=f"PC1 ({explained_var[0] * 100:.1f}% var)",
        yaxis_title=f"PC2 ({explained_var[1] * 100:.1f}% var)",
        width=800,
        height=600,
    )

    if return_data:
        return fig, hardness_data
    return fig


def compute_algorithm_footprint(
    performance: pd.DataFrame,
    features: pd.DataFrame,
    algorithm: str,
    maximize: bool = False,
    eps: float = 0.05,
    runstatus: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compute the footprint of an algorithm in feature space.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    features : pd.DataFrame
        DataFrame with feature values.
    algorithm : str
        Name of the algorithm.
    maximize : bool, default=False
        Whether to maximize performance.
    eps : float, default=0.05
        Threshold percentage from VBS.
    runstatus : pd.DataFrame or None, default=None
        Optional DataFrame with run status.

    Returns
    -------
    dict
        Dictionary with footprint data and 2D coordinates.
    """
    # Fill missing features
    features_filled = features.fillna(features.mean())

    # Scale and PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_filled.values)

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    features_df = pd.DataFrame(
        features_2d, index=features.index, columns=pd.Index([0, 1])
    )

    # Compute VBS performance
    if maximize:
        vbs_perf = performance.max(axis=1)
        threshold = vbs_perf * (1 - eps)
        footprint = performance[algorithm] >= threshold
    else:
        vbs_perf = performance.min(axis=1)
        threshold = vbs_perf * (1 + eps)
        footprint = performance[algorithm] <= threshold

    # Check runstatus if provided
    if runstatus is not None and algorithm in runstatus.columns:
        footprint = footprint & (runstatus[algorithm] == "ok")

    ok_instances = footprint[footprint].index.tolist()
    not_ok_instances = footprint[~footprint].index.tolist()

    return {
        "algorithm": algorithm,
        "footprint": footprint,
        "ok_instances": ok_instances,
        "not_ok_instances": not_ok_instances,
        "features_2d": features_df,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "coverage": len(ok_instances) / len(footprint),
    }


def plot_algorithm_footprint(
    performance: pd.DataFrame,
    features: pd.DataFrame,
    algorithm: str,
    maximize: bool = False,
    eps: float = 0.05,
    runstatus: pd.DataFrame | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot algorithm footprint in 2D PCA feature space.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    features : pd.DataFrame
        DataFrame with feature values.
    algorithm : str
        Name of the algorithm.
    maximize : bool, default=False
        Whether to maximize performance.
    eps : float, default=0.05
        Threshold percentage from VBS.
    runstatus : pd.DataFrame or None, default=None
        Optional DataFrame with run status.
    return_data : bool, default=False
        If True, also return the footprint data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, footprint_data).
    """
    footprint_data = compute_algorithm_footprint(
        performance, features, algorithm, maximize, eps, runstatus
    )

    features_2d = footprint_data["features_2d"]
    ok_instances = footprint_data["ok_instances"]
    not_ok_instances = footprint_data["not_ok_instances"]

    fig = go.Figure()

    # Plot non-footprint instances
    if not_ok_instances:
        coords_not_ok = features_2d.loc[not_ok_instances]
        fig.add_trace(
            go.Scatter(
                x=coords_not_ok[0],
                y=coords_not_ok[1],
                mode="markers",
                marker=dict(color="black", opacity=0.5, size=8),
                name="Outside footprint",
                text=not_ok_instances,
                hovertemplate="Instance: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

    # Plot footprint instances
    if ok_instances:
        coords_ok = features_2d.loc[ok_instances]
        fig.add_trace(
            go.Scatter(
                x=coords_ok[0],
                y=coords_ok[1],
                mode="markers",
                marker=dict(color="red", opacity=0.7, size=8),
                name="Within footprint",
                text=ok_instances,
                hovertemplate="Instance: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

    explained_var = footprint_data["pca_explained_variance"]
    coverage = footprint_data["coverage"]
    fig.update_layout(
        title=f"Footprint of {algorithm} (coverage: {coverage * 100:.1f}%)",
        xaxis_title=f"PC1 ({explained_var[0] * 100:.1f}% var)",
        yaxis_title=f"PC2 ({explained_var[1] * 100:.1f}% var)",
        width=800,
        height=600,
    )

    if return_data:
        return fig, footprint_data
    return fig


def summarize_performance(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float | None = None,
    runstatus: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive summary of performance data.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether to maximize performance.
    budget : float or None, default=None
        Optional cutoff value.
    runstatus : pd.DataFrame or None, default=None
        Optional DataFrame with run status.

    Returns
    -------
    dict
        Dictionary with summary statistics and information.
    """
    baselines = compute_baselines(performance, maximize, budget, runstatus)

    summary: dict[str, Any] = {
        "n_instances": len(performance),
        "n_algorithms": len(performance.columns),
        "algorithm_names": performance.columns.tolist(),
        "baselines": baselines,
        "algorithm_statistics": {},
    }

    for algo in performance.columns:
        summary["algorithm_statistics"][str(algo)] = {
            "mean": float(performance[algo].mean()),
            "std": float(performance[algo].std()),
            "min": float(performance[algo].min()),
            "max": float(performance[algo].max()),
            "median": float(performance[algo].median()),
        }

    return summary


# =============================================================================
# Critical Distance Analysis
# =============================================================================


def compute_critical_distance(
    performance: pd.DataFrame,
    alpha: float = 0.05,
    maximize: bool = False,
) -> dict[str, Any]:
    """
    Compute critical distance for comparing algorithm rankings.

    Uses the Nemenyi test. The critical distance (CD) is used to determine
    if the difference in average ranks between two algorithms is statistically
    significant. Based on the Friedman test followed by Nemenyi post-hoc test.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values, rows are instances, columns are algorithms.
    alpha : float, default=0.05
        Significance level (0.05 or 0.1 supported).
    maximize : bool, default=False
        Whether higher performance is better (True) or lower is better (False).

    Returns
    -------
    dict
        Dictionary containing CD results and statistics.
    """
    n_instances = len(performance)
    n_algorithms = len(performance.columns)
    algorithms = list(performance.columns)

    # Compute ranks for each instance (row)
    # For minimization: rank 1 = best (lowest value)
    # For maximization: rank 1 = best (highest value)
    if maximize:
        ranks = performance.rank(axis=1, ascending=False)
    else:
        ranks = performance.rank(axis=1, ascending=True)

    # Average ranks across all instances
    average_ranks = ranks.mean(axis=0)

    # Friedman test
    rank_matrix = ranks.values
    try:
        friedman_stat, friedman_p = friedmanchisquare(
            *[rank_matrix[:, i] for i in range(n_algorithms)]
        )
    except Exception:
        friedman_stat, friedman_p = np.nan, np.nan

    # Critical values for Nemenyi test (q_alpha values)
    # These are from the studentized range distribution
    # Source: Demsar (2006) - Statistical Comparisons of Classifiers over Multiple Data Sets
    q_alpha_table = {
        0.05: {
            2: 1.960,
            3: 2.343,
            4: 2.569,
            5: 2.728,
            6: 2.850,
            7: 2.949,
            8: 3.031,
            9: 3.102,
            10: 3.164,
            11: 3.219,
            12: 3.268,
            13: 3.313,
            14: 3.354,
            15: 3.391,
            16: 3.426,
            17: 3.458,
            18: 3.489,
            19: 3.517,
            20: 3.544,
        },
        0.1: {
            2: 1.645,
            3: 2.052,
            4: 2.291,
            5: 2.459,
            6: 2.589,
            7: 2.693,
            8: 2.780,
            9: 2.855,
            10: 2.920,
            11: 2.978,
            12: 3.030,
            13: 3.077,
            14: 3.120,
            15: 3.159,
            16: 3.196,
            17: 3.230,
            18: 3.261,
            19: 3.291,
            20: 3.319,
        },
    }

    # Get q_alpha value (use closest available if exact k not in table)
    if alpha not in q_alpha_table:
        alpha = 0.05  # Default to 0.05

    q_table = q_alpha_table[alpha]
    if n_algorithms in q_table:
        q_alpha = q_table[n_algorithms]
    elif n_algorithms > 20:
        # Approximate for larger k using asymptotic formula
        q_alpha = q_table[20] + 0.03 * (n_algorithms - 20)
    else:
        q_alpha = q_table[min(q_table.keys(), key=lambda x: abs(x - n_algorithms))]

    # Critical distance
    cd = float(q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_instances)))

    # Find significant differences
    significant_pairs = []
    for i, algo1 in enumerate(algorithms):
        for algo2 in algorithms[i + 1 :]:
            rank_diff = abs(average_ranks[algo1] - average_ranks[algo2])
            if rank_diff > cd:
                significant_pairs.append((str(algo1), str(algo2), float(rank_diff)))

    return {
        "average_ranks": average_ranks.to_dict(),
        "critical_distance": cd,
        "n_algorithms": n_algorithms,
        "n_instances": n_instances,
        "alpha": alpha,
        "q_alpha": q_alpha,
        "friedman_statistic": float(friedman_stat),
        "friedman_pvalue": float(friedman_p),
        "significant_differences": significant_pairs,
    }


def plot_critical_distance(
    performance: pd.DataFrame,
    alpha: float = 0.05,
    maximize: bool = False,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot a Critical Distance (CD) diagram.

    The CD diagram shows algorithms ordered by their average rank, with a horizontal
    bar indicating the critical distance. Algorithms connected by a bar are not
    significantly different.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    alpha : float, default=0.05
        Significance level for the Nemenyi test.
    maximize : bool, default=False
        Whether higher performance is better.
    return_data : bool, default=False
        If True, also return the CD data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, cd_data).
    """
    cd_data = compute_critical_distance(performance, alpha, maximize)

    avg_ranks = cd_data["average_ranks"]
    cd = cd_data["critical_distance"]

    # Sort algorithms by average rank
    sorted_algos = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])

    n_algos = len(sorted_algos)

    fig = go.Figure()

    # Draw the main axis line
    fig.add_trace(
        go.Scatter(
            x=[0.5, n_algos + 0.5],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    # Draw tick marks and labels for ranks
    for i in range(1, n_algos + 1):
        fig.add_trace(
            go.Scatter(
                x=[i, i],
                y=[-0.1, 0.1],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=i,
            y=-0.3,
            text=str(i),
            showarrow=False,
            font=dict(size=10),
        )

    # Draw CD bar at the top
    fig.add_trace(
        go.Scatter(
            x=[1, 1 + cd],
            y=[1.5, 1.5],
            mode="lines",
            line=dict(color="red", width=3),
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=1 + cd / 2,
        y=1.8,
        text=f"CD = {cd:.3f}",
        showarrow=False,
        font=dict(size=12, color="red"),
    )

    # Position algorithms - split into top and bottom
    top_algos = sorted_algos[: n_algos // 2 + n_algos % 2]
    bottom_algos = sorted_algos[n_algos // 2 + n_algos % 2 :]

    # Draw top algorithms (left side, pointing down)
    for i, algo in enumerate(top_algos):
        rank = avg_ranks[algo]
        y_pos = 0.8 + i * 0.4

        # Line from rank to algorithm name
        fig.add_trace(
            go.Scatter(
                x=[rank, rank],
                y=[0, y_pos - 0.1],
                mode="lines",
                line=dict(color="blue", width=1),
                showlegend=False,
            )
        )

        fig.add_annotation(
            x=rank,
            y=y_pos,
            text=f"{algo} ({rank:.2f})",
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
        )

    # Draw bottom algorithms (right side, pointing up)
    for i, algo in enumerate(bottom_algos):
        rank = avg_ranks[algo]
        y_pos = -0.8 - i * 0.4

        # Line from rank to algorithm name
        fig.add_trace(
            go.Scatter(
                x=[rank, rank],
                y=[0, y_pos + 0.1],
                mode="lines",
                line=dict(color="blue", width=1),
                showlegend=False,
            )
        )

        fig.add_annotation(
            x=rank,
            y=y_pos,
            text=f"{algo} ({rank:.2f})",
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
        )

    # Draw connecting bars for non-significant differences (cliques)
    # Find cliques of algorithms that are not significantly different
    cliques = _find_cd_cliques(sorted_algos, avg_ranks, cd)

    bar_y = 0.3
    for clique in cliques:
        if len(clique) > 1:
            min_rank = min(avg_ranks[a] for a in clique)
            max_rank = max(avg_ranks[a] for a in clique)
            fig.add_trace(
                go.Scatter(
                    x=[min_rank, max_rank],
                    y=[bar_y, bar_y],
                    mode="lines",
                    line=dict(color="black", width=4),
                    showlegend=False,
                )
            )
            bar_y += 0.15

    # Layout
    fig.update_layout(
        title=f"Critical Distance Diagram (α={alpha})",
        xaxis=dict(
            title="Average Rank",
            range=[0.3, n_algos + 0.7],
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-2 - len(bottom_algos) * 0.4, 2.5 + len(top_algos) * 0.4],
        ),
        width=900,
        height=500 + n_algos * 20,
        showlegend=False,
    )

    if return_data:
        return fig, cd_data
    return fig


def _find_cd_cliques(
    sorted_algos: list[str],
    avg_ranks: dict[str, float],
    cd: float,
) -> list[list[str]]:
    """
    Find cliques of algorithms that are not significantly different.

    Uses a greedy approach to find maximal cliques.

    Parameters
    ----------
    sorted_algos : list of str
        List of algorithm names sorted by average rank.
    avg_ranks : dict
        Average rank of each algorithm.
    cd : float
        Critical distance.

    Returns
    -------
    list of list of str
        List of cliques (groups of non-significantly different algorithms).
    """
    cliques = []
    used = set()

    for algo in sorted_algos:
        if algo in used:
            continue

        # Start a new clique with this algorithm
        clique = [algo]
        clique_max_rank = avg_ranks[algo]
        clique_min_rank = avg_ranks[algo]

        for other in sorted_algos:
            if other == algo or other in used:
                continue

            other_rank = avg_ranks[other]
            # Check if adding this algorithm keeps all pairs within CD
            can_add = True
            for c_algo in clique:
                if abs(avg_ranks[c_algo] - other_rank) > cd:
                    can_add = False
                    break

            if can_add:
                clique.append(other)
                clique_max_rank = max(clique_max_rank, other_rank)
                clique_min_rank = min(clique_min_rank, other_rank)

        if len(clique) > 1:
            cliques.append(clique)
            # Don't mark as used - algorithms can be in multiple cliques

    # Remove duplicate/subset cliques
    unique_cliques = []
    for clique in cliques:
        clique_set = set(clique)
        is_subset = False
        for other in unique_cliques:
            if clique_set <= set(other):
                is_subset = True
                break
        if not is_subset:
            # Remove any existing cliques that are subsets of this one
            unique_cliques = [c for c in unique_cliques if not set(c) <= clique_set]
            unique_cliques.append(clique)

    return unique_cliques


# =============================================================================
# Algorithm Similarity Analysis
# =============================================================================


def compute_algorithm_similarity(
    performance: pd.DataFrame,
    maximize: bool = False,
) -> dict[str, Any]:
    """
    Compute algorithm similarity based on win/loss/tie analysis.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether higher performance is better.

    Returns
    -------
    dict
        Dictionary containing win/tie/similarity matrices and dominance scores.
    """
    algorithms = list(performance.columns)
    n_algos = len(algorithms)
    n_instances = len(performance)

    # Initialize matrices
    wins = np.zeros((n_algos, n_algos))
    ties = np.zeros((n_algos, n_algos))

    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i == j:
                continue

            perf1 = performance[algo1].values
            perf2 = performance[algo2].values

            if maximize:
                wins[i, j] = np.sum(perf1 > perf2)
                ties[i, j] = np.sum(perf1 == perf2)
            else:
                wins[i, j] = np.sum(perf1 < perf2)
                ties[i, j] = np.sum(perf1 == perf2)

    win_matrix = pd.DataFrame(
        wins, index=pd.Index(algorithms), columns=pd.Index(algorithms)
    )
    tie_matrix = pd.DataFrame(
        ties, index=pd.Index(algorithms), columns=pd.Index(algorithms)
    )

    # Similarity: based on how often algorithms agree
    similarity = np.zeros((n_algos, n_algos))
    for i in range(n_algos):
        for j in range(n_algos):
            if i == j:
                similarity[i, j] = 1.0
            else:
                # Similarity based on symmetric difference in wins
                diff = abs(wins[i, j] - wins[j, i])
                similarity[i, j] = 1 - diff / n_instances

    similarity_matrix = pd.DataFrame(
        similarity, index=pd.Index(algorithms), columns=pd.Index(algorithms)
    )

    # Dominance score: average win rate against all other algorithms
    dominance = {}
    for i, algo in enumerate(algorithms):
        total_wins = wins[i, :].sum()
        total_comparisons = n_instances * (n_algos - 1)
        dominance[str(algo)] = (
            float(total_wins / total_comparisons) if total_comparisons > 0 else 0.0
        )

    return {
        "win_matrix": win_matrix,
        "tie_matrix": tie_matrix,
        "similarity_matrix": similarity_matrix,
        "dominance_scores": dominance,
        "n_instances": n_instances,
    }


def plot_algorithm_similarity_heatmap(
    performance: pd.DataFrame,
    maximize: bool = False,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot heatmap of algorithm similarity.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether higher performance is better.
    return_data : bool, default=False
        If True, also return the similarity data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, similarity_data).
    """
    similarity_data = compute_algorithm_similarity(performance, maximize)
    similarity_matrix = similarity_data["similarity_matrix"]

    fig = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix.values,
            x=similarity_matrix.columns.tolist(),
            y=similarity_matrix.index.tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Similarity"),
        )
    )

    fig.update_layout(
        title="Algorithm Similarity (based on pairwise comparisons)",
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        width=800,
        height=800,
    )

    if return_data:
        return fig, similarity_data
    return fig


def plot_algorithm_win_matrix(
    performance: pd.DataFrame,
    maximize: bool = False,
    normalize: bool = True,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot heatmap of win matrix.

    Shows how often row algorithm beats column algorithm.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether higher performance is better.
    normalize : bool, default=True
        If True, show as percentage instead of counts.
    return_data : bool, default=False
        If True, also return the similarity data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, similarity_data).
    """
    similarity_data = compute_algorithm_similarity(performance, maximize)
    win_matrix = similarity_data["win_matrix"]

    if normalize:
        win_matrix = win_matrix / similarity_data["n_instances"] * 100
        colorbar_title = "Win %"
        title = "Algorithm Win Rate Matrix (%)"
    else:
        colorbar_title = "Wins"
        title = "Algorithm Win Matrix (counts)"

    fig = go.Figure(
        data=go.Heatmap(
            z=win_matrix.values,
            x=win_matrix.columns.tolist(),
            y=win_matrix.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title=colorbar_title),
            text=np.round(win_matrix.values, 1),
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Column algorithm", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Row algorithm (winner)", tickfont=dict(size=10)),
        width=800,
        height=800,
    )

    if return_data:
        return fig, similarity_data
    return fig


def plot_algorithm_dendrogram(
    performance: pd.DataFrame,
    maximize: bool = False,
    method: str = "ward",
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, dict[str, Any]]:
    """
    Plot dendrogram showing hierarchical clustering of algorithms.

    Parameters
    ----------
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether higher performance is better.
    method : str, default="ward"
        Linkage method ('ward', 'single', 'complete', 'average').
    return_data : bool, default=False
        If True, also return the clustering data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, clustering_data).
    """
    similarity_data = compute_algorithm_similarity(performance, maximize)
    similarity_matrix = similarity_data["similarity_matrix"]

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix.values

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    link = linkage(condensed_dist, method=method)

    algorithms = list(similarity_matrix.columns)

    # Create dendrogram data
    # Capture dendrogram data
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    dendro_data = scipy_dendrogram(link, labels=algorithms, no_plot=True)
    sys.stdout = old_stdout

    # Build plotly figure
    fig = go.Figure()

    # Draw the dendrogram lines
    icoord = dendro_data["icoord"]
    dcoord = dendro_data["dcoord"]

    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )

    # Add algorithm labels
    leaves = dendro_data["leaves"]
    leaf_labels = [algorithms[i] for i in leaves]

    fig.update_layout(
        title=f"Algorithm Clustering Dendrogram ({method} linkage)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(5, len(algorithms) * 10, 10)),
            ticktext=leaf_labels,
            tickangle=45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(title="Distance (1 - similarity)"),
        width=900,
        height=600,
    )

    clustering_data = {
        "similarity_data": similarity_data,
        "linkage": link,
        "dendrogram": dendro_data,
        "algorithm_order": leaf_labels,
    }

    if return_data:
        return fig, clustering_data
    return fig


# =============================================================================
# Feature-Performance Correlation Analysis
# =============================================================================


def compute_feature_performance_correlation(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute correlation between features and algorithm performance.

    For each feature-algorithm pair, computes the correlation coefficient.
    This helps identify which features are predictive of algorithm performance.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with feature values, rows are instances, columns are features.
    performance : pd.DataFrame
        DataFrame with performance values, rows are instances, columns are algorithms.
    method : str, default="spearman"
        Correlation method ('spearman' or 'pearson').

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation coefficients.
    """
    # Ensure same instances
    common_instances = features.index.intersection(performance.index)
    features_sub = features.loc[common_instances]
    performance_sub = performance.loc[common_instances]

    # Fill missing features
    features_filled = features_sub.fillna(features_sub.mean())

    feature_names = list(features_filled.columns)
    algorithm_names = list(performance_sub.columns)

    correlations = np.zeros((len(feature_names), len(algorithm_names)))

    for i, feat in enumerate(feature_names):
        feat_values = features_filled[feat].values
        for j, algo in enumerate(algorithm_names):
            algo_values = performance_sub[algo].values

            if method == "spearman":
                rho, _ = spearmanr(feat_values, algo_values)
            else:
                rho = np.corrcoef(feat_values, algo_values)[0, 1]

            if np.isnan(rho):
                rho = 0.0
            correlations[i, j] = rho

    return pd.DataFrame(
        correlations, index=pd.Index(feature_names), columns=pd.Index(algorithm_names)
    )


def compute_feature_performance_difference_correlation(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    algo1: str,
    algo2: str,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute correlation with performance difference between two algorithms.

    This identifies which features predict when one algorithm outperforms another.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with feature values.
    performance : pd.DataFrame
        DataFrame with performance values.
    algo1 : str
        First algorithm name.
    algo2 : str
        Second algorithm name.
    method : str, default="spearman"
        Correlation method ('spearman' or 'pearson').

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation coefficients and p-values.
    """
    # Ensure same instances
    common_instances = features.index.intersection(performance.index)
    features_sub = features.loc[common_instances]
    performance_sub = performance.loc[common_instances]

    # Fill missing features
    features_filled = features_sub.fillna(features_sub.mean())

    # Compute performance difference
    perf_diff = performance_sub[algo1] - performance_sub[algo2]

    results = []
    for feat in features_filled.columns:
        feat_values = features_filled[feat].values

        if method == "spearman":
            rho, pval = spearmanr(feat_values, perf_diff.values)
        else:
            rho, pval = pearsonr(feat_values, perf_diff.values)

        if np.isnan(rho):
            rho, pval = 0.0, 1.0

        results.append(
            {
                "feature": feat,
                "correlation": float(rho),
                "p_value": float(pval),
                "abs_correlation": float(abs(rho)),
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("abs_correlation", ascending=False)

    return result_df


def plot_feature_performance_correlation(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    method: str = "spearman",
    top_n: int | None = None,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame]:
    """
    Plot heatmap of feature-performance correlations.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with feature values.
    performance : pd.DataFrame
        DataFrame with performance values.
    method : str, default="spearman"
        Correlation method ('spearman' or 'pearson').
    top_n : int or None, default=None
        If specified, only show top N features by max absolute correlation.
    return_data : bool, default=False
        If True, also return the correlation data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, correlation_df).
    """
    correlation_df = compute_feature_performance_correlation(
        features, performance, method
    )

    if top_n is not None and len(correlation_df) > top_n:
        # Select features with highest max absolute correlation
        max_abs_corr = correlation_df.abs().max(axis=1)
        top_features = max_abs_corr.nlargest(top_n).index
        correlation_df = correlation_df.loc[top_features]

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_df.columns.tolist(),
            y=correlation_df.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=f"Feature-Performance Correlation ({method.capitalize()})",
        xaxis=dict(title="Algorithm", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Feature", tickfont=dict(size=8)),
        width=900,
        height=max(500, 20 * len(correlation_df)),
    )

    if return_data:
        return fig, correlation_df
    return fig


def compute_feature_algorithm_selection_correlation(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    maximize: bool = False,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute correlation between features and algorithm selection.

    For each feature-algorithm pair, computes correlation with a binary indicator
    of whether that algorithm is the best for each instance.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with feature values.
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether higher performance is better.
    method : str, default="spearman"
        Correlation method ('spearman' or 'pearson').

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation coefficients.
    """
    # Ensure same instances
    common_instances = features.index.intersection(performance.index)
    features_sub = features.loc[common_instances]
    performance_sub = performance.loc[common_instances]

    # Fill missing features
    features_filled = features_sub.fillna(features_sub.mean())

    # Compute best algorithm for each instance
    if maximize:
        best_algo = performance_sub.idxmax(axis=1)
    else:
        best_algo = performance_sub.idxmin(axis=1)

    feature_names = list(features_filled.columns)
    algorithm_names = list(performance_sub.columns)

    correlations = np.zeros((len(feature_names), len(algorithm_names)))

    for i, feat in enumerate(feature_names):
        feat_values = features_filled[feat].values
        for j, algo in enumerate(algorithm_names):
            # Binary indicator: is this algorithm the best?
            is_best = (best_algo == algo).astype(int).values

            if method == "spearman":
                rho, _ = spearmanr(feat_values, is_best)
            else:
                rho = np.corrcoef(feat_values, is_best)[0, 1]

            if np.isnan(rho):
                rho = 0.0
            correlations[i, j] = rho

    return pd.DataFrame(
        correlations, index=pd.Index(feature_names), columns=pd.Index(algorithm_names)
    )


def plot_feature_selection_predictors(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    maximize: bool = False,
    method: str = "spearman",
    top_n: int = 15,
    return_data: bool = False,
) -> go.Figure | tuple[go.Figure, pd.DataFrame]:
    """
    Plot features most correlated with algorithm selection.

    Shows which features best predict when each algorithm is optimal.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with feature values.
    performance : pd.DataFrame
        DataFrame with performance values.
    maximize : bool, default=False
        Whether higher performance is better.
    method : str, default="spearman"
        Correlation method.
    top_n : int, default=15
        Number of top features to show.
    return_data : bool, default=False
        If True, also return the correlation data.

    Returns
    -------
    go.Figure or tuple
        Plotly Figure, or tuple of (Figure, correlation_df).
    """
    correlation_df = compute_feature_algorithm_selection_correlation(
        features, performance, maximize, method
    )

    # Select features with highest max absolute correlation
    max_abs_corr = correlation_df.abs().max(axis=1)
    top_features = max_abs_corr.nlargest(top_n).index
    correlation_df = correlation_df.loc[top_features]

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_df.columns.tolist(),
            y=correlation_df.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=f"Feature Correlation with Best Algorithm Selection ({method.capitalize()})",
        xaxis=dict(title="Algorithm", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Feature", tickfont=dict(size=10)),
        width=900,
        height=max(500, 25 * len(correlation_df)),
    )

    if return_data:
        return fig, correlation_df
    return fig
