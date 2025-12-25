"""
PAR (Penalized Average Runtime) transformation utilities.

This module provides functions to apply PAR-k penalization to performance data,
which is essential for algorithm selection to properly penalize timeouts.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from asf.preprocessing.feature_group_selector import MissingPrerequisiteGroupError


def apply_par(
    performance: pd.DataFrame | np.ndarray,
    budget: float,
    par_factor: float = 10.0,
) -> pd.DataFrame | np.ndarray:
    """
    Apply PAR-k (Penalized Average Runtime) transformation to performance data.

    This function replaces timeout values (values > budget) with budget * par_factor.
    This is crucial for algorithm selection because raw timeout values (e.g., 1200.999)
    look almost identical to near-timeout solves (e.g., 1199), but in practice
    timeouts should be heavily penalized.

    Parameters
    ----------
    performance : pd.DataFrame or np.ndarray
        Performance data where each value represents the runtime of an algorithm
        on an instance. Values greater than the budget indicate timeouts.
    budget : float
        The algorithm cutoff time. Values exceeding this are considered timeouts.
    par_factor : float, default=10.0
        The penalization factor. Timeouts will be replaced with budget * par_factor.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Performance data with timeouts penalized. Returns the same type as the input.

    Examples
    --------
    >>> import pandas as pd
    >>> perf = pd.DataFrame({'algo1': [100, 1201, 500], 'algo2': [200, 200, 1201]})
    >>> apply_par(perf, budget=1200, par_factor=10)
       algo1   algo2
    0    100     200
    1  12000     200
    2    500   12000
    """
    if isinstance(performance, pd.DataFrame):
        result = performance.copy()
        result = result.where(result <= budget, budget * par_factor)
        return result
    else:
        return np.where(performance <= budget, performance, budget * par_factor)


def apply_par10(
    performance: pd.DataFrame | np.ndarray,
    budget: float,
) -> pd.DataFrame | np.ndarray:
    """
    Apply PAR10 (Penalized Average Runtime with factor 10) transformation.

    Convenience function that calls apply_par with par_factor=10.

    Parameters
    ----------
    performance : pd.DataFrame or np.ndarray
        Performance data.
    budget : float
        The algorithm cutoff time.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Performance data with timeouts penalized by 10x.

    See Also
    --------
    apply_par : The general PAR-k transformation function.
    """
    return apply_par(performance, budget, par_factor=10.0)


def single_best_solver(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float | None = 5000.0,
    par: float | None = 10.0,
) -> float:
    """
    Selects the single best solver across all instances based on the aggregated performance.

    Parameters
    ----------
    performance : pd.DataFrame
        The performance data for the algorithms.
    maximize : bool, default=False
        Whether to maximize or minimize the performance.
    budget : float or None, default=5000.0
        The runtime budget. If provided with par, timeouts are penalized.
    par : float or None, default=10.0
        The penalization factor for timeouts.

    Returns
    -------
    float
        The best aggregated performance value across all instances.
    """
    if budget is not None and par is not None:
        performance_vals = np.where(performance <= budget, performance, budget * par)
    else:
        performance_vals = performance.values

    perf_sum = np.sum(performance_vals, axis=0)
    if maximize:
        return float(np.max(perf_sum))
    else:
        return float(np.min(perf_sum))


def virtual_best_solver(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float | None = 5000.0,
    par: float | None = 10.0,
) -> float:
    """
    Selects the virtual best solver for each instance by choosing the best performance per instance.

    Parameters
    ----------
    performance : pd.DataFrame
        The performance data for the algorithms.
    maximize : bool, default=False
        Whether to maximize or minimize the performance.
    budget : float or None, default=5000.0
        The runtime budget. If provided with par, timeouts are penalized.
    par : float or None, default=10.0
        The penalization factor for timeouts.

    Returns
    -------
    float
        The sum of the best performance values for each instance.
    """
    if budget is not None and par is not None:
        performance_vals = np.where(performance <= budget, performance, budget * par)
    else:
        performance_vals = performance.values

    if maximize:
        return float(np.max(performance_vals, axis=1).sum())
    else:
        return float(np.min(performance_vals, axis=1).sum())


def running_time_selector_performance(
    schedules: dict[str, list[tuple[str, float] | str]],
    performance: pd.DataFrame,
    budget: float = 5000.0,
    feature_time: pd.DataFrame | None = None,
    par: float = 10.0,
    return_per_instance: bool = False,
) -> dict[str, float] | float:
    """
    Calculates the total running time for a selector based on the given schedules and performance data.

    The schedule can contain both feature groups (strings) and algorithm selections (tuples).
    Feature groups are evaluated in order, and their computation time is only added if the
    instance is not yet solved when the feature group appears in the schedule.

    Parameters
    ----------
    schedules : dict[str, list[tuple[str, float] | str]]
        The schedules to evaluate, where each key is an instance and the value is a list of items.
        Each item can be:
        - A string: the name of a feature group to compute (uses full actual time)
        - A tuple (feature_group, budget): a feature group with a time budget
        - A tuple (algorithm, budget): an algorithm to run with its allocated budget
    performance : pd.DataFrame
        The performance data for the algorithms.
    budget : float, default=5000.0
        The budget for the scenario.
    feature_time : pd.DataFrame or None, default=None
        The feature time data for each instance. Columns should be feature group names.
    par : float, default=10.0
        The penalization factor for unsolved instances.
    return_per_instance : bool, default=False
        If True, return a dict mapping instance to running time.
        If False, return the sum of all running times.

    Returns
    -------
    dict[str, float] or float
        If return_per_instance is True, returns a dictionary mapping each instance
        to its total running time. Otherwise, returns the sum of all running times.
    """
    if feature_time is None:
        feature_time = pd.DataFrame(
            0.0,
            index=performance.index,
            columns=["feature_time"],  # type: ignore[arg-type]
        )

    total_time: dict[str, float] = {}
    for instance, schedule in schedules.items():
        allocated_times = {algorithm: 0.0 for algorithm in performance.columns}
        instance_feature_time = 0.0
        # Check if schedule contains feature groups (strings or tuples where name is in feature_time.columns)
        has_feature_groups_in_schedule = any(
            isinstance(item, str)
            or (
                isinstance(item, tuple)
                and len(item) >= 2
                and item[0] in feature_time.columns
            )
            for item in schedule
        )

        # For backward compatibility: if no feature groups in schedule, add all feature time upfront
        if not has_feature_groups_in_schedule:
            instance_feature_time = float(feature_time.loc[instance].sum())

        solved = False
        for item in schedule:
            # Check if item is a feature group (string or tuple) or algorithm selection (tuple)
            if isinstance(item, str):
                # Feature group without budget: add its full computation time if available
                if item in feature_time.columns:
                    ft_val = feature_time.loc[instance, item]
                    if hasattr(ft_val, "item"):
                        ft_val = ft_val.item()
                    instance_feature_time += (
                        0.0
                        if (
                            ft_val is None
                            or (isinstance(ft_val, float) and np.isnan(ft_val))
                        )
                        else float(ft_val)
                    )
                continue

            # It's a tuple: could be (feature_group, budget) or (algorithm, budget)
            item_name, item_budget = item
            if item_name in feature_time.columns:
                # Feature group with budget: use min(actual_time, budget)
                ft_val = feature_time.loc[instance, item_name]
                if hasattr(ft_val, "item"):
                    ft_val = ft_val.item()
                actual_ft = (
                    0.0
                    if (
                        ft_val is None
                        or (isinstance(ft_val, float) and np.isnan(ft_val))
                    )
                    else float(ft_val)
                )
                instance_feature_time += min(actual_ft, item_budget or 0.0)
                continue

            # Algorithm selection: (algorithm, algo_budget)
            algorithm, algo_budget = item_name, item_budget
            if algo_budget is None:
                algo_budget = 0.0
            remaining_budget = (
                budget - sum(allocated_times.values()) - instance_feature_time
            )
            remaining_time_to_solve = performance.loc[instance, algorithm] - (
                algo_budget + allocated_times[algorithm]
            )
            if remaining_time_to_solve <= 0:
                allocated_times[algorithm] = performance.loc[instance, algorithm]
                solved = True
                break
            elif remaining_time_to_solve <= remaining_budget:
                allocated_times[algorithm] += remaining_time_to_solve
            else:
                allocated_times[algorithm] += remaining_budget
                break

        if solved:
            total_time[instance] = sum(allocated_times.values()) + instance_feature_time
        else:
            total_time[instance] = budget * par

    if return_per_instance:
        return total_time

    return float(sum(total_time.values()))


def _validate_schedule_prerequisites(
    schedules: dict[str, list[tuple[str, float] | str]],
    feature_groups: dict[str, Any],
) -> None:
    """
    Validate that feature groups in schedules have their prerequisites computed first.

    Parameters
    ----------
    schedules : dict[str, list[tuple[str, float] | str]]
        The schedules to validate.
    feature_groups : dict[str, Any]
        Feature group definitions with 'requires' information.

    Raises
    ------
    MissingPrerequisiteGroupError
        If a feature group is used without its required prerequisites appearing first.
    """
    for instance, schedule in schedules.items():
        # Extract feature groups from this schedule in order
        schedule_feature_groups = [item for item in schedule if isinstance(item, str)]

        if not schedule_feature_groups:
            continue

        # Check that prerequisites are satisfied
        computed_groups = set()
        for fg_name in schedule_feature_groups:
            if fg_name not in feature_groups:
                computed_groups.add(fg_name)
                continue

            fg_info = feature_groups[fg_name]
            required_groups = fg_info.get("requires", [])

            for required_group in required_groups:
                if required_group not in computed_groups:
                    raise MissingPrerequisiteGroupError(
                        f"Feature group '{fg_name}' requires group '{required_group}' "
                        f"to be computed first, but it was not found before '{fg_name}' "
                        f"in the schedule for instance '{instance}'."
                    )

            computed_groups.add(fg_name)


def running_time_closed_gap(
    schedules: dict[str, list[tuple[str, float] | str]],
    performance: pd.DataFrame,
    budget: float,
    feature_time: pd.DataFrame,
    par: float = 10.0,
    feature_groups: dict[str, Any] | None = None,
) -> float:
    """
    Calculates the closed gap metric for a given selector.

    Parameters
    ----------
    schedules : dict[str, list[tuple[str, float] | str]]
        The schedules to evaluate.
    performance : pd.DataFrame
        The performance data for the algorithms.
    budget : float
        The budget for the scenario.
    feature_time : pd.DataFrame
        The feature time data for each instance.
    par : float, default=10.0
        The penalization factor for unsolved instances.
    feature_groups : dict[str, Any] or None, default=None
        Feature group definitions including prerequisite information.

    Returns
    -------
    float
        The closed gap value, representing the improvement over the single best solver.
    """
    # Validate feature group prerequisites if feature_groups is provided
    if feature_groups is not None:
        _validate_schedule_prerequisites(schedules, feature_groups)

    sbs_val = single_best_solver(performance, False, budget, par)
    vbs_val = virtual_best_solver(performance, False, budget, par)
    s_val = running_time_selector_performance(
        schedules, performance, budget, feature_time, par
    )

    if isinstance(s_val, dict):
        s_val = float(sum(s_val.values()))

    denominator = sbs_val - vbs_val
    if abs(denominator) < 1e-9:
        return 0.0

    return (sbs_val - s_val) / denominator


def precision_regret(
    schedules: dict[str, list[tuple[str, float] | str]],
    performance: pd.DataFrame,
    precision_data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> float:
    """
    Computes the sum of regrets for the given schedules.

    Parameters
    ----------
    schedules : dict[str, list[tuple[str, float] | str]]
        Selector predictions mapping instance_id to schedule.
    performance : pd.DataFrame
        Ground-truth precision table.
    precision_data : pd.DataFrame or None, default=None
        Alternative precision data to use for evaluation.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    float
        Sum of regrets for the given schedules.
    """
    regrets = []
    for instance, schedule in schedules.items():
        if not schedule or instance not in performance.index:
            continue
        item = schedule[0]
        if isinstance(item, tuple):
            selected_algo, _ = item
        else:
            # Skip feature groups at the start if needed, or handle differently
            # For precision regret, we usually expect the first algorithm
            continue

        if selected_algo not in performance.columns:
            continue

        if precision_data is not None:
            selector_precision = precision_data.loc[instance, selected_algo]
        else:
            selector_precision = performance.loc[instance, selected_algo]

        regrets.append(float(selector_precision))

    if len(regrets) == 0:
        warnings.warn("No valid schedules found for regret calculation.")
        return float("inf")
    return float(np.sum(regrets))
