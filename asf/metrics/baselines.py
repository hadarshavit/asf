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
    batch: bool = False,
) -> float | pd.Series:
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
    batch : bool, default=False
        If True, return one score per algorithm.

    Returns
    -------
    float or pd.Series
        The best aggregated performance value across all instances. In batch
        mode, returns each algorithm's aggregated performance.
    """
    if budget is not None and par is not None:
        performance_vals = np.where(performance <= budget, performance, budget * par)
    else:
        performance_vals = performance.values

    perf_sum = np.sum(performance_vals, axis=0)
    if batch:
        return pd.Series(perf_sum, index=performance.columns)

    if maximize:
        return float(np.max(perf_sum))
    else:
        return float(np.min(perf_sum))


def virtual_best_solver(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float | None = 5000.0,
    par: float | None = 10.0,
    batch: bool = False,
) -> float | pd.Series:
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
    batch : bool, default=False
        If True, return one score per algorithm.

    Returns
    -------
    float or pd.Series
        The sum of the best performance values for each instance. In batch
        mode, returns each algorithm's aggregated performance.
    """
    if budget is not None and par is not None:
        performance_vals = np.where(performance <= budget, performance, budget * par)
    else:
        performance_vals = performance.values

    if batch:
        return pd.Series(np.sum(performance_vals, axis=0), index=performance.columns)

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

    Raises
    ------
    ValueError
        If the schedule is invalid (e.g., total allocated time to algorithms is zero).
    """
    if feature_time is None:
        feature_time = pd.DataFrame(
            0.0,
            index=performance.index,
            columns=["feature_time"],
        )

    total_time: dict[str, float] = {}
    for instance, schedule in schedules.items():

        def is_feature(item: Any) -> bool:
            if isinstance(item, str):
                name = item
            elif isinstance(item, tuple) and len(item) == 2:
                name = item[0]
            else:
                return False
            return name in feature_time.columns

        def feature_cost(item: Any) -> float:
            name = item[0] if isinstance(item, tuple) else item
            value = feature_time.loc[instance, name]
            cost = 0.0 if pd.isna(value) else float(value)
            if isinstance(item, tuple) and item[1] is not None:
                cost = min(cost, item[1])
            return cost

        algorithm_items = [
            (item[0], item[1] if item[1] is not None else 0.0)
            for item in schedule
            if isinstance(item, tuple) and len(item) == 2 and not is_feature(item)
        ]
        if sum(allocated for _, allocated in algorithm_items) <= 0.0:
            raise ValueError(
                f"Instance {instance}: No algorithm time allocated in schedule {schedule}. "
            )

        # Legacy schedules without explicit feature steps pay feature costs upfront.
        explicit_features = any(is_feature(item) for item in schedule)
        elapsed = 0.0 if explicit_features else float(feature_time.loc[instance].sum())

        # Preserve the parallel-portfolio convention: a trailing block of algorithms
        # each receiving the full scenario budget runs concurrently. Interleaved
        # feature/algorithm steps must instead be evaluated in their stated order.
        budgets = [allocated for _, allocated in algorithm_items]
        seen_algorithm = False
        interleaved_features = False
        for item in schedule:
            if is_feature(item):
                interleaved_features |= seen_algorithm
            elif isinstance(item, tuple):
                seen_algorithm = True
        is_parallel = (
            len(algorithm_items) > 1
            and len(set(budgets)) == 1
            and budgets[0] >= budget
            and not interleaved_features
        )

        total_time[instance] = budget * par
        if is_parallel:
            elapsed += sum(feature_cost(item) for item in schedule if is_feature(item))
            times = [
                float(performance.loc[instance, algorithm])
                for algorithm, allocated in algorithm_items
                if algorithm in performance.columns
                and performance.loc[instance, algorithm] <= allocated
                and elapsed + performance.loc[instance, algorithm] <= budget
            ]
            if times:
                total_time[instance] = elapsed + min(times)
            continue

        for item in schedule:
            if is_feature(item):
                elapsed += feature_cost(item)
            elif isinstance(item, tuple) and len(item) == 2:
                algorithm, allocated = item
                if algorithm not in performance.columns:
                    continue
                available = min(allocated or 0.0, max(0.0, budget - elapsed))
                runtime = performance.loc[instance, algorithm]
                if runtime <= available and elapsed + runtime <= budget:
                    total_time[instance] = elapsed + float(runtime)
                    break
                elapsed += available
            if elapsed >= budget:
                break

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


def compute_solve_rate(
    schedules: dict[str, list[tuple[str, float] | str]],
    performance: pd.DataFrame,
    budget: float,
) -> float:
    """
    Compute the solve rate for selector predictions.

    For each instance in the schedules, determines if it was solved within budget.
    An instance is solved if at least one algorithm in the schedule completes within
    its allocated time.

    Parameters
    ----------
    schedules : dict[str, list[tuple[str, float] | str]]
        Selector predictions mapping instance_id to schedule/selections.
    performance : pd.DataFrame
        Performance data for the algorithms.
    budget : float
        The time budget for solving.

    Returns
    -------
    float
        Solve rate (fraction of instances solved within budget, 0-1).
    """
    # Get per-instance times using the performance metrics
    times_dict: dict[str, float] | float = running_time_selector_performance(
        schedules, performance, budget=budget, par=10.0, return_per_instance=True
    )

    if not isinstance(times_dict, dict):
        return 0.0

    # Count instances solved within budget (where time is not penalized)
    solved_count = sum(1 for time in times_dict.values() if time <= budget)
    total_count = len(times_dict)

    return float(solved_count / total_count) if total_count > 0 else 0.0
