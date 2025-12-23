import pandas as pd
import warnings
import numpy as np

from asf.preprocessing.feature_group_selector import MissingPrerequisiteGroupError


def single_best_solver(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float = 5000.0,
    par: float = 10,
) -> float:
    """
    Selects the single best solver across all instances based on the aggregated performance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate (not used in this function).
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.

    Returns:
        float: The best aggregated performance value across all instances.
    """
    if budget is not None and par is not None:
        performance = np.where(performance <= budget, performance, budget * par)

    perf_sum = np.sum(performance, axis=0)
    if maximize:
        return np.max(perf_sum)
    else:
        return np.min(perf_sum)


def virtual_best_solver(
    performance: pd.DataFrame,
    maximize: bool = False,
    budget: float = 5000.0,
    par: float = 10,
) -> float:
    """
    Selects the virtual best solver for each instance by choosing the best performance per instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate (not used in this function).
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.

    Returns:
        float: The sum of the best performance values for each instance.
    """
    if budget is not None and par is not None:
        performance = np.where(performance <= budget, performance, budget * par)

    if maximize:
        return np.max(performance, axis=1).sum()
    else:
        return np.min(performance, axis=1).sum()


def running_time_selector_performance(
    schedules: dict[str, list[tuple[str, float] | str]],
    performance: pd.DataFrame,
    budget: float = 5000,
    feature_time: pd.DataFrame | None = None,
    par: float = 10,
    return_per_instance: bool = False,
) -> dict[str, float | int]:
    """
    Calculates the total running time for a selector based on the given schedules and performance data.

    The schedule can contain both feature groups (strings) and algorithm selections (tuples).
    Feature groups are evaluated in order, and their computation time is only added if the
    instance is not yet solved when the feature group appears in the schedule.

    If the schedule contains no feature groups (only algorithm tuples) but feature_time is provided,
    all feature time is added upfront for backward compatibility.

    Args:
        schedules (dict[str, list[tuple[str, float] | str]]): The schedules to evaluate, where each key is an instance
            and the value is a list of items. Each item can be:
            - A string: the name of a feature group to compute (uses full actual time)
            - A tuple (feature_group, budget): a feature group to compute with a time budget (uses min(actual_time, budget))
            - A tuple (algorithm, budget): an algorithm to run with its allocated budget
        performance (pd.DataFrame): The performance data for the algorithms.
        budget (float): The budget for the scenario.
        par (float): The penalization factor for unsolved instances.
        feature_time (pd.DataFrame | None): The feature time data for each instance.
            Should have columns corresponding to feature group names. Defaults to zero if not provided.
        return_per_instance (bool): If True, return dict mapping instance to running time.
            If False (default), return the sum of all running times.

    Returns:
        dict[str, float | int] | float: If return_per_instance is True, returns a dictionary mapping
            each instance to its total running time. Otherwise, returns the sum of all running times.
    """
    if feature_time is None:
        feature_time = pd.DataFrame(
            0, index=performance.index, columns=["feature_time"]
        )

    total_time = {}
    for instance, schedule in schedules.items():
        allocated_times = {algorithm: 0 for algorithm in performance.columns}
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
            instance_feature_time = feature_time.loc[instance].sum()
            if hasattr(instance_feature_time, "item"):
                instance_feature_time = instance_feature_time.item()

        solved = False
        for item in schedule:
            # Check if item is a feature group (string or tuple) or algorithm selection (tuple)
            if isinstance(item, str):
                # Feature group without budget: add its full computation time if available
                if item in feature_time.columns:
                    ft_val = feature_time.loc[instance, item]
                    # guard against NaN values in feature time
                    if hasattr(ft_val, "item"):
                        ft_val = ft_val.item()
                    instance_feature_time += (
                        0.0
                        if (
                            ft_val is None
                            or (isinstance(ft_val, float) and np.isnan(ft_val))
                        )
                        else ft_val
                    )
                continue

            # It's a tuple: could be (feature_group, budget) or (algorithm, budget)
            # Distinguish by checking if first element is in feature_time columns
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
                    else ft_val
                )
                instance_feature_time += min(actual_ft, item_budget)
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

    return sum(list(total_time.values()))


def _validate_schedule_prerequisites(
    schedules: dict[str, list[tuple[str, float] | str]],
    feature_groups: dict,
) -> None:
    """
    Validate that feature groups in schedules have their prerequisites computed first.

    For each schedule, ensures that if a feature group is used, all of its required
    prerequisite groups appear before it in the schedule.

    Args:
        schedules: The schedules to validate.
        feature_groups: Feature group definitions with 'requires' information.

    Raises:
        MissingPrerequisiteGroupError: If a feature group is used without its required
            prerequisite groups appearing first in the schedule.
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
                        f"in the schedule for instance '{instance}'. "
                        f"Schedule feature groups: {schedule_feature_groups}"
                    )

            computed_groups.add(fg_name)


def running_time_closed_gap(
    schedules: dict[str, list[tuple[str, float] | str]],
    performance: pd.DataFrame,
    budget: float,
    feature_time: pd.DataFrame,
    par: float = 10,
    feature_groups: dict | None = None,
) -> float:
    """
    Calculates the closed gap metric for a given selector.

    Args:
        schedules (dict[str, list[tuple[str, float] | str]]): The schedules to evaluate.
            Each schedule can contain feature groups (strings) and algorithm selections (tuples).
        performance (pd.DataFrame): The performance data for the algorithms.
        budget (float): The budget for the scenario.
        par (float): The penalization factor for unsolved instances.
        feature_time (pd.DataFrame): The feature time data for each instance.
        feature_groups (dict | None): Feature group definitions including prerequisite information.
            When provided, validates that schedules don't use feature groups without their
            required prerequisites appearing first in the schedule.
        max_feature_time (float | None): Deprecated parameter, kept for backward compatibility. No longer used.
            Feature group budgets are now specified directly in the schedule.

    Returns:
        float: The closed gap value, representing the improvement of the selector over the single best solver
        relative to the virtual best solver.

    Raises:
        MissingPrerequisiteGroupError: If a schedule uses a feature group without its required
            prerequisite groups appearing first in the schedule.
    """
    # Validate feature group prerequisites if feature_groups is provided
    if feature_groups is not None:
        _validate_schedule_prerequisites(schedules, feature_groups)

    sbs_val = single_best_solver(performance, False, budget, par)
    vbs_val = virtual_best_solver(performance, False, budget, par)
    s_val = running_time_selector_performance(
        schedules, performance, budget, feature_time, par
    )

    return (sbs_val - s_val) / (sbs_val - vbs_val)


def precision_regret(
    schedules: dict[str, list[tuple[str, float]]],
    performance: pd.DataFrame,
    precision_data: pd.DataFrame = None,
    **kwargs,
) -> float:
    """
    Computes the sum of regrets for the given schedules based on the provided performance data.

    Args:
        schedules (dict): selector predictions: instance_id → [(algorithm, budget)]
        performance (pd.DataFrame): ground-truth precision table

    Returns:
        float: sum of regrets for the given schedules
    """
    regrets = []
    for instance, schedule in schedules.items():
        if not schedule or instance not in performance.index:
            continue
        selected_algo, _ = schedule[0]
        if selected_algo not in performance.columns:
            continue
        if precision_data is not None:
            selector_precision = precision_data.loc[instance, selected_algo]
        else:
            selector_precision = performance.loc[instance, selected_algo]
        regret = selector_precision
        regrets.append(regret)
    if len(regrets) == 0:
        warnings.warn("No valid schedules found for regret calculation.")
        return float("inf")
    return float(np.sum(regrets))
