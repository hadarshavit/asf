import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def single_best_solver(performance: pd.DataFrame, maximize: bool = False) -> float:
    """
    Selects the single best solver across all instances based on the aggregated performance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate (not used in this function).
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.

    Returns:
        float: The best aggregated performance value across all instances.
    """
    perf_sum = performance.sum(axis=0)
    if maximize:
        return perf_sum.max()
    else:
        return perf_sum.min()


def virtual_best_solver(performance: pd.DataFrame, maximize: bool = False) -> float:
    """
    Selects the virtual best solver for each instance by choosing the best performance per instance.

    Args:
        schedules (pd.DataFrame): The schedules to evaluate (not used in this function).
        performance (pd.DataFrame): The performance data for the algorithms.
        maximize (bool): Whether to maximize or minimize the performance.

    Returns:
        float: The sum of the best performance values for each instance.
    """
    if maximize:
        return performance.max(axis=1).sum()
    else:
        return performance.min(axis=1).sum()


def running_time_selector_performance(
    schedules: Dict[str, List[Tuple[str, float]]],
    performance: pd.DataFrame,
    budget: float = 5000,
    par: float = 10,
    feature_time: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[float, int]]:
    """
    Calculates the total running time for a selector based on the given schedules and performance data.

    Args:
        schedules (Dict[str, List[Tuple[str, float]]]): The schedules to evaluate, where each key is an instance
            and the value is a list of tuples (algorithm, allocated budget).
        performance (pd.DataFrame): The performance data for the algorithms.
        budget (float): The budget for the scenario.
        par (float): The penalization factor for unsolved instances.
        feature_time (Optional[pd.DataFrame]): The feature time data for each instance. Defaults to zero if not provided.

    Returns:
        Dict[str, Union[float, int]]: A dictionary mapping each instance to its total running time.
    """
    if feature_time is None:
        feature_time = pd.DataFrame(
            0, index=performance.index, columns=["feature_time"]
        )
    total_time = {}
    for instance, schedule in schedules.items():
        allocated_times = {algorithm: 0 for algorithm in performance.columns}
        solved = False
        for algorithm, algo_budget in schedule:
            remaining_budget = (
                budget
                - sum(allocated_times.values())
                - feature_time.loc[instance].sum().item()
            )
            remaining_time_to_solve = performance.loc[instance, algorithm] - (
                algo_budget + allocated_times[algorithm]
            )
            if remaining_time_to_solve < 0:
                allocated_times[algorithm] = performance.loc[instance, algorithm]
                solved = True
                break
            elif remaining_time_to_solve <= remaining_budget:
                allocated_times[algorithm] += remaining_time_to_solve
            else:
                allocated_times[algorithm] += remaining_budget
                break
        if solved:
            total_time[instance] = (
                sum(allocated_times.values()) + feature_time.loc[instance].sum().item()
            )
        else:
            total_time[instance] = budget * par

    total_time = sum(list(total_time.values()))
    return total_time


def running_time_closed_gap(
    schedules: Dict[str, List[Tuple[str, float]]],
    performance: pd.DataFrame,
    budget: float,
    feature_time: pd.DataFrame,
    par: float = 10,
) -> float:
    """
    Calculates the closed gap metric for a given selector.

    Args:
        schedules (Dict[str, List[Tuple[str, float]]]): The schedules to evaluate.
        performance (pd.DataFrame): The performance data for the algorithms.
        budget (float): The budget for the scenario.
        par (float): The penalization factor for unsolved instances.
        feature_time (pd.DataFrame): The feature time data for each instance.

    Returns:
        float: The closed gap value, representing the improvement of the selector over the single best solver
        relative to the virtual best solver.
    """
    sbs_val = single_best_solver(performance, False)
    vbs_val = virtual_best_solver(performance, False)
    s_val = running_time_selector_performance(
        schedules, performance, budget, par, feature_time
    )

    return (sbs_val - s_val) / (sbs_val - vbs_val)
