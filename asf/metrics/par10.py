"""
PAR (Penalized Average Runtime) transformation utilities.

This module provides functions to apply PAR-k penalization to performance data,
which is essential for algorithm selection to properly penalize timeouts.
"""

import numpy as np
import pandas as pd


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

    Args:
        performance (pd.DataFrame | np.ndarray): Performance data where each value
            represents the runtime of an algorithm on an instance. Values greater
            than the budget indicate timeouts.
        budget (float): The algorithm cutoff time. Values exceeding this are considered
            timeouts.
        par_factor (float, optional): The penalization factor. Timeouts will be
            replaced with budget * par_factor. Defaults to 10.0 (PAR10).

    Returns:
        pd.DataFrame | np.ndarray: Performance data with timeouts penalized.
            Returns the same type as the input.

    Examples:
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

    Args:
        performance (pd.DataFrame | np.ndarray): Performance data.
        budget (float): The algorithm cutoff time.

    Returns:
        pd.DataFrame | np.ndarray: Performance data with timeouts penalized by 10x.

    See Also:
        apply_par: The general PAR-k transformation function.
    """
    return apply_par(performance, budget, par_factor=10.0)
