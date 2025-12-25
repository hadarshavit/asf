"""
Knee-of-the-curve algorithm for algorithm pre-selection.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from asf.pre_selector.abstract_pre_selector import AbstractPreSelector


class KneeOfCurvePreSelector(AbstractPreSelector):
    """
    Knee-of-the-curve algorithm for algorithm pre-selection.

    This selector identifies the "knee" (point of maximum curvature) of the
    performance profile to select an optimal number of algorithms.

    Parameters
    ----------
    metric : Callable
        A function that takes a DataFrame of performance values and returns a single value.
    base_pre_selector : type[AbstractPreSelector]
        The base pre-selector class to use for evaluating different subset sizes.
    maximize : bool, default=False
        Whether to maximize or minimize the performance metric.
    S : float, default=1.0
        Sensitivity parameter for knee detection.
    workers : int, default=1
        Number of parallel workers for evaluation.
    max_algorithms : int or None, default=None
        Maximum number of algorithms to evaluate.
    **kwargs : Any
        Additional arguments passed to the parent class.
    """

    def __init__(
        self,
        metric: Callable[[pd.DataFrame], float],
        base_pre_selector: type[AbstractPreSelector],
        maximize: bool = False,
        S: float = 1.0,
        workers: int = 1,
        max_algorithms: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.base_pre_selector = base_pre_selector
        self.maximize = maximize
        self.S = S
        self.workers = workers
        self.max_algorithms = max_algorithms

    def fit_transform(
        self,
        performance: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame | np.ndarray:
        """
        Fit the pre-selector and transform the performance data.

        Parameters
        ----------
        performance : pd.DataFrame or np.ndarray
            The performance data.

        Returns
        -------
        pd.DataFrame or np.ndarray
            The performance data with the selected algorithms.
        """
        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],  # type: ignore[arg-type]
            )
            is_numpy = True
        else:
            performance_frame = performance
            is_numpy = False

        x: list[int] = []
        y: list[float] = []
        dfs: list[pd.DataFrame] = []

        def process(i: int) -> tuple[int, float, pd.DataFrame]:
            base_selector = self.base_pre_selector(
                n_algorithms=i + 1,
                metric=self.metric,
                maximize=self.maximize,
            )
            pre_selected_df = base_selector.fit_transform(performance_frame)
            if not isinstance(pre_selected_df, pd.DataFrame):
                pre_selected_df = pd.DataFrame(pre_selected_df)
            return i, self.metric(pre_selected_df), pre_selected_df

        max_algos = (
            self.max_algorithms
            if self.max_algorithms is not None
            else performance_frame.shape[1]
        )
        max_algos = min(max_algos, performance_frame.shape[1])

        if self.workers > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.workers)(
                delayed(process)(i) for i in range(max_algos)
            )
            results.sort(key=lambda tup: tup[0])
            for i, metric_val, pre_selected_df in results:
                x.append(i)
                y.append(metric_val)
                dfs.append(pre_selected_df)
        else:
            for i in range(max_algos):
                _, metric_val, pre_selected_df = process(i)
                x.append(i)
                y.append(metric_val)
                dfs.append(pre_selected_df)

        x_arr = np.array(x)
        y_arr = np.array(y)

        if len(x_arr) < 3:
            return performance_frame.values if is_numpy else performance_frame

        norm_x = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min())
        norm_y = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min())

        norm_y = norm_y.max() - norm_y
        y_diff = norm_y - norm_x

        local_maximas = np.where((np.diff(np.sign(np.diff(y_diff))) < 0))[0] + 1
        local_minimas = np.where((np.diff(np.sign(np.diff(y_diff))) > 0))[0] + 1

        knees: list[tuple[int, float]] = []
        for i, lmxi in enumerate(local_maximas):
            Tlmxi = y_diff[lmxi] - (self.S * np.abs(np.diff(norm_x).mean()))
            next_lmxi = (
                local_maximas[i + 1] if i + 1 < len(local_maximas) else len(y_diff)
            )
            found_knee = False
            for j in range(lmxi + 1, next_lmxi):
                if y_diff[j] < Tlmxi:
                    knees.append((lmxi, norm_x[lmxi]))
                    found_knee = True
                    break
                if j in local_minimas:
                    next_lmxi = 0
            if found_knee:
                break

        if len(knees) == 0:
            return performance_frame.values if is_numpy else performance_frame

        knee_x, _ = knees[0]
        selected_performance = dfs[knee_x]

        if is_numpy:
            selected_performance = selected_performance.to_numpy()

        return selected_performance
