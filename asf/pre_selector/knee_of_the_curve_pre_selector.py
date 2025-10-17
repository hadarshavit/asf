from asf.pre_selector.abstract_pre_selector import AbstractPreSelector
import pandas as pd
import numpy as np
from typing import Callable


class KneeOfCurvePreSelector(AbstractPreSelector):
    def __init__(
        self,
        metric: Callable,
        base_pre_selector: type[AbstractPreSelector],
        maximize=False,
        S=1.0,
        workers=1,
        **kwargs,
    ):
        """
        Initializes the MarginalContributionBasedPreSelector with the given configuration.

        Args:
            config (dict): Configuration for the pre-selector.
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.base_pre_selector = base_pre_selector
        self.maximize = maximize
        self.S = S
        self.workers = workers

    def fit_transform(
        self, performance: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """
        Selects the best subset of algorithms based on the provided performance data and metric.

        This method evaluates all possible combinations of algorithms of size `n_algorithms` and selects
        the combination that optimizes the given metric (either maximizes or minimizes, depending on the
        `maximize` flag).

        Args:
            performance (pd.DataFrame | np.ndarray): A DataFrame or ndarray containing the performance
                scores of algorithms. Rows correspond to instances, columns to algorithms.

        Returns:
            pd.DataFrame | np.ndarray: The performance data of the selected subset of algorithms,
                in the same format as the input (DataFrame or ndarray).
        """

        if isinstance(performance, np.ndarray):
            performance_frame = pd.DataFrame(
                performance,
                columns=[f"Algorithm_{i}" for i in range(performance.shape[1])],
            )
            numpy = True
        else:
            performance_frame = performance
            numpy = False

        x = []
        y = []
        dfs = []

        def process(i):
            base_pre_selector = self.base_pre_selector(
                n_algorithms=i + 1,
                metric=self.metric,
                maximize=self.maximize,
            )
            pre_selected_df = base_pre_selector.fit_transform(performance_frame)
            return i, self.metric(pre_selected_df), pre_selected_df

        if self.workers > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.workers)(
                delayed(process)(i) for i in range(performance_frame.shape[1])
            )
            # Sort results by i to maintain order
            results.sort(key=lambda tup: tup[0])
            for i, metric_val, pre_selected_df in results:
                x.append(i)
                y.append(metric_val)
                dfs.append(pre_selected_df)
        else:
            for i in range(performance_frame.shape[1]):
                base_pre_selector = self.base_pre_selector(
                    n_algorithms=i + 1,
                    metric=self.metric,
                    maximize=self.maximize,
                )
                pre_selected_df = base_pre_selector.fit_transform(performance_frame)
                x.append(i)
                y.append(self.metric(pre_selected_df))
                dfs.append(pre_selected_df)

        x = np.array(x)
        y = np.array(y)

        norm_x = (x - x.min()) / (x.max() - x.min())
        norm_y = (y - y.min()) / (y.max() - y.min())

        norm_y = norm_y.max() - norm_y
        y_diff = norm_y - norm_x

        S = 1.0
        local_maximas = np.where((np.diff(np.sign(np.diff(y_diff))) < 0))[0] + 1
        local_minimas = np.where((np.diff(np.sign(np.diff(y_diff))) > 0))[0] + 1

        knees = []
        for i, lmxi in enumerate(local_maximas):
            # Threshold for this local maximum
            Tlmxi = y_diff[lmxi] - (S * np.abs(np.diff(norm_x).mean()))
            # Find the next local maximum (or end of array)
            next_lmxi = (
                local_maximas[i + 1] if i + 1 < len(local_maximas) else len(y_diff)
            )
            found_knee = False
            for j in range(lmxi + 1, next_lmxi):
                if y_diff[j] < Tlmxi:
                    knees.append((lmxi, norm_x[lmxi]))
                    found_knee = True
                    break
                # If a local minimum is reached before threshold, reset and break
                if j in local_minimas:
                    next_lmxi = 0
            # If not found, continue to next local maximum
            if found_knee:
                break

        if len(knees) == 0:
            return performance_frame

        knee_x, knee_y = knees[0]

        selected_performance = dfs[knee_x]

        if numpy:
            selected_performance = selected_performance.to_numpy()

        return selected_performance
