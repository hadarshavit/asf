from typing import cast

import numpy as np
import pandas as pd
import pytest

from asf.pre_selector.brute_force_pre_selection import BruteForcePreSelector
from asf.pre_selector.beam_search_pre_selection import BeamSearchPreSelector
from asf.pre_selector.marginal_contribution_based import (
    MarginalContributionBasedPreSelector,
)
from asf.pre_selector.sbs_pre_selection import SBSPreSelector
from asf.pre_selector.optimize_pre_selection import OptimizePreSelection
from asf.pre_selector.knee_of_the_curve_pre_selector import (
    KneeOfCurvePreSelector,
)
from asf.pre_selector.abstract_pre_selector import AbstractPreSelector


def _sum_metric(frame: pd.DataFrame) -> float:
    return frame.to_numpy().sum()


_PARALLEL_METRIC_VALUES = {1: 1.08, 2: 0.1, 3: 1.08}


def _parallel_metric(frame: pd.DataFrame) -> float:
    return _PARALLEL_METRIC_VALUES[frame.shape[1]]


def test_brute_force_selects_lowest_sum_dataframe():
    performance = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0],
            "b": [5.0, 5.0, 5.0],
            "c": [2.0, 2.0, 2.0],
        }
    )

    selector = BruteForcePreSelector(metric=_sum_metric, n_algorithms=2, maximize=False)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert list(selected_df.columns) == ["a", "c"]


def test_brute_force_returns_numpy_for_array_input():
    performance = np.array(
        [
            [1.0, 5.0, 2.0],
            [1.0, 5.0, 2.0],
            [1.0, 5.0, 2.0],
        ]
    )

    selector = BruteForcePreSelector(metric=_sum_metric, n_algorithms=2, maximize=False)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, np.ndarray)
    assert selected.shape == (3, 2)


def test_beam_search_respects_metric_minimize():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [2.0, 3.0],
            "C": [5.0, 5.0],
            "D": [10.0, 10.0],
        }
    )

    selector = BeamSearchPreSelector(
        metric=_sum_metric,
        n_algorithms=2,
        maximize=False,
        beam_width=2,
    )

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert set(selected_df.columns) == {"A", "B"}


def test_marginal_contribution_based_maximize_selects_largest_column():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [2.0, 2.0],
            "C": [3.0, 3.0],
        }
    )

    selector = MarginalContributionBasedPreSelector(
        metric=_sum_metric,
        n_algorithms=1,
        maximize=True,
    )

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert list(selected_df.columns) == ["C"]


def test_sbs_pre_selector_handles_numpy_input():
    performance = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ]
    )

    selector = SBSPreSelector(metric=_sum_metric, n_algorithms=2, maximize=True)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, np.ndarray)
    assert selected.shape == (2, 2)


def test_sbs_pre_selector_uses_metric_for_backward_elimination():
    performance = pd.DataFrame(
        {
            "A": [1.0, 100.0, 100.0],
            "B": [100.0, 1.0, 1.0],
            "C": [40.0, 40.0, 40.0],
        }
    )

    def schedule_metric(frame: pd.DataFrame) -> float:
        return float(frame.min(axis=1).max())

    selector = SBSPreSelector(
        metric=schedule_metric,
        n_algorithms=2,
        maximize=False,
    )

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert list(selected_df.columns) == ["A", "B"]


class _RecordingOptimizer:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)
        self.called_with = None

    def __call__(self, objective_function, x0=None, bounds=None, **kwargs):
        self.called_with = {
            "bounds": bounds,
            "x0": x0,
            "kwargs": kwargs,
        }
        # Call once to ensure objective is valid.
        objective_function(self.values)

        class Result:
            def __init__(self, x):
                self.x = x

        return Result(self.values)


def test_optimize_pre_selection_uses_custom_optimizer():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [2.0, 1.0],
            "C": [3.0, 3.0],
        }
    )

    optimizer = _RecordingOptimizer([0.9, 0.8, 0.1])

    selector = OptimizePreSelection(
        metric=_sum_metric,
        n_algorithms=2,
        maximize=False,
        fmin_function=optimizer,
    )

    selected = selector.fit_transform(performance)

    assert optimizer.called_with is not None
    assert optimizer.called_with["bounds"] == [(0, 1)] * 3
    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert set(selected_df.columns) == {"A", "B"}


class _DummyBasePreSelector(AbstractPreSelector):
    def __init__(self, metric, n_algorithms, maximize=False, **kwargs):
        super().__init__(n_algorithms=n_algorithms)
        self.metric = metric
        self.n_algorithms = n_algorithms
        self.maximize = maximize

    def fit_transform(self, performance):
        return performance.iloc[:, : self.n_algorithms]


def test_knee_of_curve_pre_selector_detects_knee():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [1.5, 2.5, 3.5],
            "C": [10.0, 10.0, 10.0],
        }
    )

    metric_values = {1: 1.08, 2: 0.1, 3: 1.08}

    def metric(frame: pd.DataFrame) -> float:
        return metric_values[frame.shape[1]]

    selector = KneeOfCurvePreSelector(
        metric=metric,
        base_pre_selector=_DummyBasePreSelector,
        maximize=False,
        S=1.0,
    )

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert list(selected_df.columns) == ["A", "B"]


def test_knee_of_curve_pre_selector_returns_original_when_no_knee():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [3.0, 3.0, 3.0],
        }
    )

    selector = KneeOfCurvePreSelector(
        metric=_sum_metric,
        base_pre_selector=_DummyBasePreSelector,
        maximize=False,
    )

    selected = selector.fit_transform(performance)

    pd.testing.assert_frame_equal(selected, performance)


def test_knee_of_curve_pre_selector_parallel_numpy_detects_knee():
    pytest.importorskip("joblib")

    performance = np.array(
        [
            [1.0, 1.5, 10.0],
            [2.0, 2.5, 10.0],
            [3.0, 3.5, 10.0],
        ]
    )

    selector = KneeOfCurvePreSelector(
        metric=_parallel_metric,
        base_pre_selector=_DummyBasePreSelector,
        maximize=False,
        workers=1,
    )

    selected = selector.fit_transform(performance)

    assert isinstance(selected, np.ndarray)
    np.testing.assert_allclose(selected, performance[:, :2])


def test_optimize_pre_selection_raises_with_insufficient_algorithms():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [2.0, 1.0],
            "C": [3.0, 3.0],
        }
    )

    optimizer = _RecordingOptimizer([0.9, 0.8, 0.1])

    selector = OptimizePreSelection(
        metric=_sum_metric,
        n_algorithms=5,
        maximize=False,
        fmin_function=optimizer,
    )

    with pytest.raises(ValueError):
        selector.fit_transform(performance)


def test_optimize_pre_selection_zero_algorithms_raises():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [2.0, 1.0],
        }
    )

    optimizer = _RecordingOptimizer([0.9, 0.8])

    selector = OptimizePreSelection(
        metric=_sum_metric,
        n_algorithms=0,
        maximize=False,
        fmin_function=optimizer,
    )

    with pytest.raises(ValueError):
        selector.fit_transform(performance)


def test_brute_force_maximize_returns_best_columns():
    performance = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [5.0, 6.0],
            "C": [3.0, 4.0],
        }
    )

    selector = BruteForcePreSelector(metric=_sum_metric, n_algorithms=1, maximize=True)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert list(selected_df.columns) == ["B"]


def test_sbs_pre_selector_dataframe_minimize_orders_correctly():
    performance = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [0.5, 0.5, 0.5],
        }
    )

    selector = SBSPreSelector(metric=_sum_metric, n_algorithms=2, maximize=False)

    selected = selector.fit_transform(performance)

    assert isinstance(selected, pd.DataFrame)
    selected_df = cast(pd.DataFrame, selected)
    assert list(selected_df.columns) == ["C", "A"]
