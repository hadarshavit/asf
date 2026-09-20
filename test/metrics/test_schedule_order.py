import pandas as pd
import pytest

from asf.metrics.baselines import running_time_selector_performance


@pytest.mark.parametrize(
    "feature", ["features", ("features", 50.0), ("features", None)]
)
def test_feature_cost_is_only_paid_if_step_is_reached(feature):
    performance = pd.DataFrame({"a": [1.0], "b": [9.0]}, index=["i"])
    times = pd.DataFrame({"features": [50.0]}, index=["i"])
    early = {"i": [("a", 2.0), feature, ("b", 100.0)]}
    assert running_time_selector_performance(early, performance, 100, times) == 1.0
    late = {"i": [feature, ("a", 2.0)]}
    assert running_time_selector_performance(late, performance, 100, times) == 51.0


@pytest.mark.parametrize("runtime,expected", [(40.0, 100.0), (60.0, 1000.0)])
def test_sequential_runtime_enforces_total_budget(runtime, expected):
    performance = pd.DataFrame({"a": [200.0], "b": [runtime]}, index=["i"])
    schedule = {"i": [("a", 60.0), ("b", 60.0)]}
    assert running_time_selector_performance(schedule, performance, 100) == expected


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize(
    "feature_time,expected", [(40.0, 100.0), (41.0, 1000.0), (101.0, 1000.0)]
)
def test_feature_time_counts_toward_global_cutoff(parallel, feature_time, expected):
    performance = pd.DataFrame({"a": [60.0], "b": [80.0]}, index=["i"])
    times = pd.DataFrame({"features": [feature_time]}, index=["i"])
    algorithms = [("a", 100.0), ("b", 100.0)] if parallel else [("a", 100.0)]
    assert (
        running_time_selector_performance(
            {"i": ["features", *algorithms]}, performance, 100, times
        )
        == expected
    )


def test_failed_presolver_then_capped_features_then_solver():
    performance = pd.DataFrame({"a": [200.0], "b": [10.0]}, index=["i"])
    times = pd.DataFrame({"features": [50.0]}, index=["i"])
    schedule = {"i": [("a", 5.0), ("features", 2.0), ("b", 100.0)]}
    assert running_time_selector_performance(schedule, performance, 100, times) == 17.0
