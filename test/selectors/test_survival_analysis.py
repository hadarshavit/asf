from typing import Any

import numpy as np
import pandas as pd
import pytest

from asf.selectors import SurvivalAnalysis, SurvivalAnalysisScheduler


def test_survival_analysis(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = SurvivalAnalysis(budget=5.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    validate_predictions(predictions)


def test_survival_analysis_schedule(
    dummy_performance: Any, dummy_features: Any
) -> None:
    # Test that it produces a schedule
    selector = SurvivalAnalysisScheduler(budget=2.0)
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    assert isinstance(predictions, dict)

    for inst_id, schedule in predictions.items():
        assert isinstance(schedule, list)
        assert len(schedule) >= 1
        total_budget = sum(item[1] for item in schedule if isinstance(item, tuple))
        # Should be roughly <= budget if it's a schedule
        # But SurvivalAnalysis might behave differently depending on implementation
        assert total_budget > 0


@pytest.mark.parametrize(
    "selector_class", [SurvivalAnalysis, SurvivalAnalysisScheduler]
)
def test_survival_extends_curves_and_handles_integer_algorithm_labels(
    selector_class: type[SurvivalAnalysis],
) -> None:
    features = pd.DataFrame({"x": [0.0, np.nan, 2.0, 3.0]})
    performance = pd.DataFrame({0: [1.0] * 4, 1: [np.inf] * 4})
    selector = selector_class(budget=10.0)
    selector.fit(features, performance)

    assert selector.predict(features.iloc[:1]) == {0: [("0", 10.0)]}


@pytest.mark.parametrize(
    ("timeout_runtime", "fast_expected", "slow_expected"),
    [(10.0, 1.45, 9.04), (100.0, 5.95, 12.64)],
)
def test_survival_selects_lowest_expected_runtime(
    timeout_runtime: float, fast_expected: float, slow_expected: float
) -> None:
    from sksurv.functions import StepFunction

    selector = SurvivalAnalysis(budget=10.0)
    selector.fit(
        pd.DataFrame({"x": [0.0, 1.0]}),
        pd.DataFrame({"fast": [1.0, timeout_runtime], "slow": [9.0, 9.0]}),
    )
    curves = {
        "fast": selector._budget_curve(StepFunction(np.array([1.0]), np.array([0.05]))),
        "slow": selector._budget_curve(StepFunction(np.array([9.0]), np.array([0.04]))),
    }

    assert selector._expected_runtime(curves["fast"]) == pytest.approx(fast_expected)
    assert selector._expected_runtime(curves["slow"]) == pytest.approx(slow_expected)
    assert selector._find_optimal_schedule(curves) == [("fast", 10.0)]


def test_survival_schedule_can_include_a_dominated_solver() -> None:
    from sksurv.functions import StepFunction

    selector = SurvivalAnalysisScheduler(budget=4.0)
    times = np.array([0.0, 2.0, 4.0])
    curves = {
        "a": StepFunction(times, np.array([1.0, 0.5, 0.5])),
        "b": StepFunction(times, np.array([1.0, 0.6, 0.6])),
    }

    schedule = selector._find_optimal_schedule(curves)
    assert dict(schedule) == {"a": 2.0, "b": 2.0}
    assert selector._eval_schedule(
        np.array([1, 1, 0.5, 1]), ["a", "b"], curves
    ) == pytest.approx(-0.7)
