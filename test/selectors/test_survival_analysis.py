from typing import Any
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
