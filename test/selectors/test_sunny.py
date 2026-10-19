from typing import Any
from asf.selectors import SUNNY


def test_sunny_selector(
    dummy_performance: Any, dummy_features: Any, validate_predictions: Any
) -> None:
    selector = SUNNY()
    selector.fit(dummy_features, dummy_performance)
    predictions = selector.predict(dummy_features)
    assert isinstance(predictions, dict)
    validate_predictions(predictions)
    # Ensure it's not just a single algorithm (SUNNY can return schedules)
    for inst_id, schedule in predictions.items():
        assert isinstance(schedule, list)
        assert len(schedule) > 0
